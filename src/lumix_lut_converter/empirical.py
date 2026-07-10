from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter

from .interpolation import tetrahedral_interpolation
from .inverse import invert_lut
from .io import read_lut, sha256_file, write_cube
from .lut import LUT3D, cube_grid


CUBE_ID_RE = re.compile(r"cube_b(?P<b>\d+)_g(?P<g>\d+)_r(?P<r>\d+)")
ANCHOR_LEVELS = (0, 64, 128, 192, 255)


@dataclass(frozen=True)
class EmpiricalFitSummary:
    output_root: Path
    adapter_path: Path
    report_path: Path
    training_samples: int
    validation_samples: int


@dataclass(frozen=True)
class EmpiricalConversionSummary:
    output_root: Path
    manifest_path: Path
    converted_count: int
    copied_std_count: int


def _read_samples(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    "page_index": int(raw["page_index"]),
                    "page_filename": raw["page_filename"],
                    "style": raw["style"],
                    "file": raw["file"],
                    "kind": raw["kind"],
                    "id": raw["id"],
                    "target_rgb8": np.asarray(
                        [float(raw[f"target_{channel}"]) for channel in "rgb"]
                    ),
                    "median_rgb8": np.asarray(
                        [float(raw[f"median_{channel}"]) for channel in "rgb"]
                    ),
                    "std_rgb8": np.asarray(
                        [float(raw[f"std_{channel}"]) for channel in "rgb"]
                    ),
                }
            )
    return rows


def _anchor_arrays(
    rows: list[dict[str, Any]],
) -> dict[tuple[int, str], np.ndarray]:
    anchors: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    for row in rows:
        if row["kind"] != "anchor":
            continue
        anchors.setdefault((row["page_index"], row["style"]), {})[row["id"]] = row[
            "median_rgb8"
        ]
    return {
        key: np.stack([values[f"anchor_{level:03d}"] for level in ANCHOR_LEVELS])
        for key, values in anchors.items()
    }


def _canonical_anchors(
    anchors: dict[tuple[int, str], np.ndarray],
    style: str,
    cube_pages: list[int],
) -> np.ndarray:
    return np.median(np.stack([anchors[(page, style)] for page in cube_pages]), axis=0)


def _monotonic_mapper(observed: np.ndarray, reference: np.ndarray) -> PchipInterpolator:
    order = np.argsort(observed)
    observed = np.asarray(observed, dtype=np.float64)[order]
    reference = np.asarray(reference, dtype=np.float64)[order]
    unique, inverse = np.unique(observed, return_inverse=True)
    collapsed = np.zeros_like(unique)
    counts = np.zeros_like(unique)
    for index, group in enumerate(inverse):
        collapsed[group] += reference[index]
        counts[group] += 1
    collapsed /= counts
    collapsed = np.maximum.accumulate(collapsed)
    if len(unique) < 2:
        raise ValueError("Anchor response contains fewer than two unique code values")
    return PchipInterpolator(unique, collapsed, extrapolate=True)


def _normalise_rows(
    rows: list[dict[str, Any]],
    anchors: dict[tuple[int, str], np.ndarray],
    canonical: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    mappers: dict[tuple[int, str], tuple[PchipInterpolator, ...]] = {}
    for key, observed in anchors.items():
        _, style = key
        mappers[key] = tuple(
            _monotonic_mapper(observed[:, channel], canonical[style][:, channel])
            for channel in range(3)
        )

    normalised: list[dict[str, Any]] = []
    for row in rows:
        values = row["median_rgb8"]
        corrected = np.asarray(
            [
                mappers[(row["page_index"], row["style"])][channel](values[channel])
                for channel in range(3)
            ],
            dtype=np.float64,
        )
        normalised.append({**row, "normalised_rgb": np.clip(corrected / 255.0, 0.0, 1.0)})
    return normalised


def _build_forward_lut(
    rows: list[dict[str, Any]],
    *,
    style: str,
    cube_size: int,
    smoothing_sigma: float,
) -> LUT3D:
    table = np.full((cube_size, cube_size, cube_size, 3), np.nan, dtype=np.float64)
    for row in rows:
        if row["style"] != style or row["kind"] != "patch":
            continue
        match = CUBE_ID_RE.fullmatch(row["id"])
        if not match:
            continue
        blue = int(match.group("b"))
        green = int(match.group("g"))
        red = int(match.group("r"))
        table[blue, green, red] = row["normalised_rgb"]
    if np.any(~np.isfinite(table)):
        missing = int(np.count_nonzero(~np.isfinite(table[..., 0])))
        raise ValueError(f"Measured {style} cube is missing {missing} nodes")
    if smoothing_sigma > 0:
        table = gaussian_filter(
            table,
            sigma=(smoothing_sigma, smoothing_sigma, smoothing_sigma, 0.0),
            mode="nearest",
        )
    return LUT3D(table, title=f"Measured {style} camera response")


def _paired_samples(
    rows: list[dict[str, Any]],
    page_filter: set[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grouped: dict[tuple[int, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row["kind"] != "patch" or row["page_index"] not in page_filter:
            continue
        grouped.setdefault((row["page_index"], row["id"]), {})[row["style"]] = row

    standard: list[np.ndarray] = []
    vlog: list[np.ndarray] = []
    noise: list[float] = []
    clipped: list[bool] = []
    for styles in grouped.values():
        if "standard" not in styles or "vlog" not in styles:
            continue
        standard.append(styles["standard"]["normalised_rgb"])
        vlog.append(styles["vlog"]["normalised_rgb"])
        noise.append(
            max(
                float(np.max(styles["standard"]["std_rgb8"])),
                float(np.max(styles["vlog"]["std_rgb8"])),
            )
        )
        clipped.append(bool(np.any(styles["standard"]["median_rgb8"] >= 254.0)))
    return np.stack(standard), np.stack(vlog), np.asarray(noise), np.asarray(clipped)


def _error_metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    error = np.linalg.norm(np.asarray(reference) - np.asarray(actual), axis=1)
    channel_error = np.max(np.abs(np.asarray(reference) - np.asarray(actual)), axis=1)
    return {
        "rgb_norm_mean": float(np.mean(error)),
        "rgb_norm_p50": float(np.percentile(error, 50)),
        "rgb_norm_p95": float(np.percentile(error, 95)),
        "rgb_norm_p99": float(np.percentile(error, 99)),
        "rgb_norm_max": float(np.max(error)),
        "max_channel_code_mean": float(np.mean(channel_error) * 255.0),
        "max_channel_code_p95": float(np.percentile(channel_error, 95) * 255.0),
        "max_channel_code_max": float(np.max(channel_error) * 255.0),
    }


def fit_empirical_std_to_vlog(
    samples_path: Path,
    manifest_path: Path,
    output_root: Path,
    *,
    output_size: int = 33,
    smoothing_sigma: float = 0.15,
    max_sample_std: float = 28.0,
) -> EmpiricalFitSummary:
    samples_path = samples_path.resolve()
    manifest_path = manifest_path.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text())
    cube_size = len(manifest["cube_levels"])
    cube_pages = list(range(4, 4 + cube_size))
    validation_pages = set(range(4))

    rows = _read_samples(samples_path)
    anchors = _anchor_arrays(rows)
    canonical = {
        style: _canonical_anchors(anchors, style, cube_pages)
        for style in ("vlog", "like709", "standard")
    }
    normalised = _normalise_rows(rows, anchors, canonical)
    standard_forward = _build_forward_lut(
        normalised,
        style="standard",
        cube_size=cube_size,
        smoothing_sigma=smoothing_sigma,
    )
    vlog_forward = _build_forward_lut(
        normalised,
        style="vlog",
        cube_size=cube_size,
        smoothing_sigma=smoothing_sigma,
    )

    output_grid = cube_grid(output_size)
    inverse = invert_lut(
        standard_forward,
        output_grid,
        max_iterations=24,
        tolerance=1.0e-7,
    )
    adapter_values = tetrahedral_interpolation(vlog_forward, inverse.coordinates)
    adapter_table = adapter_values.reshape(output_size, output_size, output_size, 3)
    for index in range(output_size):
        neutral = float(np.mean(adapter_table[index, index, index]))
        adapter_table[index, index, index] = neutral
    adapter = LUT3D(
        adapter_table,
        title="LUMIX S9 measured Standard to V-Log adapter",
    )
    capture_report_path = samples_path.with_name("report.json")
    capture_quality: dict[str, Any] = {}
    capture_comment = "# Capture metadata report unavailable"
    if capture_report_path.exists():
        capture_report = json.loads(capture_report_path.read_text())
        metadata = [frame.get("metadata", {}) for frame in capture_report.get("frames", [])]
        wb_modes = sorted({item.get("WhiteBalance") for item in metadata if item.get("WhiteBalance")})
        kelvin = sorted({item.get("ColorTempKelvin") for item in metadata if item.get("ColorTempKelvin")})
        focus_modes = sorted({item.get("FocusMode") for item in metadata if item.get("FocusMode")})
        exposure_times = sorted({item.get("ExposureTime") for item in metadata if item.get("ExposureTime")})
        capture_quality = {
            "white_balance_modes": wb_modes,
            "colour_temperatures_kelvin": kelvin,
            "focus_modes": focus_modes,
            "exposure_times": exposure_times,
        }
        if wb_modes == ["Kelvin"] and len(kelvin) == 1:
            capture_comment = f"# Capture WB fixed at {kelvin[0]} K"
        else:
            capture_comment = f"# Capture WB modes: {', '.join(wb_modes) or 'unknown'}"
    adapter_path = output_root / f"STD_to_VLog_camera_fit_{output_size}.cube"
    write_cube(
        adapter_path,
        adapter,
        photo_style="STD",
        comments=(
            "# Temporary empirical fit from paired LUMIX S9 JPEG captures",
            "# Per-frame exposure/WB normalised with five neutral anchors",
            capture_comment,
            "# Exact neutral axis constrained to remain neutral",
        ),
    )

    training_standard, training_vlog, training_noise, training_clipped = _paired_samples(
        normalised, set(cube_pages)
    )
    validation_standard, validation_vlog, validation_noise, validation_clipped = _paired_samples(
        normalised, validation_pages
    )
    training_keep = training_noise <= max_sample_std
    validation_keep = validation_noise <= max_sample_std
    training_prediction = tetrahedral_interpolation(adapter, training_standard[training_keep])
    validation_prediction = tetrahedral_interpolation(adapter, validation_standard[validation_keep])
    training_nonclipped = training_keep & ~training_clipped
    validation_nonclipped = validation_keep & ~validation_clipped
    training_nonclipped_prediction = tetrahedral_interpolation(
        adapter, training_standard[training_nonclipped]
    )
    validation_nonclipped_prediction = tetrahedral_interpolation(
        adapter, validation_standard[validation_nonclipped]
    )

    inverse_metrics = _error_metrics(output_grid, inverse.recovered)
    inverse_channel_error = np.max(np.abs(output_grid - inverse.recovered), axis=1)
    report = {
        "method": (
            "Five-anchor per-frame channel normalisation; measured 9x9x9 Standard and "
            "V-Log forward LUTs; smoothed tetrahedral forward models; bounded numerical "
            "inverse of Standard; composition into V-Log"
        ),
        "samples": str(samples_path),
        "manifest": str(manifest_path),
        "output_grid": output_size,
        "measured_cube_grid": cube_size,
        "smoothing_sigma": smoothing_sigma,
        "max_sample_std_rgb8": max_sample_std,
        "capture_quality": capture_quality,
        "neutral_axis_enforced": True,
        "canonical_anchors_rgb8": {
            style: values.tolist() for style, values in canonical.items()
        },
        "inverse_standard": {
            **inverse_metrics,
            "coverage_within_1_code": float(np.mean(inverse_channel_error <= 1.0 / 255.0)),
            "coverage_within_3_codes": float(np.mean(inverse_channel_error <= 3.0 / 255.0)),
            "coverage_within_5_codes": float(np.mean(inverse_channel_error <= 5.0 / 255.0)),
            "iterations": inverse.iterations,
        },
        "training_cube": {
            "samples_total": int(len(training_standard)),
            "samples_kept": int(np.count_nonzero(training_keep)),
            **_error_metrics(training_vlog[training_keep], training_prediction),
            "nonclipped_samples": int(np.count_nonzero(training_nonclipped)),
            "nonclipped_metrics": _error_metrics(
                training_vlog[training_nonclipped], training_nonclipped_prediction
            ),
        },
        "validation_gray_ramps": {
            "samples_total": int(len(validation_standard)),
            "samples_kept": int(np.count_nonzero(validation_keep)),
            **_error_metrics(validation_vlog[validation_keep], validation_prediction),
            "nonclipped_samples": int(np.count_nonzero(validation_nonclipped)),
            "nonclipped_metrics": _error_metrics(
                validation_vlog[validation_nonclipped], validation_nonclipped_prediction
            ),
        },
        "warning": (
            "The fitted input volume is the gamut reachable by the measured Standard JPEG "
            "pipeline. Full-cube nodes outside that volume use the nearest bounded inverse."
        ),
    }
    report_path = output_root / "fit_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return EmpiricalFitSummary(
        output_root=output_root,
        adapter_path=adapter_path,
        report_path=report_path,
        training_samples=int(np.count_nonzero(training_keep)),
        validation_samples=int(np.count_nonzero(validation_keep)),
    )


def convert_collection_with_empirical_adapter(
    source_root: Path,
    adapter_path: Path,
    output_root: Path,
    *,
    output_size: int = 33,
    std_folder: str = "5_STD-base",
) -> EmpiricalConversionSummary:
    source_root = source_root.resolve()
    adapter_path = adapter_path.resolve()
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_root}")

    adapter = read_lut(adapter_path)
    identity = cube_grid(output_size)
    vlog_coordinates = tetrahedral_interpolation(adapter, identity)
    converted: list[dict[str, Any]] = []
    copied_std: list[dict[str, Any]] = []

    for source in sorted(source_root.glob("*/*")):
        if source.suffix.lower() not in {".cube", ".vlt"}:
            continue
        relative = source.relative_to(source_root)
        destination = output_root / relative.with_suffix(".cube")
        source_lut = read_lut(source)
        if relative.parts[0] == std_folder:
            values = tetrahedral_interpolation(source_lut, identity)
            comments = ("# Original Standard-base LUT; only resampled/tagged",)
            copied_std.append(
                {
                    "source": str(relative),
                    "source_grid": source_lut.size,
                    "output": str(destination.relative_to(output_root)),
                }
            )
        else:
            values = tetrahedral_interpolation(source_lut, vlog_coordinates)
            comments = (
                "# Rebased from V-Log to Standard with paired LUMIX S9 camera measurements",
                f"# Empirical adapter: {adapter_path.name}",
                f"# Source: {relative}",
            )
            converted.append(
                {
                    "source": str(relative),
                    "source_grid": source_lut.size,
                    "output": str(destination.relative_to(output_root)),
                }
            )
        output_lut = LUT3D(
            values.reshape(output_size, output_size, output_size, 3),
            title=f"{source.stem} - Standard Base (camera fit)",
        )
        write_cube(destination, output_lut, photo_style="STD", comments=comments)

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "project": "lumix-lut-converter",
        "method": (
            "Standard camera RGB -> empirical measured Standard-to-V-Log adapter -> "
            "source V-Log creative LUT"
        ),
        "source_root": str(source_root),
        "adapter": str(adapter_path),
        "adapter_sha256": sha256_file(adapter_path),
        "output_grid": output_size,
        "lumix_photo_style": "STD",
        "converted_vlog": converted,
        "standard_base_tagged": copied_std,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return EmpiricalConversionSummary(
        output_root=output_root,
        manifest_path=manifest_path,
        converted_count=len(converted),
        copied_std_count=len(copied_std),
    )
