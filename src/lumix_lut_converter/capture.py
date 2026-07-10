from __future__ import annotations

import csv
import json
import math
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage


STYLE_ORDER = ("vlog", "like709", "standard")


@dataclass(frozen=True)
class Fiducial:
    x: float
    y: float
    width: float
    height: float
    score: float
    threshold: float


@dataclass(frozen=True)
class CaptureAnalysisSummary:
    capture_root: Path
    output_root: Path
    frame_count: int
    page_count: int
    sample_count: int
    report_path: Path
    samples_path: Path


def _capture_number(path: Path) -> int:
    matches = re.findall(r"\d+", path.stem)
    if not matches:
        raise ValueError(f"Capture filename has no sequence number: {path.name}")
    return int("".join(matches))


def discover_captures(capture_root: Path) -> list[Path]:
    captures = [
        path
        for path in capture_root.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
    ]
    return sorted(captures, key=_capture_number)


def _grayscale(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image, dtype=np.float64)
    return rgb @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)


def _candidate_score(
    gray: np.ndarray,
    labels: np.ndarray,
    label: int,
    slices: tuple[slice, slice],
) -> tuple[float, float]:
    y_slice, x_slice = slices
    height = y_slice.stop - y_slice.start
    width = x_slice.stop - x_slice.start
    component = labels[slices] == label
    area = int(np.count_nonzero(component))
    fill = area / float(width * height)
    aspect = width / float(height)

    box = gray[slices]
    inner_y0 = max(0, height // 3)
    inner_y1 = min(height, height - height // 3)
    inner_x0 = max(0, width // 3)
    inner_x1 = min(width, width - width // 3)
    inner = box[inner_y0:inner_y1, inner_x0:inner_x1]
    ring_mask = np.ones((height, width), dtype=bool)
    ring_mask[inner_y0:inner_y1, inner_x0:inner_x1] = False
    ring = box[ring_mask]
    ring_contrast = max(float(np.mean(ring) - np.mean(inner)), 0.0)

    shape_score = math.exp(-2.0 * abs(math.log(max(aspect, 1.0e-6))))
    contrast_score = 1.0 + ring_contrast
    return area * max(fill, 0.10) * shape_score * contrast_score, ring_contrast


def locate_fiducials(
    image: Image.Image,
    *,
    detection_width: int = 1500,
) -> tuple[Fiducial, Fiducial, Fiducial, Fiducial]:
    """Locate the four bright-ring/dark-centre markers in TL, TR, BL, BR order."""

    image = ImageOps.exif_transpose(image).convert("RGB")
    scale = min(1.0, detection_width / image.width)
    detection_size = (round(image.width * scale), round(image.height * scale))
    small = image.resize(detection_size, Image.Resampling.BILINEAR)
    gray = _grayscale(small)
    height, width = gray.shape
    regions = (
        (0, round(0.22 * height), 0, round(0.18 * width)),
        (0, round(0.22 * height), round(0.82 * width), width),
        (round(0.75 * height), height, 0, round(0.18 * width)),
        (round(0.75 * height), height, round(0.82 * width), width),
    )

    detections: list[Fiducial] = []
    for corner, (y0, y1, x0, x1) in zip(("TL", "TR", "BL", "BR"), regions, strict=True):
        roi = gray[y0:y1, x0:x1]
        threshold = float(np.percentile(roi, 99.0))
        mask = ndimage.binary_closing(roi >= threshold, structure=np.ones((3, 3)))
        labels, _ = ndimage.label(mask)

        candidates: list[tuple[float, Fiducial]] = []
        for label, slices in enumerate(ndimage.find_objects(labels), start=1):
            if slices is None:
                continue
            y_slice, x_slice = slices
            candidate_height = y_slice.stop - y_slice.start
            candidate_width = x_slice.stop - x_slice.start
            aspect = candidate_width / float(candidate_height)
            area = int(np.count_nonzero(labels[slices] == label))
            fill = area / float(candidate_width * candidate_height)
            if not (
                7 <= candidate_height <= 80
                and 7 <= candidate_width <= 80
                and 0.58 <= aspect <= 1.72
                and area >= 20
                and fill >= 0.12
            ):
                continue

            score, _ = _candidate_score(roi, labels, label, slices)
            centre_x = x0 + (x_slice.start + x_slice.stop - 1) / 2.0
            centre_y = y0 + (y_slice.start + y_slice.stop - 1) / 2.0
            candidates.append(
                (
                    score,
                    Fiducial(
                        x=centre_x / scale,
                        y=centre_y / scale,
                        width=candidate_width / scale,
                        height=candidate_height / scale,
                        score=score,
                        threshold=threshold,
                    ),
                )
            )

        if not candidates:
            raise RuntimeError(f"Unable to locate {corner} fiducial")
        detections.append(max(candidates, key=lambda item: item[0])[1])

    return tuple(detections)  # type: ignore[return-value]


def target_fiducial_centres(width: int, height: int) -> np.ndarray:
    size = max(30, int(min(width, height) * 0.026))
    inset = max(20, size // 2)
    half = size / 2.0
    return np.asarray(
        [
            (inset + half, inset + half),
            (width - inset - size + half, inset + half),
            (inset + half, height - inset - size + half),
            (width - inset - size + half, height - inset - size + half),
        ],
        dtype=np.float64,
    )


def homography_from_points(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    destination = np.asarray(destination, dtype=np.float64)
    if source.shape != (4, 2) or destination.shape != (4, 2):
        raise ValueError("Homography requires four 2D source and destination points")

    matrix: list[list[float]] = []
    values: list[float] = []
    for (x, y), (u, v) in zip(source, destination, strict=True):
        matrix.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        values.append(u)
        matrix.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        values.append(v)
    coefficients = np.linalg.solve(np.asarray(matrix), np.asarray(values))
    return np.append(coefficients, 1.0).reshape(3, 3)


def transform_points(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    homogeneous = np.column_stack([points.reshape(-1, 2), np.ones(points.size // 2)])
    mapped = homogeneous @ np.asarray(homography, dtype=np.float64).T
    mapped = mapped[:, :2] / mapped[:, 2:3]
    return mapped.reshape(points.shape)


def _sampling_grid(rect: list[int] | tuple[int, int, int, int], samples_per_axis: int) -> np.ndarray:
    x, y, width, height = rect
    xs = np.linspace(x + 0.08 * width, x + 0.92 * width, samples_per_axis)
    ys = np.linspace(y + 0.08 * height, y + 0.92 * height, samples_per_axis)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.column_stack([grid_x.ravel(), grid_y.ravel()])


def sample_page(
    image: Image.Image,
    homography: np.ndarray,
    page: dict[str, Any],
    *,
    samples_per_axis: int = 17,
) -> list[dict[str, Any]]:
    rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.uint8)
    entries: list[tuple[str, dict[str, Any], np.ndarray]] = []
    for kind in ("patches", "anchors"):
        for patch in page[kind]:
            entries.append((kind[:-2] if kind.endswith("es") else kind[:-1], patch, _sampling_grid(
                patch["sample_rect"], samples_per_axis
            )))

    lengths = [len(entry[2]) for entry in entries]
    target_points = np.concatenate([entry[2] for entry in entries], axis=0)
    image_points = transform_points(homography, target_points)
    if (
        np.any(image_points[:, 0] < 0)
        or np.any(image_points[:, 0] > rgb.shape[1] - 1)
        or np.any(image_points[:, 1] < 0)
        or np.any(image_points[:, 1] > rgb.shape[0] - 1)
    ):
        raise RuntimeError("Projected sample coordinates fall outside the capture")

    coordinates = np.vstack([image_points[:, 1], image_points[:, 0]])
    sampled = np.column_stack(
        [
            ndimage.map_coordinates(rgb[..., channel], coordinates, order=1, mode="nearest")
            for channel in range(3)
        ]
    ).astype(np.float64)

    results: list[dict[str, Any]] = []
    offset = 0
    for (kind, patch, _), length in zip(entries, lengths, strict=True):
        values = sampled[offset : offset + length]
        offset += length
        median = np.median(values, axis=0)
        results.append(
            {
                "kind": kind,
                "id": patch["id"],
                "target_rgb8": patch["rgb8"],
                "median_rgb8": median.tolist(),
                "mean_rgb8": np.mean(values, axis=0).tolist(),
                "std_rgb8": np.std(values, axis=0).tolist(),
                "mad_rgb8": np.median(np.abs(values - median), axis=0).tolist(),
                "p01_rgb8": np.percentile(values, 1, axis=0).tolist(),
                "p99_rgb8": np.percentile(values, 99, axis=0).tolist(),
                "sample_count": length,
            }
        )
    return results


def _read_metadata(paths: list[Path]) -> dict[str, dict[str, Any]]:
    command = [
        "exiftool",
        "-j",
        "-FileName",
        "-PhotoStyle",
        "-ExposureTime",
        "-FNumber",
        "-ISO",
        "-FocalLength",
        "-WhiteBalance",
        "-ColorTempKelvin",
        "-FocusMode",
        "-ImageStabilization",
        "-LUT1Name",
        "-LUT1Opacity",
        "-LUT2Name",
        "-LUT2Opacity",
        "-HighlightWarning",
        *[str(path) for path in paths],
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {}
    records = json.loads(completed.stdout)
    return {record["FileName"]: record for record in records}


def _style_matches(style: str, metadata_style: str | None) -> bool | None:
    if metadata_style is None:
        return None
    expected = {
        "vlog": "v-log",
        "like709": "like709",
        "standard": "standard",
    }[style]
    return expected in metadata_style.lower()


def _draw_overlay(
    image: Image.Image,
    fiducials: tuple[Fiducial, Fiducial, Fiducial, Fiducial],
    homography: np.ndarray,
    canvas_width: int,
    canvas_height: int,
    output: Path,
) -> None:
    image = ImageOps.exif_transpose(image).convert("RGB")
    scale = min(1.0, 1600 / image.width)
    preview = image.resize(
        (round(image.width * scale), round(image.height * scale)),
        Image.Resampling.LANCZOS,
    )
    draw = ImageDraw.Draw(preview)
    projected = transform_points(
        homography,
        np.asarray([(0, 0), (canvas_width, 0), (canvas_width, canvas_height), (0, canvas_height)]),
    )
    polygon = [(float(x * scale), float(y * scale)) for x, y in projected]
    draw.line([*polygon, polygon[0]], fill=(0, 255, 255), width=4)
    for index, fiducial in enumerate(fiducials, start=1):
        x = fiducial.x * scale
        y = fiducial.y * scale
        radius = 10
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(255, 40, 40), width=4)
        draw.text((x + 14, y - 14), str(index), fill=(255, 255, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    preview.save(output, quality=90, subsampling=0)


def analyze_calibration_captures(
    capture_root: Path,
    manifest_path: Path,
    output_root: Path,
    *,
    style_order: tuple[str, str, str] = STYLE_ORDER,
) -> CaptureAnalysisSummary:
    capture_root = capture_root.resolve()
    manifest_path = manifest_path.resolve()
    output_root = output_root.resolve()
    manifest = json.loads(manifest_path.read_text())
    pages = manifest["pages"]
    required = len(pages) * len(style_order)
    captures = discover_captures(capture_root)
    if len(captures) < required:
        raise ValueError(f"Need at least {required} JPEGs, found {len(captures)}")
    captures = captures[:required]
    metadata = _read_metadata(captures)

    canvas_width = int(manifest["canvas"]["width"])
    canvas_height = int(manifest["canvas"]["height"])
    target_centres = target_fiducial_centres(canvas_width, canvas_height)
    output_root.mkdir(parents=True, exist_ok=True)
    overlays_root = output_root / "overlays"
    rows: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []

    for page_index, page in enumerate(pages):
        for style_index, style in enumerate(style_order):
            path = captures[page_index * len(style_order) + style_index]
            with Image.open(path) as image:
                fiducials = locate_fiducials(image)
                destination = np.asarray([(item.x, item.y) for item in fiducials])
                homography = homography_from_points(target_centres, destination)
                samples = sample_page(image, homography, page)
                _draw_overlay(
                    image,
                    fiducials,
                    homography,
                    canvas_width,
                    canvas_height,
                    overlays_root / f"{page_index:02d}_{style}_{path.stem}.jpg",
                )

            frame_metadata = metadata.get(path.name, {})
            style_match = _style_matches(style, frame_metadata.get("PhotoStyle"))
            projected_corners = transform_points(
                homography,
                np.asarray([(0, 0), (canvas_width, 0), (0, canvas_height), (canvas_width, canvas_height)]),
            )
            frames.append(
                {
                    "page_index": page_index,
                    "page_filename": page["filename"],
                    "style": style,
                    "file": path.name,
                    "metadata_style_match": style_match,
                    "metadata": frame_metadata,
                    "fiducials": [asdict(item) for item in fiducials],
                    "homography": homography.tolist(),
                    "projected_corners": projected_corners.tolist(),
                    "sample_count": len(samples),
                }
            )
            for sample in samples:
                row = {
                    "page_index": page_index,
                    "page_filename": page["filename"],
                    "style": style,
                    "file": path.name,
                    **sample,
                }
                rows.append(row)

    samples_path = output_root / "samples.csv"
    fieldnames = [
        "page_index",
        "page_filename",
        "style",
        "file",
        "kind",
        "id",
        "target_r",
        "target_g",
        "target_b",
        "median_r",
        "median_g",
        "median_b",
        "mean_r",
        "mean_g",
        "mean_b",
        "std_r",
        "std_g",
        "std_b",
        "mad_r",
        "mad_g",
        "mad_b",
        "p01_r",
        "p01_g",
        "p01_b",
        "p99_r",
        "p99_g",
        "p99_b",
        "sample_count",
    ]
    with samples_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flattened = {
                "page_index": row["page_index"],
                "page_filename": row["page_filename"],
                "style": row["style"],
                "file": row["file"],
                "kind": row["kind"],
                "id": row["id"],
                "sample_count": row["sample_count"],
            }
            for prefix, values in (
                ("target", row["target_rgb8"]),
                ("median", row["median_rgb8"]),
                ("mean", row["mean_rgb8"]),
                ("std", row["std_rgb8"]),
                ("mad", row["mad_rgb8"]),
                ("p01", row["p01_rgb8"]),
                ("p99", row["p99_rgb8"]),
            ):
                for channel, value in zip(("r", "g", "b"), values, strict=True):
                    flattened[f"{prefix}_{channel}"] = value
            writer.writerow(flattened)

    fiducial_scores = [item["score"] for frame in frames for item in frame["fiducials"]]
    style_checks = [frame["metadata_style_match"] for frame in frames]
    report = {
        "capture_root": str(capture_root),
        "manifest": str(manifest_path),
        "frame_count": len(frames),
        "page_count": len(pages),
        "sample_count": len(rows),
        "ignored_jpegs": [path.name for path in discover_captures(capture_root)[required:]],
        "summary": {
            "fiducial_score_min": min(fiducial_scores),
            "fiducial_score_median": float(np.median(fiducial_scores)),
            "metadata_style_matches": sum(value is True for value in style_checks),
            "metadata_style_mismatches": sum(value is False for value in style_checks),
            "metadata_style_unknown": sum(value is None for value in style_checks),
        },
        "frames": frames,
    }
    report_path = output_root / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return CaptureAnalysisSummary(
        capture_root=capture_root,
        output_root=output_root,
        frame_count=len(frames),
        page_count=len(pages),
        sample_count=len(rows),
        report_path=report_path,
        samples_path=samples_path,
    )
