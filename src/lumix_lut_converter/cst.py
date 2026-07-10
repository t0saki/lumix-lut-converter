from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .interpolation import tetrahedral_interpolation
from .io import read_lut, write_cube
from .lut import FloatArray, LUT3D, cube_grid


PANASONIC_VLOG_REFERENCE = (
    "https://pro-av.panasonic.net/en/cinema_camera_varicam_eva/support/pdf/"
    "VARICAM_V-Log_V-Gamut.pdf"
)

V_GAMUT_TO_BT709 = np.asarray(
    [
        [1.806576, -0.695697, -0.110879],
        [-0.170090, 1.305955, -0.135865],
        [-0.025206, -0.154468, 1.179674],
    ],
    dtype=np.float64,
)
BT709_TO_V_GAMUT = np.linalg.inv(V_GAMUT_TO_BT709)

VLOG_CUT_LINEAR = 0.01
VLOG_CUT_ENCODED = 0.181
VLOG_B = 0.00873
VLOG_C = 0.241514
VLOG_D = 0.598206


@dataclass(frozen=True)
class CSTConversionSummary:
    output_root: Path
    adapter_path: Path
    manifest_path: Path
    converted_count: int
    copied_std_count: int


def srgb_eotf(encoded: FloatArray) -> FloatArray:
    """Decode full-range sRGB code values to linear-light BT.709 primaries."""

    encoded = np.asarray(encoded, dtype=np.float64)
    return np.where(
        encoded <= 0.04045,
        encoded / 12.92,
        np.power((encoded + 0.055) / 1.055, 2.4),
    )


def srgb_oetf(linear: FloatArray) -> FloatArray:
    linear = np.asarray(linear, dtype=np.float64)
    return np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(np.maximum(linear, 0.0), 1.0 / 2.4) - 0.055,
    )


def vlog_encode(linear_reflection: FloatArray) -> FloatArray:
    """Encode scene-linear V-Gamut values with Panasonic's official V-Log curve."""

    linear = np.asarray(linear_reflection, dtype=np.float64)
    encoded = np.where(
        linear < VLOG_CUT_LINEAR,
        5.6 * linear + 0.125,
        VLOG_C * np.log10(np.maximum(linear + VLOG_B, 1.0e-15)) + VLOG_D,
    )
    return np.clip(encoded, 0.0, 1.0)


def vlog_decode(encoded: FloatArray) -> FloatArray:
    """Decode normalized V-Log code values to scene-linear V-Gamut values."""

    encoded = np.asarray(encoded, dtype=np.float64)
    return np.where(
        encoded < VLOG_CUT_ENCODED,
        (encoded - 0.125) / 5.6,
        np.power(10.0, (encoded - VLOG_D) / VLOG_C) - VLOG_B,
    )


def srgb_to_vlog(encoded_srgb: FloatArray) -> FloatArray:
    """Convert full-range sRGB code values to full-range V-Log/V-Gamut codes."""

    linear_bt709 = srgb_eotf(encoded_srgb)
    linear_v_gamut = linear_bt709 @ BT709_TO_V_GAMUT.T
    return vlog_encode(linear_v_gamut)


def build_srgb_to_vlog_adapter(size: int = 33) -> LUT3D:
    grid = cube_grid(size)
    values = srgb_to_vlog(grid)
    return LUT3D(
        values.reshape(size, size, size, 3),
        title="sRGB Standard to Panasonic V-Log/V-Gamut CST",
    )


def convert_collection_srgb_cst(
    source_root: Path,
    output_root: Path,
    *,
    output_size: int = 33,
    std_folder: str = "5_STD-base",
) -> CSTConversionSummary:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_root}")

    adapter = build_srgb_to_vlog_adapter(output_size)
    identity = cube_grid(output_size)
    vlog_coordinates = adapter.table.reshape(-1, 3)
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
            comments = ("# Original Standard/sRGB-base LUT; only resampled/tagged",)
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
                "# Rebased analytically from V-Log/V-Gamut to Standard/sRGB input",
                "# Pipeline: sRGB EOTF -> BT.709 to V-Gamut matrix -> V-Log OETF -> source LUT",
                f"# Panasonic reference: {PANASONIC_VLOG_REFERENCE}",
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
            title=f"{source.stem} - Standard sRGB Base (analytic CST)",
        )
        write_cube(destination, output_lut, photo_style="STD", comments=comments)

    adapter_path = output_root / "_technical" / f"sRGB_to_VLog_VGamut_{output_size}.cube"
    write_cube(
        adapter_path,
        adapter,
        photo_style="STD",
        comments=(
            "# Analytic colour-space transform; no camera-fit data",
            "# Use camera Standard base, sRGB colour space and ISO 100",
            f"# Panasonic reference: {PANASONIC_VLOG_REFERENCE}",
        ),
    )

    manifest = {
        "project": "lumix-lut-converter",
        "method": (
            "Standard/sRGB code -> sRGB EOTF -> linear BT.709 RGB -> official inverse "
            "Panasonic V-Gamut-to-BT.709 matrix -> official V-Log OETF -> source creative LUT"
        ),
        "source_root": str(source_root),
        "input_photo_style": "STD",
        "input_colourspace": "sRGB / BT.709 primaries, D65",
        "input_transfer": "IEC sRGB EOTF, full range",
        "intermediate_colourspace": "Panasonic V-Gamut",
        "intermediate_transfer": "Panasonic V-Log, normalized full range",
        "output_grid": output_size,
        "panasonic_reference": PANASONIC_VLOG_REFERENCE,
        "v_gamut_to_bt709": V_GAMUT_TO_BT709.tolist(),
        "bt709_to_v_gamut": BT709_TO_V_GAMUT.tolist(),
        "vlog_constants": {
            "cut_linear": VLOG_CUT_LINEAR,
            "cut_encoded": VLOG_CUT_ENCODED,
            "b": VLOG_B,
            "c": VLOG_C,
            "d": VLOG_D,
        },
        "adapter": str(adapter_path.relative_to(output_root)),
        "converted_vlog": converted,
        "standard_base_tagged": copied_std,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return CSTConversionSummary(
        output_root=output_root,
        adapter_path=adapter_path,
        manifest_path=manifest_path,
        converted_count=len(converted),
        copied_std_count=len(copied_std),
    )
