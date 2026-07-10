from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .interpolation import tetrahedral_interpolation
from .inverse import invert_v709_lut
from .io import read_lut, read_reference_zip, sha256_file, write_cube
from .lut import LUT3D, cube_grid
from .metrics import ValidationMetrics, validate_inverse


@dataclass(frozen=True)
class ConversionSummary:
    converted_count: int
    copied_std_count: int
    output_root: Path
    metrics: ValidationMetrics
    manifest_path: Path


def convert_collection(
    source_root: Path,
    reference_zip: Path,
    output_root: Path,
    *,
    output_size: int = 33,
    std_folder: str = "5_STD-base",
    black_code: int = 64,
    white_code: int = 940,
    denominator: int = 1023,
    resume: bool = False,
) -> ConversionSummary:
    source_root = source_root.resolve()
    reference_zip = reference_zip.resolve()
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()) and not resume:
        raise FileExistsError(f"Output directory is not empty: {output_root}")

    reference, reference_entry = read_reference_zip(reference_zip)
    inverse = invert_v709_lut(
        reference,
        output_size=output_size,
        black_code=black_code,
        white_code=white_code,
        denominator=denominator,
    )
    metrics = validate_inverse(
        inverse,
        black_code=black_code,
        white_code=white_code,
        denominator=denominator,
    )

    converted: list[dict[str, str | int]] = []
    copied_std: list[dict[str, str | int]] = []
    identity_points = cube_grid(output_size)

    for source in sorted(source_root.glob("*/*")):
        if source.suffix.lower() not in {".cube", ".vlt"}:
            continue
        relative = source.relative_to(source_root)
        destination = output_root / relative.with_suffix(".cube")
        source_lut = read_lut(source)

        if relative.parts[0] == std_folder:
            values = tetrahedral_interpolation(source_lut, identity_points)
            output_lut = LUT3D(
                values.reshape(output_size, output_size, output_size, 3),
                title=f"{source.stem} - Standard Base",
            )
            write_cube(
                destination,
                output_lut,
                photo_style="STD",
                comments=("# Original Standard-base LUT; only resampled/tagged",),
            )
            copied_std.append(
                {
                    "source": str(relative),
                    "source_grid": source_lut.size,
                    "output": str(destination.relative_to(output_root)),
                }
            )
            continue

        values = tetrahedral_interpolation(source_lut, inverse.vlog_coordinates)
        output_lut = LUT3D(
            values.reshape(output_size, output_size, output_size, 3),
            title=f"{source.stem} - Like709 Base",
        )
        write_cube(
            destination,
            output_lut,
            photo_style="709L",
            comments=(
                "# Rebased from V-Log using Panasonic official VLog_to_V709_forV35",
                f"# Source: {relative}",
            ),
        )
        converted.append(
            {
                "source": str(relative),
                "source_grid": source_lut.size,
                "output": str(destination.relative_to(output_root)),
            }
        )

    manifest = {
        "project": "lumix-lut-converter",
        "method": (
            "Like709 full-range RGB -> legal-range V709 -> bounded numerical inverse "
            "of Panasonic official VLog_to_V709 LUT -> source creative LUT"
        ),
        "interpolation": "tetrahedral, float64",
        "output_grid": output_size,
        "lumix_photo_style": "709L",
        "legal_range": {
            "black_code": black_code,
            "white_code": white_code,
            "denominator": denominator,
        },
        "reference_zip": str(reference_zip),
        "reference_zip_sha256": sha256_file(reference_zip),
        "reference_entry": reference_entry,
        "validation": metrics.to_dict(),
        "converted_like709": converted,
        "standard_base_tagged": copied_std,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    return ConversionSummary(
        converted_count=len(converted),
        copied_std_count=len(copied_std),
        output_root=output_root,
        metrics=metrics,
        manifest_path=manifest_path,
    )
