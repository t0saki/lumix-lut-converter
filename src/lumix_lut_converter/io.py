from __future__ import annotations

import hashlib
import re
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .lut import LUT3D


TITLE_RE = re.compile(r'^TITLE\s+["\']?(.*?)["\']?\s*$')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_lut_text(text: str, suffix: str, source: Path | None = None) -> LUT3D:
    size: int | None = None
    title = source.stem if source is not None else "Untitled"
    domain_min = np.zeros(3, dtype=np.float64)
    domain_max = np.ones(3, dtype=np.float64)
    comments: list[str] = []
    rows: list[list[float]] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            comments.append(line)
            continue
        title_match = TITLE_RE.match(line)
        if title_match:
            title = title_match.group(1)
            continue
        if line.startswith("LUT_3D_SIZE"):
            size = int(line.split()[-1])
            continue
        if line.startswith("LUT_1D_SIZE"):
            raise ValueError("1D LUTs are not supported")
        if line.startswith("DOMAIN_MIN"):
            domain_min = np.asarray([float(x) for x in line.split()[1:4]])
            continue
        if line.startswith("DOMAIN_MAX"):
            domain_max = np.asarray([float(x) for x in line.split()[1:4]])
            continue
        fields = line.split()
        if len(fields) == 3:
            try:
                rows.append([float(field) for field in fields])
            except ValueError as error:
                raise ValueError(f"Invalid LUT row: {raw_line}") from error

    if size is None:
        raise ValueError("LUT_3D_SIZE is missing")
    expected = size**3
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} LUT rows, found {len(rows)}")

    values = np.asarray(rows, dtype=np.float64)
    if suffix.lower() == ".vlt":
        values /= 4095.0
    table = values.reshape(size, size, size, 3)
    return LUT3D(
        table=table,
        title=title,
        domain_min=domain_min,
        domain_max=domain_max,
        comments=tuple(comments),
        source=source,
    )


def read_lut(path: Path) -> LUT3D:
    if path.suffix.lower() not in {".cube", ".vlt"}:
        raise ValueError(f"Unsupported LUT format: {path.suffix}")
    return parse_lut_text(path.read_text(errors="replace"), path.suffix, path)


def read_reference_zip(path: Path) -> tuple[LUT3D, str]:
    with ZipFile(path) as archive:
        names = sorted(name for name in archive.namelist() if name.lower().endswith(".cube"))
        if len(names) != 1:
            raise ValueError(f"Expected exactly one CUBE reference LUT, found {len(names)}")
        name = names[0]
        text = archive.read(name).decode("utf-8", errors="replace")
    return parse_lut_text(text, ".cube"), name


def write_cube(
    path: Path,
    lut: LUT3D,
    *,
    photo_style: str,
    comments: tuple[str, ...] = (),
    decimals: int = 10,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f'TITLE "{lut.title}"',
        f"#LUMIXPHOTOSTYLE {photo_style}",
        *comments,
        f"LUT_3D_SIZE {lut.size}",
        "",
    ]
    formatter = f"{{:.{decimals}f}} {{:.{decimals}f}} {{:.{decimals}f}}"
    for row in lut.table.reshape(-1, 3):
        # LUMIX camera LUTs are defined over normalized display code values.
        clipped = np.clip(row, 0.0, 1.0)
        lines.append(formatter.format(*clipped))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
