from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class LUT3D:
    """A normalized RGB 3D LUT stored in CUBE ordering.

    The table axes are blue, green, red because CUBE files change red fastest.
    RGB values are always float64 and are intentionally not clipped on input.
    """

    table: FloatArray
    title: str = "Untitled"
    domain_min: FloatArray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    domain_max: FloatArray = field(
        default_factory=lambda: np.ones(3, dtype=np.float64)
    )
    comments: tuple[str, ...] = ()
    source: Path | None = None

    def __post_init__(self) -> None:
        table = np.asarray(self.table, dtype=np.float64)
        domain_min = np.asarray(self.domain_min, dtype=np.float64)
        domain_max = np.asarray(self.domain_max, dtype=np.float64)
        if table.ndim != 4 or table.shape[-1] != 3:
            raise ValueError("A 3D LUT table must have shape (N, N, N, 3)")
        if not (table.shape[0] == table.shape[1] == table.shape[2]):
            raise ValueError("3D LUT axes must have equal sizes")
        if table.shape[0] < 2:
            raise ValueError("A 3D LUT must contain at least two points per axis")
        if domain_min.shape != (3,) or domain_max.shape != (3,):
            raise ValueError("LUT domains must contain three RGB values")
        if np.any(domain_max <= domain_min):
            raise ValueError("DOMAIN_MAX must be greater than DOMAIN_MIN")
        object.__setattr__(self, "table", table)
        object.__setattr__(self, "domain_min", domain_min)
        object.__setattr__(self, "domain_max", domain_max)

    @property
    def size(self) -> int:
        return int(self.table.shape[0])

    def normalise_input(self, rgb: FloatArray) -> FloatArray:
        rgb = np.asarray(rgb, dtype=np.float64)
        return (rgb - self.domain_min) / (self.domain_max - self.domain_min)


def cube_grid(size: int) -> FloatArray:
    """Return an RGB identity grid in CUBE row ordering."""
    if size < 2:
        raise ValueError("Grid size must be at least 2")
    axis = np.linspace(0.0, 1.0, size, dtype=np.float64)
    blue, green, red = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.stack([red, green, blue], axis=-1).reshape(-1, 3)


def identity_lut(size: int, title: str = "Identity") -> LUT3D:
    return LUT3D(cube_grid(size).reshape(size, size, size, 3), title=title)
