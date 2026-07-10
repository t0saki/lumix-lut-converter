from __future__ import annotations

import numpy as np

from .lut import FloatArray, LUT3D


def tetrahedral_interpolation(lut: LUT3D, rgb: FloatArray) -> FloatArray:
    """Sample *lut* with vectorised tetrahedral interpolation.

    Tetrahedral interpolation is preferred over trilinear interpolation for
    colour LUTs because it avoids the trilinear cube-centre ambiguity and is
    the common high-quality implementation used by grading systems.
    """

    original_shape = np.asarray(rgb).shape
    if not original_shape or original_shape[-1] != 3:
        raise ValueError("RGB input must end with a three-channel dimension")

    points = lut.normalise_input(np.asarray(rgb, dtype=np.float64)).reshape(-1, 3)
    points = np.clip(points, 0.0, 1.0)
    q = points * (lut.size - 1)
    lower = np.floor(q).astype(np.int32)
    fraction = q - lower
    upper = np.minimum(lower + 1, lut.size - 1)

    r0, g0, b0 = lower[:, 0], lower[:, 1], lower[:, 2]
    r1, g1, b1 = upper[:, 0], upper[:, 1], upper[:, 2]
    fr, fg, fb = fraction[:, 0:1], fraction[:, 1:2], fraction[:, 2:3]
    table = lut.table

    c000 = table[b0, g0, r0]
    c100 = table[b0, g0, r1]
    c010 = table[b0, g1, r0]
    c110 = table[b0, g1, r1]
    c001 = table[b1, g0, r0]
    c101 = table[b1, g0, r1]
    c011 = table[b1, g1, r0]
    c111 = table[b1, g1, r1]

    output = np.empty_like(c000)

    # Six tetrahedra, selected by the ordering of the fractional coordinates.
    m0 = ((fr >= fg) & (fg >= fb))[:, 0]  # r >= g >= b
    m1 = ((fr >= fb) & (fb > fg))[:, 0]   # r >= b > g
    m2 = ((fb > fr) & (fr >= fg))[:, 0]   # b > r >= g
    m3 = ((fg > fr) & (fr >= fb))[:, 0]   # g > r >= b
    m4 = ((fg >= fb) & (fb > fr))[:, 0]   # g >= b > r
    m5 = ((fb > fg) & (fg > fr))[:, 0]    # b > g > r

    output[m0] = (
        c000[m0]
        + fr[m0] * (c100[m0] - c000[m0])
        + fg[m0] * (c110[m0] - c100[m0])
        + fb[m0] * (c111[m0] - c110[m0])
    )
    output[m1] = (
        c000[m1]
        + fr[m1] * (c100[m1] - c000[m1])
        + fb[m1] * (c101[m1] - c100[m1])
        + fg[m1] * (c111[m1] - c101[m1])
    )
    output[m2] = (
        c000[m2]
        + fb[m2] * (c001[m2] - c000[m2])
        + fr[m2] * (c101[m2] - c001[m2])
        + fg[m2] * (c111[m2] - c101[m2])
    )
    output[m3] = (
        c000[m3]
        + fg[m3] * (c010[m3] - c000[m3])
        + fr[m3] * (c110[m3] - c010[m3])
        + fb[m3] * (c111[m3] - c110[m3])
    )
    output[m4] = (
        c000[m4]
        + fg[m4] * (c010[m4] - c000[m4])
        + fb[m4] * (c011[m4] - c010[m4])
        + fr[m4] * (c111[m4] - c011[m4])
    )
    output[m5] = (
        c000[m5]
        + fb[m5] * (c001[m5] - c000[m5])
        + fg[m5] * (c011[m5] - c001[m5])
        + fr[m5] * (c111[m5] - c011[m5])
    )

    return output.reshape(original_shape)
