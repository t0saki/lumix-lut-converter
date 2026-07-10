import numpy as np

from lumix_lut_converter.interpolation import tetrahedral_interpolation
from lumix_lut_converter.lut import LUT3D, cube_grid, identity_lut


def test_identity_lut_is_exact() -> None:
    rng = np.random.default_rng(20260710)
    points = rng.random((5000, 3))
    actual = tetrahedral_interpolation(identity_lut(17), points)
    np.testing.assert_allclose(actual, points, atol=1.0e-14, rtol=0.0)


def test_affine_transform_is_exact() -> None:
    grid = cube_grid(9)
    matrix = np.array(
        [[0.72, 0.11, 0.03], [0.06, 0.81, 0.05], [0.02, 0.09, 0.77]],
        dtype=np.float64,
    )
    offset = np.array([0.03, 0.02, 0.04])
    table = (grid @ matrix.T + offset).reshape(9, 9, 9, 3)
    lut = LUT3D(table)
    points = np.random.default_rng(4).random((1000, 3))
    expected = points @ matrix.T + offset
    np.testing.assert_allclose(
        tetrahedral_interpolation(lut, points), expected, atol=2.0e-14, rtol=0.0
    )
