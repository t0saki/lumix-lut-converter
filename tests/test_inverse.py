import numpy as np

from lumix_lut_converter.inverse import invert_v709_lut, legalise_rgb
from lumix_lut_converter.lut import LUT3D, cube_grid
from lumix_lut_converter.metrics import validate_inverse


def test_inverse_of_legal_range_identity() -> None:
    size = 17
    full_grid = cube_grid(size)
    legal_table = legalise_rgb(full_grid).reshape(size, size, size, 3)
    reference = LUT3D(legal_table, title="Synthetic legal reference")
    result = invert_v709_lut(reference, output_size=9, max_iterations=12)
    np.testing.assert_allclose(result.vlog_coordinates, cube_grid(9), atol=3.0e-8)
    metrics = validate_inverse(result)
    assert metrics.rgb_error_max < 1.0e-7
    assert metrics.delta_e00_max < 1.0e-4


def test_inverse_solver_handles_non_linear_reference() -> None:
    size = 17
    full_grid = cube_grid(size)
    curved = np.power(full_grid, 0.82)
    reference = LUT3D(legalise_rgb(curved).reshape(size, size, size, 3))
    result = invert_v709_lut(reference, output_size=7, max_iterations=18)
    metrics = validate_inverse(result)
    assert metrics.rgb_error_p99 < 2.0e-5
