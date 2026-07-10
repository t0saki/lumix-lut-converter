from __future__ import annotations

from dataclasses import asdict, dataclass
import warnings

warnings.filterwarnings("ignore", message='"Matplotlib" related API features are not available')
import colour
import numpy as np

from .inverse import InverseResult


@dataclass(frozen=True)
class ValidationMetrics:
    rgb_error_mean: float
    rgb_error_p95: float
    rgb_error_p99: float
    rgb_error_max: float
    delta_e00_mean: float
    delta_e00_p95: float
    delta_e00_p99: float
    delta_e00_max: float
    neutral_rgb_error_max: float
    iterations: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _delta_e00(reference_rgb: np.ndarray, test_rgb: np.ndarray) -> np.ndarray:
    colourspace = colour.RGB_COLOURSPACES["ITU-R BT.709"]
    reference_linear = colour.models.oetf_inverse_BT709(
        np.clip(reference_rgb, 0.0, 1.0)
    )
    test_linear = colour.models.oetf_inverse_BT709(np.clip(test_rgb, 0.0, 1.0))
    reference_xyz = colour.RGB_to_XYZ(reference_linear, colourspace)
    test_xyz = colour.RGB_to_XYZ(test_linear, colourspace)
    reference_lab = colour.XYZ_to_Lab(reference_xyz, colourspace.whitepoint)
    test_lab = colour.XYZ_to_Lab(test_xyz, colourspace.whitepoint)
    return colour.delta_E(reference_lab, test_lab, method="CIE 2000")


def validate_inverse(
    result: InverseResult,
    *,
    black_code: int = 64,
    white_code: int = 940,
    denominator: int = 1023,
) -> ValidationMetrics:
    black = black_code / denominator
    white = white_code / denominator
    recovered_full = (result.recovered_legal_range - black) / (white - black)
    rgb_error = np.linalg.norm(recovered_full - result.target_full_range, axis=1)
    delta_e = _delta_e00(result.target_full_range, recovered_full)

    neutral = np.isclose(result.target_full_range[:, 0], result.target_full_range[:, 1]) & np.isclose(
        result.target_full_range[:, 1], result.target_full_range[:, 2]
    )
    return ValidationMetrics(
        rgb_error_mean=float(np.mean(rgb_error)),
        rgb_error_p95=float(np.percentile(rgb_error, 95)),
        rgb_error_p99=float(np.percentile(rgb_error, 99)),
        rgb_error_max=float(np.max(rgb_error)),
        delta_e00_mean=float(np.mean(delta_e)),
        delta_e00_p95=float(np.percentile(delta_e, 95)),
        delta_e00_p99=float(np.percentile(delta_e, 99)),
        delta_e00_max=float(np.max(delta_e)),
        neutral_rgb_error_max=float(np.max(rgb_error[neutral])),
        iterations=result.iterations,
    )
