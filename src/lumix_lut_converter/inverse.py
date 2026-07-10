from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .interpolation import tetrahedral_interpolation
from .lut import FloatArray, LUT3D, cube_grid


@dataclass(frozen=True)
class InverseResult:
    vlog_coordinates: FloatArray
    target_full_range: FloatArray
    target_legal_range: FloatArray
    recovered_legal_range: FloatArray
    iterations: int


@dataclass(frozen=True)
class LUTInverseResult:
    coordinates: FloatArray
    target: FloatArray
    recovered: FloatArray
    iterations: int


def legalise_rgb(
    rgb: FloatArray,
    *,
    black_code: int = 64,
    white_code: int = 940,
    denominator: int = 1023,
) -> FloatArray:
    black = black_code / denominator
    white = white_code / denominator
    return black + np.asarray(rgb, dtype=np.float64) * (white - black)


def invert_v709_lut(
    reference: LUT3D,
    *,
    output_size: int = 33,
    black_code: int = 64,
    white_code: int = 940,
    denominator: int = 1023,
    max_iterations: int = 24,
    tolerance: float = 2.0e-8,
) -> InverseResult:
    """Numerically invert an official V-Log -> legal V709 3D LUT.

    A nearest-node search seeds a bounded, vectorised damped Gauss-Newton
    solver. The forward model uses tetrahedral interpolation throughout.
    """

    target_full = cube_grid(output_size)
    target_legal = legalise_rgb(
        target_full,
        black_code=black_code,
        white_code=white_code,
        denominator=denominator,
    )

    inverse = invert_lut(
        reference,
        target_legal,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    return InverseResult(
        vlog_coordinates=inverse.coordinates,
        target_full_range=target_full,
        target_legal_range=target_legal,
        recovered_legal_range=inverse.recovered,
        iterations=inverse.iterations,
    )


def invert_lut(
    reference: LUT3D,
    target_rgb: FloatArray,
    *,
    max_iterations: int = 24,
    tolerance: float = 2.0e-8,
) -> LUTInverseResult:
    """Numerically invert a 3D LUT for arbitrary target RGB coordinates."""

    target = np.asarray(target_rgb, dtype=np.float64)
    original_shape = target.shape
    if not original_shape or original_shape[-1] != 3:
        raise ValueError("Target RGB must end with a three-channel dimension")
    target = target.reshape(-1, 3)

    reference_inputs = cube_grid(reference.size)
    reference_outputs = reference.table.reshape(-1, 3)
    tree = cKDTree(reference_outputs)
    _, nearest = tree.query(target, workers=-1)
    points = reference_inputs[nearest].copy()
    best_points = points.copy()
    best_squared_error = np.full(len(points), np.inf, dtype=np.float64)

    epsilon = 1.0e-5
    damping = 2.0e-7
    completed_iterations = 0

    for iteration in range(max_iterations):
        completed_iterations = iteration + 1
        current = tetrahedral_interpolation(reference, points)
        residual = current - target
        squared_error = np.einsum("ij,ij->i", residual, residual)
        improved = squared_error < best_squared_error
        best_squared_error[improved] = squared_error[improved]
        best_points[improved] = points[improved]

        if float(np.percentile(np.sqrt(squared_error), 99)) <= tolerance:
            break

        jacobian = np.empty((len(points), 3, 3), dtype=np.float64)
        for channel in range(3):
            plus = points.copy()
            minus = points.copy()
            plus[:, channel] = np.minimum(plus[:, channel] + epsilon, 1.0)
            minus[:, channel] = np.maximum(minus[:, channel] - epsilon, 0.0)
            span = np.maximum(plus[:, channel] - minus[:, channel], 1.0e-12)
            derivative = (
                tetrahedral_interpolation(reference, plus)
                - tetrahedral_interpolation(reference, minus)
            ) / span[:, None]
            jacobian[:, :, channel] = derivative

        transposed = np.swapaxes(jacobian, 1, 2)
        normal = transposed @ jacobian
        normal[:, 0, 0] += damping
        normal[:, 1, 1] += damping
        normal[:, 2, 2] += damping
        right_hand_side = (transposed @ residual[..., None])[..., 0]
        step = np.linalg.solve(normal, right_hand_side[..., None])[..., 0]
        step_norm = np.linalg.norm(step, axis=1, keepdims=True)
        step *= np.minimum(1.0, 0.10 / np.maximum(step_norm, 1.0e-15))

        candidate_points = points
        candidate_error = squared_error
        for scale in (1.0, 0.5, 0.25, 0.125):
            candidate = np.clip(points - scale * step, 0.0, 1.0)
            candidate_residual = (
                tetrahedral_interpolation(reference, candidate) - target
            )
            error = np.einsum("ij,ij->i", candidate_residual, candidate_residual)
            choose = error < candidate_error
            if np.any(choose):
                candidate_points = candidate_points.copy()
                candidate_error = candidate_error.copy()
                candidate_points[choose] = candidate[choose]
                candidate_error[choose] = error[choose]
        points = candidate_points

    final_current = tetrahedral_interpolation(reference, points)
    final_residual = final_current - target
    final_squared_error = np.einsum("ij,ij->i", final_residual, final_residual)
    improved = final_squared_error < best_squared_error
    best_points[improved] = points[improved]
    recovered = tetrahedral_interpolation(reference, best_points)

    return LUTInverseResult(
        coordinates=best_points.reshape(original_shape),
        target=target.reshape(original_shape),
        recovered=recovered.reshape(original_shape),
        iterations=completed_iterations,
    )
