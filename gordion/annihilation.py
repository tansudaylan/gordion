"""Capture--annihilation population evolution without evaporation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AnnihilationHistory:
    """Captured population and annihilation rate at one elapsed time."""

    captured_particles: float
    annihilation_rate_per_second: float
    equilibrium_fraction: float
    equilibration_timescale_s: float


def evolve_capture_annihilation(
    capture_rate_per_second: float,
    annihilation_coefficient_per_second: float,
    age_s: float,
) -> AnnihilationHistory:
    """Solve constant capture and s-wave annihilation from an empty reservoir.

    The model solves ``dN/dt = C - C_A N^2`` with no evaporation. The effective
    coefficient ``C_A`` may encode a particle cross section and an assumed
    spatial distribution. Callers supply it as an input.
    """

    values = (capture_rate_per_second, annihilation_coefficient_per_second, age_s)
    if not np.isfinite(values).all() or min(values) < 0.0:
        raise ValueError(
            "capture rate, annihilation coefficient, and age must be finite and nonnegative"
        )
    if capture_rate_per_second == 0.0 or annihilation_coefficient_per_second == 0.0:
        captured_particles = capture_rate_per_second * age_s  # [particles]
        return AnnihilationHistory(captured_particles, 0.0, 0.0, float("inf"))

    equilibration_timescale_s = 1.0 / np.sqrt(
        capture_rate_per_second * annihilation_coefficient_per_second
    )  # [s]
    equilibrium_population = np.sqrt(
        capture_rate_per_second / annihilation_coefficient_per_second
    )  # [particles]
    time_factor = np.tanh(age_s / equilibration_timescale_s)
    captured_particles = equilibrium_population * time_factor
    equilibrium_fraction = float(time_factor**2)
    annihilation_rate_per_second = (
        0.5 * annihilation_coefficient_per_second * captured_particles**2
    )  # [s^-1]
    return AnnihilationHistory(
        float(captured_particles),
        float(annihilation_rate_per_second),
        equilibrium_fraction,
        float(equilibration_timescale_s),
    )