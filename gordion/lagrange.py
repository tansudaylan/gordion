"""Co-rotating Lagrange points and local dark-matter phase-space samples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .capture import G
from .phase_space import HaloModel, PhaseSpaceEnsemble, _sample_halo_velocity


@dataclass(frozen=True)
class BinarySystem:
    """Circular restricted-three-body system in SI units."""

    primary_mass_kg: float
    secondary_mass_kg: float
    separation_m: float
    primary_name: str = "primary"
    secondary_name: str = "secondary"

    def __post_init__(self) -> None:
        values = (self.primary_mass_kg, self.secondary_mass_kg, self.separation_m)
        if not np.isfinite(values).all() or min(values) <= 0.0:
            raise ValueError("binary masses and separation must be finite and positive")

    @property
    def mass_fraction(self) -> float:
        return self.secondary_mass_kg / (self.primary_mass_kg + self.secondary_mass_kg)

    @property
    def angular_velocity_rad_s(self) -> float:
        return np.sqrt(G * (self.primary_mass_kg + self.secondary_mass_kg)
                       / self.separation_m**3)


def lagrange_points(system: BinarySystem) -> dict[str, np.ndarray]:
    """Return L1--L5 coordinates in barycentric co-rotating SI coordinates."""

    mu = system.mass_fraction
    primary_x = -mu
    secondary_x = 1.0 - mu

    def force_balance(x: float) -> float:
        first = (1.0 - mu) * (x + mu) / abs(x + mu) ** 3
        second = mu * (x - 1.0 + mu) / abs(x - 1.0 + mu) ** 3
        return x - first - second

    epsilon = 1e-10
    l1_x = brentq(force_balance, primary_x + epsilon, secondary_x - epsilon)
    l2_x = brentq(force_balance, secondary_x + epsilon, 4.0)
    l3_x = brentq(force_balance, -4.0, primary_x - epsilon)
    scale = system.separation_m
    return {
        "L1": np.array([l1_x * scale, 0.0, 0.0]),
        "L2": np.array([l2_x * scale, 0.0, 0.0]),
        "L3": np.array([l3_x * scale, 0.0, 0.0]),
        "L4": np.array([(0.5 - mu) * scale, np.sqrt(3.0) * 0.5 * scale, 0.0]),
        "L5": np.array([(0.5 - mu) * scale, -np.sqrt(3.0) * 0.5 * scale, 0.0]),
    }


def sample_lagrange_phase_space(
    system: BinarySystem,
    number: int,
    halo: HaloModel = HaloModel(),
    *,
    cloud_width_fraction: float = 0.01,
    seed: int | None = None,
) -> dict[str, PhaseSpaceEnsemble]:
    """Sample halo phase space around all five co-rotating equilibrium points.

    The local velocity magnitude includes the two-body gravitational potential
    through energy conservation. Velocity directions follow the sampled halo
    direction, then transform to the co-rotating frame. The cloud is an
    explicitly chosen spatial sampling width, not a claim that DM is trapped
    around a Lagrange point.
    """

    if number < 1:
        raise ValueError("number must be positive")
    if not np.isfinite(cloud_width_fraction) or cloud_width_fraction <= 0.0:
        raise ValueError("cloud_width_fraction must be finite and positive")
    rng = np.random.default_rng(seed)
    points = lagrange_points(system)
    mu = system.mass_fraction
    primary_position = np.array([-mu * system.separation_m, 0.0, 0.0])
    secondary_position = np.array([(1.0 - mu) * system.separation_m, 0.0, 0.0])
    velocity_infinity = _sample_halo_velocity(halo, number, rng)
    speed_infinity = np.linalg.norm(velocity_infinity, axis=1)
    direction = velocity_infinity / speed_infinity[:, None]
    omega = system.angular_velocity_rad_s
    result = {}
    for label, point in points.items():
        position = point + rng.normal(
            0.0, cloud_width_fraction * system.separation_m, size=(number, 3)
        )
        distance_primary = np.linalg.norm(position - primary_position, axis=1)
        distance_secondary = np.linalg.norm(position - secondary_position, axis=1)
        potential = (-G * system.primary_mass_kg / distance_primary
                     - G * system.secondary_mass_kg / distance_secondary)
        local_speed = np.sqrt(speed_infinity**2 - 2.0 * potential)
        inertial_velocity = local_speed[:, None] * direction
        rotating_velocity = inertial_velocity - np.cross(
            np.array([0.0, 0.0, omega]), position
        )
        result[label] = PhaseSpaceEnsemble(
            position, rotating_velocity, local_speed / speed_infinity
        )
    return result