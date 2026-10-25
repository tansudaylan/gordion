"""Co-rotating Lagrange points and local dark-matter phase-space samples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from .capture import G
from .phase_space import HaloModel, PhaseSpaceEnsemble, _sample_halo_velocity

ROUTH_CRITICAL_MASS_FRACTION = 0.5 * (1.0 - np.sqrt(23.0 / 27.0))


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


@dataclass(frozen=True)
class LagrangeRetentionResult:
    """Retention summary for particles initially placed near L4 or L5."""

    retained_count: int
    retained_fraction: float
    median_max_distance_fraction: float
    p84_max_distance_fraction: float
    sample_count: int
    periods: float
    retention_radius_fraction: float


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


def triangular_points_stable(system: BinarySystem) -> bool:
    """Return the ideal circular restricted-three-body L4/L5 stability result.

    This linear criterion assumes point masses on circular orbits. Capture and
    accumulation require a separate dynamical model.
    """

    return bool(system.mass_fraction < ROUTH_CRITICAL_MASS_FRACTION)


def simulate_lagrange_retention(
    system: BinarySystem,
    point_name: str,
    number: int,
    *,
    periods: float = 20.0,
    cloud_width_fraction: float = 0.01,
    velocity_dispersion_fraction: float = 0.01,
    retention_radius_fraction: float = 0.1,
    samples_per_period: int = 32,
    seed: int | None = None,
) -> LagrangeRetentionResult:
    """Follow particles initially placed near stable L4 or L5 points.

    The calculation uses the dimensionless circular restricted three-body
    problem. Positions are in binary separations and velocities in ``a * omega``.
    The returned fraction counts particles that remain within the chosen radius
    of the equilibrium point throughout the integration. It describes survival
    after placement. Halo capture and accumulation require separate calculations.
    """

    if point_name not in ("L4", "L5") or not triangular_points_stable(system):
        raise ValueError("retention requires a linearly stable L4 or L5 point")
    values = (
        periods,
        cloud_width_fraction,
        velocity_dispersion_fraction,
        retention_radius_fraction,
    )
    if not np.isfinite(values).all() or min(values) < 0.0:
        raise ValueError("retention parameters must be finite and nonnegative")
    if min(periods, retention_radius_fraction) <= 0.0:
        raise ValueError("periods and retention radius must be positive")
    if (
        isinstance(number, bool)
        or not isinstance(number, (int, np.integer))
        or isinstance(samples_per_period, bool)
        or not isinstance(samples_per_period, (int, np.integer))
        or number < 1
        or samples_per_period < 1
    ):
        raise ValueError("number and samples_per_period must be positive")

    points = lagrange_points(system)
    equilibrium_position = points[point_name] / system.separation_m
    rng = np.random.default_rng(seed)
    initial_positions = equilibrium_position + rng.normal(
        0.0, cloud_width_fraction, size=(number, 3)
    )
    initial_velocities = rng.normal(
        0.0, velocity_dispersion_fraction, size=(number, 3)
    )
    initial_state = np.column_stack((initial_positions, initial_velocities)).ravel()
    mass_fraction = system.mass_fraction

    def derivative(_time: float, flattened_state: np.ndarray) -> np.ndarray:
        states = flattened_state.reshape(number, 6)
        position_x, position_y, position_z = states[:, :3].T
        velocity_x, velocity_y, velocity_z = states[:, 3:].T
        primary_distance_cubed = (
            (position_x + mass_fraction) ** 2
            + position_y**2
            + position_z**2
        ) ** 1.5
        secondary_distance_cubed = (
            (position_x - 1.0 + mass_fraction) ** 2
            + position_y**2
            + position_z**2
        ) ** 1.5
        acceleration_x = (
            2.0 * velocity_y + position_x
            - (1.0 - mass_fraction) * (position_x + mass_fraction)
            / primary_distance_cubed
            - mass_fraction * (position_x - 1.0 + mass_fraction)
            / secondary_distance_cubed
        )
        acceleration_y = (
            -2.0 * velocity_x + position_y
            - (1.0 - mass_fraction) * position_y / primary_distance_cubed
            - mass_fraction * position_y / secondary_distance_cubed
        )
        acceleration_z = (
            -(1.0 - mass_fraction) * position_z / primary_distance_cubed
            - mass_fraction * position_z / secondary_distance_cubed
        )
        return np.column_stack((
            velocity_x,
            velocity_y,
            velocity_z,
            acceleration_x,
            acceleration_y,
            acceleration_z,
        )).ravel()

    final_time = 2.0 * np.pi * periods
    time_samples = np.linspace(
        0.0, final_time, int(np.ceil(periods * samples_per_period)) + 1
    )
    solution = solve_ivp(
        derivative,
        (0.0, final_time),
        initial_state,
        method="DOP853",
        t_eval=time_samples,
        rtol=1e-9,
        atol=1e-11,
    )
    if not solution.success:
        raise RuntimeError(f"CR3BP integration failed: {solution.message}")
    trajectories = solution.y.reshape(number, 6, -1)[:, :3, :]
    distances = np.linalg.norm(
        trajectories - equilibrium_position[None, :, None], axis=1
    )
    maximum_distances = np.max(distances, axis=1)
    retained_count = int(np.count_nonzero(
        maximum_distances <= retention_radius_fraction
    ))
    return LagrangeRetentionResult(
        retained_count=retained_count,
        retained_fraction=retained_count / number,
        median_max_distance_fraction=float(np.median(maximum_distances)),
        p84_max_distance_fraction=float(np.percentile(maximum_distances, 84)),
        sample_count=number,
        periods=periods,
        retention_radius_fraction=retention_radius_fraction,
    )


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