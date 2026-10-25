"""Six-dimensional dark-matter phase-space sampling and scattering transport."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np

from .bodies import CelestialBody
from .capture import G, GEV_TO_KG, NuclearSpecies, _nucleus_cross_section


@dataclass(frozen=True)
class HaloModel:
    """Maxwellian truncated in the Galactic frame, then shifted to the body."""

    density_gev_cm3: float = 0.4  # [GeV/cm^3]
    speed_scale_m_s: float = 220000.0  # [m/s]
    galactic_escape_speed_m_s: float = 544000.0  # [m/s]
    body_velocity_m_s: tuple[float, float, float] = (0.0, 0.0, 0.0)  # [m/s]

    def __post_init__(self) -> None:
        values = (self.density_gev_cm3, self.speed_scale_m_s,
                  self.galactic_escape_speed_m_s, *self.body_velocity_m_s)
        if not np.isfinite(values).all():
            raise ValueError("halo parameters must be finite")
        if min(self.density_gev_cm3, self.speed_scale_m_s,
               self.galactic_escape_speed_m_s) <= 0.0:
            raise ValueError("halo density and speeds must be positive")
        if len(self.body_velocity_m_s) != 3:
            raise ValueError("body_velocity_m_s must have three components")


@dataclass(frozen=True)
class PhaseSpaceEnsemble:
    """Weighted samples in position-velocity coordinates ``(x,y,z,vx,vy,vz)``."""

    position_m: np.ndarray
    velocity_m_s: np.ndarray
    weight: np.ndarray

    def __post_init__(self) -> None:
        position = np.asarray(self.position_m, dtype=float)
        velocity = np.asarray(self.velocity_m_s, dtype=float)
        weight = np.asarray(self.weight, dtype=float)
        if position.ndim != 2 or position.shape[1] != 3:
            raise ValueError("position_m must have shape (n, 3)")
        if velocity.shape != position.shape or weight.shape != (position.shape[0],):
            raise ValueError("velocity_m_s and weight must match position_m")
        if not np.isfinite(position).all() or not np.isfinite(velocity).all():
            raise ValueError("phase-space coordinates must be finite")
        if not np.isfinite(weight).all() or np.any(weight < 0.0):
            raise ValueError("phase-space weights must be finite and nonnegative")
        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "velocity_m_s", velocity)
        object.__setattr__(self, "weight", weight)

    @property
    def coordinates(self) -> np.ndarray:
        """Return the six phase-space coordinates in SI units."""

        return np.column_stack((self.position_m, self.velocity_m_s))


@dataclass(frozen=True)
class TransportResult:
    """Scattering trajectories, their status, and first-capture phase points."""

    trajectory_samples: PhaseSpaceEnsemble
    sample_particle_index: np.ndarray
    particle_status: np.ndarray
    scatter_count: np.ndarray
    capture_phase_space: PhaseSpaceEnsemble
    capture_rate_per_second: float


def _sample_halo_velocity(halo: HaloModel, number: int,
                          rng: np.random.Generator) -> np.ndarray:
    sigma = halo.speed_scale_m_s / np.sqrt(2.0)
    body_velocity = np.asarray(halo.body_velocity_m_s, dtype=float)
    accepted: list[np.ndarray] = []
    remaining = number
    while remaining:
        galactic_velocity = rng.normal(
            0.0, sigma, size=(max(remaining * 2, 32), 3)
        )
        selected = galactic_velocity[
            np.linalg.norm(galactic_velocity, axis=1)
            < halo.galactic_escape_speed_m_s
        ]
        if selected.size:
            take = selected[:remaining]
            accepted.append(take)
            remaining -= len(take)
    return np.concatenate(accepted, axis=0) - body_velocity


def sample_unbound_phase_space(body: CelestialBody, number: int,
                               halo: HaloModel = HaloModel(), *,
                               seed: int | None = None,
                               sanity_checks: bool = False) -> PhaseSpaceEnsemble:
    """Sample the unbound halo distribution inside a spherical body.

    Positions are uniform in volume and importance-weighted by gravitational
    focusing. At fixed position, asymptotic Maxwellian velocities are mapped to
    local speeds with energy conservation while retaining their directions.
    This is the straight-ray Liouville approximation; it omits scattering and
    directional bending. Use ``simulate_capture_trajectories`` to include
    elastic collisions and bound capture.
    """

    if number < 1:
        raise ValueError("number must be positive")
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=(number, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    radius = body.radius_m * rng.random(number) ** (1.0 / 3.0)
    position = radius[:, None] * direction
    asymptotic_velocity = _sample_halo_velocity(halo, number, rng)
    asymptotic_speed = np.linalg.norm(asymptotic_velocity, axis=1)
    local_speed = np.sqrt(asymptotic_speed**2 + body.escape_speed_m_s(radius)**2)
    velocity_direction = asymptotic_velocity / asymptotic_speed[:, None]
    local_velocity = local_speed[:, None] * velocity_direction
    weights = local_speed / asymptotic_speed
    if sanity_checks:
        specific_energy = 0.5 * np.sum(local_velocity**2, axis=1) + body.potential_j_kg(radius)
        if np.any(specific_energy < -1e-8 * np.max(local_speed**2)):
            raise RuntimeError("unbound halo samples have negative specific energy")
    return PhaseSpaceEnsemble(position, local_velocity, weights)


def _unit_perpendicular(vector: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    candidate = rng.normal(size=3)
    candidate -= np.dot(candidate, vector) * vector
    norm = np.linalg.norm(candidate)
    while norm < 1e-12:
        candidate = rng.normal(size=3)
        candidate -= np.dot(candidate, vector) * vector
        norm = np.linalg.norm(candidate)
    return candidate / norm


def sample_surface_inflow(body: CelestialBody, number: int,
                          halo: HaloModel = HaloModel(), *,
                          seed: int | np.random.SeedSequence | None = None
                          ) -> tuple[PhaseSpaceEnsemble, np.ndarray]:
    """Sample gravitationally focused halo trajectories at the body surface.

    Returns the incident phase points and their flux weights in m^3/s. A
    uniformly sampled impact area is drawn from the focused cross section.
    """

    if number < 1:
        raise ValueError("number must be positive")
    rng = np.random.default_rng(seed)
    velocity_infinity = _sample_halo_velocity(halo, number, rng)
    speed_infinity = np.linalg.norm(velocity_infinity, axis=1)
    vesc = body.escape_speed_m_s(body.radius_m)
    speed_surface = np.sqrt(speed_infinity**2 + vesc**2)
    bmax = body.radius_m * np.sqrt(1.0 + (vesc / speed_infinity) ** 2)
    impact = bmax * np.sqrt(rng.random(number))
    sin_angle = np.clip(impact * speed_infinity / (body.radius_m * speed_surface), 0.0, 1.0)
    incoming_direction = velocity_infinity / speed_infinity[:, None]
    position = np.empty((number, 3))
    velocity = np.empty((number, 3))
    for index in range(number):
        transverse = _unit_perpendicular(incoming_direction[index], rng)
        cos_angle = np.sqrt(1.0 - sin_angle[index] ** 2)
        radial_direction = -cos_angle * incoming_direction[index] + sin_angle[index] * transverse
        position[index] = body.radius_m * radial_direction
        velocity[index] = speed_surface[index] * incoming_direction[index]
    flux_weight = pi * bmax**2 * speed_infinity
    return PhaseSpaceEnsemble(position, velocity, np.ones(number)), flux_weight


def _acceleration(body: CelestialBody, position_m: np.ndarray) -> np.ndarray:
    radius = np.linalg.norm(position_m)
    if radius == 0.0:
        return np.zeros(3)
    enclosed_mass = body.enclosed_mass_kg(radius)
    return -G * enclosed_mass * position_m / radius**3


def _scatter_velocity(velocity_m_s: np.ndarray, dm_mass_gev: float,
                      species: NuclearSpecies,
                      rng: np.random.Generator) -> np.ndarray:
    target_mass = species.mass_number
    center_of_mass = dm_mass_gev / (dm_mass_gev + target_mass) * velocity_m_s
    relative_speed = np.linalg.norm(velocity_m_s)
    relative_mass_fraction = target_mass / (dm_mass_gev + target_mass)
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    return center_of_mass + relative_mass_fraction * relative_speed * direction


def _check_elastic_scatter(initial_velocity: np.ndarray, final_dm_velocity: np.ndarray,
                           dm_mass_gev: float, target_mass_gev: float) -> None:
    target_velocity = dm_mass_gev / target_mass_gev * (initial_velocity - final_dm_velocity)
    initial_momentum = dm_mass_gev * initial_velocity
    final_momentum = dm_mass_gev * final_dm_velocity + target_mass_gev * target_velocity
    initial_energy = 0.5 * dm_mass_gev * np.dot(initial_velocity, initial_velocity)
    final_energy = (0.5 * dm_mass_gev * np.dot(final_dm_velocity, final_dm_velocity)
                    + 0.5 * target_mass_gev * np.dot(target_velocity, target_velocity))
    momentum_error = np.linalg.norm(final_momentum - initial_momentum) / max(
        np.linalg.norm(initial_momentum), 1.0
    )
    energy_error = abs(final_energy - initial_energy) / max(initial_energy, 1.0)
    if momentum_error > 1e-12 or energy_error > 1e-12:
        raise RuntimeError("elastic-scatter kinematics failed momentum or energy conservation")


def simulate_capture_trajectories(
    body: CelestialBody,
    number: int,
    dm_mass_gev: float,
    sigma_cm2: float,
    halo: HaloModel = HaloModel(),
    *,
    nucleon_mass_gev: float = 0.94,
    scale_from_nucleon: bool = True,
    step_fraction: float = 0.02,
    max_steps: int = 5000,
    sample_stride: int = 4,
    seed: int | None = None,
    sanity_checks: bool = False,
) -> TransportResult:
    """Follow gravitational trajectories and isotropic elastic nuclear scatters.

    A velocity-Verlet step advances each incident particle through the body.
    The per-step collision probability is ``1-exp(-sum(n_i sigma_i ds))``.
    Nuclei are cold and stationary, scatter isotropically in the center-of-mass
    frame, and body rotation, thermal motion, inelastic channels, and DM
    self-interactions are omitted. Capture is the first interior state with
    negative specific orbital energy.
    """

    if number < 1 or max_steps < 1 or sample_stride < 1:
        raise ValueError("number, max_steps, and sample_stride must be positive")
    if not np.isfinite((dm_mass_gev, sigma_cm2, nucleon_mass_gev, step_fraction)).all():
        raise ValueError("transport parameters must be finite")
    if min(dm_mass_gev, nucleon_mass_gev, step_fraction) <= 0.0 or sigma_cm2 < 0.0:
        raise ValueError("masses and step_fraction must be positive; sigma cannot be negative")
    inflow_seed, scattering_seed = np.random.SeedSequence(seed).spawn(2)
    rng = np.random.default_rng(scattering_seed)
    initial, flux_weight = sample_surface_inflow(
        body, number, halo, seed=inflow_seed
    )
    sample_position: list[np.ndarray] = []
    sample_velocity: list[np.ndarray] = []
    sample_weight: list[float] = []
    sample_particle: list[int] = []
    capture_position = np.full((number, 3), np.nan)
    capture_velocity = np.full((number, 3), np.nan)
    statuses = np.full(number, "step_limit", dtype="U16")
    scatter_counts = np.zeros(number, dtype=int)
    speed_scale = max(body.escape_speed_m_s(body.radius_m), halo.speed_scale_m_s)
    dt = step_fraction * body.radius_m / speed_scale  # [s]

    for particle in range(number):
        position = initial.position_m[particle].copy()
        velocity = initial.velocity_m_s[particle].copy()
        alive = True
        for step in range(max_steps):
            acceleration = _acceleration(body, position)
            half_velocity = velocity + 0.5 * dt * acceleration
            next_position = position + dt * half_velocity
            midpoint = 0.5 * (position + next_position)
            midpoint_radius = np.linalg.norm(midpoint)
            scattered = False
            if midpoint_radius < body.radius_m:
                layer = body.layers[body.layer_index(midpoint_radius)]
                density = body.density_kg_m3(midpoint_radius)
                species_opacity = np.array([
                    density * element.mass_fraction / (element.mass_number * GEV_TO_KG)
                    * _nucleus_cross_section(element, nucleon_mass_gev, dm_mass_gev,
                                             sigma_cm2, scale_from_nucleon) * 1e-4
                    for element in layer.composition
                ])
                path_length = np.linalg.norm(next_position - position)
                optical_step = float(species_opacity.sum() * path_length)
                if rng.random() < -np.expm1(-min(optical_step, 700.0)):
                    target_index = int(rng.choice(len(layer.composition),
                                                  p=species_opacity / species_opacity.sum()))
                    initial_scatter_velocity = half_velocity.copy()
                    target_species = layer.composition[target_index]
                    velocity = _scatter_velocity(
                        half_velocity, dm_mass_gev, target_species, rng,
                    )
                    if sanity_checks:
                        _check_elastic_scatter(initial_scatter_velocity, velocity,
                                               dm_mass_gev, target_species.mass_number)
                    scatter_counts[particle] += 1
                    scattered = True
                else:
                    velocity = half_velocity
            else:
                velocity = half_velocity

            position = next_position
            velocity += 0.5 * dt * _acceleration(body, position)
            radius = np.linalg.norm(position)
            energy = 0.5 * np.dot(velocity, velocity) + body.potential_j_kg(radius)
            if step % sample_stride == 0 and radius <= body.radius_m:
                sample_position.append(position.copy())
                sample_velocity.append(velocity.copy())
                sample_weight.append(flux_weight[particle] * dt * sample_stride)
                sample_particle.append(particle)
            if scattered and radius < body.radius_m and energy < 0.0:
                statuses[particle] = "captured"
                capture_position[particle] = position
                capture_velocity[particle] = velocity
                alive = False
                break
            if radius >= body.radius_m and np.dot(position, velocity) > 0.0:
                statuses[particle] = "escaped" if energy >= 0.0 else "bound"
                alive = False
                break
        if alive:
            statuses[particle] = "step_limit"

    if sample_position:
        path_ensemble = PhaseSpaceEnsemble(
            np.asarray(sample_position), np.asarray(sample_velocity), np.asarray(sample_weight)
        )
    else:
        path_ensemble = PhaseSpaceEnsemble(np.empty((0, 3)), np.empty((0, 3)), np.empty(0))
    captured = statuses == "captured"
    capture_weights = flux_weight[captured]
    capture_ensemble = PhaseSpaceEnsemble(
        capture_position[captured], capture_velocity[captured], capture_weights
    )
    estimated_rate = (halo.density_gev_cm3 * 1e6 / dm_mass_gev
                      * float(capture_weights.sum()) / number)
    if sanity_checks:
        if np.any(np.linalg.norm(capture_position[captured], axis=1) > body.radius_m):
            raise RuntimeError("captured phase points lie outside the body")
        if not np.isfinite(estimated_rate) or estimated_rate < 0.0:
            raise RuntimeError("estimated capture rate is invalid")
    return TransportResult(path_ensemble, np.asarray(sample_particle, dtype=int),
                           statuses, scatter_counts, capture_ensemble, estimated_rate)