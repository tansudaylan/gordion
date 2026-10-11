"""Spherical celestial-body profiles for capture and phase-space calculations."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from math import pi
from typing import Iterable

import numpy as np

from .capture import G, NuclearSpecies, _composition


EARTH_MASS_KG = 5.9722e24  # [kg]
EARTH_RADIUS_M = 6.3710e6  # [m]
MOON_MASS_KG = 7.342e22  # [kg]
MOON_RADIUS_M = 1.7374e6  # [m]
MARS_MASS_KG = 6.4171e23  # [kg]
MARS_RADIUS_M = 3.3895e6  # [m]
SUN_MASS_KG = 1.9885e30  # [kg]
SUN_RADIUS_M = 6.957e8  # [m]
JUPITER_MASS_KG = 1.8981e27  # [kg]
JUPITER_RADIUS_M = 6.9911e7  # [m]
EARTH_MASS_JUPITER = 317.83
EARTH_RADIUS_JUPITER = 0.08921


@dataclass(frozen=True)
class BodyLayer:
    """A spherical shell specified by enclosed radial and mass fractions."""

    outer_radius_fraction: float
    mass_fraction: float
    composition: tuple[NuclearSpecies, ...]

    def __post_init__(self) -> None:
        if not 0.0 < self.outer_radius_fraction <= 1.0:
            raise ValueError("outer_radius_fraction must be in (0, 1]")
        if not 0.0 < self.mass_fraction <= 1.0:
            raise ValueError("mass_fraction must be in (0, 1]")
        object.__setattr__(self, "composition", _composition(self.composition))
        if not np.isclose(sum(item.mass_fraction for item in self.composition), 1.0,
                          rtol=0.0, atol=1e-8):
            raise ValueError("each body layer's elemental mass fractions must sum to one")


@dataclass(frozen=True)
class CelestialBody:
    """A nonrotating spherical body with piecewise-uniform density shells."""

    name: str
    mass_kg: float
    radius_m: float
    layers: tuple[BodyLayer, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("body name must not be empty")
        if not np.isfinite((self.mass_kg, self.radius_m)).all() or min(self.mass_kg, self.radius_m) <= 0.0:
            raise ValueError("body mass and radius must be finite and positive")
        if not self.layers:
            raise ValueError("at least one radial layer is required")
        outer = np.array([layer.outer_radius_fraction for layer in self.layers])
        fractions = np.array([layer.mass_fraction for layer in self.layers])
        if np.any(np.diff(outer) <= 0.0) or not np.isclose(outer[-1], 1.0):
            raise ValueError("layer outer radii must increase and end at one body radius")
        if not np.isclose(fractions.sum(), 1.0, rtol=0.0, atol=1e-8):
            raise ValueError("layer mass fractions must sum to one")

    @cached_property
    def _shell_geometry(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        edges_m = np.concatenate(([0.0], np.array(
            [layer.outer_radius_fraction for layer in self.layers], dtype=float
        ) * self.radius_m))
        shell_masses_kg = np.array([layer.mass_fraction for layer in self.layers]) * self.mass_kg
        volumes_m3 = (4.0 * pi / 3.0) * (edges_m[1:]**3 - edges_m[:-1]**3)
        densities = shell_masses_kg / volumes_m3
        return edges_m, shell_masses_kg, volumes_m3, densities

    @property
    def composition(self) -> tuple[NuclearSpecies, ...]:
        combined: dict[tuple[str, float], float] = {}
        for layer in self.layers:
            for element in layer.composition:
                key = (element.name, element.mass_number)
                combined[key] = combined.get(key, 0.0) + layer.mass_fraction * element.mass_fraction
        return tuple(NuclearSpecies(fraction, mass_number, name)
                     for (name, mass_number), fraction in combined.items())

    @property
    def surface_gravity_m_s2(self) -> float:
        return G * self.mass_kg / self.radius_m**2

    def density_kg_m3(self, radius_m: float | np.ndarray) -> float | np.ndarray:
        radius = np.asarray(radius_m, dtype=float)
        if not np.isfinite(radius).all() or np.any(radius < 0.0):
            raise ValueError("radius_m must contain finite nonnegative values")
        edges, _, _, densities = self._shell_geometry
        index = np.searchsorted(edges[1:], np.minimum(radius, self.radius_m), side="left")
        result = np.where(radius <= self.radius_m, densities[np.minimum(index, len(densities) - 1)], 0.0)
        return float(result) if result.ndim == 0 else result

    def layer_index(self, radius_m: float) -> int:
        if not np.isfinite(radius_m) or radius_m < 0.0 or radius_m > self.radius_m:
            raise ValueError("radius_m must lie inside the body")
        return int(min(np.searchsorted(self._shell_geometry[0][1:], radius_m, side="left"),
                       len(self.layers) - 1))

    def composition_at(self, radius_m: float) -> tuple[NuclearSpecies, ...]:
        return self.layers[self.layer_index(radius_m)].composition

    def enclosed_mass_kg(self, radius_m: float | np.ndarray) -> float | np.ndarray:
        radius = np.asarray(radius_m, dtype=float)
        if not np.isfinite(radius).all() or np.any(radius < 0.0):
            raise ValueError("radius_m must contain finite nonnegative values")
        clipped = np.minimum(radius, self.radius_m)
        edges, masses, volumes, densities = self._shell_geometry
        enclosed = np.zeros_like(clipped)
        for inner, outer, mass, density, volume in zip(edges[:-1], edges[1:], masses, densities, volumes):
            radial_volume = np.clip(clipped**3 - inner**3, 0.0, outer**3 - inner**3)
            enclosed += density * (4.0 * pi / 3.0) * radial_volume
        enclosed = np.where(radius >= self.radius_m, self.mass_kg, enclosed)
        return float(enclosed) if enclosed.ndim == 0 else enclosed

    def potential_j_kg(self, radius_m: float | np.ndarray) -> float | np.ndarray:
        """Return gravitational potential per unit mass with zero at infinity."""

        radius = np.asarray(radius_m, dtype=float)
        if not np.isfinite(radius).all() or np.any(radius < 0.0):
            raise ValueError("radius_m must contain finite nonnegative values")
        interior_radius = np.minimum(radius, self.radius_m)
        integral = np.zeros_like(interior_radius)
        edges, shell_masses, _, densities = self._shell_geometry
        enclosed_before = 0.0
        for inner, outer, shell_mass, density in zip(
            edges[:-1], edges[1:], shell_masses, densities
        ):
            active = interior_radius < outer
            lower = np.maximum(interior_radius, inner)
            volume_coefficient = 4.0 * pi * density / 3.0
            constant_mass = enclosed_before - volume_coefficient * inner**3
            inverse_radius_integral = np.zeros_like(integral)
            if abs(constant_mass) > 1e-10:
                inverse_radius_integral = constant_mass * (1.0 / np.maximum(lower, 1e-30) - 1.0 / outer)
            shell_integral = G * (
                inverse_radius_integral
                + 0.5 * volume_coefficient * (outer**2 - lower**2)
            )
            integral += np.where(active, shell_integral, 0.0)
            enclosed_before += shell_mass
        inside = -G * self.mass_kg / self.radius_m - integral
        outside = -G * self.mass_kg / np.maximum(radius, self.radius_m)
        result = np.where(radius <= self.radius_m, inside, outside)
        return float(result) if result.ndim == 0 else result

    def escape_speed_m_s(self, radius_m: float | np.ndarray) -> float | np.ndarray:
        potential = np.asarray(self.potential_j_kg(radius_m))
        speed = np.sqrt(np.maximum(-2.0 * potential, 0.0))
        return float(speed) if speed.ndim == 0 else speed


def _species(*items: tuple[str, float, float]) -> tuple[NuclearSpecies, ...]:
    return tuple(NuclearSpecies(fraction, mass_number, name)
                 for name, mass_number, fraction in items)


def solar_system_bodies() -> dict[str, CelestialBody]:
    """Return approximate layered Earth, Moon, and Mars profiles.

    Layer mass fractions and compositions are transparent bulk models, not
    seismological inversions. See the Gordion guide for their limitations.
    """

    return {
        "Earth": CelestialBody("Earth", EARTH_MASS_KG, EARTH_RADIUS_M, (
            BodyLayer(0.546, 0.325, _species(("Fe", 56.0, 0.85), ("Ni", 58.0, 0.05), ("S", 32.0, 0.10))),
            BodyLayer(1.0, 0.675, _species(("O", 16.0, 0.44), ("Mg", 24.0, 0.23),
                                          ("Si", 28.0, 0.21), ("Fe", 56.0, 0.06),
                                          ("Ca", 40.0, 0.025), ("Al", 27.0, 0.035))),
        )),
        "Moon": CelestialBody("Moon", MOON_MASS_KG, MOON_RADIUS_M, (
            BodyLayer(0.25, 0.025, _species(("Fe", 56.0, 0.85), ("S", 32.0, 0.15))),
            BodyLayer(1.0, 0.975, _species(("O", 16.0, 0.43), ("Mg", 24.0, 0.20),
                                          ("Si", 28.0, 0.21), ("Fe", 56.0, 0.12),
                                          ("Ca", 40.0, 0.03), ("Al", 27.0, 0.01))),
        )),
        "Mars": CelestialBody("Mars", MARS_MASS_KG, MARS_RADIUS_M, (
            BodyLayer(0.50, 0.18, _species(("Fe", 56.0, 0.85), ("S", 32.0, 0.15))),
            BodyLayer(1.0, 0.82, _species(("O", 16.0, 0.44), ("Mg", 24.0, 0.22),
                                         ("Si", 28.0, 0.21), ("Fe", 56.0, 0.10),
                                         ("Ca", 40.0, 0.02), ("Al", 27.0, 0.01))),
        )),
    }


def two_layer_exoplanet(name: str, mass_earth: float, radius_earth: float,
                        core_mass_fraction: float, *,
                        core_density_kg_m3: float = 11000.0) -> CelestialBody:
    """Construct an explicit iron-core/silicate-mantle exoplanet profile."""

    if not np.isfinite((mass_earth, radius_earth, core_mass_fraction,
                        core_density_kg_m3)).all():
        raise ValueError("exoplanet parameters must be finite")
    if mass_earth <= 0.0 or radius_earth <= 0.0 or core_density_kg_m3 <= 0.0:
        raise ValueError("planet mass, radius, and core density must be positive")
    if not 0.0 <= core_mass_fraction <= 1.0:
        raise ValueError("core_mass_fraction must be between zero and one")
    mass_kg = mass_earth * EARTH_MASS_KG
    radius_m = radius_earth * EARTH_RADIUS_M
    mantle = _species(("O", 16.0, 0.44), ("Mg", 24.0, 0.23), ("Si", 28.0, 0.21),
                      ("Fe", 56.0, 0.06), ("Ca", 40.0, 0.025), ("Al", 27.0, 0.035))
    core = _species(("Fe", 56.0, 0.90), ("Ni", 58.0, 0.10))
    if core_mass_fraction == 0.0:
        return CelestialBody(name, mass_kg, radius_m,
                             (BodyLayer(1.0, 1.0, mantle),))
    if core_mass_fraction == 1.0:
        return CelestialBody(name, mass_kg, radius_m,
                             (BodyLayer(1.0, 1.0, core),))
    core_mass = core_mass_fraction * mass_kg
    core_radius = (3.0 * core_mass / (4.0 * pi * core_density_kg_m3)) ** (1.0 / 3.0)
    if core_radius >= radius_m:
        raise ValueError("core density and mass fraction leave no mantle volume")
    return CelestialBody(name, mass_kg, radius_m, (
        BodyLayer(core_radius / radius_m, core_mass_fraction, core),
        BodyLayer(1.0, 1.0 - core_mass_fraction, mantle),
    ))