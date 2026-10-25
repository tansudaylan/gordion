"""Gordion dark-matter phase-space and capture calculations."""

from .annihilation import AnnihilationHistory, evolve_capture_annihilation
from .bodies import BodyLayer, CelestialBody, solar_system_bodies, two_layer_exoplanet
from .catalog import ExoplanetRecord, fetch_confirmed_exoplanets
from .capture import (
    AsteriaCapture,
    CaptureModel,
    CaptureResult,
    NuclearSpecies,
    escape_speed,
)
from .lagrange import (
    BinarySystem,
    LagrangeRetentionResult,
    lagrange_points,
    sample_lagrange_phase_space,
    simulate_lagrange_retention,
    triangular_points_stable,
)
from .paths import get_data_path, get_repository_path, get_visuals_path
from .phase_space import (
    HaloModel,
    PhaseSpaceEnsemble,
    TransportResult,
    sample_surface_inflow,
    sample_unbound_phase_space,
    simulate_capture_trajectories,
)

__all__ = [
    "AsteriaCapture",
    "AnnihilationHistory",
    "BinarySystem",
    "BodyLayer",
    "CaptureModel",
    "CaptureResult",
    "CelestialBody",
    "ExoplanetRecord",
    "HaloModel",
    "LagrangeRetentionResult",
    "NuclearSpecies",
    "PhaseSpaceEnsemble",
    "TransportResult",
    "escape_speed",
    "evolve_capture_annihilation",
    "fetch_confirmed_exoplanets",
    "get_data_path",
    "get_repository_path",
    "get_visuals_path",
    "lagrange_points",
    "sample_lagrange_phase_space",
    "sample_surface_inflow",
    "sample_unbound_phase_space",
    "simulate_lagrange_retention",
    "simulate_capture_trajectories",
    "solar_system_bodies",
    "two_layer_exoplanet",
    "triangular_points_stable",
]
