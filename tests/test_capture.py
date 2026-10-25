"""Regression tests for the Asteria-compatible capture-rate implementation."""

import numpy as np
import pytest
import subprocess
import sys

from gordion.bodies import (
    EARTH_RADIUS_M,
    JUPITER_MASS_KG,
    JUPITER_RADIUS_M,
    two_layer_exoplanet,
)
from gordion import evolve_capture_annihilation
from gordion.capture import AsteriaCapture, CaptureModel, NuclearSpecies


JUPITER_RADIUS_M = 7.0e7  # [m]
JUPITER_MASS_KG = 1.898e27  # [kg]
HALO_DENSITY_GEV_CM3 = 0.4  # [GeV/cm^3]
HALO_SPEED_M_S = 270000.0  # [m/s]


def test_capture_annihilation_evolution_matches_analytic_solution():
    capture_rate_per_second = 4.0  # [s^-1]
    annihilation_coefficient_per_second = 0.25  # [s^-1]
    age_s = 2.0  # [s]
    result = evolve_capture_annihilation(
        capture_rate_per_second, annihilation_coefficient_per_second, age_s
    )
    expected_fraction = np.tanh(age_s) ** 2

    assert result.captured_particles == pytest.approx(4.0 * np.tanh(age_s))
    assert result.equilibrium_fraction == pytest.approx(expected_fraction)
    assert result.annihilation_rate_per_second == pytest.approx(
        0.5 * capture_rate_per_second * expected_fraction
    )


def test_capture_annihilation_without_annihilation_has_linear_growth():
    capture_rate_per_second = 2.0  # [s^-1]
    age_s = 3.0  # [s]
    result = evolve_capture_annihilation(capture_rate_per_second, 0.0, age_s)

    assert result.captured_particles == pytest.approx(6.0)
    assert result.annihilation_rate_per_second == 0.0
    assert result.equilibrium_fraction == 0.0
    assert np.isinf(result.equilibration_timescale_s)


@pytest.mark.parametrize(
    "parameters",
    ((-1.0, 1.0, 1.0), (1.0, -1.0, 1.0), (1.0, 1.0, -1.0),
     (np.nan, 1.0, 1.0)),
)
def test_capture_annihilation_rejects_invalid_inputs(parameters):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        evolve_capture_annihilation(*parameters)


@pytest.fixture
def jupiter_capture():
    return CaptureModel(((0.75, 1.0), (0.25, 4.0)))


def test_capture_runs_without_asteria():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import builtins
original_import = builtins.__import__
def block_asteria(name, *args, **kwargs):
    if name.lower().startswith('asteria'):
        raise ImportError('Asteria is unavailable')
    return original_import(name, *args, **kwargs)
builtins.__import__ = block_asteria
from gordion import CaptureModel
model = CaptureModel(((0.75, 1.0), (0.25, 4.0)))
assert model.calculate(7e7, 1.898e27, 0.4, 270000, 10, 1e-34).rate_per_second > 0
""",
        ],
        check=True,
    )


def test_legacy_capture_name_remains_compatible():
    assert AsteriaCapture is CaptureModel


@pytest.mark.parametrize(
    ("sigma_cm2", "expected_rate"),
    (
        (1e-38, 1.3084135912028517e21),
        (1e-34, 1.2154380534286965e26),
        (1e-30, 1.6743441908017205e26),
    ),
)
def test_capture_matches_asteria_v1_jupiter_reference(jupiter_capture, sigma_cm2, expected_rate):
    result = jupiter_capture.multicapture(
        JUPITER_RADIUS_M, JUPITER_MASS_KG, HALO_DENSITY_GEV_CM3,
        HALO_SPEED_M_S, 0.94, sigma_cm2, 10.0,
    )

    assert result == pytest.approx(expected_rate, rel=3e-3)


def test_nucleon_scaling_and_nucleus_cross_section_differ(jupiter_capture):
    scaled = jupiter_capture.multicapture(
        JUPITER_RADIUS_M, JUPITER_MASS_KG, HALO_DENSITY_GEV_CM3,
        HALO_SPEED_M_S, 0.94, 1e-34, 10.0,
    )
    unscaled = jupiter_capture.multicapture_noA(
        JUPITER_RADIUS_M, JUPITER_MASS_KG, HALO_DENSITY_GEV_CM3,
        HALO_SPEED_M_S, 0.94, 1e-34, 10.0,
    )

    assert scaled > unscaled > 0.0


@pytest.mark.parametrize(
    ("sigma_cm2", "expected_rate"),
    (
        (1e-38, 1.805279673e19),
        (1e-32, 1.726779289e26),
        (1e-28, 1.748296812e26),
    ),
)
def test_no_scaling_matches_asteria_v1_jupiter_reference(
    jupiter_capture, sigma_cm2, expected_rate
):
    result = jupiter_capture.multicapture_noA(
        JUPITER_RADIUS_M, JUPITER_MASS_KG, HALO_DENSITY_GEV_CM3,
        HALO_SPEED_M_S, 0.94, sigma_cm2, 10.0,
    )

    assert result == pytest.approx(expected_rate, rel=2e-3)


def test_capture_result_exposes_regime_and_optical_depth(jupiter_capture):
    result = jupiter_capture.calculate(
        JUPITER_RADIUS_M, JUPITER_MASS_KG, HALO_DENSITY_GEV_CM3,
        HALO_SPEED_M_S, 10.0, 1e-34,
    )

    assert result.regime == "multiscatter"
    assert result.optical_depth > 1.5
    assert 0.0 <= result.reflection_fraction <= 1.0
    assert result.rate_per_second <= result.geometric_rate_per_second


def test_heavy_asymptotic_capture_reaches_reflection_limited_rate():
    mass_jupiter = 0.00103  # [M_J]
    radius_jupiter = 0.067  # [R_J]
    mass_earth = mass_jupiter * 317.83  # [M_earth]
    radius_earth = radius_jupiter * JUPITER_RADIUS_M / EARTH_RADIUS_M  # [R_earth]
    bulk_density = mass_earth / radius_earth**3 * 5513.4  # [kg m^-3]
    core_density = max(11000.0, 1.05 * 0.33 * bulk_density)  # [kg m^-3]
    body = two_layer_exoplanet(
        "Rocky asymptotic-regime benchmark",
        mass_earth,
        radius_earth,
        0.33,
        core_density_kg_m3=core_density,
    )
    result = CaptureModel(body.composition).calculate(
        body.radius_m,
        body.mass_kg,
        HALO_DENSITY_GEV_CM3,
        HALO_SPEED_M_S,
        1000.0,
        1e-32,
    )

    assert result.regime == "asymptotic-multiscatter"
    assert result.optical_depth >= 1e6
    assert result.rate_per_second >= 0.9 * result.geometric_rate_per_second
    assert result.rate_per_second <= result.geometric_rate_per_second


def test_zero_cross_section_has_zero_capture(jupiter_capture):
    result = jupiter_capture.calculate(
        JUPITER_RADIUS_M, JUPITER_MASS_KG, HALO_DENSITY_GEV_CM3,
        HALO_SPEED_M_S, 10.0, 0.0,
    )
    assert result.rate_per_second == 0.0
    assert result.regime == "no-scattering"


def test_composition_is_instance_scoped():
    hydrogen = CaptureModel((NuclearSpecies(1.0, 1.0, "H"),))
    iron = CaptureModel((NuclearSpecies(1.0, 56.0, "Fe"),))
    hydrogen_tau = hydrogen.optical_depth(JUPITER_RADIUS_M, JUPITER_MASS_KG, 10.0, 1e-36)
    iron_tau = iron.optical_depth(JUPITER_RADIUS_M, JUPITER_MASS_KG, 10.0, 1e-36)
    assert hydrogen_tau != iron_tau


def test_mixed_target_mass_is_optical_depth_weighted():
    composition = (NuclearSpecies(0.4, 16.0, "O"), NuclearSpecies(0.6, 56.0, "Fe"))
    model = CaptureModel(composition)
    optical_depths = np.array([
        species.mass_fraction * CaptureModel((NuclearSpecies(1.0, species.mass_number),))
        .optical_depth(JUPITER_RADIUS_M, JUPITER_MASS_KG, 10.0, 1e-36)
        for species in composition
    ])
    expected = sum(depth * species.mass_number
                   for depth, species in zip(optical_depths, composition)) / sum(optical_depths)

    target_mass = model._effective_target_mass(
        JUPITER_RADIUS_M, JUPITER_MASS_KG, 10.0, 1e-36, 0.94, True
    )

    assert target_mass == pytest.approx(expected)
