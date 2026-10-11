"""Physical checks for Gordion's spherical body profiles."""

import numpy as np
import pytest

from gordion.bodies import (
    BodyLayer,
    CelestialBody,
    EARTH_MASS_KG,
    EARTH_RADIUS_M,
    G,
    solar_system_bodies,
    two_layer_exoplanet,
)
from gordion.capture import NuclearSpecies


def test_solar_system_profiles_close_mass_and_have_finite_potentials():
    for body in solar_system_bodies().values():
        assert body.enclosed_mass_kg(body.radius_m) == pytest.approx(body.mass_kg)
        assert np.isfinite(body.potential_j_kg(np.array([0.0, body.radius_m / 2.0,
                                                       body.radius_m]))).all()
        assert body.escape_speed_m_s(body.radius_m) == pytest.approx(
            np.sqrt(2.0 * G * body.mass_kg / body.radius_m)
        )


def test_uniform_sphere_potential_matches_analytic_solution():
    body = CelestialBody("uniform", EARTH_MASS_KG, EARTH_RADIUS_M, (
        BodyLayer(1.0, 1.0, (NuclearSpecies(1.0, 28.0, "Si"),)),
    ))
    radius = np.array([0.0, 0.5, 1.0]) * EARTH_RADIUS_M
    expected = -G * EARTH_MASS_KG / (2.0 * EARTH_RADIUS_M) * (
        3.0 - (radius / EARTH_RADIUS_M) ** 2
    )
    assert body.potential_j_kg(radius) == pytest.approx(expected, rel=1e-12)


def test_cmf_changes_density_profile_but_preserves_mass_radius():
    low_cmf = two_layer_exoplanet("low CMF", 5.0, 1.5, 0.1)
    high_cmf = two_layer_exoplanet("high CMF", 5.0, 1.5, 0.7)
    assert low_cmf.enclosed_mass_kg(low_cmf.radius_m) == pytest.approx(low_cmf.mass_kg)
    assert high_cmf.enclosed_mass_kg(high_cmf.radius_m) == pytest.approx(high_cmf.mass_kg)
    assert low_cmf.density_kg_m3(0.6 * low_cmf.radius_m) != pytest.approx(
        high_cmf.density_kg_m3(0.6 * high_cmf.radius_m)
    )
    assert low_cmf.escape_speed_m_s(low_cmf.radius_m) == pytest.approx(
        high_cmf.escape_speed_m_s(high_cmf.radius_m)
    )