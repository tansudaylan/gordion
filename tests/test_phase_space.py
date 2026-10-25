"""Tests for six-dimensional sampling and elastic scattering transport."""

import numpy as np
import pytest

from gordion.bodies import solar_system_bodies
from gordion.lagrange import (
    ROUTH_CRITICAL_MASS_FRACTION,
    BinarySystem,
    lagrange_points,
    sample_lagrange_phase_space,
    simulate_lagrange_retention,
    triangular_points_stable,
)
from gordion.phase_space import (
    HaloModel,
    _sample_halo_velocity,
    sample_unbound_phase_space,
    simulate_capture_trajectories,
)


def test_unbound_samples_are_finite_six_dimensional_and_focused():
    earth = solar_system_bodies()["Earth"]
    halo = HaloModel()
    samples = sample_unbound_phase_space(earth, 256, halo, seed=11)
    radius = np.linalg.norm(samples.position_m, axis=1)
    speed = np.linalg.norm(samples.velocity_m_s, axis=1)
    assert samples.coordinates.shape == (256, 6)
    assert np.isfinite(samples.coordinates).all()
    assert np.all(radius <= earth.radius_m)
    assert np.all(speed >= earth.escape_speed_m_s(radius))
    assert np.all(samples.weight >= 1.0)


def test_shifted_halo_velocity_is_truncated_in_the_galactic_frame():
    halo = HaloModel(body_velocity_m_s=(232000.0, -35000.0, 0.0))  # [m s^-1]
    relative_velocity = _sample_halo_velocity(
        halo, 4096, np.random.default_rng(13)
    )
    galactic_velocity = relative_velocity + np.asarray(halo.body_velocity_m_s)

    assert np.all(
        np.linalg.norm(galactic_velocity, axis=1)
        < halo.galactic_escape_speed_m_s
    )
    np.testing.assert_allclose(
        relative_velocity.mean(axis=0),
        -np.asarray(halo.body_velocity_m_s),
        atol=15000.0,  # [m s^-1]
    )


def test_transport_returns_capture_phase_points_and_passes_scatter_sanity():
    earth = solar_system_bodies()["Earth"]
    result = simulate_capture_trajectories(
        earth, 32, 10.0, 1e-30, HaloModel(), step_fraction=0.05,
        max_steps=1000, sample_stride=5, seed=4, sanity_checks=True,
    )
    assert result.trajectory_samples.coordinates.shape[1] == 6
    assert result.capture_phase_space.coordinates.shape[1] == 6
    assert set(np.unique(result.particle_status)).issubset(
        {"captured", "escaped", "bound", "step_limit"}
    )
    captured = result.particle_status == "captured"
    assert np.all(result.scatter_count[captured] > 0)
    assert np.isfinite(result.capture_rate_per_second)
    assert result.capture_rate_per_second >= 0.0


def test_transport_is_reproducible_with_independent_random_streams():
    earth = solar_system_bodies()["Earth"]
    arguments = (earth, 8, 10.0, 1e-30, HaloModel())
    options = {
        "step_fraction": 0.05,
        "max_steps": 1000,
        "sample_stride": 5,
        "seed": 17,
    }

    first = simulate_capture_trajectories(*arguments, **options)
    second = simulate_capture_trajectories(*arguments, **options)

    np.testing.assert_array_equal(first.particle_status, second.particle_status)
    np.testing.assert_array_equal(
        first.trajectory_samples.coordinates,
        second.trajectory_samples.coordinates,
    )
    np.testing.assert_array_equal(
        first.capture_phase_space.coordinates,
        second.capture_phase_space.coordinates,
    )


def test_lagrange_points_are_equilibria_and_samples_have_six_coordinates():
    system = BinarySystem(1.9885e30, 5.9722e24, 1.495978707e11)
    points = lagrange_points(system)
    mu = system.mass_fraction

    def residual(position_m):
        x, y, z = position_m / system.separation_m
        r1 = np.sqrt((x + mu) ** 2 + y**2 + z**2)
        r2 = np.sqrt((x - 1.0 + mu) ** 2 + y**2 + z**2)
        return np.array([
            x - (1.0 - mu) * (x + mu) / r1**3 - mu * (x - 1.0 + mu) / r2**3,
            y - (1.0 - mu) * y / r1**3 - mu * y / r2**3,
            -(1.0 - mu) * z / r1**3 - mu * z / r2**3,
        ])

    assert set(points) == {"L1", "L2", "L3", "L4", "L5"}
    assert max(np.linalg.norm(residual(point)) for point in points.values()) < 1e-10
    samples = sample_lagrange_phase_space(system, 24, seed=5)
    assert all(value.coordinates.shape == (24, 6) for value in samples.values())
    assert all(np.isfinite(value.coordinates).all() for value in samples.values())


@pytest.mark.parametrize(
    ("secondary_fraction", "expected"),
    ((0.01, True),
     (ROUTH_CRITICAL_MASS_FRACTION, False),
     (0.04, False)),
)
def test_triangular_point_stability_uses_routh_limit(secondary_fraction, expected):
    primary_mass_kg = 1.0 - secondary_fraction  # [kg, scaled test system]
    secondary_mass_kg = secondary_fraction  # [kg, scaled test system]
    separation_m = 1.0  # [m, scaled test system]
    system = BinarySystem(primary_mass_kg, secondary_mass_kg, separation_m)
    assert triangular_points_stable(system) is expected


def test_lagrange_retention_keeps_an_exact_equilibrium_stationary():
    primary_mass_kg = 0.99  # [kg, scaled test system]
    secondary_mass_kg = 0.01  # [kg, scaled test system]
    separation_m = 1.0  # [m, scaled test system]
    system = BinarySystem(primary_mass_kg, secondary_mass_kg, separation_m)
    result = simulate_lagrange_retention(
        system,
        "L4",
        8,
        periods=4.0,
        cloud_width_fraction=0.0,
        velocity_dispersion_fraction=0.0,
        seed=12,
    )

    assert result.retained_count == 8
    assert result.retained_fraction == 1.0
    assert result.median_max_distance_fraction < 1e-8
    assert result.p84_max_distance_fraction < 1e-8


def test_lagrange_retention_requires_a_stable_triangular_point():
    primary_mass_kg = 0.96  # [kg, scaled test system]
    secondary_mass_kg = 0.04  # [kg, scaled test system]
    separation_m = 1.0  # [m, scaled test system]
    system = BinarySystem(primary_mass_kg, secondary_mass_kg, separation_m)

    with pytest.raises(ValueError, match="linearly stable L4 or L5"):
        simulate_lagrange_retention(system, "L4", 8, seed=12)


def test_lagrange_retention_summary_is_reproducible():
    primary_mass_kg = 0.99  # [kg, scaled test system]
    secondary_mass_kg = 0.01  # [kg, scaled test system]
    separation_m = 1.0  # [m, scaled test system]
    system = BinarySystem(primary_mass_kg, secondary_mass_kg, separation_m)
    options = {
        "periods": 5.0,
        "cloud_width_fraction": 1e-3,
        "velocity_dispersion_fraction": 1e-4,
        "retention_radius_fraction": 0.1,
        "seed": 4,
    }
    first = simulate_lagrange_retention(system, "L4", 32, **options)
    second = simulate_lagrange_retention(system, "L4", 32, **options)

    assert first == second
    assert 0.0 <= first.retained_fraction <= 1.0
    assert 0.0 <= first.median_max_distance_fraction
    assert first.p84_max_distance_fraction >= first.median_max_distance_fraction