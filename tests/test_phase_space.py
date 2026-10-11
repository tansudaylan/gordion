"""Tests for six-dimensional sampling and elastic scattering transport."""

import numpy as np
import pytest

from gordion.bodies import solar_system_bodies
from gordion.lagrange import BinarySystem, lagrange_points, sample_lagrange_phase_space
from gordion.phase_space import (
    HaloModel,
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