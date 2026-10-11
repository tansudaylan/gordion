#!/usr/bin/env python3
"""Estimate capture and six-dimensional dark-matter distributions in planets."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("agg")

import matplotlib.pyplot as plt
import numpy as np
from tdpy.verbosity import print

from gordion.bodies import (
    EARTH_MASS_KG,
    EARTH_RADIUS_M,
    JUPITER_MASS_KG,
    JUPITER_RADIUS_M,
    SUN_MASS_KG,
    CelestialBody,
    solar_system_bodies,
    two_layer_exoplanet,
)
from gordion.capture import AsteriaCapture, G
from gordion.catalog import ExoplanetRecord, fetch_confirmed_exoplanets
from gordion.lagrange import BinarySystem, lagrange_points, sample_lagrange_phase_space
from gordion.paths import get_data_path
from gordion.phase_space import (
    HaloModel,
    sample_unbound_phase_space,
    simulate_capture_trajectories,
)


DM_MASS_GEV = 10.0  # [GeV]
NUCLEON_SIGMA_CM2 = 1e-36  # [cm^2]
NUCLEON_MASS_GEV = 0.94  # [GeV]
HALO_DENSITY_GEV_CM3 = 0.4  # [GeV/cm^3]
HALO_SPEED_M_S = 270000.0  # [m/s]
LOCAL_STANDARD_OF_REST_M_S = 232000.0  # [m/s]
EARTH_MASSES_PER_JUPITER = JUPITER_MASS_KG / EARTH_MASS_KG
EARTH_RADII_PER_JUPITER = JUPITER_RADIUS_M / EARTH_RADIUS_M


def _visual_directory() -> Path:
    path = Path(__file__).resolve().parent / "visuals"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _data_directory() -> Path:
    path = get_data_path() / "planetary_phase_space"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def _save_figure(figure, name: str, tag: str) -> Path:
    path = _visual_directory() / f"{name}_{tag}.png"
    print(f"Writing to {path}...")
    figure.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(figure)
    return path


def _write_rows(path: Path, fieldnames: list[str], rows) -> None:
    print(f"Writing to {path}...")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _halo(body_velocity_m_s=(LOCAL_STANDARD_OF_REST_M_S, 0.0, 0.0)) -> HaloModel:
    return HaloModel(
        density_gev_cm3=HALO_DENSITY_GEV_CM3,
        speed_scale_m_s=220000.0,
        galactic_escape_speed_m_s=544000.0,
        body_velocity_m_s=body_velocity_m_s,
    )


def _plot_solar_system(samples: int, transport_particles: int,
                       sanity_checks: bool, seed: int, tag: str) -> list[dict]:
    bodies = solar_system_bodies()
    figure, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    summary = []
    orbital_speeds_m_s = {"Earth": 29780.0, "Moon": 30800.0, "Mars": 24100.0}  # [m/s]
    for row, (name, body) in enumerate(bodies.items()):
        halo = _halo((232000.0, orbital_speeds_m_s[name], 0.0))
        unbound = sample_unbound_phase_space(body, samples, halo, seed=seed + row)
        transport = simulate_capture_trajectories(
            body,
            transport_particles,
            DM_MASS_GEV,
            NUCLEON_SIGMA_CM2,
            halo,
            step_fraction=0.04,
            max_steps=2500,
            sample_stride=4,
            seed=seed + row,
            sanity_checks=sanity_checks,
        )
        model = AsteriaCapture(body.composition)
        capture = model.calculate(
            body.radius_m, body.mass_kg, HALO_DENSITY_GEV_CM3,
            HALO_SPEED_M_S, DM_MASS_GEV, NUCLEON_SIGMA_CM2,
            NUCLEON_MASS_GEV,
        )
        free_speed = np.linalg.norm(unbound.velocity_m_s, axis=1) / 1000.0
        axes[row, 0].hist(free_speed, bins=24, weights=unbound.weight,
                          density=True, histtype="step", linewidth=1.7,
                          color="#007360", label="Unbound halo, focused")
        if len(transport.trajectory_samples.weight):
            path_speed = np.linalg.norm(transport.trajectory_samples.velocity_m_s, axis=1) / 1000.0
            axes[row, 0].hist(path_speed, bins=24,
                              weights=transport.trajectory_samples.weight,
                              density=True, histtype="step", linewidth=1.5,
                              color="#cc6677", label="Scattering paths")
        if len(transport.capture_phase_space.weight):
            captured_speed = np.linalg.norm(transport.capture_phase_space.velocity_m_s, axis=1) / 1000.0
            axes[row, 0].hist(captured_speed, bins=12,
                              weights=transport.capture_phase_space.weight,
                              density=True, histtype="step", linewidth=1.5,
                              color="#332288", label="First-capture states")
        axes[row, 0].set(xlabel="Speed [km s$^{-1}$]", ylabel="Weighted density",
                         title=f"{name}: speed distribution")
        axes[row, 0].legend(frameon=False, fontsize=8)
        if len(transport.trajectory_samples.weight):
            path = transport.trajectory_samples
            speed = np.linalg.norm(path.velocity_m_s, axis=1) / 1000.0
            axes[row, 1].scatter(path.position_m[:, 0] / body.radius_m,
                                 path.velocity_m_s[:, 0] / 1000.0,
                                 c=speed, s=7, alpha=0.55, cmap="viridis",
                                 rasterized=True)
        axes[row, 1].set(xlabel="$x/R$", ylabel="$v_x$ [km s$^{-1}$]",
                         title="Position-velocity projection")
        summary.append({
            "body": name,
            "mass_kg": body.mass_kg,
            "radius_m": body.radius_m,
            "cmf_model": body.layers[0].mass_fraction,
            "asteria_capture_rate_s-1": capture.rate_per_second,
            "asteria_regime": capture.regime,
            "optical_depth": capture.optical_depth,
            "transport_captured_fraction": float(np.mean(transport.particle_status == "captured")),
            "transport_particles": transport_particles,
        })
    figure.suptitle("Dark matter in the Earth, Moon, and Mars", fontsize=15)
    _save_figure(figure, "solar_system_phase_space", tag)
    _write_rows(_data_directory() / f"solar_system_capture_{tag}.csv",
                list(summary[0]), summary)
    return summary


def _catalog_body(record: ExoplanetRecord, cmf: float = 0.33) -> CelestialBody:
    mass_earth = record.mass_jupiter * EARTH_MASSES_PER_JUPITER
    radius_earth = record.radius_jupiter * EARTH_RADII_PER_JUPITER
    bulk_density = mass_earth / radius_earth**3 * 5513.4  # [kg m^-3]
    core_density = max(11000.0, 1.05 * cmf * bulk_density)  # [kg m^-3]
    return two_layer_exoplanet(record.name, mass_earth, radius_earth, cmf,
                               core_density_kg_m3=core_density)


def _plot_exoplanets(records: list[ExoplanetRecord], samples_per_planet: int,
                     seed: int, tag: str) -> int:
    sample_rows = []
    summary_rows = []
    speed_all = []
    for index, record in enumerate(records):
        body = _catalog_body(record)
        orbital_speed = 0.0
        if record.period_days is not None and record.host_mass_solar is not None:
            star_mass = record.host_mass_solar * SUN_MASS_KG
            period_s = record.period_days * 86400.0  # [s]
            semimajor_axis = (G * (star_mass + body.mass_kg) * period_s**2
                              / (4.0 * np.pi**2)) ** (1.0 / 3.0)
            orbital_speed = np.sqrt(G * (star_mass + body.mass_kg) / semimajor_axis)
        halo = _halo((LOCAL_STANDARD_OF_REST_M_S, orbital_speed, 0.0))
        samples = sample_unbound_phase_space(body, samples_per_planet, halo,
                                             seed=seed + index)
        speed = np.linalg.norm(samples.velocity_m_s, axis=1)
        mass_earth = record.mass_jupiter * EARTH_MASSES_PER_JUPITER
        radius_earth = record.radius_jupiter * EARTH_RADII_PER_JUPITER
        summary_rows.append({
            "planet": record.name,
            "host": record.host_name,
            "mass_provenance": record.mass_provenance,
            "mass_earth": mass_earth,
            "radius_earth": radius_earth,
            "assumed_cmf": 0.33,
            "mean_density_kg_m3": body.mass_kg / (4.0 * np.pi * body.radius_m**3 / 3.0),
            "surface_escape_speed_km_s": body.escape_speed_m_s(body.radius_m) / 1000.0,
            "mean_interior_dm_speed_km_s": float(np.average(speed, weights=samples.weight) / 1000.0),
            "p16_interior_dm_speed_km_s": float(np.percentile(speed, 16) / 1000.0),
            "p84_interior_dm_speed_km_s": float(np.percentile(speed, 84) / 1000.0),
        })
        speed_all.extend(speed / 1000.0)
        for point, velocity, weight in zip(samples.position_m, samples.velocity_m_s, samples.weight):
            sample_rows.append({
                "planet": record.name,
                "host": record.host_name,
                "mass_provenance": record.mass_provenance,
                "x_m": point[0], "y_m": point[1], "z_m": point[2],
                "vx_m_s": velocity[0], "vy_m_s": velocity[1], "vz_m_s": velocity[2],
                "weight": weight,
            })
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    masses = np.array([row["mass_earth"] for row in summary_rows])
    escape = np.array([row["surface_escape_speed_km_s"] for row in summary_rows])
    mean_speed = np.array([row["mean_interior_dm_speed_km_s"] for row in summary_rows])
    points = axes[0].scatter(masses, escape, c=mean_speed, s=16, alpha=0.72,
                             cmap="viridis", rasterized=True)
    axes[0].set(xscale="log", xlabel="Planet mass [Earth masses]",
                ylabel="Surface escape speed [km s$^{-1}$]",
                title="Catalog planets with measured mass and radius")
    fig.colorbar(points, ax=axes[0], label="Mean interior DM speed [km s$^{-1}$]")
    axes[1].hist(speed_all, bins=32, color="#007360", histtype="step", linewidth=1.5)
    axes[1].set(xlabel="Interior DM speed [km s$^{-1}$]", ylabel="Sample count",
                title="Unbound 6D catalog samples")
    _save_figure(fig, "all_exoplanet_velocity_distribution", tag)
    output_dir = _data_directory()
    _write_rows(output_dir / f"exoplanet_phase_space_summary_{tag}.csv",
                list(summary_rows[0]), summary_rows)
    _write_rows(output_dir / f"exoplanet_phase_space_samples_{tag}.csv",
                list(sample_rows[0]), sample_rows)
    return len(summary_rows)


def _plot_cmf_grid(seed: int, tag: str) -> int:
    templates = (("Earth-size rocky", 1.0, 1.0),
                 ("Super-Earth", 5.0, 1.5),
                 ("Sub-Neptune", 8.0, 2.5))
    cmfs = (0.0, 0.1, 0.3, 0.5, 0.7)
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(cmfs)))
    fig, axes = plt.subplots(len(templates), 3, figsize=(14, 10), constrained_layout=True)
    rows = []
    radial_fraction = np.linspace(0.01, 1.0, 180)
    for row_index, (template, mass, radius) in enumerate(templates):
        for cmf, color in zip(cmfs, colors):
            body = two_layer_exoplanet(template, mass, radius, cmf)
            radial = radial_fraction * body.radius_m
            density = body.density_kg_m3(radial)
            escape = body.escape_speed_m_s(radial)
            samples = sample_unbound_phase_space(body, 128, _halo(),
                                                 seed=seed + row_index * 10 + int(cmf * 100))
            mean_speed = float(np.average(np.linalg.norm(samples.velocity_m_s, axis=1),
                                          weights=samples.weight))
            cap = AsteriaCapture(body.composition).calculate(
                body.radius_m, body.mass_kg, HALO_DENSITY_GEV_CM3,
                HALO_SPEED_M_S, DM_MASS_GEV, NUCLEON_SIGMA_CM2,
                NUCLEON_MASS_GEV,
            )
            axes[row_index, 0].plot(radial_fraction, density, color=color,
                                    label=f"CMF={cmf:.1f}")
            axes[row_index, 1].plot(radial_fraction, escape / 1000.0, color=color)
            axes[row_index, 2].scatter(cmf, cap.rate_per_second, color=color, s=30)
            rows.append({
                "template": template, "mass_earth": mass, "radius_earth": radius,
                "cmf": cmf, "inner_profile_density_kg_m3": body.density_kg_m3(
                    max(body.layers[0].outer_radius_fraction * body.radius_m * 0.5, 1.0)),
                "mean_interior_dm_speed_km_s": mean_speed / 1000.0,
                "capture_rate_s-1": cap.rate_per_second,
            })
        axes[row_index, 0].set(ylabel="Density [kg m$^{-3}$]", title=template)
        axes[row_index, 1].set(ylabel="Escape speed [km s$^{-1}$]")
        axes[row_index, 2].set(ylabel="Capture rate [s$^{-1}$]", xlabel="Core mass fraction")
        axes[row_index, 2].set_yscale("log")
        axes[row_index, 0].set_xlabel("Radius / planet radius")
        axes[row_index, 1].set_xlabel("Radius / planet radius")
    axes[0, 0].legend(frameon=False, fontsize=8)
    _save_figure(fig, "exoplanet_cmf_density_profiles", tag)
    _write_rows(_data_directory() / f"cmf_profile_summary_{tag}.csv",
                list(rows[0]), rows)
    return len(rows)


def _catalog_lagrange_systems(records: list[ExoplanetRecord]) -> list[BinarySystem]:
    selected = []
    used_hosts = set()
    for record in sorted(records, key=lambda item: (
        item.period_days is None,
        item.period_days if item.period_days is not None else float("inf"),
    )):
        if record.host_name in used_hosts or record.period_days is None or record.host_mass_solar is None:
            continue
        if record.host_name not in {"HD 108236", "TRAPPIST-1", "WASP-121"}:
            continue
        planet_mass_kg = record.mass_jupiter * JUPITER_MASS_KG
        star_mass_kg = record.host_mass_solar * SUN_MASS_KG
        period_s = record.period_days * 86400.0  # [s]
        separation = (G * (star_mass_kg + planet_mass_kg) * period_s**2 / (4.0 * np.pi**2)) ** (1.0 / 3.0)
        selected.append(BinarySystem(star_mass_kg, planet_mass_kg, separation,
                                     record.host_name, record.name))
        used_hosts.add(record.host_name)
    return selected


def _plot_lagrange_points(records: list[ExoplanetRecord], samples: int,
                          seed: int, tag: str) -> int:
    earth_a = 1.495978707e11  # [m]
    jupiter_a = 5.2044 * earth_a  # [m]
    systems = [
        BinarySystem(SUN_MASS_KG, EARTH_MASS_KG, earth_a, "Sun", "Earth"),
        BinarySystem(SUN_MASS_KG, JUPITER_MASS_KG, jupiter_a, "Sun", "Jupiter"),
    ]
    systems.extend(_catalog_lagrange_systems(records))
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, 5))
    fig, axes = plt.subplots(len(systems), 2, figsize=(12, 3.4 * len(systems)),
                             squeeze=False, constrained_layout=True)
    rows = []
    for system_index, system in enumerate(systems):
        points = lagrange_points(system)
        phase = sample_lagrange_phase_space(system, samples, _halo(), seed=seed + system_index)
        location_axis, speed_axis = axes[system_index]
        mass_fraction = system.mass_fraction
        location_axis.scatter(-mass_fraction, 0.0, color="black", s=42,
                              marker="*", zorder=4, label=system.primary_name)
        location_axis.scatter(1.0 - mass_fraction, 0.0, color="#cc6677", s=24,
                              zorder=4, label=system.secondary_name)
        for point_index, (label, location) in enumerate(points.items()):
            x_position = location[0] / system.separation_m
            y_position = location[1] / system.separation_m
            location_axis.scatter(x_position, y_position,
                                  color=colors[point_index], s=23, zorder=3)
            label_offset = {"L1": (-20, 8), "L2": (5, 8), "L3": (-12, 8),
                            "L4": (5, 4), "L5": (5, -10)}[label]
            location_axis.annotate(label, (x_position, y_position), xytext=label_offset,
                                   textcoords="offset points", fontsize=8)
            velocity = np.linalg.norm(phase[label].velocity_m_s, axis=1) / 1000.0
            speed_axis.hist(velocity, bins=24, density=True, histtype="step",
                            color=colors[point_index], alpha=0.8, label=label)
            for speed in velocity:
                rows.append({"primary": system.primary_name,
                             "secondary": system.secondary_name,
                             "point": label,
                             "x_rotating_m": location[0],
                             "y_rotating_m": location[1],
                             "z_rotating_m": location[2],
                             "sample_speed_rotating_km_s": speed})
        location_axis.set(xlabel="Barycentric x / separation",
                          ylabel="Barycentric y / separation",
                          title=f"{system.primary_name}–{system.secondary_name}",
                          aspect="equal", xlim=(-1.2, 1.3), ylim=(-1.15, 1.15))
        speed_axis.set(xlabel="Co-rotating DM speed [km s$^{-1}$]",
                       ylabel="Probability density", title="Local velocity distribution")
        speed_axis.legend(frameon=False, fontsize=8, ncol=5)
    _save_figure(fig, "lagrange_point_phase_space", tag)
    _write_rows(_data_directory() / f"lagrange_phase_space_{tag}.csv",
                list(rows[0]), rows)
    return len(systems)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", "solar", "exoplanets", "cmf", "lagrange"),
                        default="all")
    parser.add_argument("--samples", type=int, default=512,
                        help="phase-space samples per Solar System body or Lagrange point")
    parser.add_argument("--transport-particles", type=int, default=128)
    parser.add_argument("--catalog-samples", type=int, default=12)
    parser.add_argument("--limit-planets", type=int,
                        help="limit catalog rows for a quick development run")
    parser.add_argument("--offline", action="store_true",
                        help="require the cached NASA Exoplanet Archive table")
    parser.add_argument("--refresh", action="store_true",
                        help="download a fresh NASA Exoplanet Archive table")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--sanity-checks", action="store_true",
                        help="check elastic-scatter energy and momentum conservation")
    arguments = parser.parse_args()
    if arguments.samples < 1 or arguments.catalog_samples < 1 or arguments.transport_particles < 1:
        parser.error("sample counts must be positive")
    tag = _run_tag()
    need_catalog = arguments.mode in {"all", "exoplanets", "lagrange"}
    records = []
    if need_catalog:
        catalog_cache = get_data_path() / "exoplanet_archive" / "planetary_systems.csv"
        if arguments.offline and not catalog_cache.is_file():
            parser.error("--offline requires a cached data/exoplanet_archive/planetary_systems.csv")
        records = fetch_confirmed_exoplanets(
            refresh=arguments.refresh and not arguments.offline,
            cache_path=catalog_cache,
        )
    if arguments.limit_planets is not None:
        records = records[:arguments.limit_planets]
    if arguments.mode in {"all", "solar"}:
        _plot_solar_system(arguments.samples, arguments.transport_particles,
                           arguments.sanity_checks, arguments.seed, tag)
    if arguments.mode in {"all", "exoplanets"}:
        _plot_exoplanets(records, arguments.catalog_samples, arguments.seed, tag)
    if arguments.mode in {"all", "cmf"}:
        _plot_cmf_grid(arguments.seed, tag)
    if arguments.mode in {"all", "lagrange"}:
        _plot_lagrange_points(records, arguments.samples, arguments.seed, tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())