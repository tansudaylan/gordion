# Gordion

The package takes its name from [Gordion](https://en.wikipedia.org/wiki/Gordion), the ancient Phrygian capital in central Anatolia associated with the Gordian-knot tradition. This Gordion follows how gravity and collisions change dark-matter trajectories and lead to capture.

Capturing dark matter can take many collisions, and the motion after each one
matters. I use Gordion to calculate capture across interaction regimes and sample
the resulting phase space in modeled celestial bodies. Its capture model follows the
multi-regime treatment of Leane & Smirnov (2023), also released as Asteria.
Around that core, Gordion provides layered body profiles, elastic-scattering
trajectories, weighted six-dimensional samples, and co-rotating Lagrange-point
distributions. It also evaluates capture--annihilation histories for a supplied
effective annihilation coefficient, and tests the ideal circular restricted
three-body stability of L4 and L5.

The library, planetary examples, and notebook run without Asteria. Only the
optional benchmark comparison imports it.

Gordion follows particles through spherical gravity and samples isotropic
elastic scattering on stationary nuclei. Earth, Moon, and Mars use explicit
piecewise-uniform layers. Exoplanets use two-layer iron-core/silicate-mantle
models, with core density chosen to match the catalog mass and radius. I use
these profiles to compare explicit interior assumptions. Mass and radius alone
leave a planet's interior structure underdetermined.

`CaptureModel` accepts one bulk composition and calculates the corresponding
capture rate. `CelestialBody` retains radial layers for the trajectory
integrator, which selects the local density and composition at each scattering
step. The analytic capture-rate grid does not integrate radial profiles.

`HaloModel` truncates the Maxwellian at the Galactic escape speed before
transforming velocities into the modeled body's frame. Its `body_velocity_m_s`
sets that frame transformation.

## Install

```bash
python -m pip install -e '.[examples]'
export GORDION_PATH=/path/to/gordion
```

`GORDION_PATH` identifies the repository. Runtime data and generated figures
are kept under its ignored `data/` and `visuals/` directories.

## Examples

To run the Solar System, exoplanet-catalog, CMF-grid, and Lagrange-point
calculations in one pass, run this from the repository root:

```bash
python examples/planetary_phase_space/run.py --mode all
```

To apply several interior assumptions to every eligible catalog planet, use:

```bash
python examples/planetary_phase_space/run.py --mode all --offline --catalog-cmfs 0.1 0.33 0.7 --bokeh
```

This produces one model for every planet and core mass fraction, rather than
limiting the comparison to the synthetic CMF templates. Both summary and full
6D sample tables label the assumed core mass fraction. Summaries also retain
inner and outer model densities, and the interactive view shows these values
on hover. The 0.33 default uses the same seeded samples as earlier runs.
Models of the same planet share halo random draws to compare the interior
assumptions with less Monte Carlo variation. Each model keeps its own positions,
velocities, and weights. These are unbound interior halo samples. Scattering
and first-capture samples are calculated separately in the Solar System mode.

Install `python -m pip install -e '.[bokeh]'` and add `--bokeh` with `--mode exoplanets` or `--mode all` to write an interactive mass, escape-speed, and phase-space-speed HTML plot beside the static catalog figure.

The phase-space example uses white backgrounds with black axes. Add
`--typeplotback dark` to use black backgrounds with white axes in both static
and interactive plots. Its plotting functions accept the same selection from
Python and otherwise inherit an enclosing TDpy `plot_style()` theme.

The NASA Exoplanet Archive table downloads on the first run and is cached under
`data/`. Use `--offline` to rerun from that snapshot or `--refresh` to retrieve
the current table. Choose `--mode solar`, `--mode exoplanets`, `--mode cmf`, or
`--mode lagrange` to run one part at a time. Timestamped figures go to
`examples/planetary_phase_space/visuals/`. Capture summaries and the catalog's
6D samples stay under the ignored `data/planetary_phase_space/` directory.

The [interactive notebook](examples/planetary_phase_space/PlanetaryDarkMatterPhaseSpace.ipynb)
works through the same calculations for Earth, the Moon, Mars, the exoplanet
catalog, CMF grids, and Lagrange points.

The catalog-wide exoplanet run keeps every row with positive mass and radius and
assumes a core mass fraction of 0.33. It uses an iron-rich core and silicate
mantle, raising the core density when needed to leave a positive mantle density
at the catalog mass and radius. It also keeps each archive mass's provenance
because some values are minimum masses or mass-radius estimates. CMF figures
vary the core fraction explicitly. Lagrange-point samples are reported in the
co-rotating frame. They show passing halo particles in the sampled neighborhoods.
The Routh criterion assumes point masses on circular orbits. Capture,
dissipation, and residence times require separate dynamical calculations.

`gordion.simulate_lagrange_retention` starts particles near a stable L4 or L5
point and evolves them in the dimensionless circular restricted three-body
problem. It counts particles that stay within a chosen radius of the equilibrium
point throughout the chosen number of binary periods. The result describes
survival after placement. Halo capture must be calculated separately.

`gordion.evolve_capture_annihilation` follows an initially empty dark-matter
population under constant capture and s-wave annihilation, with evaporation
omitted. The caller supplies the effective annihilation coefficient and its
spatial interpretation. Photon and neutrino spectra require annihilation yields
and transport models.

By default, the Solar System run samples 512 unbound velocities and follows
128 scattering trajectories per body. The catalog sweep samples 12 phase-space
points per eligible planet. Increase these counts for smoother distributions,
or run one mode at a time to keep compute and output sizes manageable.

## Example figures

The Solar System panel puts unbound samples beside scattering paths. The catalog
plot summarizes six-dimensional samples for 6,371 planets with positive archived
mass and radius in the 2026-10-11 snapshot. The CMF and Lagrange figures use
explicit model profiles and co-rotating coordinates.

![Earth, Moon, and Mars phase-space samples with scattering trajectories](examples/planetary_phase_space/visuals/solar_system_phase_space_20261011_224225.png)

![Interior dark-matter speed estimates across the measured-mass/radius exoplanet catalog](examples/planetary_phase_space/visuals/all_exoplanet_velocity_distribution_20261011_224241.png)

![Density profiles, escape speeds, and Gordion capture rates across CMF models](examples/planetary_phase_space/visuals/exoplanet_cmf_density_profiles_20261011_224241.png)

![Co-rotating L1-L5 locations and velocity distributions in planetary systems](examples/planetary_phase_space/visuals/lagrange_point_phase_space_20261011_224241.png)

![Gordion multi-regime capture rates for Earth, Moon, and Mars](examples/planetary_phase_space/visuals/solar_system_asteria_capture_20261011_224241.png)

## Capture Model and Attribution

`gordion.capture.CaptureModel` implements the capture-rate treatment
by Leane & Smirnov (2023), including nucleon-to-nucleus scaling,
single and multiple scattering, the weak/intermediate/strong interaction
regimes, finite-target stopping, and the fitted reflection factor for light
dark matter. Composition is instance-scoped, so independent bodies can be
evaluated safely in the same process.

The former `AsteriaCapture` name remains an alias for compatibility.

The scalar `speed_m_s` accepted by the rate model follows Asteria's effective
Maxwellian-speed convention. It is not the mean speed of the shifted,
truncated distribution sampled by `HaloModel` for trajectories. The paper
analysis uses 270 km/s as the RMS effective speed corresponding approximately
to a 220 km/s Maxwellian parameter.

## What Gordion Adds

The Asteria v1.0 Python release provides capture-rate functions configured
through a module-level composition. Gordion retains both nucleon-scaled and
nucleus-level rate conventions, while storing composition on each model
instance. Six Jupiter rate comparisons span both conventions and the
single-scatter, multiscatter, and asymptotic regimes. They agree within 0.3\%
and establish parity at those points, not improved capture-rate accuracy.

Beyond rate calculations, Gordion exposes weighted six-dimensional particle
states, scattering trajectories through piecewise-uniform body layers, and
first-capture states with their particle indices and flux weights. It also
provides constant-capture annihilation histories and restricted-three-body
tools for co-rotating Lagrange points. Asteria's paper also uses scattering
simulations to study reflection. Gordion's addition is an inspectable
per-particle transport result and the related orbital and annihilation
workflows.

## Optional Asteria Benchmark

Download the Python v1.0 release cited below and place its `asteria.py` on
`PYTHONPATH`. Then run the comparison from the Gordion repository root.

```bash
PYTHONPATH=/path/to/Asteria/Python python examples/asteria_benchmark/run.py
```

This example imports Asteria only when invoked. It compares Jupiter capture
rates for nucleon-scaled and nucleus-level cross sections at weak, intermediate,
and strong interaction strengths. It prints both rates and their relative
differences. The reported regimes come from Gordion's model.
With NumPy 2, the example restores `np.math` inside Asteria's module because the
v1.0 release expects it.

Reference: Leane & Smirnov (2023), *Dark Matter Capture in Celestial Objects:
Treatment Across Kinematic and Interaction Regimes*,
[arXiv:2309.00669](https://arxiv.org/abs/2309.00669). Asteria v1.0 is
[CC-BY 4.0 on Zenodo](https://doi.org/10.5281/zenodo.8150110).
