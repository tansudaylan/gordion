# Gordion

Gordion calculates dark-matter capture and phase-space distributions in
celestial bodies. It ports the multi-regime capture treatment in Asteria and
adds layered planetary profiles, elastic-scattering trajectories, weighted
six-dimensional samples, and co-rotating Lagrange-point distributions.

The trajectory model follows particles through spherical gravity and samples
isotropic, elastic scattering on stationary nuclei. The benchmark Earth, Moon,
and Mars profiles are explicit piecewise-uniform layers. Exoplanet profiles are
two-layer iron-core/silicate-mantle models whose mass and radius match the input
catalog values. These assumptions are useful for comparing systems; they are
not interior-structure measurements.

## Install

```bash
python -m pip install -e '.[examples]'
export GORDION_PATH=/path/to/gordion
```

`GORDION_PATH` identifies the repository. Runtime data and generated figures
are kept under its ignored `data/` and `visuals/` directories.

## Examples

Run the Solar System, exoplanet-catalog, CMF-grid, and Lagrange-point examples
from the repository root:

```bash
python examples/planetary_phase_space/run.py --mode all
```

The NASA Exoplanet Archive table is downloaded on the first run and cached
under `data/`. Use `--offline` to rerun from that snapshot or `--refresh` to
retrieve the current table. `--mode solar`, `--mode exoplanets`, `--mode cmf`,
and `--mode lagrange` run one part at a time. Timestamped figures are written
directly to `examples/planetary_phase_space/visuals/`. Capture summaries and
the full catalog's 6D samples are kept under the ignored
`data/planetary_phase_space/` directory.

The [interactive notebook](examples/planetary_phase_space/PlanetaryDarkMatterPhaseSpace.ipynb)
walks through the same calculations for Earth, the Moon, Mars, the exoplanet
catalog, CMF grids, and Lagrange points.

The archive-wide exoplanet example uses every catalog row with positive mass
and radius and assumes a core mass fraction of 0.33. Its two-layer profile uses
an iron-rich core and silicate mantle; the core density is raised when needed to
leave a positive mantle density at the catalog mass and radius. The code retains
the archive's mass provenance because some reported masses are minimum masses
or mass-radius estimates. CMF figures vary the core fraction explicitly. The
Lagrange-point example reports samples in a co-rotating frame. It does not claim
those particles are trapped near an equilibrium point.

At the defaults, the Solar System run samples 512 unbound velocities per body
and follows 128 scattering trajectories per body. The catalog sweep samples 12
phase-space points per eligible planet. Increase these options for smoother
distributions; run one mode at a time to limit compute and output size.

## Example figures

The Solar System panel compares unbound and scattering-path samples. The
catalog plot summarizes 6D samples for 6,371 planets with positive archived
mass and radius in the 2026-10-11 snapshot. The CMF and Lagrange figures use
explicit model profiles and co-rotating coordinates.

![Earth, Moon, and Mars phase-space samples with scattering trajectories](examples/planetary_phase_space/visuals/solar_system_phase_space_20261011_224225.png)

![Interior dark-matter speed estimates across the measured-mass/radius exoplanet catalog](examples/planetary_phase_space/visuals/all_exoplanet_velocity_distribution_20261011_224241.png)

![Density profiles, escape speeds, and Asteria capture rates across CMF models](examples/planetary_phase_space/visuals/exoplanet_cmf_density_profiles_20261011_224241.png)

![Co-rotating L1-L5 locations and velocity distributions in planetary systems](examples/planetary_phase_space/visuals/lagrange_point_phase_space_20261011_224241.png)

![Asteria multi-regime capture rates for Earth, Moon, and Mars](examples/planetary_phase_space/visuals/solar_system_asteria_capture_20261011_224241.png)

## Asteria

`gordion.capture.AsteriaCapture` ports the released Asteria v1.0 capture-rate
calculation by Leane & Smirnov (2023), including nucleon-to-nucleus scaling,
single and multiple scattering, the weak/intermediate/strong interaction
regimes, finite-target stopping, and the fitted reflection factor for light
dark matter. Composition is instance-scoped, so independent bodies can be
evaluated safely in the same process.

Reference: Leane & Smirnov (2023), *Dark Matter Capture in Celestial Objects:
Treatment Across Kinematic and Interaction Regimes*,
[arXiv:2309.00669](https://arxiv.org/abs/2309.00669). Asteria v1.0 is
[CC-BY 4.0 on Zenodo](https://doi.org/10.5281/zenodo.8150110).

## Runtime paths

Install the package and set its repository path before using repository-local data or visualization directories:

```bash
python -m pip install -e .
export GORDION_PATH=/path/to/gordion
```

`GORDION_PATH` identifies the repository root. Runtime inputs and generated figures belong in `data/` and `visuals/`, which are ignored by Git.
