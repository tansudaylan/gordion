Getting started
===============

Install Gordion and the optional plotting dependency from the repository root:

.. code-block:: console

   python -m pip install -e '.[examples]'
   export GORDION_PATH=/path/to/gordion

Run all maintained examples with:

.. code-block:: console

   python examples/planetary_phase_space/run.py --mode all

The first exoplanet-catalog run queries the NASA Exoplanet Archive and stores a
CSV snapshot under ``data/``. Add ``--offline`` to reuse it or ``--refresh`` to
retrieve an updated table. Timestamped figures are written directly under the
example's ``visuals/`` directory. Capture summaries and the full catalog's 6D
samples are stored in Gordion's ignored ``data/planetary_phase_space/`` folder.

The supported modes are:

* ``solar`` samples phase space and elastic-scattering trajectories inside
  Earth, the Moon, and Mars and plots Gordion capture rates.
* ``exoplanets`` analyzes every archive row with positive mass and radius. Its
  interior profile assumes a 0.33 core mass fraction, and the catalog's mass
  provenance is retained in the outputs.
* ``cmf`` compares core mass fractions and radial profiles for several
  explicitly specified planet mass-radius pairs.
* ``lagrange`` computes L1--L5 in the circular restricted three-body problem
  and samples velocity distributions in the co-rotating frame.

An interactive walkthrough is available in
``examples/planetary_phase_space/PlanetaryDarkMatterPhaseSpace.ipynb``.

Optional Asteria comparison
------------------------------

Gordion runs independently of Asteria. Only the optional benchmark example
imports the reference implementation. Download Asteria's Python v1.0 release
and expose its ``asteria.py`` module on ``PYTHONPATH``.

.. code-block:: console

  PYTHONPATH=/path/to/Asteria/Python python examples/asteria_benchmark/run.py

The example reports Jupiter capture rates from both implementations and their
relative differences for both cross-section conventions.

Physics assumptions
-------------------

The self-contained ``gordion.CaptureModel`` follows the homogeneous-composition
formalism of Leane & Smirnov (2023). The old ``AsteriaCapture`` name remains
a compatibility alias and does not require the reference package.
The trajectory calculation instead uses spherical, piecewise-uniform layers,
cold stationary targets, isotropic elastic scattering, and Newtonian gravity.
It identifies capture when a scattered particle's specific orbital energy
becomes negative. It does not model thermal target motion, inelastic channels,
annihilation, DM self-interactions, or post-capture thermalization.

Unbound interior samples use Liouville's theorem and energy conservation to map
a shifted Maxwellian speed distribution into the body's potential. Their
straight-ray direction mapping is an approximation. The Lagrange-point samples
include the binary gravitational potential and co-rotating velocity transform,
but do not model capture or stable trapping in the restricted three-body
problem.

References
----------

Leane & Smirnov (2023), ``Dark Matter Capture in Celestial Objects: Treatment
Across Kinematic and Interaction Regimes``.
`arXiv:2309.00669 <https://arxiv.org/abs/2309.00669>`_. The Asteria v1.0 source
is available under CC-BY 4.0 at
`Zenodo DOI 10.5281/zenodo.8150110 <https://doi.org/10.5281/zenodo.8150110>`_.

NASA Exoplanet Archive, Planetary Systems table,
`TAP service <https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html>`_.