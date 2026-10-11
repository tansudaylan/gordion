Gordion
=======

Gordion calculates dark-matter capture and phase-space distributions in
celestial bodies. Its independent capture module implements the multi-regime
treatment of Leane & Smirnov (2023), also released as Asteria. Asteria is not a
runtime dependency. Layered planetary profiles and trajectory scattering add spatial and
velocity distributions, including captured phase points. All particle outputs
are represented in six-dimensional position-velocity space.

The body and exoplanet examples use explicit density and composition models.
Their results depend on those assumptions and should not be read as measured
planetary interiors.

.. toctree::
	:maxdepth: 2
	:caption: User guide

	getting_started
	api