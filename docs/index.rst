Gordion
=======

I use Gordion to calculate dark-matter capture and map the resulting phase
space inside explicit body models. Its independent capture module implements the multi-regime
treatment of Leane & Smirnov (2023), also released as Asteria. Asteria is not a
runtime dependency. Layered planetary profiles and trajectory scattering add spatial and
velocity distributions, including captured phase points. All particle outputs
are represented in six-dimensional position-velocity space.

The body and exoplanet examples use explicit density and composition models.
Those choices shape the capture rates and velocity distributions. They are
assumed interior models, not measurements of planetary structure.

.. toctree::
	:maxdepth: 2
	:caption: User guide

	getting_started
	api