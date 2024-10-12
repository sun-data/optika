Introduction
============

:mod:`optika` is a Python package for designing optical systems inspired by
`Zemax <https://en.wikipedia.org/wiki/Zemax>`_.
It allows the user to compute the spectral response and resolution of an
arbitrary optical system and optimize it using :mod:`scipy.optimize`.
The main design goals of :mod:`optika` are to

* Use :mod:`astropy.units` to specify the parameters of an optical system.
* Automatically compute the field of view and entrance pupil of a given optical
  system.
* Allow for :math:`n`-dimensional configurations of an optical system by allowing
  its parameters to be array-like.
* Compute uncertainties in the performance of an optical system using
  the Monte-Carlo method.

To satisfy the last two design goals, :mod:`optika` uses the
purpose-built :mod:`named_arrays` package as a backend.
:mod:`named_arrays` is an implementation of a
`named tensor <https://nlp.seas.harvard.edu/NamedTensor>`_,
which allows the user to name the axes in an :math:`n`-dimensional array.
This makes specifying :math:`n`-dimensional configurations in :mod:`optika`
easier since the user doesn't have to manually insert singleton dimensions
to broadcast orthogonal configuration changes against each other.
Furthermore, :mod:`named_arrays` provides an implementation of a 3D vector,
:class:`named_arrays.Cartesian3dVectorArray`, which is convenient to use since
many of the inputs and outputs of :mod:`optika` can be represented as 3D vectors.

Features
--------

* Sequential raytrace modeling of an optical system
* Stratified random sampling of input rays for faster convergence
* Image simulation of a given scene using an optical system
* Spherical, conical, and toroidal surface sag profiles
* Circular, rectangular, and polygonal apertures
* Support for mirrors and arbitrary multilayer coatings
* Diffraction grating support

  * Constant, polynomial and holographic ruling spacing
  * Sinusoidal, square, rectangular, sawtooth, and triangular ruling profiles

* CCD/CMOS sensor simulation

  * Quantum efficiency
  * Charge diffusion

Missing Features
----------------

* **Polarization**. Different polarization states are not propagated through the
  system.
* **Physical Optics**. Only geometric optics is supported right now, but adding
  a Fourier optics propagator is a longstanding goal of the project.
* **Glass Optical Constants**. :mod:`optika` has a wide array of optical
  constants from sources such as :cite:t:`Palik1997,Henke1993`,
  but it does not yet have a database for different types of glass like Zemax
  does.

Differences from Zemax
----------------------

* The position and orientation of surfaces in :mod:`optika` are specified in
  global coordinates instead of coordinates relative to the last surface.

* The field of view is automatically calculated, there is no need to set the
  extent of the field.

* Diffraction grating rulings are now a parameter of an optical surface.
  There is no need to change the type of surface to allow for different ruling
  designs.


Tutorials
=========

Jupyter notebook examples on how to use :mod:`optika`.

.. toctree::
    :maxdepth: 1

    tutorials/prime_focus


API Reference
=============

An in-depth description of the interfaces in this package.

.. autosummary::
    :toctree: _autosummary
    :template: module_custom.rst
    :recursive:

    optika


References
==========

.. bibliography::

|


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
