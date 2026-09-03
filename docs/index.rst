optika
======

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
:class:`~named_arrays.Cartesian3dVectorArray`, which is convenient to use since
many of the inputs and outputs of :mod:`optika` can be represented as 3D vectors.

Installation
------------

:mod:`optika` is published on PyPI and can be installed using::

    pip install optika


Features
--------

* Sequential raytrace modeling of an optical system
* Stratified random sampling of input rays for faster convergence
* Image simulation of a given scene using an optical system
* Fast linear forward model approximating a raytraced system, for imaging many
  scenes without raytracing each one
* Spherical, conical, and toroidal surface sag profiles
* Circular, rectangular, and polygonal apertures
* Support for mirrors and arbitrary multilayer coatings
* Refractive glass materials with Sellmeier dispersion (e.g. N-BK7, F2)
* Diffraction grating support

  * Constant, polynomial and holographic ruling spacing
  * Sinusoidal, square, rectangular, sawtooth, and triangular ruling profiles

* CCD/CMOS sensor simulation

  * Quantum efficiency
  * Noise simulation
  * Charge diffusion

Limitations
-----------

* **Polarization**. Different polarization states are not propagated through the
  system.
* **Physical Optics**. Only geometric optics is supported right now, but adding
  a Fourier optics propagator is a longstanding goal of the project.
* **Glass Catalog**. :mod:`optika` has a wide array of optical
  constants from sources such as :cite:t:`Palik1997` and :cite:t:`Henke1993`,
  and the :class:`~optika.materials.Glass` material provides Sellmeier dispersion
  for a few common glasses (e.g. N-BK7, F2), but it does not yet have a
  comprehensive glass database like Zemax does.

Differences from Zemax
----------------------

* The position and orientation of surfaces in :mod:`optika` are specified in
  global coordinates instead of coordinates relative to the last surface.

* The field of view is automatically calculated, there is no need to set the
  extent of the field.

* Diffraction grating rulings are now a parameter of an optical surface.
  There is no need to change the type of surface to allow for different ruling
  designs.


Examples
========

Compute the transmissivity of a thin filter, such as the aluminum filters used
to reject visible light on solar instruments.

.. jupyter-execute::

    import matplotlib.pyplot as plt
    import astropy.units as u
    import named_arrays as na
    import optika

    # Define the wavelengths at which to compute the transmissivity
    wavelength = na.geomspace(100, 800, axis="wavelength", num=201) * u.AA

    # Compute the efficiency of a 100 nm layer of aluminum
    reflectivity, transmissivity = optika.materials.multilayer_efficiency(
        wavelength=wavelength,
        layers=optika.materials.Layer(
            chemical="Al",
            thickness=1000 * u.AA,
        ),
    )

    # Plot the transmissivity, which drops sharply at the aluminum L edge
    fig, ax = plt.subplots(constrained_layout=True)
    na.plt.plot(wavelength, transmissivity.average, ax=ax, axis="wavelength");
    ax.set_xscale("log");
    ax.set_xlabel(f"wavelength ({wavelength.unit:latex_inline})");
    ax.set_ylabel("transmissivity");

|

Compute the effective quantum efficiency of a back-illuminated CCD, and compare
it to the theoretical maximum for the same sensor.

.. jupyter-execute::

    # Define the wavelengths at which to compute the quantum efficiency
    wavelength = na.geomspace(10, 10000, axis="wavelength", num=1001) * u.AA

    # Compute the effective quantum efficiency of the sensor
    eqe = optika.sensors.quantum_efficiency_effective(
        wavelength=wavelength,
    )

    # Compute the quantum efficiency of an ideal back surface
    eqe_max = optika.sensors.quantum_efficiency_effective(
        wavelength=wavelength,
        cce_backsurface=1,
    )

    # Plot both
    fig, ax = plt.subplots(constrained_layout=True)
    na.plt.plot(wavelength, eqe, ax=ax, axis="wavelength", label="effective");
    na.plt.plot(wavelength, eqe_max, ax=ax, axis="wavelength", label="ideal back surface");
    ax.set_xscale("log");
    ax.set_xlabel(f"wavelength ({wavelength.unit:latex_inline})");
    ax.set_ylabel("quantum efficiency");
    ax.legend();

|

For a complete optical system, including a raytrace and a simulated image, see
the worked Newtonian telescope in
:class:`optika.systems.SequentialSystem`.

|


Tutorials
=========

Jupyter notebook examples on how to use :mod:`optika`.

.. toctree::
    :maxdepth: 1

    tutorials/prime_focus
    tutorials/fzp_focus


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
