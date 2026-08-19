# optika

[![tests](https://github.com/sun-data/optika/actions/workflows/tests.yml/badge.svg)](https://github.com/sun-data/optika/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/sun-data/optika/graph/badge.svg?token=tBcex8q72g)](https://codecov.io/gh/sun-data/optika)
[![Black](https://github.com/sun-data/optika/actions/workflows/black.yml/badge.svg)](https://github.com/sun-data/optika/actions/workflows/black.yml)
[![Ruff](https://github.com/sun-data/optika/actions/workflows/ruff.yml/badge.svg)](https://github.com/sun-data/optika/actions/workflows/ruff.yml)
[![Documentation Status](https://readthedocs.org/projects/optika/badge/?version=latest)](https://optika.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/optika.svg)](https://badge.fury.io/py/optika)

A Python library for simulating optical systems, similar to Zemax.

`optika` computes the spectral response and resolution of an arbitrary optical system, and can optimize it using `scipy.optimize`.
Surfaces carry their own sag profile, aperture, material, and rulings, and are placed in global coordinates, so a system is an ordinary Python object that can be built, modified, and swept over programmatically.

Because every parameter can be an array from [`named-arrays`](https://github.com/sun-data/named-arrays), a whole configuration space of designs propagates through the raytrace at once, and an uncertain parameter carries its uncertainty through to the performance of the system.

More information is available in the [documentation](https://optika.readthedocs.io/en/latest/).

## Installation

Optika can be installed using pip:

```bash
pip install optika
```

## Features

- Sequential raytrace modeling of an optical system
- Stratified random sampling of input rays for faster convergence
- Image simulation of a given scene using an optical system
- A fast linear forward model approximating a raytraced system, for imaging many scenes without raytracing each one
- Spherical, conical, and toroidal surface sag profiles
- Circular, rectangular, and polygonal apertures
- Mirrors and arbitrary multilayer coatings
- Refractive glass materials with Sellmeier dispersion (e.g. N-BK7, F2)
- Diffraction gratings, with constant, polynomial, and holographic ruling spacing, and sinusoidal, square, rectangular, sawtooth, and triangular ruling profiles
- CCD/CMOS sensor simulation, including quantum efficiency, noise, and charge diffusion
- n-dimensional configurations of the optical system using [named-arrays](https://github.com/sun-data/named-arrays)
- Uncertainty propagation using [named-arrays](https://github.com/sun-data/named-arrays)

## Key concepts

**Surfaces are placed in global coordinates.**
Unlike Zemax, where each surface is positioned relative to the one before it, an `optika` surface carries a `transformation` giving its position and orientation in the coordinate system of the whole instrument.
Moving one surface therefore does not move everything downstream of it.

**The field of view and entrance pupil are computed, not specified.**
The apertures of the surfaces determine them, so marking a surface with `is_pupil_stop` or `is_field_stop` is enough.

**Rulings are a property of a surface.**
A diffraction grating is an ordinary surface with a `rulings` field, so switching between ruling designs does not mean switching to a different type of surface.

**Any parameter can be an array.**
Giving a parameter an extra named axis sweeps the system over that axis, and every ray traced through it carries that axis along, which is how `optika` explores a configuration space without a loop.
An [`UncertainScalarArray`](https://named-arrays.readthedocs.io/en/latest/_autosummary/named_arrays.UncertainScalarArray.html) parameter propagates its uncertainty through the raytrace by the Monte Carlo method.

## Example Gallery

[Simulate a Newtonian telescope](https://optika.readthedocs.io/en/latest/_autosummary/optika.systems.SequentialSystem.html#optika.systems.SequentialSystem)
using `optika`

![Newtonian telescope example](https://optika.readthedocs.io/en/latest/_images/optika.systems.SequentialSystem_0_0.png)
![image simulation](https://optika.readthedocs.io/en/latest/_images/optika.systems.SequentialSystem_1_0.png)

Compute the [reflectivity of a multilayer mirror](https://optika.readthedocs.io/en/latest/_autosummary/optika.materials.multilayer_efficiency.html#optika.materials.multilayer_efficiency)
by specifying the materials and thicknesses of the layers.

![multilayer example](https://optika.readthedocs.io/en/latest/_images/optika.materials.multilayer_efficiency_1_1.png)

Model the [quantum efficiency of a backilluminated CCD](https://optika.readthedocs.io/en/latest/_autosummary/optika.sensors.quantum_efficiency_effective.html#optika.sensors.quantum_efficiency_effective)

![QE example](https://optika.readthedocs.io/en/latest/_images/optika.sensors.quantum_efficiency_effective_0_0.png)

Compute the [transmissivity of a thin filter](https://optika.readthedocs.io/en/latest/#examples),
such as the aluminum filters used to reject visible light on solar instruments.

```python
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
```
![aluminum filter example](https://optika.readthedocs.io/en/latest/_images/index_0_0.png)

## Development

Install the package in editable mode along with its test dependencies, and run the test suite using [pytest](https://docs.pytest.org):
```bash
pip install -e .[test]
pytest
```

This project is formatted using [black](https://black.readthedocs.io), linted using [ruff](https://docs.astral.sh/ruff), and type-checked using [mypy](https://mypy-lang.org), all of which are checked by continuous integration:
```bash
black .
ruff check .
mypy optika
```

To build the documentation locally:
```bash
pip install -e .[doc]
sphinx-build docs docs/_build/html
```
