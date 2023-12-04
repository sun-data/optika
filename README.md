# optika

[![tests](https://github.com/sun-data/optika/actions/workflows/tests.yml/badge.svg)](https://github.com/sun-data/optika/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/sun-data/optika/graph/badge.svg?token=tBcex8q72g)](https://codecov.io/gh/sun-data/optika)
[![Black](https://github.com/sun-data/optika/actions/workflows/black.yml/badge.svg)](https://github.com/sun-data/optika/actions/workflows/black.yml)
[![Documentation Status](https://readthedocs.org/projects/optika/badge/?version=latest)](https://optika.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/optika.svg)](https://badge.fury.io/py/optika)

A Python library for simulating optical systems, similar to Zemax.

## Installation

Optika can be installed using pip:

```bash
pip install optika
```

## Features
- Sequential raytrace modeling
- Spherical, conical and toroidal surface sag profiles
- Ruled surfaces, constant and variable line spacing
- Circular, rectangular, and polygonal apertures
- n-dimensional configurations of the optical system using [named-arrays](https://github.com/sun-data/named-arrays)
- uncertainity propagation using [named-arrays](https://github.com/sun-data/named-arrays)

## Example Gallery

An [example](https://optika.readthedocs.io/en/latest/_autosummary/optika.systems.SequentialSystem.html#optika.systems.SequentialSystem)
of how to raytrace a Newtonian telescope using Optika:

![Newtonian telescope example](https://optika.readthedocs.io/en/latest/_images/optika.systems.SequentialSystem_0_0.png)