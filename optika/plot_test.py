import matplotlib.pyplot as plt
import named_arrays as na
import numpy as np

import optika.plot


def test_is_3d_2d():
    fig, ax = plt.subplots()
    assert not optika.plot.is_3d(ax)
    plt.close(fig)


def test_is_3d_3d():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    assert optika.plot.is_3d(ax)
    plt.close(fig)


def test_is_3d_mixed():
    """Only if every axes is a 3D one, since one drawing goes to all of them."""
    fig = plt.figure()
    flat = fig.add_subplot(121)
    solid = fig.add_subplot(122, projection="3d")

    def axes(*a):
        return na.ScalarArray(np.array(a, dtype=object), axes=("ax",))

    assert not optika.plot.is_3d(axes(flat, solid))
    assert optika.plot.is_3d(axes(solid, solid))
    plt.close(fig)


def test_kwargs_filled():
    """The colour asked for becomes the edge, and the face is left blank."""
    result = optika.plot.kwargs_filled({"color": "tab:red", "linewidth": 2})

    assert result["edgecolors"] == "tab:red"
    assert result["facecolors"] == "white"
    assert result["linewidth"] == 2
    assert "color" not in result


def test_kwargs_filled_facecolor():
    """A face colour given explicitly is kept."""
    result = optika.plot.kwargs_filled({"color": "black", "facecolor": "none"})

    assert result["facecolors"] == "none"
    assert result["edgecolors"] == "black"
    assert "facecolor" not in result
