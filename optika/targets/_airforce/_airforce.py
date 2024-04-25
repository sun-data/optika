import pathlib
import matplotlib.pyplot as plt
from svglib import svglib
from reportlab.graphics import renderPM
import named_arrays as na

__all__ = [
    "airforce",
]


def airforce(
    axis_x: str,
    axis_y: str,
    num_x: int = 1000,
    num_y: int = 1000,
) -> na.ScalarArray:
    """
    A `1951 USAF resolution test target <https://en.wikipedia.org/wiki/1951_USAF_resolution_test_chart>`_
    used for testing the performance of optical systems.

    Parameters
    ----------
    axis_x:
        The name of the horizontal axis.
    axis_y:
        The name of the vertical axis.
    num_x:
        The number of pixels along the horizontal axis.
    num_y:
        The number of pixels along the vertical axis.

    Examples
    --------

    Load and display the test target

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import named_arrays as na
        import optika

        # Load the test target
        target = optika.targets.airforce("x", "y")

        # Display the test target
        fig, ax = plt.subplots(constrained_layout=True)
        na.plt.pcolormesh(C=target);
    """
    path = pathlib.Path(__file__).parent / "USAF-1951.svg"

    fn = "tmp.png"
    drawing = svglib.svg2rlg(path)

    dpi = 100

    width = num_x / dpi
    height = num_y / dpi

    scale_x = width / (drawing.width / 72)
    scale_y = height / (drawing.height / 72)

    drawing.scale(scale_x, scale_y)

    drawing.width = width * 72
    drawing.height = height * 72

    renderPM.drawToFile(drawing, fn, fmt="PNG", dpi=dpi)
    img = plt.imread(fn)

    img = img.astype(float)[::-1].sum(~0)

    img = img / img.max()

    img = -img + 1

    return na.ScalarArray(
        ndarray=img,
        axes=(axis_y, axis_x),
    )
