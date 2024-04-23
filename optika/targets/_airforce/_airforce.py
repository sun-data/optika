import io
import pathlib
import numpy as np
import cairosvg
import PIL
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
    buf = io.BytesIO()
    cairosvg.svg2png(
        url=str(path),
        write_to=buf,
        output_width=num_x,
        output_height=num_y,
    )
    img = np.array(PIL.Image.open(buf))

    return na.ScalarArray(
        ndarray=img.astype(float)[::-1, :, 3],
        axes=(axis_x, axis_y),
    )

