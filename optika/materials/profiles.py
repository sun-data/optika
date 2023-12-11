import abc
import dataclasses
import numpy as np
import scipy.special
import astropy.units as u
import named_arrays as na

__all__ = [
    "AbstractInterfaceProfile",
    "ErfInterfaceProfile",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractInterfaceProfile(
    abc.ABC,
):
    """
    Abstract interface describing the :cite:t:`Stearns1989`
    interface profile between two layers in a multilayer stack.
    """

    @property
    @abc.abstractmethod
    def width(self) -> u.Quantity | na.AbstractScalar:
        """
        Characteristic length scale of the interface profile.
        """

    @abc.abstractmethod
    def __call__(self, z: u.Quantity | na.AbstractScalar) -> na.AbstractScalar:
        """
        Calculate the fraction of atoms that are in the new layer vs. those
        that are in the current layer.

        Parameters
        ----------
        z
            the depth in the current layer
        """

    @abc.abstractmethod
    def reflectivity(
        self,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractScalar:
        """
        Calculate the loss of the reflectivity due to this interface profile.

        Parameters
        ----------
        wavelength
            the wavelength of the incident light
        direction
            the propagation direction of the incident light, expressed in
            direction cosines.
        """


@dataclasses.dataclass(eq=False, repr=False)
class ErfInterfaceProfile(
    AbstractInterfaceProfile,
):
    """
    Error function interface profile between two layers in a multilayer stack.

    Examples
    --------

    Plot an error function interface profile as a function of depth

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import named_arrays as na
        import optika

        # Define an array of widths
        width = na.linspace(1, 2, axis="width", num=5) * u.nm

        # Define the interface profile
        p = optika.materials.profiles.ErfInterfaceProfile(width=width)

        # Define an array of depths into the material
        z = na.linspace(-5, 5, axis="z", num=101) * u.nm

        # Plot the interface profile as a function of depth
        fig, ax = plt.subplots(constrained_layout=True);
        na.plt.plot(z, p(z), ax=ax, axis="z", label=width);
        ax.set_xlabel(f"depth ({z.unit:latex_inline})");
        ax.set_ylabel(f"interface profile");
        ax.legend();

    Plot the reflectivity of the error function interface profile as a function
    of incidence angle

    .. jupyter-execute::

        # Define a wavelength
        wavelength = 304 * u.AA

        # Define an array of incidence angles
        angle = na.linspace(-90, 90, axis="angle", num=101) * u.deg

        # define an array of direction cosines based off of the incidence angles
        direction = na.Cartesian3dVectorArray(
            x=np.sin(angle),
            y=0,
            z=np.cos(angle),
        )

        # calculate the reflectivity for the given angles
        reflectivity = p.reflectivity(wavelength, direction)

        # Plot the reflectivity of the interface profile as a function of
        # incidence angle
        fig, ax = plt.subplots(constrained_layout=True);
        na.plt.plot(angle, reflectivity, ax=ax, axis="angle", label=width);
        ax.set_xlabel(f"angle ({angle.unit:latex_inline})");
        ax.set_ylabel(f"reflectivity");
        ax.legend();

    """
    width: u.Quantity | na.AbstractScalar = 0 * u.nm
    r"""
    the width of the Gaussian in the intergrand of :math:`\text{erf}(x)`
    """

    def __call__(self, z: u.Quantity | na.AbstractScalar) -> na.AbstractScalar:
        width = self.width
        x = z / (np.sqrt(2) * width)

        result = (1 + scipy.special.erf(x)) / 2

        return result

    def reflectivity(
        self,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractScalar:
        s = 4 * np.pi * direction.z / wavelength
        result = np.exp(-np.square(s * self.width) / 2)
        return result
