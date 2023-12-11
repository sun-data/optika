from typing import TypeVar
import abc
import astropy.units as u
import named_arrays as na
import optika

__all__ = [
    "AbstractImagingSensor",
    "AbstractCCD",
]


MaterialT = TypeVar("MaterialT", bound=optika.materials.AbstractMaterial)


class AbstractImagingSensor(
    optika.surfaces.AbstractSurface[
        None,
        MaterialT,
        optika.apertures.RectangularAperture,
        optika.apertures.RectangularAperture,
        None,
    ],
):
    @property
    def sag(self) -> None:
        return None

    @property
    def rulings(self) -> None:
        return None


class AbstractCCD(
    AbstractImagingSensor[MaterialT],
):
    @property
    @abc.abstractmethod
    def quantum_efficiency_effective(self, rays: optika.rays.RayVectorArray):
        r"""
        Compute the effective quantum efficiency (EQE) introduced by :cite:t:`Stern1994`,
        which includes both the recombination rate and the actual quantum efficiency.

        Parameters
        ----------
        rays
            the incident light rays

        Notes
        -----
        Our goal is to recover Equation 11 in :cite:t:`Stern1994`.
        From inspecting Equations 6 and 9 in :cite:t:`Stern1994`,
        we can see that the effective quantum efficiency is:

        .. math::
            :label: eqe

            \text{EQE} = T_\lambda \int_0^\infty \alpha \eta(x) e^{-\alpha x} \; dx

        where :math:`T_\lambda` is the net transmission of photons through the backsurface
        oxide layer (accounting for both absorption and reflections),
        :math:`\alpha` is the absorption coefficient of silicon,
        :math:`x` is the distance from the backsurface,
        and :math:`\eta(x)` is the differential charge collection efficiency (CCE).

        :cite:t:`Stern1994` assumes that the differential CCE takes the following
        linear form,

        .. math::
            :label: differential-cce

            \eta(x) = \begin{cases}
                \eta_0 + (1 - \eta_0) x / W, & x < W \\
                1, & x > W,
            \end{cases}

        where :math:`\eta_0` is the differential CCE at the backsurface,
        and :math:`W` is the width of the implant region.

        Plugging Equation :eq:`differential-cce` into Equation :eq:`eqe` yields

        .. math::

            \text{EQE} &= \alpha T_\lambda \left\{
                            \int_0^W \left[ \eta_0 + \left( \frac{1 - \eta_0}{W} \right) x \right] e^{-\alpha x} \; dx
                            + \int_W^\infty e^{-\alpha x} \; dx
            \right\} \\
            &= \alpha T_\lambda \left\{
                \eta_0 \int_0^W e^{-\alpha x} \; dx
                + \left( \frac{1 - \eta_0}{W} \right) \int_0^W x e^{-\alpha x} \; dx
                + \int_W^\infty e^{-\alpha x} \; dx
            \right\} \\
            &= \alpha T_\lambda \left\{
                -\left[ \frac{\eta_0}{\alpha} e^{-\alpha x} \right|_0^W
                - \left( \frac{1 - \eta_0}{W} \right) \left[ \left( \frac{\alpha x + 1}{\alpha^2} \right) e^{-\alpha x} \right|_0^W
                - \left[ \frac{1}{\alpha} e^{-\alpha x} \right|_W^\infty
            \right\} \\
            &= T_\lambda \left\{
                - \left[ \eta_0 (e^{-\alpha W} - 1) \right]
                - \left( \frac{1 - \eta_0}{\alpha W} \right) \left[ (\alpha W + 1) e^{-\alpha W} - 1 \right]
                - \left[ 0 - e^{-\alpha W} \right]
            \right\} \\
            &= T_\lambda \left\{
                - \eta_0 e^{-\alpha W}
                + \eta_0
                - e^{-\alpha W}
                + \eta_0 e^{-\alpha W}
                + \left( \frac{1 - \eta_0}{\alpha W} \right) (1 - e^{-\alpha W})
                + e^{-\alpha W}
            \right\} \\
            &= T_\lambda \left\{
                \eta_0
                + \left( \frac{1 - \eta_0}{\alpha W} \right) (1 - e^{-\alpha W})
            \right\} \\

        Compute the limit of :math:`W` approaching infinity

        .. math::

            \lim_{W \to \infty} \text{EQE} = T_\lambda \eta_0

        Compute the limit of :math:`W` approaching zero

        .. math::

            \lim_{W \to 0} \text{EQE} &= T_\lambda \left\{ \eta_0 + \left( \frac{1 - \eta_0}{\alpha W} \right) (1 - (1 - \alpha W + \alpha^2 W^2 + ...))  \right\} \\
                                        &= T_\lambda

        Compute the limit of :math:`\eta_0` approaching 1

        .. math::

            \lim_{\eta_0 \to 1} = T_\lambda

        """
