Stop Finding
============

One of the design goals of :mod:`optika` is to automatically compute the
field of view and entrance pupil of an optical system (see
:attr:`~optika.systems.AbstractSequentialSystem.field_min`,
:attr:`~optika.systems.AbstractSequentialSystem.field_max`,
:attr:`~optika.systems.AbstractSequentialSystem.pupil_min`, and
:attr:`~optika.systems.AbstractSequentialSystem.pupil_max`).
Unlike Zemax, the user never has to specify the extent of the field or pupil.
This page describes the strategy :mod:`optika` uses to discover that extent and
to sample the field and pupil for imaging, since the strategy spans several
methods and the reasoning behind it is easy to lose.


Overview
--------

The trajectory of a ray through a sequential system is fixed by two surfaces:

* the **field stop**, which limits the region of the object that is imaged, and
* the **pupil stop** (or aperture stop), which limits the bundle of rays
  accepted from each point of the object.

A surface is marked as a stop by setting ``is_field_stop=True`` or
``is_pupil_stop=True`` on it. Every ray that survives the system passes inside
both stops, so a ray that grazes the *border* of one stop while passing through
a chosen point of the other traces out the boundary of the accepted light. This
is what :attr:`~optika.systems.AbstractSequentialSystem.rayfunction_stops`
returns: a :class:`~optika.rays.RayFunctionArray`, defined on the first surface
of the system, whose rays are constructed to strike prescribed points on both
the field stop and the pupil stop. The field of view, the entrance pupil, and
the sampling of rays used for image simulation are all derived from it.

The difficulty is that a ray is launched from the *first* surface, but the
constraints live on the *stop* surfaces further downstream. There is no
closed-form expression for the launch coordinate that lands a ray on a given
point of a given stop, so the launch coordinate is found by root-finding.


Two principles
--------------

Everything below follows from two physical requirements. They are worth stating
up front because they dictate *where* each coordinate is measured, and getting
the measurement plane wrong produces answers that look plausible but are subtly
biased.

**The field is anchored on the object plane.**
    A point-spread function is the image of a single point of the object. For
    the simulated PSF to be correct, every ray in a given field bundle must
    share one object direction (for a distant object) or one object position
    (for a nearby object). If the field were instead anchored on an internal
    surface, rays with the same field label but different pupil labels would
    correspond to slightly different object points, smearing the PSF. So the
    field coordinate is always resolved on the object plane.

**The pupil is measured at the entrance pupil.**
    The entrance pupil is the image of the pupil stop in object space, i.e. the
    plane on which the incoming wavefront is uniform for a uniform distant
    source. Measuring and sampling the pupil *there*, rather than on the
    angular object plane or on the pupil stop itself, matters for two reasons:

    * **Robustness.** For a system with a tiny entrance aperture (a feed optic
      much smaller than the beam, as in FURST), the entrance-pupil extent is
      essentially independent of field, so rays aimed through it always land on
      the aperture. Measured instead as an angle on the object plane, the same
      extent swings with field and a single global box makes most field angles
      miss the aperture.

    * **Radiometry.** Vignetting is the fraction of the accepted bundle that
      survives to the detector, an area integral over the pupil. Sampling
      uniformly on the entrance pupil is equal-area sampling in the plane where
      the wavefront is uniform, so the vignetting fraction is an unweighted mean
      of the surviving samples. Sampling uniformly on the pupil stop instead
      would, for a system with pupil distortion (such as the grating in ESIS,
      which is the pupil stop), place unequal areas of the entrance pupil under
      each sample and bias the result.

The consequence is that the field is resolved as an object-space coordinate and
the pupil as an entrance-pupil coordinate, and the machinery below exists to
connect those object-space coordinates to the physical stop surfaces.


Input coordinates
-----------------

A stop ray is labelled by three input coordinates, gathered together in an
:class:`~optika.vectors.ObjectVectorArray`:

``wavelength``
    The vacuum wavelength of the ray. It is carried through the solve
    unchanged; it only matters because dispersive surfaces (gratings) bend
    different wavelengths differently.

``field``
    A two-dimensional coordinate locating the ray on the field stop.

``pupil``
    A two-dimensional coordinate locating the ray on the pupil stop.

Both ``field`` and ``pupil`` may be given in either **normalized** or
**physical** units, and the units alone tell :mod:`optika` how to interpret
them:

.. list-table::
    :header-rows: 1
    :widths: 25 25 50

    * - Units
      - Meaning
      - Interpretation
    * - dimensionless
      - normalized
      - A value in :math:`[-1, 1]`, denormalized against the bounding box of
        the relevant stop's aperture.
    * - length (e.g. ``mm``)
      - physical position
      - A position on the relevant surface.
    * - angle (e.g. ``deg``)
      - physical direction
      - A direction, converted to direction cosines with
        :func:`~optika.direction`.

This convention (dimensionless means normalized, length means position, angle
means direction) is used consistently throughout the stop-finding code; see
``_coordinates_are_normalized`` and
:attr:`~optika.systems.AbstractSequentialSystem.object_is_at_infinity`.


The two-point ray solve
-----------------------

The core primitive is the private method ``_shoot_rays``. Given a
``subsystem`` (a contiguous slice of the system's surfaces), a wavelength, a
grid ``grid_first`` on the first surface, and a grid ``grid_last`` on the last
surface, it finds the launch ray that connects them. Both grids are given in
*physical* units; denormalizing a normalized stop coordinate against its
aperture is the caller's job, so at this boundary a dimensionless grid is an
unambiguous direction cosine rather than a normalized coordinate.

A ray leaving the first surface has two degrees of freedom that are *not*
pinned by ``grid_first``. If ``grid_first`` is a **position**, the free degrees
of freedom are the launch **direction**; if ``grid_first`` is a **direction**,
they are the launch **position**. ``_shoot_rays`` reads which case applies from
the units of ``grid_first`` (length means position, angle or dimensionless
direction cosine means direction), builds a
:class:`~optika.rays.RayVectorArray` with the fixed coordinate filled in, and
solves for the free coordinate so that the ray lands on ``grid_last`` at the
last surface. The residual whose root is sought is the miss distance at the
last surface,

.. math::

    \vec{r}(\vec{a}) = \vec{g}_\text{trial}(\vec{a}) - \vec{g}_\text{last},

where :math:`\vec{a}` is the trial value of the free coordinate and
:math:`\vec{g}_\text{trial}` is where the resulting ray actually crosses the
target coordinate of the last surface (its position if ``grid_last`` is a
position, its direction if ``grid_last`` is a direction). This residual is
evaluated by ``_ray_error`` and driven to zero with
:func:`named_arrays.optimize.root_newton`.

Because ``grid_last`` may itself be either a position or a direction, the same
routine works in either direction along the system. In particular, running it
with the subsystem reversed and an angular ``grid_last`` back-traces rays to a
target *direction* on an object at infinity, which is how the field extent on
the object plane is recovered.

Two details make the solve robust across systems of wildly different physical
scale:

* **Seeding.** The initial guess aims each ray from the first surface toward a
  sensible target: directly at its point on the last surface when no surface
  with optical power lies in between (the guess is then nearly exact), and
  otherwise at the center of the first powered surface (a mirror, a curved sag,
  or a ruled surface, found by ``_anchor_surface``). This keeps the guess
  inside the basin of convergence even for strongly off-axis feed or fold
  mirrors.

* **Scaling.** Both the convergence tolerance and the finite-difference step
  used to estimate the Jacobian are scaled by the size of the target aperture,
  so that a millimeter-scale spectrograph and a meter-scale telescope are
  solved to the same *relative* precision. The default absolute step of
  :func:`named_arrays.jacobian` is otherwise below the floating-point noise
  floor of the raytrace and yields a Jacobian made of noise.

``_shoot_rays`` returns the launch rays *at the first surface* (in global
coordinates), with both the given and the solved coordinate filled in, so that
propagating them through the subsystem reproduces ``grid_last``.


The strategy
------------

The two principles rule out computing a single global field box and a single
global pupil box, because the entrance-pupil extent depends on field. Instead
the field of view and the per-field entrance pupil are calibrated in stages,
and the expensive root-finding is confined to a coarse grid.

**1. Field extent on the object plane.**
    A small number of rays connecting the field-stop border to the pupil-stop
    border are traced back to the object plane, and their extent gives the
    field of view. This is the object-plane field domain that every later stage
    samples within.

**2. Per-field entrance-pupil extent.**
    The object surface is treated as the field stop, and a *coarse* grid of
    field directions, sampled evenly across the field of view from stage 1, is
    connected to every point on the *wire* (border) of the pupil stop with
    ``_shoot_rays``. Only the wire is needed: it is the boundary of the pupil
    stop, so its image on the entrance pupil is the boundary of the entrance
    pupil, and the minimum and maximum of those positions give a per-field
    entrance-pupil bounding box. This is the stage that captures the
    field dependence of the pupil (the pupil distortion), and it is the only
    stage that pays for dense root-finding, kept affordable by using a coarse
    field grid and only the one-dimensional pupil-stop wire.

**3. Interpolation.**
    The per-field bounding box from stage 2 is a smooth function of field (it is
    the pupil distortion), so it is fit on the coarse field grid and
    interpolated onto the dense field grid, in the same spirit as
    :class:`~optika.distortion.PolynomialDistortionModel` and the vignetting
    models. The coarse grid must be fine enough to resolve the distortion, not
    the scene.

**4. Dense forward trace.**
    For image simulation, each dense ray is fully specified in object space: its
    direction is an object-plane field angle, and its position is a normalized
    pupil coordinate mapped onto the interpolated per-field entrance-pupil box.
    Both coordinates are known without any root-finding, so the dense pass is a
    pure *forward* trace through the system. This is what keeps a
    :math:`1000 \times 1000` field affordable: the Newton solves live entirely
    in the coarse calibration of stage 2, never in the dense grid.

Because a uniform grid on the entrance-pupil box is equal-area in the plane
where the wavefront is uniform, the vignetted fraction of that grid (with the
``where`` keyword marking the surviving rays) is the vignetting directly, with
no Jacobian weight. The axis-aligned bounding box slightly over-covers a
rotated or astigmatic entrance pupil, but the excess samples fall outside the
aperture and are removed by vignetting, so the result is correct if marginally
less sample-efficient.


The object as the field stop
----------------------------

Stage 2 relies on being able to mark the **object surface itself** as the field
stop, with an angular (dimensionless, sine-of-half-angle) aperture. The
``field`` coordinate is then a direction on the object plane rather than a
position on an internal surface, which is exactly the object-plane anchoring the
first principle requires. It is also the only workable option in two common
cases where an internal field stop has no solution:

* **Spectrographs**, where the field stop is the detector. A single wavelength
  illuminates only part of the detector, so connecting the *border* of the
  detector to the pupil stop has no solution at that wavelength.

* **Systems with a tiny entrance aperture**, where a single global pupil box,
  back-projected to the object, misses the feed optic for most field angles.

When ``_shoot_rays`` cannot connect an internal field stop at a single
wavelength, the raised error points the user toward this option. Whether the
object is treated as being at infinity is inferred from the units of its
aperture: a length aperture is a finite object, a dimensionless aperture is an
object at infinity (see
:attr:`~optika.systems.AbstractSequentialSystem.object_is_at_infinity`).


Field of view and entrance pupil
--------------------------------

The field of view and entrance pupil are read off from the stop rayfunction by
reducing over *both* the field and pupil axes:

* :attr:`~optika.systems.AbstractSequentialSystem.field_min` /
  :attr:`~optika.systems.AbstractSequentialSystem.field_max` give the corners
  of the field of view, expressed as angles (via :func:`~optika.angles`) when
  the object is at infinity and as positions otherwise.

* :attr:`~optika.systems.AbstractSequentialSystem.pupil_min` /
  :attr:`~optika.systems.AbstractSequentialSystem.pupil_max` give the corners
  of the entrance pupil, expressed as positions when the object is at infinity
  and as angles otherwise.

The two swap roles with object distance because field and pupil are conjugate:
for a distant object the field is naturally angular and the pupil is a physical
aperture, while for a nearby object the field is a physical extent and the
pupil subtends an angle.


Conventions and assumptions
---------------------------

A few conventions are load-bearing and worth stating explicitly:

* **Dimensionless means normalized at the system input, and a direction cosine
  inside the solve.** A ``field`` or ``pupil`` grid supplied to the system is
  interpreted as normalized when it is dimensionless (see the table above).
  Denormalization happens before ``_shoot_rays`` is called, so once inside the
  solve a dimensionless grid is unambiguously a physical direction cosine.

* **The pupil is measured in an object-space plane.** The entrance-pupil extent
  and the sampling grid live on the entrance pupil, not on the pupil stop.
  Measuring on the pupil stop would reintroduce the distortion bias the second
  principle exists to avoid.

* **Grids are in the local frame** of the surface they belong to. Aperture
  denormalization produces coordinates in the surface's own frame, and
  ``_shoot_rays`` returns launch rays in the global frame after applying the
  first surface's transformation.

* **Angles are direction cosines.** :func:`~optika.direction` and
  :func:`~optika.angles` convert between a pair of azimuth/elevation angles and
  a three-dimensional direction cosine, and are inverses of each other.
