Stop Finding
============

One of the design goals of :mod:`optika` is to automatically compute the
field of view and entrance pupil of an optical system (see
:attr:`~optika.systems.AbstractSequentialSystem.field_min`,
:attr:`~optika.systems.AbstractSequentialSystem.field_max`,
:attr:`~optika.systems.AbstractSequentialSystem.pupil_min`, and
:attr:`~optika.systems.AbstractSequentialSystem.pupil_max`).
Unlike Zemax, the user never has to specify the extent of the field or pupil.
This page describes how :mod:`optika` discovers that extent and samples the
field and pupil for imaging, since the machinery spans several methods and the
reasoning behind it is easy to lose.

Every private method named on this page is defined on
:class:`~optika.systems.AbstractSequentialSystem`, in
``optika/systems/_sequential.py``.


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
holds: a :class:`~optika.rays.RayFunctionArray` of the rays which graze the
border of both stops, carried back to the object surface and expressed in its
frame. The field of view, the entrance pupil, and the sampling of rays used for
image simulation are all derived from it.

The difficulty is that there is no closed-form expression for the ray which
connects a given point of one stop to a given point of the other, so those rays
are found by root-finding. The solve runs between the two stops only; the rays
are then carried back to the object by ordinary propagation.


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

A ray is labeled by three input coordinates, gathered together in an
:class:`~optika.vectors.ObjectVectorArray`:

``wavelength``
    The vacuum wavelength of the ray. It is carried through the solve
    unchanged; it only matters because dispersive surfaces (gratings) bend
    different wavelengths differently.

``field``
    The ray's object point: a direction if the object is at infinity, and a
    position on the object surface otherwise.

``pupil``
    The ray's coordinate on the entrance pupil, in the same frame: a position
    on the object surface if the object is at infinity, and a direction
    otherwise. Field and pupil swap roles with object distance because they
    are conjugate.

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
      - A value in :math:`[-1, 1]`, mapped onto the field of view (for a
        field) or onto the entrance pupil of that field point (for a pupil),
        both read off
        :attr:`~optika.systems.AbstractSequentialSystem.rayfunction_stops`.
    * - length (e.g. ``mm``)
      - physical position
      - A position on the object surface.
    * - angle (e.g. ``deg``)
      - physical direction
      - A direction, converted to direction cosines with
        :func:`~optika.direction`.

This convention (dimensionless means normalized, length means position, angle
means direction) is used consistently throughout the stop-finding code. There
is no helper which decides it; each site tests the unit of the grid it was
handed against :func:`~named_arrays.unit_normalized`, and reads
:attr:`~optika.systems.AbstractSequentialSystem.object_is_at_infinity` to know
which of position and direction the field is.


The two-point ray solve
-----------------------

The core primitive is ``_solve_rays``. Given a ``subsystem`` (a contiguous
slice of the system's surfaces), a wavelength, a grid ``grid_first`` on the
first surface, and a grid ``grid_last`` on the last surface, it finds the
launch ray that connects them. Both grids are given in *physical* units, in
the local frame of their own surface; at this boundary a dimensionless grid is
an unambiguous direction cosine rather than a normalized coordinate.

A ray leaving the first surface has two degrees of freedom that are *not*
pinned by ``grid_first``. If ``grid_first`` is a **position**, the free degrees
of freedom are the launch **direction**; if ``grid_first`` is a **direction**,
they are the launch **position**. ``_solve_rays`` reads which case applies from
the units of ``grid_first``, builds a :class:`~optika.rays.RayVectorArray` with
the fixed coordinate filled in, and solves for the free coordinate so that the
ray lands on ``grid_last`` at the last surface. The residual whose root is
sought is the miss distance at the last surface,

.. math::

    \vec{r}(\vec{a}) = \vec{g}_\text{trial}(\vec{a}) - \vec{g}_\text{last},

where :math:`\vec{a}` is the trial value of the free coordinate and
:math:`\vec{g}_\text{trial}` is where the resulting ray actually crosses the
target coordinate of the last surface (its position if ``grid_last`` is a
position, its direction if ``grid_last`` is a direction). This residual is
evaluated by ``_ray_error`` and driven to zero with
:func:`named_arrays.optimize.root_newton`.

Because either grid may be a position or a direction, the same routine serves
whichever of the two stops comes first in the system, and whether the launch
surface is an internal stop or the object itself with an angular aperture.

Two details make the solve robust across systems of wildly different physical
scale:

* **Seeding.** The initial guess aims each ray from the first surface toward a
  sensible target, found by ``_aim_point``: directly at its point on the last
  surface when no surface with optical power lies in between (the guess is
  then nearly exact), and otherwise at the center of the first powered surface
  (a mirror, a curved sag, or a ruled surface, found by ``_anchor_surface``).
  This keeps the guess inside the basin of convergence even for strongly
  off-axis feed or fold mirrors.

* **Scaling.** Both the convergence tolerance and the finite-difference step
  used to estimate the Jacobian are scaled by the size of the target aperture,
  so that a millimeter-scale spectrograph and a meter-scale telescope are
  solved to the same *relative* precision. The default absolute step of
  :func:`named_arrays.jacobian` is otherwise below the floating-point noise
  floor of the raytrace and yields a Jacobian made of noise.

``_solve_rays`` returns the launch rays *at the first surface*, in global
coordinates, with both the given and the solved coordinate filled in, so that
propagating them through the subsystem reproduces ``grid_last``.


The strategy
------------

The field of view and the entrance pupil are calibrated in four stages, and
only the first of them does any root-finding.

**1. Connect the stops.**
    ``_calc_rayfunction_stops_only`` makes one call to ``_solve_rays`` per
    wavelength, connecting every point on the *wire* (border) of the pupil
    stop to every point on the wire of the field stop, plus one more point at
    the field stop's center. Only the wires are needed: the boundary of a stop
    maps to the boundary of the accepted light. The center of the field stop
    is a point on that stop like any on its wire, so it rides through the
    same solve as one more sample and costs a fraction of it; the fit in stage
    3 needs it.

**2. Carry the rays to the object.**
    ``_calc_rayfunction_stops`` propagates the solved rays back through the
    surfaces before the first stop to the object surface, with no further
    root-finding, and expresses them in the object surface's frame. The
    direction is flipped there so that the rays point into the system. This
    is :attr:`~optika.systems.AbstractSequentialSystem.rayfunction_stops`,
    less the center sample, which is kept in
    ``_rayfunction_stops_with_center`` for stage 3 since it is not part of the
    field's outline. :attr:`~optika.systems.AbstractSequentialSystem.field_min`
    and the other three corners are reductions of these rays over both stop
    axes.

**3. Fit the entrance pupil per field point.**
    ``_calc_pupil_fit`` fits the corners of the entrance pupil as quadratics
    in field, from the samples along the field's edge and the one at its
    center. The next section says why.

**4. Denormalize.**
    ``_denormalize_grid`` maps a normalized field onto the field of view and a
    normalized pupil onto each field point's fitted pupil, so that every ray of
    a dense grid is fully specified in object space and the dense pass through
    the system is a pure *forward* trace. This is what keeps a
    :math:`1000 \times 1000` field affordable: the Newton solves live entirely
    in stage 1, never in the dense grid.

Because a uniform grid on the entrance-pupil box is equal-area in the plane
where the wavefront is uniform, the vignetted fraction of that grid is the
vignetting directly, with no Jacobian weight. The axis-aligned box slightly
over-covers a rotated or astigmatic entrance pupil, but the excess samples fall
outside the aperture and are removed by vignetting, so the result is correct if
marginally less sample-efficient.


The object as the field stop
----------------------------

The object surface may itself be marked as the field stop, with an angular
(dimensionless, sine-of-half-angle) aperture. The ``field`` coordinate is then
a direction on the object plane, which is exactly the object-plane anchoring
the first principle requires, and stage 2 has nothing to carry back. It is also
the only workable option for a **spectrograph** whose field stop is the
detector: a single wavelength illuminates only part of the detector, so
connecting the *border* of the detector to the pupil stop has no solution at
that wavelength. When ``_solve_rays`` cannot connect an internal field stop, the
error it raises points the user toward this option.

Whether the object is treated as being at infinity is inferred from the units
of its aperture: a length aperture is a finite object, a dimensionless aperture
is an object at infinity (see
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


The entrance pupil of each field point
--------------------------------------

The stop rayfunction gives one entrance pupil for the whole field: the box
which every field point's pupil fits inside. That box is larger than any
individual field point's pupil whenever the pupil walks across the field,
which it does whenever the pupil stop is far from the entrance pupil. Rays
drawn uniformly in it are then mostly outside the pupil of the field point
they belong to, and are thrown away at the stop.

``_calc_pupil_fit`` calibrates the pupil per field point instead. It fits a
quadratic in the field to the lower-left and upper-right corners of the pupil,
using the samples along the edge of the field stop plus the one at its center,
and ``_denormalize_grid`` evaluates that fit at each field point being traced.
The center sample matters because the edge of a round field lies on a single
conic, along which a quadratic cannot tell a constant from a radial term; a
sample away from the edge is what pins down the size of the pupil in the
interior.

Two things keep it honest:

* Each field point's fitted box is held inside the box shared by every field
  point, so a fit which extrapolates cannot send rays outside the pupil the
  stops actually admit. Where the held box collapses, inverts, or is not a
  number, which a system limited by its field stop rather than its pupil stop
  can make it, that field point takes the shared box instead.

* If the fit is singular, which a field stop degenerate in one component makes
  it, the whole calibration returns :obj:`None` and every field point takes
  the shared box. This is never worse than not having the fit at all.

The fit is in the field alone; the wavelength rides along as a broadcast axis,
since the fit is only ever evaluated at the wavelengths it was made at. It is
cached on the system, along with the stop rays it is built from, because both
depend on nothing but the wavelength.

One consequence for :meth:`~optika.systems.AbstractSequentialSystem.area_effective`:
it draws each ray at a random position inside its field cell, and once the
pupil is resolved per field point its box is no larger than the pupil, so the
box at the center of a cell would clip the rays drawn toward the cell's edges
by however far the pupil walks across one cell. Each cell is therefore given
the union of the boxes at its vertices, by ``_pupil_over_field_cells``.


Frames
------

Every grid and every set of rays belongs to a stated frame, and mixing two of
them is the most common way to get a wrong answer here. Four conventions are in
play, and each is load-bearing:

* A grid produced from an aperture is in **that surface's own frame**, since
  an aperture and a sag are only defined there.

* ``_solve_rays`` takes its launch grid in the launch surface's frame and
  returns the solved rays in the **global** frame, converting once at each
  end.

* ``_calc_rayfunction_stops`` returns its rays in the **object surface's**
  frame, because that is the frame in which the field and pupil of an input
  grid are read. The entrance pupil is fit from those rays alone, edge and
  center together, so its samples cannot disagree about the frame.

* :meth:`~optika.systems.AbstractSequentialSystem.rayfunction` returns its
  rays in the **sensor's** frame, which is the sensor's own transformation
  with the system's
  :attr:`~optika.systems.SequentialSystem.transformation` composed on top,
  since that is where ``surfaces_all`` places the sensor. Its counterpart
  :meth:`~optika.systems.AbstractSequentialSystem.raytrace` returns global
  rays instead.

Nothing in the type system enforces any of this, so a function which takes or
returns rays should say which frame they are in. What does enforce it is
``test_rayfunction_is_invariant_under_a_rigid_motion``, which moves every
surface of a system together. That describes the same instrument in a
different frame, so nothing the system reports may change; a quantity measured
in the global frame by mistake moves with the motion and fails. Building a
system at an angle on purpose does not test this, because tilting one surface
changes the instrument and can stop any light reaching the sensor, which no
amount of broken frame handling would then make worse.
