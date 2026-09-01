"""
3D bubble shape models: volume & surface area from the 2D split-ellipse fit.

All models take the same three measured quantities (as produced by
``estimate_a_b1_b2_split_fit``) and differ only in the assumed 3D revolution:

    a   : full MAJOR-axis length (tip to tip)        -> semi-major  A = a / 2
    b1  : front SEMI-minor axis (a radius)
    b2  : rear  SEMI-minor axis (a radius)

Two competing models
--------------------
``prolate`` — revolution about the MAJOR axis.
    The unseen (depth) axis is assumed equal to the in-plane minor axis ``b``.
        V = (pi * a / 3) * (b1**2 + b2**2)
    The volume is evaluated directly from the two measured half-widths.

``oblate`` — revolution about the MINOR axis  (Mikaelian et al. 2015).
    The unseen (depth) axis is assumed equal to the in-plane MAJOR semi-axis
    a/2, i.e. the bubble is circular in the plane perpendicular to its short
    axis (a flattened, rising-bubble shape). Two half-oblate-spheroids.
        V = (pi * a**2 / 6) * (b1 + b2)
    This matches Mikaelian et al. (Exp. Therm. Fluid Sci. 64:1-12, 2015),
    eq. (h) p.6:  V = pi*a^2*b1/6 + pi*a^2*b2/6.

Surface definitions
-------------------
The fit splits the silhouette ACROSS the major axis, giving two semi-minor
radii b1 (front) and b2 (rear). Volume is additive over the two halves in both
models, so V is insensitive to how the halves are joined. Surface is not.

For the OBLATE model the two halves are revolved about the minor axis and both
end on the SAME equatorial circle of radius a/2 — the b1/b2 mismatch lies along
the axis of revolution, so one dome is simply taller than the other. They glue
edge to edge and the closed surface is exactly the sum of the two curved halves.

For the PROLATE model, one asymmetric projection does not uniquely determine a
surface of revolution. The reported surface uses the VOLUME-EQUIVALENT SYMMETRIC
prolate spheroid, radius b_eq = sqrt((b1^2 + b2^2)/2). This is an explicit
volume-preserving axisymmetric regularization: the body is closed and smooth, and
(4/3)*pi*b_eq^2*(a/2) == (pi*a/3)*(b1^2 + b2^2). It does not claim to reproduce
the asymmetric measured silhouette.

The two models give equal volumes when a*(b1 + b2) == 2*(b1^2 + b2^2). For a
symmetric fit (b1 == b2 == b) that reduces to a == 2*b, i.e. a sphere, and the
ratio V_oblate / V_prolate is then the aspect ratio a / (2*b); for an
asymmetric fit the condition also admits non-circular silhouettes.

All functions are vectorised: ``a``, ``b1``, ``b2`` may be floats or numpy
arrays of broadcastable shape. Volumes are in mm^3 and surfaces in mm^2 when
inputs are in mm. Axis triples that are not finite and strictly positive are
not physically measurable bubbles; both the volume and the surface are returned
as NaN for those elements.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "volume_surface_prolate",
    "volume_surface_oblate",
    "volume_surface",
]

_SPHERE_REL_TOL = 1e-6


def _valid_axes(a, b1, b2):
    """Element-wise mask of physically admissible axis triples.

    A measurable bubble has finite, strictly positive axes. Zero or negative
    axes arise only from degenerate fits (e.g. a zero-area mask) and must not
    produce a volume: ``(a=2, b1=0, b2=1)`` is a bubble with no thickness, and
    ``(a=-2, b1=1, b2=1)`` has no geometric meaning at all. Both volume and
    surface consult this mask so they never disagree about which rows are real.
    """
    with np.errstate(invalid="ignore"):
        return (
            np.isfinite(a) & np.isfinite(b1) & np.isfinite(b2) & (a > 0.0) & (b1 > 0.0) & (b2 > 0.0)
        )


def _half_spheroid_surface(c_eq, p_pol):
    """Curved surface area of HALF a spheroid of revolution about its polar axis.

    The spheroid has equatorial radius ``c_eq`` and polar semi-axis ``p_pol``;
    it is cut by the equatorial plane (perpendicular to the polar axis) and this
    returns the lateral curved area of one half (so two equal halves sum to the
    full spheroid surface).

    Handles all three regimes element-wise:
      * p_pol < c_eq  -> oblate  (polar axis is the short one)
      * p_pol > c_eq  -> prolate (polar axis is the long one)
      * p_pol ~ c_eq  -> sphere
    """
    c = np.asarray(c_eq, dtype=float)
    p = np.asarray(p_pol, dtype=float)
    c, p = np.broadcast_arrays(c, p)
    out = np.full(c.shape, np.nan, dtype=float)

    valid = (c > 0) & (p > 0)
    scale = np.maximum(np.maximum(c, p), 1e-12)
    rel = np.where(valid, np.abs(p - c) / scale, 0.0)

    sphere = valid & (rel < _SPHERE_REL_TOL)
    oblate = valid & (p < c) & ~sphere
    prolate = valid & (p > c) & ~sphere

    # Sphere: half curved surface = 2*pi*r^2  (r = c = p)
    out = np.where(sphere, 2.0 * np.pi * c**2, out)

    with np.errstate(divide="ignore", invalid="ignore"):
        # Oblate: e = sqrt(1 - (p/c)^2);  half = pi*c^2 + (pi*p^2/e)*artanh(e)
        e_ob = np.sqrt(np.clip(1.0 - (p / c) ** 2, 0.0, None))
        e_ob_safe = np.where(e_ob > 0, e_ob, 1.0)
        ob_val = np.pi * c**2 + (np.pi * p**2 / e_ob_safe) * np.arctanh(
            np.clip(e_ob, 0.0, 1.0 - 1e-12)
        )
        out = np.where(oblate, ob_val, out)

        # Prolate: e = sqrt(1 - (c/p)^2); half = pi*c^2 + pi*(c*p/e)*arcsin(e)
        e_pr = np.sqrt(np.clip(1.0 - (c / p) ** 2, 0.0, None))
        e_pr_safe = np.where(e_pr > 0, e_pr, 1.0)
        pr_val = np.pi * c**2 + np.pi * (c * p / e_pr_safe) * np.arcsin(np.clip(e_pr, 0.0, 1.0))
        out = np.where(prolate, pr_val, out)

    return out


def _as_input(value):
    return np.asarray(value, dtype=float)


def _broadcast_axes(a, b1, b2):
    """Broadcast the three measured axes to a common shape as float arrays."""
    return np.broadcast_arrays(_as_input(a), _as_input(b1), _as_input(b2))


def volume_surface_prolate(a, b1, b2):
    """Prolate model (revolution about the MAJOR axis); depth axis = minor.

    ``V = (pi*a/3)*(b1^2 + b2^2)`` — additive over the two measured halves.

    The surface is that of the volume-equivalent SYMMETRIC prolate spheroid of
    equatorial radius ``b_eq = sqrt((b1^2 + b2^2)/2)`` and polar semi-axis
    ``a/2``. This regularization preserves the model volume while making the
    unobserved axisymmetric surface assumption explicit; it does not reproduce an
    asymmetric measured silhouette. See the module docstring.

    ``b_eq`` preserves the volume exactly: ``(4/3)*pi*b_eq^2*(a/2)`` is
    identically ``(pi*a/3)*(b1^2 + b2^2)``, so gas holdup is unaffected by this
    choice; only interfacial area depends on it.
    """
    a, b1, b2 = _broadcast_axes(a, b1, b2)
    valid = _valid_axes(a, b1, b2)
    A = 0.5 * a  # semi-major (revolution / polar axis)

    with np.errstate(invalid="ignore"):
        V = (np.pi * a / 3.0) * (b1**2 + b2**2)
        # Volume-equivalent symmetric equatorial radius; A > b_eq -> prolate.
        b_eq = np.sqrt(0.5 * (b1**2 + b2**2))
        S = 2.0 * _half_spheroid_surface(b_eq, A)
    return np.where(valid, V, np.nan), np.where(valid, S, np.nan)


def volume_surface_oblate(a, b1, b2):
    """Oblate model of Mikaelian et al. 2015 (revolution about the MINOR axis);
    depth axis = major semi-axis a/2.

    V = (pi*a^2/6)*(b1 + b2);  surface = two half-oblate-spheroids.

    Both halves share the same equatorial circle of radius ``a/2``, so they glue
    exactly and the sum of the two curved halves is the closed surface.
    """
    a, b1, b2 = _broadcast_axes(a, b1, b2)
    valid = _valid_axes(a, b1, b2)
    c = 0.5 * a  # equatorial radius (in-plane major semi-axis = depth axis)

    with np.errstate(invalid="ignore"):
        V = (np.pi * a**2 / 6.0) * (b1 + b2)
        # Each half: equatorial radius = c, polar semi-axis = b_i (b_i < c -> oblate)
        S = _half_spheroid_surface(c, b1) + _half_spheroid_surface(c, b2)
    return np.where(valid, V, np.nan), np.where(valid, S, np.nan)


def volume_surface(a, b1, b2, model: str = "oblate"):
    """Dispatch to a named shape model. ``model`` in {"oblate", "prolate"}."""
    if model == "oblate":
        return volume_surface_oblate(a, b1, b2)
    if model == "prolate":
        return volume_surface_prolate(a, b1, b2)
    raise ValueError(f"Unknown shape model {model!r}; use 'oblate' or 'prolate'.")
