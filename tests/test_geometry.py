"""
Geometry unit tests for the bubble shape models.

These tests encode the analytic volume and surface area for the shapes used by the pipeline.

Prolate model
-------------
A bubble's prolate limb is modelled by revolution about the MAJOR axis:
  - principal (major) axis length ``a``  → semi-major ``A = a / 2``
  - front half: equatorial (minor) semi-axis ``b1``
  - rear  half: equatorial (minor) semi-axis ``b2``

Volume of one half-spheroid with polar semi-axis ``A`` and equatorial radius
``B`` is ``(2/3)·π·A·B²``. Summing the two halves:

    V = (2/3)·π·A·(b1² + b2²) = (2/3)·π·(a/2)·(b1² + b2²) = (π·a/3)·(b1² + b2²)

so the prefactor is ``π·a/3``.

The prolate surface is the volume-equivalent symmetric spheroid,
``b_eq = sqrt((b1² + b2²)/2)``. This closed, smooth axisymmetric regularization
preserves the model volume and does not claim to reproduce an asymmetric silhouette.
"""

import math

import cv2
import numpy as np
import pytest

from klarity.parsing import sphere_metrics_from_mask
from klarity.shape_models import (
    _half_spheroid_surface,
    volume_surface_oblate,
    volume_surface_prolate,
)

# ---------------------------------------------------------------------------
# Analytic ground-truth volumes for the two competing 3D models.
# A bubble is fit as: a = full major axis, b1/b2 = front/rear semi-minor axes.
# ---------------------------------------------------------------------------


def _true_prolate_volume(a, b1, b2):
    """Revolution about the MAJOR axis: two half prolate spheroids
    (semi-major A=a/2, equatorial radius b1 / b2)."""
    A = a / 2.0
    return (2.0 / 3.0) * math.pi * A * b1**2 + (2.0 / 3.0) * math.pi * A * b2**2


def _true_oblate_volume(a, b1, b2):
    """Revolution about the MINOR axis (Mikaelian): two half oblate spheroids
    (equatorial radius c=a/2, polar semi-axis b1 / b2)."""
    c = a / 2.0
    return (2.0 / 3.0) * math.pi * c**2 * b1 + (2.0 / 3.0) * math.pi * c**2 * b2


def _analytic_prolate_volume(a: float, b1: float, b2: float) -> float:
    """Analytic volume of the two-half-prolate-spheroid model."""
    return (math.pi * a / 3.0) * (b1**2 + b2**2)


# ---------------------------------------------------------------------------
# Prolate volume and surface invariants.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d", [0.5, 1.0, 2.0, 4.0])
def test_prolate_surface_of_sphere(d):
    """A sphere of diameter d (a=d, b1=b2=d/2) must have surface area pi*d^2."""
    _, S = volume_surface_prolate(d, d / 2.0, d / 2.0)
    assert float(S) == pytest.approx(math.pi * d**2, rel=1e-6)


@pytest.mark.parametrize("d", [0.5, 1.0, 2.0, 4.0])
def test_prolate_volume_of_sphere(d):
    """A sphere of diameter d must have volume pi*d^3/6."""
    V, _ = volume_surface_prolate(d, d / 2.0, d / 2.0)
    assert float(V) == pytest.approx(math.pi / 6.0 * d**3, rel=1e-6)


@pytest.mark.parametrize(
    "a,b1,b2",
    [
        (4.0, 1.0, 1.0),  # symmetric prolate spheroid
        (4.0, 1.5, 0.8),  # asymmetric (egg-shaped)
        (2.0, 0.9, 0.9),
    ],
)
def test_prolate_volume_matches_analytic(a, b1, b2):
    V, _ = volume_surface_prolate(a, b1, b2)
    assert float(V) == pytest.approx(_analytic_prolate_volume(a, b1, b2), rel=1e-6)


# ---------------------------------------------------------------------------
# Shape models (klarity.shape_models) — validated against analytic truth.
# ---------------------------------------------------------------------------

_SHAPES = [
    (2.0, 1.0, 1.0),  # sphere d=2
    (4.0, 1.0, 1.0),  # oblate 4:2 (symmetric)
    (6.0, 1.0, 1.0),  # oblate 6:2 (flatter)
    (4.0, 1.5, 0.8),  # asymmetric
]


@pytest.mark.parametrize("a,b1,b2", _SHAPES)
def test_prolate_model_volume_matches_analytic(a, b1, b2):
    V, _ = volume_surface_prolate(a, b1, b2)
    assert float(V) == pytest.approx(_true_prolate_volume(a, b1, b2), rel=1e-9)


@pytest.mark.parametrize("a,b1,b2", _SHAPES)
def test_oblate_model_volume_matches_mikaelian(a, b1, b2):
    """Mikaelian eq. (h): V = (pi*a^2/6)*(b1+b2)."""
    V, _ = volume_surface_oblate(a, b1, b2)
    assert float(V) == pytest.approx(math.pi * a**2 / 6.0 * (b1 + b2), rel=1e-9)
    assert float(V) == pytest.approx(_true_oblate_volume(a, b1, b2), rel=1e-9)


@pytest.mark.parametrize(
    "model,d", [("oblate", 1.0), ("oblate", 3.0), ("prolate", 1.0), ("prolate", 3.0)]
)
def test_both_models_recover_sphere(model, d):
    """Both models must reduce to a true sphere (V=pi*d^3/6, S=pi*d^2) at a=d, b=d/2."""
    from klarity.shape_models import volume_surface

    V, S = volume_surface(d, d / 2.0, d / 2.0, model=model)
    assert float(V) == pytest.approx(math.pi / 6.0 * d**3, rel=1e-6)
    assert float(S) == pytest.approx(math.pi * d**2, rel=1e-6)


def test_oblate_vs_prolate_ratio_is_aspect_ratio():
    """For symmetric bubbles, V_oblate / V_prolate == a / (2b)."""
    a, b = 6.0, 1.0
    Vo, _ = volume_surface_oblate(a, b, b)
    Vp, _ = volume_surface_prolate(a, b, b)
    assert float(Vo) / float(Vp) == pytest.approx(a / (2.0 * b), rel=1e-9)


def test_shape_models_are_vectorised():
    a = np.array([2.0, 4.0, 6.0])
    b1 = np.array([1.0, 1.0, 1.0])
    b2 = np.array([1.0, 1.0, 1.0])
    V, S = volume_surface_oblate(a, b1, b2)
    assert V.shape == (3,) and S.shape == (3,)
    assert np.all(np.isfinite(V)) and np.all(np.isfinite(S))


# ---------------------------------------------------------------------------
# Asymmetric prolate surface: regularization and volume invariance.
# ---------------------------------------------------------------------------

_ASYMMETRIC = [(4.0, 1.5, 0.8), (3.0, 1.2, 0.4), (6.0, 1.0, 0.95)]


@pytest.mark.parametrize("a,b1,b2", _ASYMMETRIC)
def test_prolate_surface_is_volume_equivalent_symmetric_spheroid(a, b1, b2):
    """S must be the closed surface of the spheroid at b_eq = sqrt((b1^2+b2^2)/2)."""
    b_eq = math.sqrt((b1**2 + b2**2) / 2.0)
    expected = 2.0 * float(_half_spheroid_surface(b_eq, a / 2.0))
    _, S = volume_surface_prolate(a, b1, b2)
    assert float(S) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("a,b1,b2", _ASYMMETRIC + [(2.0, 1.0, 1.0)])
def test_prolate_regularization_is_volume_preserving(a, b1, b2):
    """The regularized radius reproduces the model volume exactly."""
    V, _ = volume_surface_prolate(a, b1, b2)
    b_eq = math.sqrt((b1**2 + b2**2) / 2.0)
    assert float(V) == pytest.approx((4.0 / 3.0) * math.pi * b_eq**2 * (a / 2.0), rel=1e-14)
    assert float(V) == pytest.approx((math.pi * a / 3.0) * (b1**2 + b2**2), rel=1e-14)


# ---------------------------------------------------------------------------
# Domain validation: volume and surface must agree on which rows are real.
# ---------------------------------------------------------------------------

_INVALID_AXES = [
    (0.0, 1.0, 1.0),  # zero major axis
    (-2.0, 1.0, 1.0),  # negative major axis
    (2.0, 0.0, 1.0),  # zero radius -> a bubble with no thickness
    (2.0, -1.0, 1.0),  # negative radius
    (2.0, 1.0, 0.0),
    (math.nan, 1.0, 1.0),
    (2.0, math.nan, 1.0),
    (2.0, 1.0, math.inf),
]


@pytest.mark.parametrize("model", [volume_surface_prolate, volume_surface_oblate])
@pytest.mark.parametrize("a,b1,b2", _INVALID_AXES)
def test_invalid_axes_give_nan_volume_and_surface(model, a, b1, b2):
    """Neither quantity may be produced for an axis triple that is not measurable."""
    V, S = model(a, b1, b2)
    assert math.isnan(float(V))
    assert math.isnan(float(S))


def test_validation_is_elementwise_not_all_or_nothing():
    """One bad row must not poison the valid rows around it."""
    a = np.array([2.0, 0.0, 4.0])
    b1 = np.array([1.0, 1.0, 1.5])
    b2 = np.array([1.0, 1.0, 0.8])
    V, S = volume_surface_prolate(a, b1, b2)
    assert np.isfinite(V[[0, 2]]).all() and np.isfinite(S[[0, 2]]).all()
    assert np.isnan(V[1]) and np.isnan(S[1])


@pytest.mark.parametrize("model", [volume_surface_prolate, volume_surface_oblate])
def test_eccentricity_limits_are_finite(model):
    """Near-sphere (e -> 0) and highly elongated (e -> 1) must both stay finite."""
    near_sphere = model(2.0 * (1.0 + 1e-12), 1.0, 1.0)
    elongated = model(1000.0, 0.001, 0.001)
    for V, S in (near_sphere, elongated):
        assert np.isfinite(float(V)) and np.isfinite(float(S))
        assert float(V) > 0.0 and float(S) > 0.0


def test_near_sphere_approaches_the_sphere_from_both_sides():
    """Perturbing a sphere either way must move S continuously off pi*d^2."""
    d = 2.0
    exact = math.pi * d**2
    for eps in (-1e-4, 1e-4):
        _, S = volume_surface_prolate(d, (d / 2.0) * (1 + eps), (d / 2.0) * (1 + eps))
        assert float(S) == pytest.approx(exact, rel=1e-3)


def test_shape_models_broadcast_scalars_against_arrays():
    a = np.array([2.0, 4.0])
    V, S = volume_surface_prolate(a, np.array([0.9, 1.5]), 0.8)
    assert V.shape == (2,) and S.shape == (2,)
    assert np.all(np.isfinite(V)) and np.all(np.isfinite(S))


@pytest.mark.parametrize("model", [volume_surface_prolate, volume_surface_oblate])
def test_shape_models_reject_incompatible_array_shapes(model):
    with pytest.raises(ValueError, match="shape mismatch"):
        model(np.ones(2), np.ones(3), 1.0)


# ---------------------------------------------------------------------------
# Mikaelian two-half-ellipse least-squares fit (klarity.ellipse_fits).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("A,b1,b2", [(120, 60, 60), (120, 70, 45), (90, 50, 50)])
def test_two_half_ellipse_fit_recovers_axes(A, b1, b2):
    """Rasterise a known two-half-ellipse (axis-aligned) and check recovery."""
    from klarity.ellipse_fits import fit_two_half_ellipse_lsq

    pad = 20
    H = 2 * (max(b1, b2) + pad)
    W = 2 * (A + pad)
    cx0, cy0 = W // 2, H // 2

    th_up = np.linspace(0, np.pi, 400)
    th_lo = np.linspace(np.pi, 2 * np.pi, 400)
    xs = np.concatenate([A * np.cos(th_up), A * np.cos(th_lo)])
    ys = np.concatenate([b1 * np.sin(th_up), b2 * np.sin(th_lo)])
    poly = np.stack([cx0 + xs, cy0 + ys], axis=1).astype(np.int32)

    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [poly.reshape(-1, 1, 2)], 1)

    est = fit_two_half_ellipse_lsq(mask.astype(bool))
    assert est is not None
    cx, cy, ang, A_fit, b1_fit, b2_fit = est
    # angle may come out ~0 or ~180; axes are what matter
    assert A_fit == pytest.approx(A, rel=0.06)
    # b1/b2 may swap depending on the principal-axis sign; compare as a set
    assert sorted([b1_fit, b2_fit]) == pytest.approx(sorted([b1, b2]), rel=0.08)


# ---------------------------------------------------------------------------
# Area-based sphere metrics are independent of the ellipsoid path.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("r_px", [40, 80, 160])
def test_sphere_metrics_from_disc_mask(r_px):
    """Rasterise a disc of known radius; area-based d, V, S should match within
    discretisation error."""
    pad = 20
    size = 2 * (r_px + pad)
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (size // 2, size // 2), r_px, 1, thickness=-1)
    mask = mask.astype(bool)

    ps = 0.01  # mm/px
    d_mm, V_mm3, S_mm2 = sphere_metrics_from_mask(mask, ps)

    d_true = 2.0 * r_px * ps
    assert d_mm == pytest.approx(d_true, rel=0.02)
    assert V_mm3 == pytest.approx(math.pi / 6.0 * d_true**3, rel=0.05)
    assert S_mm2 == pytest.approx(math.pi * d_true**2, rel=0.03)


# ---------------------------------------------------------------------------
# Guarded ellipsoid fit used by the pipeline (Mikaelian -> fitEllipse fallback).
# The critical property is the axis CONVENTION: estimate_a_b1_b2_ellipsoid must
# return the FULL major-axis length so that volume_surface_oblate (which halves
# it internally) gets the right value; ``a`` appears squared in the oblate model.
# ---------------------------------------------------------------------------


def _render_two_half_ellipse(A_semi, b1_semi, b2_semi, pad=25):
    """Rasterise an axis-aligned two-half-ellipse with the given SEMI-axes (px)."""
    H = 2 * (max(b1_semi, b2_semi) + pad)
    W = 2 * (A_semi + pad)
    cx0, cy0 = W // 2, H // 2
    th_up = np.linspace(0, np.pi, 400)
    th_lo = np.linspace(np.pi, 2 * np.pi, 400)
    xs = np.concatenate([A_semi * np.cos(th_up), A_semi * np.cos(th_lo)])
    ys = np.concatenate([b1_semi * np.sin(th_up), b2_semi * np.sin(th_lo)])
    poly = np.stack([cx0 + xs, cy0 + ys], axis=1).astype(np.int32)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [poly.reshape(-1, 1, 2)], 1)
    return mask.astype(bool)


@pytest.mark.parametrize("A_semi,b1_semi,b2_semi", [(120, 60, 60), (120, 70, 45), (90, 50, 50)])
def test_ellipsoid_fit_returns_full_major_axis(A_semi, b1_semi, b2_semi):
    """a_mm must be the FULL major axis (= 2*A_semi*ps), b1/b2 the SEMI-minors."""
    from klarity.parsing import estimate_a_b1_b2_ellipsoid

    ps = 0.01  # mm/px
    mask = _render_two_half_ellipse(A_semi, b1_semi, b2_semi)
    est = estimate_a_b1_b2_ellipsoid(mask, ps)
    assert est is not None
    _, _, _, a_mm, b1_mm, b2_mm = est

    assert a_mm == pytest.approx(2.0 * A_semi * ps, rel=0.06)  # FULL major
    assert sorted([b1_mm, b2_mm]) == pytest.approx(sorted([b1_semi * ps, b2_semi * ps]), rel=0.08)


@pytest.mark.parametrize("A_semi,b1_semi,b2_semi", [(120, 60, 60), (120, 70, 45)])
def test_ellipsoid_fit_volume_matches_analytic_oblate(A_semi, b1_semi, b2_semi):
    """End-to-end: fit -> volume_surface_oblate -> analytic oblate volume."""
    from klarity.parsing import estimate_a_b1_b2_ellipsoid

    ps = 0.01
    mask = _render_two_half_ellipse(A_semi, b1_semi, b2_semi)
    _, _, _, a_mm, b1_mm, b2_mm = estimate_a_b1_b2_ellipsoid(mask, ps)
    V, _ = volume_surface_oblate(a_mm, b1_mm, b2_mm)

    V_true = _true_oblate_volume(2.0 * A_semi * ps, b1_semi * ps, b2_semi * ps)
    assert float(V) == pytest.approx(V_true, rel=0.10)


def test_ellipsoid_fit_returns_none_for_degenerate_mask():
    """A mask too small to fit any ellipse yields None (caller uses sphere fallback)."""
    from klarity.parsing import estimate_a_b1_b2_ellipsoid

    mask = np.zeros((20, 20), dtype=bool)
    mask[10, 9:12] = True  # a 3-px line: < 5 contour points
    assert estimate_a_b1_b2_ellipsoid(mask, 0.01) is None
