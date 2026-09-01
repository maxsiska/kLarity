"""
2D ellipse-fitting methods used to turn a bubble mask into (major, minor) axes.

These are the in-plane *fits* (they define the drawn outline), as distinct from
the 3D shape models in ``shape_models.py`` (which only turn a fixed fit into a
volume/surface).

Currently provided
------------------
``fit_two_half_ellipse_lsq`` — Mikaelian et al. 2015 procedure (their step g):
    least-squares fit of two half-ellipses sharing the same centre and major
    axis, split along the major axis, with independent front/rear semi-minor
    axes b1, b2 (fore-and-aft asymmetry along the MINOR / symmetry axis).

This differs from ``parsing.estimate_a_b1_b2_split_fit`` (the current pipeline
method), which measures the minor axis from the *maximum* perpendicular distance
at the quarter-points and places the asymmetry along the MAJOR axis.
"""

from __future__ import annotations

import typing

import cv2
import numpy as np


def _principal_axis(points_centered: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    cov = np.cov(points_centered.T)
    evals, evecs = np.linalg.eigh(cov)
    u = evecs[:, int(np.argmax(evals))]
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.array([-u[1], u[0]])
    angle_deg = float(np.degrees(np.arctan2(u[1], u[0])))
    return u, v, angle_deg


def fit_two_half_ellipse_lsq(
    mask: np.ndarray,
) -> typing.Optional[typing.Tuple[float, float, float, float, float, float]]:
    """Mikaelian two-half-ellipse least-squares fit.

    Returns ``(cx, cy, angle_deg, A_px, b1_px, b2_px)`` in PIXELS, where ``A_px``
    is the SEMI-major axis and ``b1_px``/``b2_px`` are the front/rear semi-minor
    axes, or ``None`` if the fit fails.

    Method: rotate the contour into the principal-axis frame (u=major, v=minor),
    split points by the sign of v, and solve the linear least-squares system
        u^2 * (1/A^2) + v^2 * (1/b_side^2) = 1
    for the shared 1/A^2 and the per-side 1/b1^2, 1/b2^2.
    """
    m = (mask > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    pts = cnt[:, 0, :].astype(float)
    if pts.shape[0] < 6:
        return None

    mom = cv2.moments(cnt)
    if mom["m00"] == 0:
        return None
    cx, cy = mom["m10"] / mom["m00"], mom["m01"] / mom["m00"]

    P = pts - np.array([cx, cy])
    u, v, angle_deg = _principal_axis(P)
    uu = P @ u
    vv = P @ v
    n = pts.shape[0]
    rhs = np.ones(n, dtype=float)

    def _solve(v0: float):
        """Linear LSQ for (1/A^2, 1/b1^2, 1/b2^2) given a split-line offset v0."""
        vs = vv - v0
        front = vs >= 0
        design = np.zeros((n, 3), dtype=float)
        design[:, 0] = uu**2
        design[front, 1] = vs[front] ** 2
        design[~front, 2] = vs[~front] ** 2
        sol, *_ = np.linalg.lstsq(design, rhs, rcond=None)
        resid = float(np.sum((design @ sol - 1.0) ** 2))
        return resid, sol

    # The centroid is offset from the half-ellipses' meeting line for a
    # fore-and-aft asymmetric shape, so search the split offset v0 along the
    # minor axis (free centre, as in Mikaelian's fit).
    v_span = float(vv.max() - vv.min())
    best_v0, best = 0.0, _solve(0.0)
    for v0 in np.linspace(-0.35 * v_span, 0.35 * v_span, 35):
        cur = _solve(float(v0))
        if cur[0] < best[0]:
            best_v0, best = float(v0), cur
    # local refinement
    step = 0.7 * v_span / 35.0
    for v0 in np.linspace(best_v0 - step, best_v0 + step, 15):
        cur = _solve(float(v0))
        if cur[0] < best[0]:
            best_v0, best = float(v0), cur

    p, q1, q2 = best[1]
    if not (p > 0 and q1 > 0 and q2 > 0):
        return None

    A_px = 1.0 / np.sqrt(p)
    b1_px = 1.0 / np.sqrt(q1)
    b2_px = 1.0 / np.sqrt(q2)
    # centre sits on the meeting line, shifted from the centroid by best_v0 along v
    cx_c = cx + best_v0 * v[0]
    cy_c = cy + best_v0 * v[1]
    return float(cx_c), float(cy_c), float(angle_deg), float(A_px), float(b1_px), float(b2_px)


def two_half_ellipse_polyline(
    cx: float,
    cy: float,
    angle_deg: float,
    A_px: float,
    b1_px: float,
    b2_px: float,
    num: int = 120,
) -> np.ndarray:
    """Closed outline of the Mikaelian two-half-ellipse fit (asymmetry along the
    minor axis, split along the major axis). Returns int32 (N, 2) image points."""
    th = np.deg2rad(angle_deg)
    ct, st = np.cos(th), np.sin(th)
    t_up = np.linspace(0.0, np.pi, num)  # v >= 0  -> b1
    t_lo = np.linspace(np.pi, 2.0 * np.pi, num)  # v <= 0  -> b2
    xs = np.concatenate([A_px * np.cos(t_up), A_px * np.cos(t_lo)])
    ys = np.concatenate([b1_px * np.sin(t_up), b2_px * np.sin(t_lo)])
    X = cx + xs * ct - ys * st
    Y = cy + xs * st + ys * ct
    return np.stack([X, Y], axis=1).astype(np.int32)
