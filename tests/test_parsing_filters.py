"""
Unit tests for decision helpers in ``klarity.parsing`` that support detection filtering
and sphere/ellipsoid classification.

* ``_size_gate_keep_indices`` — the size-aware confidence gate must measure the
  mask area fraction in the SAME coordinate space as the mask tensor
  (network/letterbox), not against the original image area. Using the original
  area changes the intended threshold.

* ``_use_sphere_model`` — the hybrid sphere/ellipsoid gate must group the two
  sphere criteria before the NaN guard: ``(size_thresh or near) and not
  isnan(d)``. Written without parentheses, Python binds it as ``size_thresh or
  (near and not isnan(d))`` (``and`` > ``or``), which would pick the sphere model
  for a small bubble even when its sphere diameter is NaN.
"""

import math

import numpy as np
import pandas as pd
import pytest

from klarity.parsing import (
    BAND_STATUS_ONE_MODEL_ONLY,
    BAND_STATUS_SPHERE_DEGENERATE,
    CLIP_RATIO_THRESHOLD,
    _size_gate_keep_indices,
    _spheroid_band,
    _use_sphere_model,
    drop_border_clipped,
    drop_zero_area_masks,
    is_blank_frame,
    mask_solidity,
)
from klarity.shape_models import volume_surface_oblate, volume_surface_prolate

# ---------------------------------------------------------------------------
# Size-aware confidence gate
# ---------------------------------------------------------------------------

CONF_SMALL = 0.12
CONF_LARGE = 0.60
LARGE_FRAC = 0.08


def test_size_gate_uses_mask_canvas_not_original_image():
    """A mask ~12 % of the network canvas is 'large' and must clear conf_large.

    The foreground count and canvas area must be expressed in the same coordinate space.
    """
    mask_areas = np.array([40000.0])  # foreground px, network space
    canvas = 640.0 * 512.0  # network/letterbox canvas (frac = 0.122 -> large)
    confs = np.array([0.30])  # below conf_large, above conf_small

    keep = _size_gate_keep_indices(mask_areas, canvas, confs, CONF_SMALL, CONF_LARGE, LARGE_FRAC)
    assert keep == []


def test_size_gate_small_and_large_branches():
    canvas = 640.0 * 512.0
    areas = np.array(
        [
            1000.0,  # tiny mask -> small branch
            0.15 * canvas,  # large mask, mid conf -> dropped (needs 0.60)
            0.15 * canvas,  # large mask, high conf -> kept
        ]
    )
    confs = np.array([0.15, 0.50, 0.70])
    keep = _size_gate_keep_indices(areas, canvas, confs, CONF_SMALL, CONF_LARGE, LARGE_FRAC)
    assert keep == [0, 2]


def test_size_gate_small_mask_low_conf_still_dropped_below_conf_small():
    canvas = 640.0 * 512.0
    areas = np.array([500.0])  # small
    confs = np.array([0.05])  # below conf_small
    keep = _size_gate_keep_indices(areas, canvas, confs, CONF_SMALL, CONF_LARGE, LARGE_FRAC)
    assert keep == []


def test_size_gate_frac_exactly_at_threshold_is_large():
    canvas = 640.0 * 512.0
    areas = np.array([LARGE_FRAC * canvas])  # frac == large_frac -> large (>=)
    confs = np.array([0.30])  # below conf_large
    keep = _size_gate_keep_indices(areas, canvas, confs, CONF_SMALL, CONF_LARGE, LARGE_FRAC)
    assert keep == []


def test_size_gate_empty_input():
    keep = _size_gate_keep_indices(
        np.array([]), 0.0, np.array([]), CONF_SMALL, CONF_LARGE, LARGE_FRAC
    )
    assert keep == []


def test_size_gate_nonpositive_canvas_keeps_all():
    # Degenerate canvas -> do not divide by zero; keep everything (defensive).
    areas = np.array([1.0, 2.0])
    confs = np.array([0.9, 0.9])
    keep = _size_gate_keep_indices(areas, 0.0, confs, CONF_SMALL, CONF_LARGE, LARGE_FRAC)
    assert keep == [0, 1]


# ---------------------------------------------------------------------------
# Hybrid sphere/ellipsoid gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "size_thresh,near,d_mm,expected",
    [
        (True, False, 1.0, True),  # small enough, valid diameter
        (False, True, 1.0, True),  # near-spherical, valid diameter
        (True, True, 1.0, True),  # both criteria
        (False, False, 1.0, False),  # neither -> ellipsoid
        # NaN sphere diameter must veto the sphere model in every case.
        (True, False, float("nan"), False),
        (True, True, float("nan"), False),
        (False, True, float("nan"), False),
        (False, False, float("nan"), False),
    ],
)
def test_use_sphere_model(size_thresh, near, d_mm, expected):
    assert _use_sphere_model(size_thresh, near, d_mm) is expected


def test_use_sphere_model_requires_finite_diameter():
    # A small bubble with NaN sphere diameter must not be classified as a sphere.
    assert _use_sphere_model(True, False, float("nan")) is False


# ---------------------------------------------------------------------------
# Classification-respecting spheroid band (store-both prolate/oblate)
# ---------------------------------------------------------------------------


def test_spheroid_band_sphere_class_is_degenerate():
    # Sphere-classified bubble: no depth ambiguity -> lower == upper == sphere.
    d, V, S = 1.0, math.pi / 6.0, math.pi  # d = 1 mm sphere
    band = _spheroid_band(True, d, V, S, 999.0, 999.0, 999.0, 999.0)
    assert (band.d_lower, band.volume_lower, band.surface_lower) == (d, V, S)
    assert (band.d_upper, band.volume_upper, band.surface_upper) == (d, V, S)
    assert band.status == BAND_STATUS_SPHERE_DEGENERATE


def test_spheroid_band_ellipsoid_ordering_and_ratio():
    # Flattened bubble a = 4, b1 = b2 = 1 (aspect 2): lower = prolate,
    # upper = oblate, and V_up / V_lo = aspect = a / (2 b) exactly.
    a, b1, b2 = 4.0, 1.0, 1.0
    V_pro, S_pro = volume_surface_prolate(a, b1, b2)
    V_obl, S_obl = volume_surface_oblate(a, b1, b2)
    band = _spheroid_band(
        False, float("nan"), float("nan"), float("nan"), V_pro, S_pro, V_obl, S_obl
    )
    assert band.volume_lower == V_pro and band.volume_upper == V_obl
    assert band.surface_lower == min(S_pro, S_obl) and band.surface_upper == max(S_pro, S_obl)
    assert band.volume_lower < band.volume_upper
    assert band.volume_upper / band.volume_lower == pytest.approx(a / (2.0 * b1))
    # volume-equivalent diameters follow d = (6V/pi)^(1/3)
    assert band.d_lower == pytest.approx((6.0 * V_pro / math.pi) ** (1.0 / 3.0))
    assert band.d_upper == pytest.approx((6.0 * V_obl / math.pi) ** (1.0 / 3.0))
    assert band.d_lower < band.d_upper


def test_spheroid_band_asymmetric_swap_keeps_ordering():
    # For a strongly fore/aft-asymmetric fit,
    # V_oblate/V_prolate = a(b1+b2)/(2(b1^2+b2^2)) can drop below 1.
    a, b1, b2 = 2.2, 1.5, 0.5
    # process_image passes plain floats to _spheroid_band; mirror that here
    V_pro, S_pro = (float(x) for x in volume_surface_prolate(a, b1, b2))
    V_obl, S_obl = (float(x) for x in volume_surface_oblate(a, b1, b2))
    assert V_obl < V_pro  # the swap case
    band = _spheroid_band(
        False, float("nan"), float("nan"), float("nan"), V_pro, S_pro, V_obl, S_obl
    )
    assert band.volume_lower == V_obl
    assert band.volume_upper == V_pro
    assert band.volume_lower <= band.volume_upper and band.d_lower <= band.d_upper
    # the AREA interval is sorted on area regardless of which model won on volume
    assert band.surface_lower <= band.surface_upper
    assert {band.surface_lower, band.surface_upper} == {float(S_pro), float(S_obl)}


def test_spheroid_band_sphere_limit_collapses():
    # At the sphere limit a = 2 b the two models coincide -> band width zero.
    a, b = 2.0, 1.0
    V_pro, S_pro = volume_surface_prolate(a, b, b)
    V_obl, S_obl = volume_surface_oblate(a, b, b)
    band = _spheroid_band(
        False, float("nan"), float("nan"), float("nan"), V_pro, S_pro, V_obl, S_obl
    )
    assert band.volume_lower == pytest.approx(band.volume_upper)
    assert band.d_lower == pytest.approx(band.d_upper)
    assert band.surface_lower == pytest.approx(band.surface_upper)


# ---------------------------------------------------------------------------
# Mask solidity (merge/truncation screen; replaces the retired confidence gate)
# ---------------------------------------------------------------------------


def _disk(cx: float, cy: float, r: float, shape=(200, 200)) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2


def test_mask_solidity_single_disk_near_one():
    # A single convex silhouette must have solidity ~1.
    assert mask_solidity(_disk(100, 100, 60)) > 0.98


def test_mask_solidity_dumbbell_below_merge_threshold():
    # Two overlapping disks form a concave waist — the merged-detection
    # signature the solidity screen is intended to detect (< ~0.95).
    dumbbell = _disk(60, 100, 45) | _disk(150, 100, 45)
    s = mask_solidity(dumbbell)
    assert s < 0.95
    assert s > 0.5  # still a substantial fraction of its hull


def test_mask_solidity_degenerate_masks_are_nan():
    assert np.isnan(mask_solidity(np.zeros((50, 50), bool)))


def test_spheroid_band_one_usable_model_is_not_a_band():
    """A model pair with only one complete member yields no band, and says so.

    A single finite member cannot define an interval.
    """
    band = _spheroid_band(
        False, float("nan"), float("nan"), float("nan"), 0.0, 0.0, float("nan"), float("nan")
    )
    assert band.status == BAND_STATUS_ONE_MODEL_ONLY
    assert math.isnan(band.d_lower) and math.isnan(band.d_upper)
    assert math.isnan(band.volume_lower) and math.isnan(band.volume_upper)


# ---------------------------------------------------------------------------
# Blank/black acquisition-frame rejection (is_blank_frame)
# ---------------------------------------------------------------------------
#
# Some water-setting acquisitions write one black (illumination-off) frame per
# 50-frame burst. The exact-zero guard misses them because a little sensor speckle
# survives, yet the model paints a spurious ~59%-of-frame mask that becomes a fake
# ~11 mm "sphere" dominating gas holdup. Measured 2026-07-17: black frames mean
# intensity 0.0; every illuminated frame >= 87. A low mean-intensity cut separates
# them with a wide margin.


def test_is_blank_frame_flags_pure_black_frame():
    black = np.zeros((1024, 1280, 3), dtype=np.uint8)
    assert is_blank_frame(black)


def test_is_blank_frame_flags_black_frame_with_sensor_speckle():
    # A handful of bright speckle pixels keeps the mean ~0 -> exact-zero test misses
    # it, but is_blank_frame still flags it. This is the real failure mode.
    speckle = np.zeros((1024, 1280, 3), dtype=np.uint8)
    rng = np.random.default_rng(0)
    ys, xs = rng.integers(0, 1024, 50), rng.integers(0, 1280, 50)
    speckle[ys, xs] = 255
    assert not np.all(speckle == 0)  # the exact-zero guard would NOT skip it
    assert float(speckle.mean()) < 1.0
    assert is_blank_frame(speckle)


def test_is_blank_frame_passes_illuminated_frames():
    # Bright-field water frames measured >= 147; the dimmest turbid 0.25% xanthan
    # frame was 87. Both must pass at the default threshold.
    assert not is_blank_frame(np.full((1024, 1280, 3), 147, dtype=np.uint8))
    assert not is_blank_frame(np.full((1024, 1280, 3), 87, dtype=np.uint8))


def test_is_blank_frame_threshold_is_tunable():
    frame = np.full((10, 10, 3), 20, dtype=np.uint8)
    assert is_blank_frame(frame, mean_thresh=30)  # 20 < 30 -> blank
    assert not is_blank_frame(frame, mean_thresh=15)  # 20 >= 15 -> not blank


# ---------------------------------------------------------------------------
# Zero-area phantom-mask filtering invariants
# ---------------------------------------------------------------------------
#
# YOLO can emit a detection whose segmentation mask thresholds to 0 px. These
# carry mask_area == 0 and NaN volume/diameter (model_used == "sphere_fallback").
# Volume/diameter aggregates skip them (NaN-aware), but a raw .size() count would
# include them and inflate number density (~1%, up to ~5% at low shear / high
# viscosity). drop_zero_area_masks removes them so counts match the volume metrics.


def _bubble_frame():
    return pd.DataFrame(
        {
            "mask_area": [100.0, 0.0, 250.0, 0.0, 5.0],
            "volume_mm3_chosen": [1.2, float("nan"), 3.4, float("nan"), 0.1],
            "model_used": [
                "sphere",
                "sphere_fallback",
                "asym_ellipsoid",
                "sphere_fallback",
                "sphere",
            ],
        }
    )


def test_drop_zero_area_masks_removes_only_zero_area_rows():
    out = drop_zero_area_masks(_bubble_frame())
    assert len(out) == 3
    assert (out["mask_area"] > 0).all()
    assert list(out["model_used"]) == ["sphere", "asym_ellipsoid", "sphere"]


def test_drop_zero_area_masks_makes_count_match_volume_count():
    # The scientific invariant: after filtering, a raw row count equals the count of
    # measurable (non-NaN volume) bubbles, keeping the number-density numerator physical.
    out = drop_zero_area_masks(_bubble_frame())
    assert len(out) == out["volume_mm3_chosen"].notna().sum()


def test_drop_zero_area_masks_drops_nan_area():
    df = pd.DataFrame({"mask_area": [10.0, float("nan"), 3.0]})
    out = drop_zero_area_masks(df)
    assert len(out) == 2
    assert (out["mask_area"] > 0).all()


def test_drop_zero_area_masks_noop_without_column():
    df = pd.DataFrame({"x": [1, 2, 3]})
    out = drop_zero_area_masks(df)
    assert len(out) == 3


# ---------------------------------------------------------------------------
# Border-truncated bubbles
# ---------------------------------------------------------------------------
#
# The published policy uses the chord-to-diameter ratio
# border_contact_px / equivalent_diameter. Rows above 1.0 are excluded; border contact
# alone and ratios <= 1.0 are retained unchanged. No reconstructed geometry is substituted.


def _clip_frame():
    return pd.DataFrame(
        {
            # ratios: 0.0 (interior), 0.5 (grazing), 1.0 (exactly at threshold), 2.0 (deep)
            "border_contact_px": [0.0, 25.0, 60.0, 80.0],
            "equivalent_diameter": [50.0, 50.0, 60.0, 40.0],
            # Scientific values must pass through unchanged for every retained row.
            "volume_mm3_chosen": [1.0, 2.0, 3.0, 4.0],
        }
    )


def test_published_border_clip_ratio_threshold_is_one():
    assert CLIP_RATIO_THRESHOLD == pytest.approx(1.0)


def test_drop_border_clipped_removes_only_severe_truncation():
    out = drop_border_clipped(_clip_frame())
    assert list(out["border_contact_px"]) == [0.0, 25.0, 60.0]
    assert list(out["volume_mm3_chosen"]) == [1.0, 2.0, 3.0]


def test_drop_border_clipped_keeps_the_threshold_itself():
    # ratio == 1.0 is kept; the filter is strictly greater-than, matching the pinned
    # Task 2b policy (drop where the border chord EXCEEDS the equivalent diameter).
    out = drop_border_clipped(_clip_frame())
    ratio = out["border_contact_px"] / out["equivalent_diameter"]
    assert ratio.max() == pytest.approx(1.0)


def test_drop_border_clipped_keeps_border_contact_below_threshold_unchanged():
    df = _clip_frame().loc[[1]].copy()
    out = drop_border_clipped(df)
    pd.testing.assert_frame_equal(out, df)


def test_drop_border_clipped_threshold_is_configurable():
    out = drop_border_clipped(_clip_frame(), threshold=0.4)
    assert list(out["border_contact_px"]) == [0.0]


def test_drop_border_clipped_keeps_rows_with_undefined_ratio():
    # A NaN or zero equivalent diameter says nothing about truncation, so the row is
    # kept here and removed (if at all) by the zero-area filter instead.
    df = pd.DataFrame(
        {
            "border_contact_px": [10.0, 10.0, float("nan")],
            "equivalent_diameter": [0.0, float("nan"), 50.0],
        }
    )
    assert len(drop_border_clipped(df)) == 3


def test_drop_border_clipped_noop_without_columns():
    df = pd.DataFrame({"x": [1, 2, 3]})
    assert len(drop_border_clipped(df)) == 3
