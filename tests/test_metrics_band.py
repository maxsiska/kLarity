"""Unit tests for the prolate/oblate spheroid band in ``klarity.metrics``.

The rules are:

* sphere-classified bubbles have no out-of-plane depth ambiguity -> lower == upper ==
  the chosen (sphere) value, regardless of the stored prolate/oblate spheroid values;
* ellipsoid-classified bubbles take the VOLUME min/max for the volume interval and,
  INDEPENDENTLY, the AREA min/max for the area interval (ties -> prolate is lower);
Row 2 of the fixture exercises different volume and area orderings.
"""

import math

import numpy as np
import pandas as pd

from klarity import metrics


def _frame() -> pd.DataFrame:
    # rows: sphere; ellipsoid (usual order); ellipsoid with volume and area order OPPOSED
    # (V_prolate=6 > V_oblate=4 but A_prolate=3 > A_oblate=2, so the volume-lower model is
    # the area-larger one); tie on volume with distinct areas.
    return pd.DataFrame(
        {
            "model_used": ["sphere", "asym_ellipsoid", "asym_ellipsoid", "asym_ellipsoid"],
            "volume_mm3_prolate": [9.0, 2.0, 6.0, 3.0],
            "volume_mm3_oblate": [9.0, 5.0, 4.0, 3.0],
            "surface_area_mm2_prolate": [7.0, 1.5, 3.0, 1.1],
            "surface_area_mm2_oblate": [7.0, 2.5, 2.0, 2.2],
            "bubble_volume_mm3": [1.0, 5.0, 4.0, 3.0],  # chosen = oblate for ellipsoids
            "bubble_surface_area_mm2": [1.0, 2.5, 2.0, 2.2],
        }
    )


def test_sphere_band_is_degenerate_at_chosen():
    v_lo, v_up, a_lo, a_up = metrics.spheroid_band_arrays(_frame())[:4]
    # sphere row: lower == upper == chosen (NOT the stored prolate/oblate=9.0)
    assert v_lo[0] == 1.0 and v_up[0] == 1.0
    assert a_lo[0] == 1.0 and a_up[0] == 1.0


def test_ellipsoid_usual_order_prolate_is_lower():
    band = metrics.spheroid_band_arrays(_frame())
    # row 1: V_prolate=2 < V_oblate=5 and A_prolate=1.5 < A_oblate=2.5 -- orders agree,
    # so the independently sorted intervals coincide here.
    assert band.v_lower[1] == 2.0 and band.v_upper[1] == 5.0
    assert band.a_lower[1] == 1.5 and band.a_upper[1] == 2.5


def test_area_interval_is_sorted_on_area_not_on_volume():
    """Row 2 has V_prolate > V_oblate while A_prolate > A_oblate.

    The volume interval must take oblate as its lower member, but the AREA interval must
    still be sorted on area. Pairing the areas to the volume order here would return
    a_lower=2.0 > a_upper=3.0, which violates the interval invariant.
    """
    band = metrics.spheroid_band_arrays(_frame())
    assert band.v_lower[2] == 4.0 and band.v_upper[2] == 6.0  # oblate, prolate
    assert band.a_lower[2] == 2.0 and band.a_upper[2] == 3.0  # oblate, prolate -- sorted


def test_area_interval_is_never_inverted():
    """The invariant that failed for 0.54% of stored production rows."""
    band = metrics.spheroid_band_arrays(_frame(), classification_respecting=False)
    finite = np.isfinite(band.a_lower) & np.isfinite(band.a_upper)
    assert np.all(band.a_lower[finite] <= band.a_upper[finite])
    finite_v = np.isfinite(band.v_lower) & np.isfinite(band.v_upper)
    assert np.all(band.v_lower[finite_v] <= band.v_upper[finite_v])


def test_ellipsoid_tie_keeps_prolate_as_lower():
    band = metrics.spheroid_band_arrays(_frame())
    # row 3: equal volumes -> prolate is the lower member (matches parsing._spheroid_band)
    assert band.v_lower[3] == 3.0 and band.v_upper[3] == 3.0
    # the area interval is still sorted on area
    assert band.a_lower[3] == 1.1 and band.a_upper[3] == 2.2


def test_band_brackets_the_chosen_value():
    df = _frame()
    v_lo, v_up = metrics.spheroid_band_arrays(df)[:2]
    chosen = df["bubble_volume_mm3"].to_numpy()
    assert np.all(v_lo <= chosen + 1e-9)
    assert np.all(chosen <= v_up + 1e-9)


# ---------------------------------------------------------------------------
# Unconditional band (sphere/ellipsoid threshold removed from reporting)
# ---------------------------------------------------------------------------


def test_unconditional_band_applies_to_sphere_classified_rows():
    """The depth axis is unobserved for a sphere-classified bubble too.

    With ``classification_respecting=False`` the sphere row must take the stored
    prolate/oblate values instead of collapsing to the chosen sphere value -- this is
    what removes the threshold from the reported holdup and interfacial area.
    """
    df = _frame()
    df.loc[0, ["volume_mm3_prolate", "volume_mm3_oblate"]] = [8.0, 9.0]
    df.loc[0, ["surface_area_mm2_prolate", "surface_area_mm2_oblate"]] = [6.0, 7.0]

    v_lo, v_up, a_lo, a_up = metrics.spheroid_band_arrays(df, classification_respecting=False)[:4]
    assert v_lo[0] == 8.0 and v_up[0] == 9.0
    assert a_lo[0] == 6.0 and a_up[0] == 7.0

    # ...while the default convention still collapses it to the chosen sphere value.
    v_lo_c, v_up_c = metrics.spheroid_band_arrays(df, classification_respecting=True)[:2]
    assert v_lo_c[0] == 1.0 and v_up_c[0] == 1.0


def test_unconditional_band_agrees_on_ellipsoid_rows():
    """Ellipsoid-classified rows are unaffected by the convention switch."""
    df = _frame()
    band_default = metrics.spheroid_band_arrays(df, classification_respecting=True)
    band_uncond = metrics.spheroid_band_arrays(df, classification_respecting=False)
    for arr_default, arr_uncond in zip(band_default, band_uncond):
        np.testing.assert_allclose(arr_default[1:], arr_uncond[1:])


def test_missing_spheroid_falls_back_to_chosen_in_both_conventions():
    """sphere_fallback rows (ellipse fit failed) have NaN spheroids, not a NaN band.

    Their only measurement is the sphere value, so both conventions must return it
    rather than propagating NaN into the summed per-frame band.
    """
    df = _frame()
    df.loc[1, "model_used"] = "sphere_fallback"
    df.loc[1, ["volume_mm3_prolate", "volume_mm3_oblate"]] = [np.nan, np.nan]
    df.loc[1, ["surface_area_mm2_prolate", "surface_area_mm2_oblate"]] = [np.nan, np.nan]

    for respecting in (True, False):
        v_lo, v_up, a_lo, a_up = metrics.spheroid_band_arrays(
            df, classification_respecting=respecting
        )[:4]
        assert v_lo[1] == 5.0 and v_up[1] == 5.0
        assert a_lo[1] == 2.5 and a_up[1] == 2.5


# ---------------------------------------------------------------------------
# Incomplete and invalid model rows must stay visible, never become zeros
# ---------------------------------------------------------------------------


def _frame_level_input(**overrides) -> pd.DataFrame:
    """One-bubble frame-level input with the MultiIndex compute_frame_metrics expects."""
    index = pd.MultiIndex.from_tuples(
        [("placement_1", "setting", "rep_1", 1, 1)],
        names=[
            "placement",
            "reactor_setting",
            "replicate",
            "burst_index",
            "image_number_in_burst",
        ],
    )
    row = {
        "model_used": ["asym_ellipsoid"],
        "bubble_volume_mm3": [2.0],
        "bubble_surface_area_mm2": [3.0],
        "equivalent_diameter_mm": [1.0],
        "volume_mm3_prolate": [2.0],
        "volume_mm3_oblate": [4.0],
        "surface_area_mm2_prolate": [1.5],
        "surface_area_mm2_oblate": [2.5],
    }
    row.update({k: [v] for k, v in overrides.items()})
    return pd.DataFrame(row, index=index)


def test_all_nan_scientific_group_stays_missing():
    """A frame with no usable geometry must not report zero physical quantities."""
    bubbles = _frame_level_input(
        model_used="sphere_fallback",
        volume_mm3_prolate=np.nan,
        volume_mm3_oblate=np.nan,
        surface_area_mm2_prolate=np.nan,
        surface_area_mm2_oblate=np.nan,
        bubble_volume_mm3=np.nan,
        bubble_surface_area_mm2=np.nan,
        equivalent_diameter_mm=np.nan,
    )
    out = metrics.compute_frame_metrics_from_bubbles(
        bubbles,
        placement_level="placement",
        setting_level="reactor_setting",
        classification_respecting_band=False,
    )
    for col in (
        "V_total_mm3",
        "A_total_mm2",
        "diameter_sum_mm",
        "V_sphere_mm3",
        "A_sphere_mm2",
        "V_total_lower_mm3",
        "V_total_upper_mm3",
        "A_total_lower_mm2",
        "A_total_upper_mm2",
    ):
        assert np.isnan(out[col].iloc[0]), f"{col} became {out[col].iloc[0]!r}, expected NaN"


def test_band_requires_both_surfaces_not_just_both_volumes():
    """Two finite volumes with a missing surface is not a usable band row.

    Deciding availability on the volumes alone admitted a row whose summed area was then
    silently zero.
    """
    df = _frame()
    df.loc[1, "surface_area_mm2_oblate"] = np.nan
    band = metrics.spheroid_band_arrays(df, classification_respecting=False)
    assert np.isnan(band.v_lower[1]) and np.isnan(band.v_upper[1])
    assert np.isnan(band.a_lower[1]) and np.isnan(band.a_upper[1])


# ---------------------------------------------------------------------------
# Per-bubble band written at processing time (klarity.parsing._spheroid_band)
# ---------------------------------------------------------------------------


def test_parsing_band_status_two_models():
    from klarity import parsing

    band = parsing._spheroid_band(False, 1.0, 1.0, 1.0, 2.0, 3.0, 5.0, 2.5)
    assert band.status == parsing.BAND_STATUS_TWO_MODELS
    # volume interval takes prolate as lower; area interval is sorted on area, so the
    # area-lower member is the oblate one despite its larger volume
    assert (band.volume_lower, band.volume_upper) == (2.0, 5.0)
    assert (band.surface_lower, band.surface_upper) == (2.5, 3.0)


def test_parsing_band_status_sphere_is_degenerate():
    from klarity import parsing

    band = parsing._spheroid_band(True, 1.2, 0.9, 4.5, 2.0, 3.0, 5.0, 2.5)
    assert band.status == parsing.BAND_STATUS_SPHERE_DEGENERATE
    assert band.volume_lower == band.volume_upper == 0.9
    assert band.surface_lower == band.surface_upper == 4.5


def test_parsing_band_status_one_model_only_is_recorded_not_collapsed():
    """One complete model is not a band; the row must say so rather than look ordinary."""
    from klarity import parsing

    band = parsing._spheroid_band(False, 1.0, 1.0, 1.0, 2.0, 3.0, np.nan, np.nan)
    assert band.status == parsing.BAND_STATUS_ONE_MODEL_ONLY
    assert math.isnan(band.volume_lower) and math.isnan(band.volume_upper)


def test_parsing_band_status_invalid_when_neither_model_is_usable():
    from klarity import parsing

    band = parsing._spheroid_band(False, 1.0, 1.0, 1.0, np.nan, np.nan, np.nan, np.nan)
    assert band.status == parsing.BAND_STATUS_INVALID
    assert math.isnan(band.volume_lower)


def test_parsing_band_volume_and_area_orders_may_disagree():
    """The production case: smaller volume, larger area."""
    from klarity import parsing

    band = parsing._spheroid_band(False, 1.0, 1.0, 1.0, 4.0, 9.0, 6.0, 7.0)
    assert band.volume_lower == 4.0
    assert band.surface_lower == 7.0 and band.surface_upper == 9.0
    assert band.surface_lower <= band.surface_upper


# ---------------------------------------------------------------------------
# Volume-equivalent diameter (the reported size definition)
# ---------------------------------------------------------------------------


def test_volume_equivalent_diameter_inverts_the_sphere_volume():
    """d_V must return the diameter of the sphere of that volume, exactly."""
    diameters = np.array([0.1, 1.0, 2.5, 8.0])
    volumes = np.pi * diameters**3 / 6.0
    np.testing.assert_allclose(metrics.volume_equivalent_diameter_mm(volumes), diameters)


def test_volume_equivalent_diameter_is_nan_for_non_positive_volume():
    """Zero/negative/NaN volume has no equivalent diameter; it must not raise."""
    out = metrics.volume_equivalent_diameter_mm(np.array([0.0, -1.0, np.nan]))
    assert np.all(np.isnan(out))
