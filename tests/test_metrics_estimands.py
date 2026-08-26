"""Condition-level estimands used by public tables and figures."""

import numpy as np
import pandas as pd
import pytest

from klarity import metrics


def _frames() -> pd.DataFrame:
    """Two deliberately unequal frames, so means of ratios are visibly wrong."""
    return pd.DataFrame(
        {
            "placement": ["placement_1", "placement_1"],
            "xanthan": ["000 xanthan", "000 xanthan"],
            "rpm_val": [100.0, 100.0],
            "aer_val": [55.0, 55.0],
            "diameter_sum_mm": [20.0, 3.0],
            "diameter_count": [10.0, 1.0],
            "V_total_mm3": [100.0, 300.0],
            "A_total_mm2": [50.0, 60.0],
            "V_obs_mm3": [1000.0, 1000.0],
            "V_obs_mL": [1.0, 1.0],
            "V_liquid_mm3": [900.0, 700.0],
            "n_bubbles_total": [10.0, 1.0],
            "mean_diameter_mm": [2.0, 3.0],
            "epsilon_obs": [0.1, 0.3],
            "a_obs_m2_m3": [50.0, 60.0],
            "a_specific_m2_m3": [500.0, 200.0],
            "a_L_m2_m3": [1000.0 * 50.0 / 900.0, 1000.0 * 60.0 / 700.0],
            "V_total_mid_mm3": [100.0, 300.0],
            "A_total_mid_mm2": [50.0, 60.0],
        }
    )


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        ("mean_diameter_mm", 23.0 / 11.0),
        ("epsilon_obs", 400.0 / 2000.0),
        ("a_obs_m2_m3", 1000.0 * 110.0 / 2000.0),
        ("a_specific_m2_m3", 1000.0 * 110.0 / 400.0),
        ("a_L_m2_m3", 1000.0 * 110.0 / 1600.0),
    ],
)
def test_audited_metrics_are_ratios_of_summed_contributions(metric, expected):
    assert metrics.condition_metric_estimand(_frames(), metric) == pytest.approx(expected)


def test_unregistered_count_metric_is_the_typical_valid_frame():
    assert metrics.condition_metric_estimand(_frames(), "n_bubbles_total") == pytest.approx(5.5)


def test_valid_zero_detection_frame_has_metric_specific_physical_effect():
    frames = pd.DataFrame(
        {
            "diameter_sum_mm": [20.0, 0.0],
            "diameter_count": [10.0, 0.0],
            "V_total_mm3": [100.0, 0.0],
            "A_total_mm2": [50.0, 0.0],
            "V_obs_mm3": [1000.0, 1000.0],
            "V_liquid_mm3": [900.0, 1000.0],
            "V_total_mid_mm3": [100.0, 0.0],
            "A_total_mid_mm2": [50.0, 0.0],
        }
    )

    # It enters observed/liquid-volume denominators, but creates no bubbles, gas or area.
    assert metrics.condition_metric_estimand(frames, "epsilon_obs") == pytest.approx(0.05)
    assert metrics.condition_metric_estimand(frames, "a_obs_m2_m3") == pytest.approx(25.0)
    assert metrics.condition_metric_estimand(frames, "a_L_m2_m3") == pytest.approx(
        1000.0 * 50.0 / 1900.0
    )

    # Bubble-population ratios are unchanged because the zero frame adds neither member.
    assert metrics.condition_metric_estimand(frames, "mean_diameter_mm") == pytest.approx(2.0)
    assert metrics.condition_metric_estimand(frames, "a_specific_m2_m3") == pytest.approx(500.0)


def test_grid_aggregation_defaults_to_the_condition_estimand():
    out, metric_col = metrics.aggregate_frames_for_grid(_frames(), "a_specific_m2_m3")
    assert metric_col == "a_specific_m2_m3_condition"
    assert len(out) == 1
    assert out.loc[0, metric_col] == pytest.approx(1000.0 * 110.0 / 400.0)


def test_grid_frame_mean_remains_explicitly_available_for_exploration():
    out, metric_col = metrics.aggregate_frames_for_grid(
        _frames(), "a_specific_m2_m3", reducer="mean"
    )
    assert metric_col == "a_specific_m2_m3_mean"
    assert out.loc[0, metric_col] == pytest.approx(350.0)


def test_ratio_jackknife_matches_mean_standard_error_for_constant_denominator():
    frames = _frames()
    expected_iid = np.std([0.1, 0.3], ddof=1) / np.sqrt(2.0)
    assert metrics.condition_metric_standard_error(frames, "epsilon_obs") == pytest.approx(
        expected_iid
    )
    assert metrics.condition_metric_standard_error(
        frames, "epsilon_obs", n_eff=1.0
    ) == pytest.approx(expected_iid * np.sqrt(2.0))


def test_registered_estimand_fails_if_contribution_columns_are_missing():
    with pytest.raises(KeyError, match="contribution columns"):
        metrics.condition_metric_estimand(
            pd.DataFrame({"mean_diameter_mm": [1.0, 2.0]}), "mean_diameter_mm"
        )
