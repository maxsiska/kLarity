"""Published condition estimands in setting-comparison plots."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from klarity.viz import plot_settings_comparison

SETTING = "100 rpm 55 lmin 000 xanthan"


def _frames() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "placement": ["placement_1", "placement_1"],
            "reactor_setting": [SETTING, SETTING],
            "mean_diameter_mm": [2.0, 3.0],
            "diameter_sum_mm": [20.0, 3.0],
            "diameter_count": [10.0, 1.0],
        }
    )


def _point_y(estimand: str) -> float:
    fig = plot_settings_comparison(
        _frames(),
        settings=[SETTING],
        metrics=["mean_diameter_mm"],
        y_labels=["diameter"],
        placements=["placement_1"],
        estimand=estimand,
        show=False,
    )
    point = float(fig.axes[0].containers[0].lines[0].get_ydata()[0])
    plt.close(fig)
    return point


def test_condition_plot_uses_pooled_bubble_mean():
    assert _point_y("condition") == pytest.approx(23.0 / 11.0)


def test_exploratory_frame_mean_remains_explicitly_available():
    assert _point_y("frame_mean") == pytest.approx(2.5)


def test_unknown_estimand_raises():
    with pytest.raises(ValueError, match="estimand"):
        _point_y("ambiguous")


def test_supplied_effective_sample_size_must_cover_every_condition():
    with pytest.raises(KeyError, match="N_eff is missing"):
        plot_settings_comparison(
            _frames(),
            settings=[SETTING],
            metrics=["mean_diameter_mm"],
            y_labels=["diameter"],
            placements=["placement_1"],
            estimand="condition",
            n_eff_lookup={},
            show=False,
        )
