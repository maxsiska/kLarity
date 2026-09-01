"""Horizontal spread of settings within a placement in setting comparisons.

Every setting used to be drawn at the same x, so with five aeration setpoints -- each
carrying a marker, a CI and a depth band -- the groups overlapped into an unreadable
stack. ``dodge`` spreads them symmetrically about the placement tick.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from klarity.viz import plot_settings_comparison

PLACEMENTS = ["placement_1", "placement_2", "placement_3"]
SETTINGS = [f"100 rpm {a} lmin 000 xanthan" for a in (45, 55, 70, 80, 90)]
METRIC = "epsilon_obs_mid"


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        [
            {"placement": p, "reactor_setting": s, METRIC: float(v)}
            for s in SETTINGS
            for p in PLACEMENTS
            for v in rng.normal(0.02, 0.002, size=10)
        ]
    )


def _band_for(df, half_width=0.004):
    means = df.groupby(["reactor_setting", "placement"])[METRIC].mean()
    return {(METRIC, s, p): (m - half_width, m + half_width) for (s, p), m in means.items()}


def _plot(df, **kwargs):
    return plot_settings_comparison(
        df,
        settings=SETTINGS,
        metrics=[METRIC],
        y_labels=["eps"],
        placements=PLACEMENTS,
        show=False,
        **kwargs,
    )


def _series_x(ax):
    """x of the marker line of each CI container, in setting order."""
    return [c.lines[0].get_xdata() for c in ax.containers]


def test_settings_are_drawn_at_distinct_x(df):
    ax = _plot(df).axes[0]
    first_placement = np.array([xs[0] for xs in _series_x(ax)])
    assert len(np.unique(first_placement)) == len(SETTINGS)


def test_dodge_zero_restores_the_stacked_layout(df):
    ax = _plot(df, dodge=0).axes[0]
    first_placement = np.array([xs[0] for xs in _series_x(ax)])
    assert np.allclose(first_placement, 0.0)


def test_group_is_symmetric_about_the_placement_tick(df):
    """The tick must keep marking the placement, not drift to one side of its group."""
    ax = _plot(df, dodge=0.5).axes[0]
    for i in range(len(PLACEMENTS)):
        offsets = np.array([xs[i] for xs in _series_x(ax)]) - i
        assert offsets.mean() == pytest.approx(0.0)
        assert offsets.min() == pytest.approx(-0.25)
        assert offsets.max() == pytest.approx(0.25)


def test_ticks_stay_on_the_placements(df):
    ax = _plot(df, dodge=0.5).axes[0]
    assert np.allclose(ax.get_xticks(), np.arange(len(PLACEMENTS)))
    assert [t.get_text() for t in ax.get_xticklabels()] == [
        "Position 1",
        "Position 2",
        "Position 3",
    ]


def test_groups_do_not_collide_with_their_neighbours(df):
    """dodge < 1 must leave clear space between adjacent placements."""
    ax = _plot(df, dodge=0.5).axes[0]
    xs = np.array(_series_x(ax))
    assert xs[:, 0].max() < xs[:, 1].min()


def test_band_stays_on_the_same_x_as_its_own_marker(df):
    """A band drawn at the undodged x would detach from the series it belongs to."""
    ax = _plot(df, dodge=0.5, band_lookup=_band_for(df)).axes[0]
    # containers alternate band, CI per setting
    for i in range(0, len(ax.containers), 2):
        band_segments = ax.containers[i].lines[2][0].get_segments()
        ci_x = ax.containers[i + 1].lines[0].get_xdata()
        band_x = np.array([seg[0, 0] for seg in band_segments])
        assert np.allclose(band_x, ci_x)


def test_single_setting_is_centred(df):
    fig = plot_settings_comparison(
        df,
        settings=SETTINGS[:1],
        metrics=[METRIC],
        y_labels=["eps"],
        placements=PLACEMENTS,
        show=False,
    )
    assert np.allclose(fig.axes[0].containers[0].lines[0].get_xdata(), np.arange(3))
