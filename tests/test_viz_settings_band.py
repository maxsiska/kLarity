"""The systematic depth band drawn alongside the sampling CI in setting comparisons.

``plot_settings_comparison(band_lookup=...)`` overlays a second interval for gas holdup
and specific interfacial area, whose value depends on the unobserved depth axis. The two
intervals mean different things and must stay separable: the CI shrinks as frames
accumulate, the band does not shrink at all.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from klarity.viz import plot_settings_comparison

PLACEMENTS = ["placement_1", "placement_2"]
SETTINGS = ["75 rpm 55 lmin 000 xanthan", "150 rpm 55 lmin 000 xanthan"]


@pytest.fixture
def df():
    """Two settings x two placements x 20 frames, deterministic."""
    rng = np.random.default_rng(0)
    rows = []
    for setting in SETTINGS:
        for placement in PLACEMENTS:
            for value in rng.normal(0.02, 0.002, size=20):
                rows.append(
                    {
                        "placement": placement,
                        "reactor_setting": setting,
                        "epsilon_obs_mid": float(value),
                    }
                )
    return pd.DataFrame(rows)


def _band_for(df, half_width):
    """Band of +-half_width about each condition mean -- centred like the real one."""
    means = df.groupby(["reactor_setting", "placement"])["epsilon_obs_mid"].mean()
    return {
        ("epsilon_obs_mid", setting, placement): (m - half_width, m + half_width)
        for (setting, placement), m in means.items()
    }


def _plot(df, **kwargs):
    return plot_settings_comparison(
        df,
        settings=SETTINGS,
        metrics=["epsilon_obs_mid"],
        y_labels=[r"$\varepsilon$ [-]"],
        placements=PLACEMENTS,
        show=False,
        **kwargs,
    )


def test_no_band_lookup_draws_one_interval_per_setting(df):
    ax = _plot(df).axes[0]
    assert len(ax.containers) == len(SETTINGS)


def test_band_lookup_adds_a_second_interval_per_setting(df):
    ax = _plot(df, band_lookup=_band_for(df, 0.004)).axes[0]
    assert len(ax.containers) == 2 * len(SETTINGS)


def test_band_is_drawn_behind_the_confidence_interval(df):
    """A band in front would hide the CI it is meant to sit behind."""
    ax = _plot(df, band_lookup=_band_for(df, 0.004)).axes[0]
    band, ci = ax.containers[0], ax.containers[1]
    assert band.lines[2][0].get_zorder() < ci.lines[2][0].get_zorder()


def test_band_is_visually_distinct_from_the_confidence_interval(df):
    """Wider and translucent, so the two do not read as one interval."""
    ax = _plot(df, band_lookup=_band_for(df, 0.004)).axes[0]
    band_bars, ci_bars = ax.containers[0].lines[2][0], ax.containers[1].lines[2][0]
    assert band_bars.get_linewidth() > ci_bars.get_linewidth()
    assert band_bars.get_alpha() is not None and band_bars.get_alpha() < 1.0


def test_interval_key_legend_only_appears_with_a_band(df):
    """With both intervals present the figure has to say which is which."""
    without = _plot(df).axes[0]
    assert without.get_legend() is None

    with_band = _plot(df, band_lookup=_band_for(df, 0.004)).axes[0]
    key = with_band.get_legend()
    assert key is not None
    labels = {t.get_text() for t in key.get_texts()}
    assert labels == {"prolate–oblate band", "95% CI"}


def test_metrics_without_a_band_entry_are_skipped(df):
    """Count and diameter have no depth band; they must share the call unharmed."""
    df = df.assign(n_bubbles_total=1.0)
    fig = plot_settings_comparison(
        df,
        settings=SETTINGS,
        metrics=["epsilon_obs_mid", "n_bubbles_total"],
        y_labels=["eps", "n"],
        placements=PLACEMENTS,
        band_lookup=_band_for(df, 0.004),  # covers epsilon only
        show=False,
    )
    assert len(fig.axes[0].containers) == 2 * len(SETTINGS)  # banded
    assert len(fig.axes[1].containers) == len(SETTINGS)  # not banded


def test_partial_band_coverage_draws_only_the_covered_points(df):
    full = _band_for(df, 0.004)
    partial = {k: v for k, v in full.items() if k[2] == "placement_1"}
    ax = _plot(df, band_lookup=partial).axes[0]
    assert len(ax.containers) == 2 * len(SETTINGS)
    # one banded point per setting, not two
    assert len(ax.containers[0].lines[2][0].get_segments()) == 1


def test_band_not_bracketing_the_mean_raises(df):
    """A band centred on a different quantity would be silently misleading."""
    offset = {k: (lo + 1.0, hi + 1.0) for k, (lo, hi) in _band_for(df, 0.004).items()}
    with pytest.raises(ValueError, match="does not bracket the plotted mean"):
        _plot(df, band_lookup=offset)


def test_zero_width_band_is_accepted(df):
    """lower == upper == mean is degenerate but consistent, not an error."""
    ax = _plot(df, band_lookup=_band_for(df, 0.0)).axes[0]
    assert len(ax.containers) == 2 * len(SETTINGS)


def test_band_keys_that_match_nothing_raise(df):
    """A lookup covering the metric but keyed differently would draw no band at all.

    The realistic slip is a setting string built from floats ("45.0 lmin"), which matches
    no plotted point and silently yields the band-free figure the parameter exists to
    prevent.
    """
    mismatched = {
        ("epsilon_obs_mid", setting.replace(" rpm", ".0 rpm"), placement): value
        for (_, setting, placement), value in _band_for(df, 0.004).items()
    }
    with pytest.raises(ValueError, match="none matched the plotted points"):
        _plot(df, band_lookup=mismatched)


def test_metric_absent_from_the_lookup_does_not_raise(df):
    """Count and diameter carry no band; their absence is intended, not a key mismatch."""
    df = df.assign(n_bubbles_total=1.0)
    fig = plot_settings_comparison(
        df,
        settings=SETTINGS,
        metrics=["n_bubbles_total"],
        y_labels=["n"],
        placements=PLACEMENTS,
        band_lookup=_band_for(df, 0.004),  # epsilon only — nothing for this metric
        show=False,
    )
    assert len(fig.axes[0].containers) == len(SETTINGS)
