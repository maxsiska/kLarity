"""
Layout tests for the metric-grid heat maps.

``plot_metric_grid_from_agg_all_aeration`` exists to show the full five-point aeration
series (45/55/70/80/90 L min^-1) instead of the four levels the reference figures use.
Its contract is that the extra row costs nothing in page space: the figure footprint must
stay exactly what ``plot_metric_grid_from_agg`` produces for the same tile grid, so the
two figure families can sit side by side in the same document.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from klarity.viz import (
    plot_metric_grid_from_agg,
    plot_metric_grid_from_agg_all_aeration,
)

PLACEMENTS = [f"placement_{i}" for i in range(1, 7)]
XANTHAN_LEVELS = ["000 xanthan", "0125 xanthan", "025 xanthan"]
RPM_LEVELS = [75.0, 100.0, 125.0, 150.0]
AER_LEVELS_4 = [45.0, 55.0, 70.0, 90.0]
AER_LEVELS_5 = [45.0, 55.0, 70.0, 80.0, 90.0]


@pytest.fixture
def agg() -> pd.DataFrame:
    """Full synthetic (placement, xanthan, rpm, aeration) grid with a deterministic metric."""
    rows = []
    for placement in PLACEMENTS:
        for xanthan in XANTHAN_LEVELS:
            for rpm in RPM_LEVELS:
                for aer in AER_LEVELS_5:
                    rows.append(
                        {
                            "placement": placement,
                            "xanthan": xanthan,
                            "rpm_val": rpm,
                            "aer_val": aer,
                            # Values in mm; the exact scale is irrelevant to layout.
                            "mean_diameter_mm": rpm / 100.0 + aer / 1000.0,
                        }
                    )
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _grid_axes(fig):
    """The tile axes, i.e. everything except the colorbar axes added last."""
    return fig.axes[:-1]


def test_footprint_matches_reference_with_extra_aeration_row(agg):
    """Five aeration levels must not change the figure size the reference function sets."""
    plot_metric_grid_from_agg(
        agg,
        metric_col="mean_diameter_mm",
        placements_keep=PLACEMENTS,
        xanthan_levels=XANTHAN_LEVELS,
        rpm_levels_keep=RPM_LEVELS,
        aer_levels_keep=AER_LEVELS_4,
    )
    reference_size = tuple(plt.gcf().get_size_inches())
    plt.close("all")

    plot_metric_grid_from_agg_all_aeration(
        agg,
        metric_col="mean_diameter_mm",
        placements_keep=PLACEMENTS,
        xanthan_levels=XANTHAN_LEVELS,
        rpm_levels_keep=RPM_LEVELS,
        aer_levels_keep=AER_LEVELS_5,
    )
    assert tuple(plt.gcf().get_size_inches()) == pytest.approx(reference_size)


def test_all_aeration_levels_are_drawn(agg):
    """Every aeration level present in agg gets a row, 80 L/min included."""
    work = plot_metric_grid_from_agg_all_aeration(agg, metric_col="mean_diameter_mm")

    assert sorted(work["aer_val"].unique()) == AER_LEVELS_5
    for ax in _grid_axes(plt.gcf()):
        assert len(ax.get_yticks()) == len(AER_LEVELS_5)
    # 6 placements x 3 xanthan levels, one tile each.
    assert len(_grid_axes(plt.gcf())) == len(PLACEMENTS) * len(XANTHAN_LEVELS)


def test_aeration_filter_still_applies(agg):
    """The default is 'keep everything', but an explicit filter must still win."""
    work = plot_metric_grid_from_agg_all_aeration(
        agg, metric_col="mean_diameter_mm", aer_levels_keep=AER_LEVELS_4
    )
    assert sorted(work["aer_val"].unique()) == AER_LEVELS_4


def test_annotation_font_stays_within_limits(agg):
    """Auto-sized annotations must respect the requested bounds as the grid densifies."""
    limits = (6.0, 8.0)
    plot_metric_grid_from_agg_all_aeration(
        agg,
        metric_col="mean_diameter_mm",
        placements_keep=PLACEMENTS,
        xanthan_levels=XANTHAN_LEVELS,
        annotation_fontsize_limits=limits,
    )
    ax = _grid_axes(plt.gcf())[0]
    # 4 rpm x 5 aeration cells are annotated; the placement label on the first column
    # is a text artist too, so filter to the numeric annotations by cell count.
    sizes = {
        round(t.get_fontsize(), 3) for t in ax.texts if t.get_text().replace(".", "").isdigit()
    }
    assert sizes, "no numeric cell annotations were drawn"
    assert all(limits[0] <= s <= limits[1] for s in sizes)


def test_explicit_annotation_fontsize_is_honored(agg):
    plot_metric_grid_from_agg_all_aeration(
        agg, metric_col="mean_diameter_mm", annotation_fontsize=5.5
    )
    ax = _grid_axes(plt.gcf())[0]
    sizes = {
        round(t.get_fontsize(), 3) for t in ax.texts if t.get_text().replace(".", "").isdigit()
    }
    assert sizes == {5.5}


def test_missing_columns_raise(agg):
    with pytest.raises(KeyError):
        plot_metric_grid_from_agg_all_aeration(
            agg.drop(columns=["aer_val"]), metric_col="mean_diameter_mm"
        )
