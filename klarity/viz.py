from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional, Union

import matplotlib
import matplotlib.lines
import matplotlib.pyplot
import matplotlib.ticker
import numpy
import pandas

from klarity import metrics as metric_utils
from klarity.parsing import parse_setting

matplotlib.rcParams["font.serif"] = ["Times New Roman"]
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["mathtext.fontset"] = "custom"
matplotlib.rcParams["mathtext.rm"] = "Times New Roman"
matplotlib.rcParams["mathtext.it"] = "Times New Roman:italic"
matplotlib.rcParams["mathtext.bf"] = "Times New Roman:bold"

A4_TEXT_WIDTH_IN = 6.27  # (21.0 - 2*2.54) / 2.54
A4_TEXT_HEIGHT_IN = 9.69  # (29.7 - 2*2.54) / 2.54
MAX_FIG_HEIGHT_IN = 8.5  # leave ~1.2 in for caption


class Colors:
    light_red = numpy.array((223, 83, 62)) / 255
    light_blue = numpy.array((69, 145, 247)) / 255
    dark_red = numpy.array((122, 25, 24)) / 255
    dark_blue = numpy.array((0, 0, 255)) / 255
    alt_blue = numpy.array((59, 117, 175)) / 255
    light_gray = numpy.array((179, 179, 179)) / 255
    lavender = "#E6E6FA"
    magnolia = "#F4F0F7"
    gray = "#4D4D4D"
    current = "#00635D"
    eminence = "#713685"
    pink = "#E97DC3"
    green = "#08A238"
    taupe = "#8F564D"
    orange = "#FF8430"
    blue = "#3083DC"
    black = "k"
    floral = "#AE6EC4"
    red = "red"
    purple = "purple"


custom_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "metric-blend", ["#FFEBFE", "#52004E"]
)


matplotlib.rcParams["text.color"] = Colors.gray
matplotlib.rcParams["axes.labelcolor"] = Colors.gray
matplotlib.rcParams["axes.titlecolor"] = Colors.gray
matplotlib.rcParams["xtick.color"] = Colors.black
matplotlib.rcParams["ytick.color"] = Colors.black

color_cycle = [
    Colors.green,
    Colors.orange,
    Colors.blue,
    Colors.pink,
    Colors.eminence,
    Colors.current,
    Colors.taupe,
]

placement_level_x = -0.45

# Tolerance when checking that a supplied systematic band brackets the plotted mean,
# as a fraction of the largest mean on the series. Relative rather than absolute so the
# same check works for gas holdup (order 1e-3) and interfacial area (order 1e2).
# See plot_settings_comparison(band_lookup=...).
_BAND_REL_TOL = 1e-6


def nice_tick_step(span: float, max_ticks: int = 5) -> float:
    """
    Smallest "nice" major-tick spacing that yields at most `max_ticks` ticks over `span`.

    Candidate steps are 1, 2, 2.5 or 5 times a power of ten, so tick labels stay short.
    Ticks sit on multiples of the step and the axis limits are arbitrary, so the worst
    case is ``floor(span / step) + 1`` ticks inside the view; the smallest candidate
    satisfying that bound is returned.

    span: axis range in data units (same units as the axis, e.g. mm)
    max_ticks: upper bound on the number of major ticks (>= 2)
    """
    if not numpy.isfinite(span) or span <= 0:
        raise ValueError(f"span must be finite and positive, got {span!r}")
    if max_ticks < 2:
        raise ValueError(f"max_ticks must be >= 2, got {max_ticks!r}")

    # Any admissible step satisfies step > span / max_ticks, so starting one decade
    # below that scale cannot skip the answer.
    decade = 10.0 ** numpy.floor(numpy.log10(span / max_ticks))
    for scale in (decade / 10.0, decade, decade * 10.0):
        for multiple in (1.0, 2.0, 2.5, 5.0):
            step = float(multiple * scale)
            # +1e-9 keeps the count conservative when span / step is an integer in
            # exact arithmetic but lands just below it in floating point.
            if numpy.floor(span / step + 1e-9) + 1 <= max_ticks:
                return step
    return float(10.0 * decade)


METRIC_SPECS = {
    "mean_diameter_mm": dict(
        title="Pooled mean bubble diameter",
        cbar="Pooled mean bubble diameter [mm]",
        estimand=metric_utils.condition_metric_population("mean_diameter_mm"),
        robust=False,
        vmin=None,
        vmax=None,
        annotation_decimals=2,
    ),
    # Gas holdup (volume) and interfacial-area density (surface) depend on the unobserved
    # prolate/oblate depth axis, so they are reported as the band MIDPOINT plus a companion
    # band-width panel (± relative half-width, %). See metrics.spheroid_band_arrays.
    "epsilon_obs_mid": dict(
        title="Local gas holdup (prolate–oblate midpoint)",
        cbar=r"Local gas holdup $\varepsilon$ [-]",
        estimand=metric_utils.condition_metric_population("epsilon_obs_mid"),
        robust=True,
        vmin=None,
        vmax=None,
        annotation_decimals=3,
    ),
    "epsilon_obs_band_pct": dict(
        title="Gas holdup: prolate–oblate band width",
        cbar=r"band half-width [$\pm$%]",
        estimand=metric_utils.condition_metric_population("epsilon_obs_band_pct"),
        robust=True,
        vmin=None,
        vmax=None,
        annotation_decimals=1,
    ),
    "a_obs_m2_m3_mid": dict(
        title="Local interfacial area density (prolate–oblate midpoint)",
        cbar=r"Interfacial area [m$^{2}$ m$^{-3}$]",
        estimand=metric_utils.condition_metric_population("a_obs_m2_m3_mid"),
        robust=True,
        vmin=None,
        vmax=None,
        annotation_decimals=1,
    ),
    "a_obs_m2_m3_band_pct": dict(
        title="Interfacial area: prolate–oblate band width",
        cbar=r"band half-width [$\pm$%]",
        estimand=metric_utils.condition_metric_population("a_obs_m2_m3_band_pct"),
        robust=True,
        vmin=None,
        vmax=None,
        annotation_decimals=1,
    ),
    # The sphere-fraction panels (frac_sphere_count/_volume/_surface) are deliberately NOT
    # listed because reported quantities are classification-independent. The columns are
    # still computed per frame, and `aspect_ratio` + `d_mm_sphere`
    # are kept per bubble, so the split can be reconstructed at any cutoff on demand.
    "n_bubbles_per_mL": dict(
        title="Bubble number density",
        cbar="bubbles/mL",
        estimand=metric_utils.condition_metric_population("n_bubbles_per_mL"),
        robust=True,
        vmin=None,
        vmax=None,
        annotation_decimals=2,
    ),
    "a_specific_m2_m3": dict(
        title="Gas-volume-specific interfacial area",
        cbar=r"A / V$_\mathrm{gas}$ [m$^{2}$ m$^{-3}$]",
        estimand=metric_utils.condition_metric_population("a_specific_m2_m3"),
        robust=True,
        vmin=None,
        vmax=None,
        annotate=False,
    ),
    "a_L_m2_m3": dict(
        title="Liquid-volume interfacial area density",
        cbar=r"Liquid-volume interfacial area density [m$^{2}$ m$^{-3}$]",
        estimand=metric_utils.condition_metric_population("a_L_m2_m3"),
        robust=True,
        vmin=None,
        vmax=None,
        annotation_decimals=1,
    ),
}


def plot_replicate_distribution(
    sub: pandas.DataFrame,
    output_path: Optional[Path] = None,
    value_col: str = "equivalent_diameter_mm",
    bins: int = 80,
    frequency: bool = False,
    dpi: int = 1500,
    title_prefix: str = "Replicate",
    ax: Optional[matplotlib.pyplot.Axes] = None,
    color: Optional[str] = None,
    annotate: bool = True,
    xlim: Optional[tuple[float, float]] = None,
    title: Optional[str] = None,
    legend_label: Optional[str] = None,
    annotation_fields: tuple = ("mean", "avg_bpf"),  # <<< choose which stats to show
):
    """
    Draw a histogram with GIF-style aesthetics. If `ax` is given, draw into that axis.
    Returns a stats dict or None if no data.
    """
    if sub.empty:
        return None

    flat = sub.reset_index()
    if value_col not in flat.columns:
        return None

    x = flat[value_col].dropna().to_numpy()
    if x.size == 0:
        return None

    # compute stats
    xmin_local, xmax_local = float(numpy.min(x)), float(numpy.max(x))
    xmin, xmax = xlim if xlim is not None else (xmin_local, xmax_local)
    mean_val = float(numpy.mean(x))
    total_bubbles = int(len(x))

    # (we still compute these in case you want them later; we just don't display them now)
    if {"burst_index", "image_number_in_burst"}.issubset(flat.columns):
        n_frames = flat[["burst_index", "image_number_in_burst"]].drop_duplicates().shape[0]
    else:
        n_frames = (
            flat["image_number_in_burst"].nunique()
            if "image_number_in_burst" in flat.columns
            else 0
        )
    avg_bpf = (total_bubbles / n_frames) if n_frames else 0.0

    created_fig = False
    if ax is None:
        fig, ax = matplotlib.pyplot.subplots(figsize=(6, 4))
        created_fig = True

    ax.set_xlim(xmin, xmax)

    if frequency:
        counts, bin_edges = numpy.histogram(x, bins=bins, range=(xmin, xmax))
        freqs = counts / counts.sum()
        bin_width = bin_edges[1] - bin_edges[0]
        ax.bar(
            bin_edges[:-1],
            freqs,
            width=bin_width,
            align="edge",
            edgecolor="none",
            alpha=0.85,
            color=color,
            label=legend_label,
        )
    else:
        ax.hist(
            x,
            bins=bins,
            range=(xmin, xmax),
            edgecolor="none",
            alpha=0.85,
            color=color,
            label=legend_label,
        )

    if legend_label:
        ax.legend(
            loc="upper right",
        )

    # Titles & labels
    if title is not None:
        ax.set_title(title)
    elif created_fig:
        ax.set_title(f"{title_prefix} — Total bubbles: {total_bubbles}")

    if created_fig:
        ax.set_xlabel(value_col.replace("_", " ").title())
        ax.set_ylabel("Frequency [-]" if frequency else "Count [-]")

    # Flexible annotation content
    if annotate:
        parts = []
        if "mean" in annotation_fields:
            parts.append(rf"$\overline{{d}}$: {mean_val:.2f} mm")
        if "total" in annotation_fields:
            parts.append(f"Total Bubbles: {total_bubbles}")
        if "avg_bpf" in annotation_fields:
            parts.append(r"$n_{\mathrm{bubbles}}$: " + f"{avg_bpf:.1f}")
        if "frames" in annotation_fields:
            parts.append(f"Frames: {n_frames}")
        if parts:
            ax.text(
                0.98,
                0.98,
                "\n".join(parts),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7,
                zorder=5,
            )

    if created_fig and (output_path is not None):
        matplotlib.pyplot.tight_layout()
        matplotlib.pyplot.savefig(output_path, dpi=dpi)
        matplotlib.pyplot.close(fig)

    return {
        "xmin": xmin,
        "xmax": xmax,
        "mean": mean_val,
        "total_bubbles": total_bubbles,
        "frames": n_frames,
        "avg_bpf": avg_bpf,
    }


def plot_all_xanthan_grids(
    bubble_level_df: pandas.DataFrame,
    placements: list[str],
    value_col: str = "equivalent_diameter_mm",
    bins: int = 80,
    xmax_percentile: float = 99.5,
    outdir: Union[str, Path] = "visc_comparison",
    fname_prefix: str = "visc_compare_settings",
    color_map: Optional[dict] = None,
    x_max_ticks: int = 5,  # cap on x-axis major tick labels per subplot
    x_tick_step: Optional[float] = None,  # fixed x tick spacing [mm]; overrides x_max_ticks
):
    """
    For each (rpm, aeration) combination implicit in the 'setting' level of the MultiIndex,
    create a grid plot comparing xanthan levels for the given placements.

    We look for xanthan levels in the canonical order:
        ["000 xanthan", "0125 xanthan", "025 xanthan"]
    and only include those that actually exist for that (rpm, aeration).
    """

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Figure out which index level corresponds to the "setting" strings
    # ------------------------------------------------------------------
    index_names = list(bubble_level_df.index.names)

    # Try preferred name first
    try:
        setting_level = index_names.index("setting")
    except ValueError:
        # Fallback: assume 'placement' is the first level and
        # the "setting" is the next one (like in grid_xanthan_by_placement)
        if "placement" in index_names:
            placement_level = index_names.index("placement")
            # pick the first level that is not 'placement'
            candidates = [i for i, name in enumerate(index_names) if i != placement_level]
            if not candidates:
                raise RuntimeError(f"Could not infer setting level from index names: {index_names}")
            setting_level = candidates[0]
        else:
            # last-resort fallback: assume level 1 is the setting
            if len(index_names) < 2:
                raise RuntimeError(
                    f"MultiIndex has too few levels to infer settings: {index_names}"
                )
            setting_level = 1

    # Collect all unique setting strings from the chosen level
    setting_index = bubble_level_df.index.get_level_values(setting_level).unique()

    # ------------------------------------------------------------------
    # Group settings by (rpm, aeration) using parse_setting
    # ------------------------------------------------------------------
    groups = defaultdict(list)  # (rpm, aeration) -> list of (xanthan, setting_string)
    for setting in setting_index:
        rpm, aeration, xanthan = parse_setting(str(setting))
        groups[(rpm, aeration)].append((xanthan, setting))

    desired_xan_order = ["000 xanthan", "0125 xanthan", "025 xanthan"]

    # ------------------------------------------------------------------
    # For each (rpm, aeration) group, call grid_xanthan_by_placement
    # ------------------------------------------------------------------
    for (rpm, aeration), xan_list in groups.items():
        # map xanthan string -> full setting string
        xan_to_setting = {xan: s for xan, s in xan_list}

        # keep only desired xanthan levels, in desired order, if present
        settings_ordered = [
            xan_to_setting[xan] for xan in desired_xan_order if xan in xan_to_setting
        ]

        # skip if only one desired xanthan level is available (no real comparison)
        if len(settings_ordered) < 2:
            print(
                f"[skip] {rpm} · {aeration}: only {len(settings_ordered)} of "
                f"desired xanthan levels present."
            )
            continue

        title = f"{rpm} · {aeration}"
        prefix = f"{fname_prefix}_" f"{rpm.replace(' ', '_')}_" f"{aeration.replace(' ', '_')}"

        print(f"[plot] {title} with settings: {settings_ordered}")

        for frequency, suffix in [(False, "count"), (True, "frequency")]:
            grid_xanthan_by_placement(
                bubble_level_df=bubble_level_df,
                placements=placements,
                settings=settings_ordered,
                settings_title=title,
                value_col=value_col,
                bins=bins,
                fname_prefix=f"{prefix}_{suffix}",
                outdir=outdir,
                color_map=color_map,
                xmax_percentile=xmax_percentile,
                frequency=frequency,
                x_max_ticks=x_max_ticks,
                x_tick_step=x_tick_step,
            )


def grid_xanthan_by_placement(
    bubble_level_df: pandas.DataFrame,
    rpm: Optional[str] = None,  # used if settings=None
    aeration: Optional[str] = None,  # used if settings=None
    placements: list = (),
    xanthan_values: Optional[list] = None,  # used if settings=None; any length (1,2,3,...)
    replicates: Optional[list] = None,  # pool these; None = all available
    value_col: str = "equivalent_diameter_mm",
    bins: int = 80,
    outdir: Union[str, Path] = "visc_comparison",
    fname_prefix: Optional[str] = None,
    reverse_rows: bool = True,
    color_map: Optional[dict] = None,  # keys = viscosity labels ("000 xanthan", ...)
    settings: Optional[list] = None,  # explicit list of settings (MultiIndex level 'setting')
    settings_title: Optional[str] = None,  # optional suptitle override when using `settings`
    xmax_percentile: Optional[float] = 99.5,
    frequency: bool = False,
    x_max_ticks: int = 5,  # cap on x-axis major tick labels per subplot
    x_tick_step: Optional[float] = None,  # fixed x tick spacing [mm]; overrides x_max_ticks
):
    """
    Behavior:
      - Histograms have distinct colors per column (viscosity label).
      - Mean dashed line is ALWAYS red.
      - Xanthan labels are derived from the setting.
      - Binning respects xlim_cap via xlim_eff.
      - Column headers shown as wt%.
      - No 'Replicate' titles stamped on every subplot.
      - X tick spacing is explicit when `x_tick_step` is supplied; otherwise an adaptive
        step derived from the shared x-range limits label density (see `x_max_ticks`).
    """
    import re

    idx = pandas.IndexSlice
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    def format_position_label(placement: str) -> str:
        m = re.search(r"(\d+)$", str(placement))
        if m:
            return f"Position {m.group(1)}"
        return f"Position {placement}"

    def extract_xanthan_label_from_setting(s: str) -> str:
        parts = str(s).split()
        if "xanthan" in parts:
            ix = parts.index("xanthan")
            if ix >= 1:
                return f"{parts[ix-1]} xanthan"
            return "xanthan"
        return str(s)

    _XANTHAN_WT = {
        "000": "0.000",
        "0125": "0.125",
        "025": "0.250",
    }

    def format_xanthan_wt(label: str) -> str:
        s = str(label)
        if "xanthan" not in s:
            return s
        token = s.split()[0]
        wt = _XANTHAN_WT.get(token, token)
        return f"{wt} wt% xanthan"

    # ------------------------------------------------------------------
    # Decide columns (settings vs rpm/aeration/xanthan)
    # ------------------------------------------------------------------
    if settings is not None:
        col_keys = list(settings)
        col_labels = [extract_xanthan_label_from_setting(s) for s in col_keys]
        col_titles = [format_xanthan_wt(lbl) for lbl in col_labels]

        if color_map is None:
            # stable unique order in appearance
            seen = []
            for lbl in col_labels:
                if lbl not in seen:
                    seen.append(lbl)
            color_map = {lbl: color_cycle[i % len(color_cycle)] for i, lbl in enumerate(seen)}

        fname_mid = "_".join([str(lbl).replace(" ", "_") for lbl in col_labels])

    else:
        assert (
            rpm is not None
            and aeration is not None
            and xanthan_values is not None
            and len(xanthan_values) >= 1
        ), "Provide either `settings` OR (rpm, aeration, xanthan_values)."

        col_keys = [f"{rpm} {aeration} {x}" for x in xanthan_values]
        col_labels = list(xanthan_values)
        col_titles = [format_xanthan_wt(x) for x in xanthan_values]

        if color_map is None:
            color_map = {lbl: color_cycle[i % len(color_cycle)] for i, lbl in enumerate(col_labels)}

        fname_mid = f"{rpm}_{aeration}".replace(" ", "_")

    if reverse_rows:
        placements = list(reversed(placements))

    # ------------------------------------------------------------------
    # Pool data & collect global x-range
    # ------------------------------------------------------------------
    pooled: dict[tuple, pandas.DataFrame] = {}
    all_vals = []

    for placement in placements:
        for setting_key, label in zip(col_keys, col_labels):
            if replicates is None:
                try:
                    reps_avail = (
                        bubble_level_df.loc[idx[placement, setting_key, :, :, :]]
                        .index.get_level_values("replicate")
                        .unique()
                        .tolist()
                    )
                except KeyError:
                    reps_avail = []
            else:
                reps_avail = replicates

            dfs = []
            for rep in reps_avail:
                try:
                    df_sub = bubble_level_df.loc[idx[placement, setting_key, rep, :, :]]
                except KeyError:
                    continue
                if not df_sub.empty and (value_col in df_sub.columns):
                    dfs.append(df_sub)

            pooled_df = (
                pandas.concat(dfs) if dfs else pandas.DataFrame(columns=bubble_level_df.columns)
            )
            pooled[(placement, label)] = pooled_df

            if not pooled_df.empty:
                all_vals.append(pooled_df[value_col].dropna().to_numpy())

    if not all_vals:
        print("[skip] No data matching the requested placements/settings.")
        return None

    x_all = numpy.concatenate(all_vals)
    xlim_eff = (
        float(numpy.min(x_all)),
        float(numpy.percentile(x_all, xmax_percentile if xmax_percentile is not None else 99.5)),
    )

    # ------------------------------------------------------------------
    # Create subplot grid
    # ------------------------------------------------------------------
    nrows, ncols = len(placements), len(col_labels)

    cell_h = (A4_TEXT_WIDTH_IN / ncols) * 0.8
    fig_height = cell_h * nrows

    fig, axes = matplotlib.pyplot.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(A4_TEXT_WIDTH_IN, fig_height),
        sharex=True,
        sharey="row",
        squeeze=False,
    )

    # Column headers
    for c, title_ in enumerate(col_titles):
        axes[0, c].set_title(title_)

    # Row labels on the Left side
    for r, placement in enumerate(placements):
        ax = axes[r, 0]
        ax.text(
            placement_level_x,
            0.5,
            format_position_label(placement),
            transform=ax.transAxes,
            rotation=90,
            va="center",
            ha="left",
            color=Colors.gray,
        )

    # ------------------------------------------------------------------
    # Fill grid
    # ------------------------------------------------------------------
    for r, placement in enumerate(placements):
        for c, label in enumerate(col_labels):
            ax = axes[r, c]
            sub = pooled[(placement, label)]
            col_color = color_map.get(label, None)

            if sub.empty:
                ax.set_xlim(*xlim_eff)
                if c == 0:
                    ax.set_ylabel("Count")
                else:
                    ax.tick_params(labelleft=False)
                if r == nrows - 1:
                    ax.set_xlabel("Bubble Diameter [mm]")
                continue

            stats = plot_replicate_distribution(
                sub=sub,
                output_path=None,
                value_col=value_col,
                bins=bins,
                ax=ax,
                color=col_color,  # <<< per-column histogram color
                frequency=frequency,
                annotate=True,
                xlim=xlim_eff,
                title_prefix="",  # <<< no "Replicate" titles
                title=None,
                legend_label=None,
            )

            ax.set_xlim(*xlim_eff)

            # Mean dashed line: ALWAYS red
            if isinstance(stats, dict) and "mean" in stats and numpy.isfinite(stats["mean"]):
                ax.axvline(
                    float(stats["mean"]),
                    linestyle="--",
                    linewidth=1.0,
                    color=Colors.red,
                    zorder=4,
                )

            y_label = "Frequency" if frequency else "Count"

            if c == 0:
                ax.set_ylabel(y_label)
            else:
                ax.tick_params(labelleft=False)

            if r == nrows - 1:
                ax.set_xlabel("Bubble Diameter [mm]")

    # X ticks: pick a step from the shared x-range so labels do not overlap in narrow
    # subplots while limiting label density on wide distributions.
    x_step = (
        float(x_tick_step)
        if x_tick_step is not None
        else nice_tick_step(xlim_eff[1] - xlim_eff[0], max_ticks=x_max_ticks)
    )

    for ax_row in axes:
        for ax in ax_row:
            ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(x_step))
            ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(x_step / 2))
            if frequency:
                ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
                ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.05))

    matplotlib.pyplot.tight_layout(rect=(0.05, 0, 1, 0.97))

    stem = ((f"{fname_prefix}_" if fname_prefix else "") + f"{fname_mid}_grid").replace(" ", "_")
    out_paths = []
    for ext, dpi in [("svg", 150), ("pdf", 150)]:
        out_path = Path(outdir) / f"{stem}.{ext}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print("Wrote", out_path)
        out_paths.append(out_path)
    matplotlib.pyplot.close(fig)
    return out_paths


# ==============================
# Plot overviews
# ==============================


def plot_metric_grid_from_agg(
    agg: pandas.DataFrame,
    *,
    metric_col: str,
    placement_col: str = "placement",
    xanthan_col: str = "xanthan",
    rpm_val_col: str = "rpm_val",
    aer_val_col: str = "aer_val",
    # ordering (soft)
    placements: Optional[list] = None,
    xanthan_order: Optional[list] = None,
    # filtering (hard)
    placements_keep: Optional[list] = None,
    xanthan_levels: Optional[list] = None,
    rpm_levels_keep: Optional[list] = None,
    aer_levels_keep: Optional[list] = None,
    # display maps
    placement_label_map: Optional[dict] = None,
    xanthan_label_map: Optional[dict] = None,
    axis_label_map: Optional[dict] = None,
    title: Optional[str] = None,
    colorbar_label: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    robust: bool = False,
    missing_text: str = "no data",
    # Cell annotations
    annotate_cells: bool = True,
    annotation_decimals: int = 1,
    auto_contrast: bool = True,
    # Viscosity axis
    viscosity_map: Optional[dict] = None,  # {xanthan_level: {rpm: viscosity}}
    viscosity_label: str = "µ (Pa·s)",
    viscosity_decimals: int = 3,
    viscosity_label_pad: float = 10,  # Distance between xanthan title and viscosity label
    # PV map: replace rpm tick labels with P/V values per xanthan level
    pv_map: Optional[dict] = None,  # {xanthan_level: {rpm: pv_value}}
    # existing
    dpi: int = 300,
    outpath: Optional[Union[str, Path]] = None,
):
    """
    Metric-agnostic grid plot:
      rows = placements
      cols = xanthan levels
      inside each tile: rpm (x) vs aeration (y) heatmap of metric_col

    Requires agg to contain:
      placement_col, xanthan_col, rpm_val_col, aer_val_col, metric_col

    Viscosity mapping:
      viscosity_map: Dict mapping xanthan levels to {rpm: viscosity} dicts
                     Example: {
                         '000 xanthan': {75: 0.001, 100: 0.001, 125: 0.001, 150: 0.001},
                         '0125 xanthan': {75: 0.045, 100: 0.032, 125: 0.025, 150: 0.020},
                         '025 xanthan': {75: 0.120, 100: 0.085, 125: 0.065, 150: 0.050}
                     }
    """
    req = [placement_col, xanthan_col, rpm_val_col, aer_val_col, metric_col]
    missing = [c for c in req if c not in agg.columns]
    if missing:
        raise KeyError(f"agg missing required columns: {missing}")

    work = agg.dropna(subset=req).copy()
    if work.empty:
        print("[skip] empty after dropping NaNs")
        return None

    # --- optional filtering (hard) ---
    if placements_keep is not None:
        work = work[work[placement_col].isin(placements_keep)]
    if xanthan_levels is not None:
        work = work[work[xanthan_col].isin(xanthan_levels)]
    if rpm_levels_keep is not None:
        work = work[work[rpm_val_col].isin(rpm_levels_keep)]
    if aer_levels_keep is not None:
        work = work[work[aer_val_col].isin(aer_levels_keep)]

    if work.empty:
        print("[skip] empty after applying level filters")
        return None

    def _disp(mapper, key):
        return mapper.get(key, str(key)) if mapper else str(key)

    # placements order (reverse numeric-ish)
    all_p = list(work[placement_col].unique())

    if placements_keep is not None:
        placements_used = [p for p in placements_keep if p in all_p]
    elif placements is not None:
        primary = [p for p in placements if p in all_p]
        rest = [p for p in all_p if p not in primary]
        placements_used = primary + rest
    else:
        placements_used = all_p

    def _placement_key(p):
        import re

        m = re.search(r"(\d+)$", str(p))
        return int(m.group(1)) if m else 0

    placements_used = list(reversed(sorted(placements_used, key=_placement_key)))

    # xanthan order
    all_x = sorted(work[xanthan_col].unique())
    if xanthan_order is not None:
        primary = [x for x in xanthan_order if x in all_x]
        rest = [x for x in all_x if x not in primary]
        xan_used = primary + rest
    else:
        xan_used = all_x

    # axis levels
    rpm_levels = (
        sorted(work[rpm_val_col].unique()) if rpm_levels_keep is None else list(rpm_levels_keep)
    )
    aer_levels = (
        sorted(work[aer_val_col].unique()) if aer_levels_keep is None else list(aer_levels_keep)
    )

    # global color scale
    vals = pandas.to_numeric(work[metric_col], errors="coerce").dropna()
    if vals.empty:
        print("[skip] metric_col has no numeric values")
        return None

    if vmin is None or vmax is None:
        if robust:
            lo, hi = numpy.percentile(vals.to_numpy(), [5, 95])
        else:
            lo, hi = float(vals.min()), float(vals.max())
        if vmin is None:
            vmin = float(lo)
        if vmax is None:
            vmax = float(hi)
        if numpy.isclose(vmin, vmax):
            vmin -= 1e-6
            vmax += 1e-6

    # Helper function for auto-contrast
    def _get_text_color(value, vmin, vmax, cmap):
        """Choose white or black text based on background luminance"""
        if not auto_contrast:
            return "white"
        if numpy.isnan(value):
            return "black"
        norm_val = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.5
        norm_val = numpy.clip(norm_val, 0, 1)
        rgba = cmap(norm_val)
        r, g, b = rgba[:3]
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return "white" if luminance < 0.5 else "black"

    nrows, ncols = len(placements_used), len(xan_used)
    cell_height = (A4_TEXT_WIDTH_IN / ncols) * 0.68
    fig_height = cell_height * nrows + 1.3

    fig, axes = matplotlib.pyplot.subplots(
        nrows=nrows, ncols=ncols, figsize=(A4_TEXT_WIDTH_IN, fig_height), sharex=False, sharey=True
    )
    if nrows == 1 and ncols == 1:
        axes = numpy.array([[axes]])
    elif nrows == 1:
        axes = numpy.array([axes])
    elif ncols == 1:
        axes = numpy.array([[ax] for ax in axes])

    xlab = axis_label_map.get("x", "rpm") if axis_label_map else "rpm"
    ylab = axis_label_map.get("y", "l/min") if axis_label_map else "l/min"

    last_im = None
    last_cmap = None

    for i, p in enumerate(placements_used):
        for j, x in enumerate(xan_used):
            ax = axes[i, j]
            sub = work[(work[placement_col] == p) & (work[xanthan_col] == x)]

            ax.set_xticks(numpy.arange(len(rpm_levels)))
            ax.set_yticks(numpy.arange(len(aer_levels)))

            if sub.empty:
                ax.set_facecolor("#f0f0f0")
                ax.text(
                    0.5,
                    0.5,
                    missing_text,
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color=Colors.gray,
                )
            else:
                mat = numpy.full((len(aer_levels), len(rpm_levels)), numpy.nan, dtype="float32")
                for _, row in sub.iterrows():
                    rv = row[rpm_val_col]
                    av = row[aer_val_col]
                    if rv not in rpm_levels or av not in aer_levels:
                        continue
                    r = rpm_levels.index(rv)
                    a = aer_levels.index(av)
                    mat[a, r] = float(row[metric_col])

                im = ax.imshow(mat, origin="lower", vmin=vmin, vmax=vmax, aspect="auto")
                last_im = im
                last_cmap = im.get_cmap()

                # Add text annotations
                if annotate_cells:
                    for a_idx in range(len(aer_levels)):
                        for r_idx in range(len(rpm_levels)):
                            val = mat[a_idx, r_idx]
                            if not numpy.isnan(val):
                                text_color = _get_text_color(val, vmin, vmax, last_cmap)

                                if annotation_decimals == 0:
                                    text = f"{val:.0f}"
                                else:
                                    text = f"{val:.{annotation_decimals}f}"

                                ax.text(
                                    r_idx,
                                    a_idx,
                                    text,
                                    ha="center",
                                    va="center",
                                    color=text_color,
                                    fontsize=8,
                                )

            # titles/labels
            if i == 0:
                ax.set_title(_disp(xanthan_label_map, x), pad=11, fontsize=9)

                # Add viscosity axis on top row
                if viscosity_map is not None and x in viscosity_map:
                    ax2 = ax.twiny()  # Create secondary x-axis
                    ax2.set_xlim(ax.get_xlim())  # Match primary axis limits
                    ax2.set_xticks(numpy.arange(len(rpm_levels)))
                    ax2.tick_params(labelsize=9)

                    # Create viscosity labels
                    visc_labels = []
                    for rpm in rpm_levels:
                        if rpm in viscosity_map[x]:
                            visc = viscosity_map[x][rpm]
                            if viscosity_decimals == 0:
                                visc_labels.append(f"{visc:.0f}")
                            else:
                                visc_labels.append(f"{visc:.{viscosity_decimals}f}")
                        else:
                            visc_labels.append("")

                    ax2.set_xticklabels(visc_labels, fontsize=9)

                    # Add viscosity label on ALL columns (not just first)
                    ax2.set_xlabel(viscosity_label, labelpad=viscosity_label_pad, fontsize=9)

            if j == 0:
                ax.text(
                    placement_level_x,
                    0.5,
                    _disp(placement_label_map, p),
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=9,
                )
                ax.set_ylabel(ylab)

                ax.set_yticks(numpy.arange(len(aer_levels)))
                ax.set_yticklabels(
                    [str(int(v)) if float(v).is_integer() else str(v) for v in aer_levels],
                    fontsize=9,
                )
                ax.tick_params(axis="y", labelleft=True, labelsize=9)
            else:
                ax.tick_params(axis="y", labelleft=False, labelsize=9)

            if i == nrows - 1:
                if pv_map is not None and x in pv_map:
                    xtick_labels = [
                        f"{pv_map[x][v]:.3f}" if v in pv_map[x] else str(v) for v in rpm_levels
                    ]
                else:
                    xtick_labels = [
                        str(int(v)) if float(v).is_integer() else str(v) for v in rpm_levels
                    ]
                ax.set_xticklabels(xtick_labels, ha="center", fontsize=9)
                ax.set_xlabel(xlab)
                ax.tick_params(axis="x", labelsize=9)
            else:
                ax.set_xticklabels([])

    matplotlib.pyplot.tight_layout(rect=(0, 0, 0.94, 0.95))

    if last_im is not None:
        cax = fig.add_axes((0.945, 0.15, 0.015, 0.7))
        cbar = fig.colorbar(last_im, cax=cax)
        cbar.ax.tick_params(labelsize=9)
        cbar.set_label(colorbar_label or metric_col)

    if outpath is not None:
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        matplotlib.pyplot.savefig(
            outpath,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.02,
        )
        print("Wrote", outpath)

    return work


def plot_metric_grid_from_agg_all_aeration(
    agg: pandas.DataFrame,
    *,
    metric_col: str,
    placement_col: str = "placement",
    xanthan_col: str = "xanthan",
    rpm_val_col: str = "rpm_val",
    aer_val_col: str = "aer_val",
    # ordering (soft)
    placements: Optional[list] = None,
    xanthan_order: Optional[list] = None,
    # filtering (hard)
    placements_keep: Optional[list] = None,
    xanthan_levels: Optional[list] = None,
    rpm_levels_keep: Optional[list] = None,
    aer_levels_keep: Optional[list] = None,
    # display maps
    placement_label_map: Optional[dict] = None,
    xanthan_label_map: Optional[dict] = None,
    axis_label_map: Optional[dict] = None,
    title: Optional[str] = None,
    colorbar_label: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    robust: bool = False,
    missing_text: str = "no data",
    # Cell annotations
    annotate_cells: bool = True,
    annotation_decimals: int = 1,
    auto_contrast: bool = True,
    annotation_fontsize: Optional[float] = None,
    annotation_fontsize_limits: tuple[float, float] = (6.0, 8.0),
    # Viscosity axis
    viscosity_map: Optional[dict] = None,  # {xanthan_level: {rpm: viscosity}}
    viscosity_label: str = "µ (Pa·s)",
    viscosity_decimals: int = 3,
    viscosity_label_pad: float = 10,
    # PV map: replace rpm tick labels with P/V values per xanthan level
    pv_map: Optional[dict] = None,  # {xanthan_level: {rpm: pv_value}}
    # existing
    dpi: int = 300,
    max_fig_height_in: Optional[float] = None,
    outpath: Optional[Union[str, Path]] = None,
):
    """
    Full-aeration variant of :func:`plot_metric_grid_from_agg`.

    Same grid (rows = placements, cols = xanthan levels, tiles = rpm x aeration), but
    intended for the complete aeration series -- including the 80 L/min setpoint that the
    reference figures drop. ``aer_levels_keep`` therefore defaults to ``None``, i.e. every
    aeration level present in ``agg``.

    Layout contract
    ---------------
    The figure footprint is byte-for-byte the same rule as the reference function:
    width = ``A4_TEXT_WIDTH_IN``, height = ``(A4_TEXT_WIDTH_IN / n_xanthan) * 0.68 *
    n_placements + 1.3``. That rule depends on the number of *tiles*, not on the number of
    aeration rows inside a tile, so adding the fifth aeration level does not change the
    figure size at all -- the heat map cells get shorter instead. For the 6 x 3 paper grid
    that is 6.27 x 9.83 in either way, and a data cell goes from 25.3 x 19.6 pt (4 levels)
    to 25.3 x 15.7 pt (5 levels).

    Because the cells shrink, the in-cell annotations are sized from the *rendered* axes
    box rather than fixed at 8 pt: the font is set to half the cell height, clipped to
    ``annotation_fontsize_limits``. At four aeration levels that reproduces the reference
    8 pt; at five it eases off to ~7.9 pt. Pass ``annotation_fontsize`` to override.

    ``max_fig_height_in`` optionally clamps the height (e.g. to ``A4_TEXT_HEIGHT_IN``).
    It is off by default so this variant stays visually flush with figures produced by the
    reference function, which at six placement rows already runs slightly past the A4 text
    block and is trimmed by ``bbox_inches="tight"`` on save.

    ``title`` is accepted for signature compatibility with the reference function and, as
    there, is deliberately not drawn -- a suptitle would change the height and break the
    layout contract above. The metric name belongs in the figure caption.

    Returns the filtered frame that was plotted, or ``None`` if nothing was left to plot.
    """
    req = [placement_col, xanthan_col, rpm_val_col, aer_val_col, metric_col]
    missing = [c for c in req if c not in agg.columns]
    if missing:
        raise KeyError(f"agg missing required columns: {missing}")

    work = agg.dropna(subset=req).copy()
    if work.empty:
        print("[skip] empty after dropping NaNs")
        return None

    # --- optional filtering (hard) ---
    if placements_keep is not None:
        work = work[work[placement_col].isin(placements_keep)]
    if xanthan_levels is not None:
        work = work[work[xanthan_col].isin(xanthan_levels)]
    if rpm_levels_keep is not None:
        work = work[work[rpm_val_col].isin(rpm_levels_keep)]
    if aer_levels_keep is not None:
        work = work[work[aer_val_col].isin(aer_levels_keep)]

    if work.empty:
        print("[skip] empty after applying level filters")
        return None

    def _disp(mapper, key):
        return mapper.get(key, str(key)) if mapper else str(key)

    # placements order (reverse numeric-ish)
    all_p = list(work[placement_col].unique())

    if placements_keep is not None:
        placements_used = [p for p in placements_keep if p in all_p]
    elif placements is not None:
        primary = [p for p in placements if p in all_p]
        rest = [p for p in all_p if p not in primary]
        placements_used = primary + rest
    else:
        placements_used = all_p

    def _placement_key(p):
        import re

        m = re.search(r"(\d+)$", str(p))
        return int(m.group(1)) if m else 0

    placements_used = list(reversed(sorted(placements_used, key=_placement_key)))

    # xanthan order
    all_x = sorted(work[xanthan_col].unique())
    if xanthan_order is not None:
        primary = [x for x in xanthan_order if x in all_x]
        rest = [x for x in all_x if x not in primary]
        xan_used = primary + rest
    else:
        xan_used = all_x

    # axis levels -- aeration defaults to everything present, so 80 L/min is kept
    rpm_levels = (
        sorted(work[rpm_val_col].unique()) if rpm_levels_keep is None else list(rpm_levels_keep)
    )
    aer_levels = (
        sorted(work[aer_val_col].unique()) if aer_levels_keep is None else list(aer_levels_keep)
    )

    # global color scale
    vals = pandas.to_numeric(work[metric_col], errors="coerce").dropna()
    if vals.empty:
        print("[skip] metric_col has no numeric values")
        return None

    if vmin is None or vmax is None:
        if robust:
            lo, hi = numpy.percentile(vals.to_numpy(), [5, 95])
        else:
            lo, hi = float(vals.min()), float(vals.max())
        if vmin is None:
            vmin = float(lo)
        if vmax is None:
            vmax = float(hi)
        if numpy.isclose(vmin, vmax):
            vmin -= 1e-6
            vmax += 1e-6

    # Helper function for auto-contrast
    def _get_text_color(value, vmin, vmax, cmap):
        """Choose white or black text based on background luminance"""
        if not auto_contrast:
            return "white"
        if numpy.isnan(value):
            return "black"
        norm_val = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.5
        norm_val = numpy.clip(norm_val, 0, 1)
        rgba = cmap(norm_val)
        r, g, b = rgba[:3]
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return "white" if luminance < 0.5 else "black"

    nrows, ncols = len(placements_used), len(xan_used)
    # Identical to the reference layout: depends on the tile grid, not on how many
    # aeration rows sit inside a tile.
    cell_height = (A4_TEXT_WIDTH_IN / ncols) * 0.68
    fig_height = cell_height * nrows + 1.3
    if max_fig_height_in is not None:
        fig_height = min(fig_height, max_fig_height_in)

    fig, axes = matplotlib.pyplot.subplots(
        nrows=nrows, ncols=ncols, figsize=(A4_TEXT_WIDTH_IN, fig_height), sharex=False, sharey=True
    )
    if nrows == 1 and ncols == 1:
        axes = numpy.array([[axes]])
    elif nrows == 1:
        axes = numpy.array([axes])
    elif ncols == 1:
        axes = numpy.array([[ax] for ax in axes])

    xlab = axis_label_map.get("x", "rpm") if axis_label_map else "rpm"
    ylab = axis_label_map.get("y", "l/min") if axis_label_map else "l/min"

    last_im = None
    last_cmap = None
    # Cell values are kept so the annotation pass can run after tight_layout, once the
    # true cell height (and hence a legible font size) is known.
    matrices: dict[tuple[int, int], numpy.ndarray] = {}

    for i, p in enumerate(placements_used):
        for j, x in enumerate(xan_used):
            ax = axes[i, j]
            sub = work[(work[placement_col] == p) & (work[xanthan_col] == x)]

            ax.set_xticks(numpy.arange(len(rpm_levels)))
            ax.set_yticks(numpy.arange(len(aer_levels)))

            if sub.empty:
                ax.set_facecolor("#f0f0f0")
                ax.text(
                    0.5,
                    0.5,
                    missing_text,
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color=Colors.gray,
                )
            else:
                mat = numpy.full((len(aer_levels), len(rpm_levels)), numpy.nan, dtype="float32")
                for _, row in sub.iterrows():
                    rv = row[rpm_val_col]
                    av = row[aer_val_col]
                    if rv not in rpm_levels or av not in aer_levels:
                        continue
                    r = rpm_levels.index(rv)
                    a = aer_levels.index(av)
                    mat[a, r] = float(row[metric_col])

                im = ax.imshow(mat, origin="lower", vmin=vmin, vmax=vmax, aspect="auto")
                last_im = im
                last_cmap = im.get_cmap()
                matrices[(i, j)] = mat

            # titles/labels
            if i == 0:
                ax.set_title(_disp(xanthan_label_map, x), pad=11, fontsize=9)

                # Add viscosity axis on top row
                if viscosity_map is not None and x in viscosity_map:
                    ax2 = ax.twiny()  # Create secondary x-axis
                    ax2.set_xlim(ax.get_xlim())  # Match primary axis limits
                    ax2.set_xticks(numpy.arange(len(rpm_levels)))
                    ax2.tick_params(labelsize=9)

                    # Create viscosity labels
                    visc_labels = []
                    for rpm in rpm_levels:
                        if rpm in viscosity_map[x]:
                            visc = viscosity_map[x][rpm]
                            if viscosity_decimals == 0:
                                visc_labels.append(f"{visc:.0f}")
                            else:
                                visc_labels.append(f"{visc:.{viscosity_decimals}f}")
                        else:
                            visc_labels.append("")

                    ax2.set_xticklabels(visc_labels, fontsize=9)

                    # Add viscosity label on ALL columns (not just first)
                    ax2.set_xlabel(viscosity_label, labelpad=viscosity_label_pad, fontsize=9)

            if j == 0:
                ax.text(
                    placement_level_x,
                    0.5,
                    _disp(placement_label_map, p),
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=9,
                )
                ax.set_ylabel(ylab)

                ax.set_yticks(numpy.arange(len(aer_levels)))
                ax.set_yticklabels(
                    [str(int(v)) if float(v).is_integer() else str(v) for v in aer_levels],
                    fontsize=9,
                )
                ax.tick_params(axis="y", labelleft=True, labelsize=9)
            else:
                ax.tick_params(axis="y", labelleft=False, labelsize=9)

            if i == nrows - 1:
                if pv_map is not None and x in pv_map:
                    xtick_labels = [
                        f"{pv_map[x][v]:.3f}" if v in pv_map[x] else str(v) for v in rpm_levels
                    ]
                else:
                    xtick_labels = [
                        str(int(v)) if float(v).is_integer() else str(v) for v in rpm_levels
                    ]
                ax.set_xticklabels(xtick_labels, ha="center", fontsize=9)
                ax.set_xlabel(xlab)
                ax.tick_params(axis="x", labelsize=9)
            else:
                ax.set_xticklabels([])

    matplotlib.pyplot.tight_layout(rect=(0, 0, 0.94, 0.95))

    # Annotate only once the layout is settled: the cell height that decides whether the
    # numbers still fit is only known after tight_layout has sized the axes.
    if annotate_cells and matrices:
        if annotation_fontsize is None:
            fig.canvas.draw()
            axes_box = axes[0, 0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            cell_height_pt = axes_box.height * 72.0 / len(aer_levels)
            lo_pt, hi_pt = annotation_fontsize_limits
            # Half the cell height leaves roughly the cell's own height again as clearance
            # above and below the digits; 8 pt (the reference value) is the cap.
            fontsize = float(numpy.clip(0.5 * cell_height_pt, lo_pt, hi_pt))
        else:
            fontsize = float(annotation_fontsize)

        for (i, j), mat in matrices.items():
            ax = axes[i, j]
            for a_idx in range(len(aer_levels)):
                for r_idx in range(len(rpm_levels)):
                    val = mat[a_idx, r_idx]
                    if numpy.isnan(val):
                        continue
                    text_color = _get_text_color(val, vmin, vmax, last_cmap)

                    if annotation_decimals == 0:
                        text = f"{val:.0f}"
                    else:
                        text = f"{val:.{annotation_decimals}f}"

                    ax.text(
                        r_idx,
                        a_idx,
                        text,
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=fontsize,
                    )

    if last_im is not None:
        cax = fig.add_axes((0.945, 0.15, 0.015, 0.7))
        cbar = fig.colorbar(last_im, cax=cax)
        cbar.ax.tick_params(labelsize=9)
        cbar.set_label(colorbar_label or metric_col)

    if outpath is not None:
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        matplotlib.pyplot.savefig(
            outpath,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.02,
        )
        print("Wrote", outpath)

    return work


def plot_metric_grid_from_frames(
    frame_level_df: pandas.DataFrame,
    *,
    metric_col: str,
    placement_col: str = "placement",
    setting_col: str = "reactor_setting",
    # Either provide these numeric/categorical columns, OR set derive_from_setting=True
    rpm_col: str = "rpm_val",
    aer_col: str = "aer_val",
    xanthan_col: str = "xanthan",
    derive_from_setting: bool = True,
    # Layout controls
    placements: Optional[list] = None,
    xanthan_values: Optional[list] = None,
    rpm_values: Optional[list] = None,
    aer_values: Optional[list] = None,
    reducer: Union[str, Callable[[pandas.Series], float]] = "median",
    # Color scaling
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    global_scale: bool = True,
    cmap: str = "viridis",
    # Output
    title: Optional[str] = None,
    outdir: Union[str, Path] = "frame_metric_grids",
    fname: Optional[str] = None,
    dpi: int = 200,
) -> Path:
    """
    Heatmap grid for frame-level metrics, collapsed to one value per (placement, setting).

    Rows: placements
    Columns: xanthan levels
    Each subplot: rpm (x) × aeration (y), color = reducer(metric across frames)

    Parameters
    ----------
    frame_level_df:
        One row per frame, containing at least placement_col, setting_col, metric_col.
        If derive_from_setting=True, this function will add rpm_val/aer_val/xanthan via parse_setting
        (through your existing enrich_with_setting_info).
    metric_col:
        Metric to visualize (e.g., "epsilon_obs", "a_obs_m2_m3", "n_bubbles_per_mL").
    reducer:
        How to collapse frames → one number per setting. Recommended:
          - "median" for robust central tendency
          - "q95" to visualize upper tail / bursts of high holdup
          - "cv" to visualize stability across frames
    global_scale:
        If True and vmin/vmax not provided, compute vmin/vmax from all plotted cells so colors
        are comparable across subplots.

    Returns
    -------
    Path to the saved PNG.
    """
    df = frame_level_df

    # Ensure we have rpm/aer/xanthan available for pivoting.
    if derive_from_setting:
        missing = [c for c in (rpm_col, aer_col, xanthan_col) if c not in df.columns]
        if missing:
            # Uses your module function, which depends on parse_setting.
            df = df.copy(deep=False)
            df = metric_utils.enrich_with_setting_info(df, setting_col=setting_col)

    # Collapse frames -> one value per (placement, setting).
    reduce_fn = metric_utils._resolve_reducer(reducer)
    per_setting = (
        df.groupby([placement_col, setting_col], observed=True, sort=False)[metric_col]
        .apply(reduce_fn)
        .rename(metric_col)
        .reset_index()
    )

    # Attach rpm/aer/xanthan for layout (use first occurrence per setting; they’re constant per setting).
    meta_cols = [placement_col, setting_col, rpm_col, aer_col, xanthan_col]
    meta = df[meta_cols].drop_duplicates(subset=[placement_col, setting_col])
    per_setting = per_setting.merge(meta, on=[placement_col, setting_col], how="left")

    # Determine axes values.
    if placements is None:
        placements = list(pandas.unique(per_setting[placement_col]))
    if xanthan_values is None:
        xanthan_values = list(pandas.unique(per_setting[xanthan_col]))
    if rpm_values is None:
        rpm_values = sorted(pandas.unique(per_setting[rpm_col].astype(float)))
    if aer_values is None:
        aer_values = sorted(pandas.unique(per_setting[aer_col].astype(float)))

    # Precompute global vmin/vmax if requested.
    if global_scale and (vmin is None or vmax is None):
        values = per_setting[metric_col].to_numpy(dtype="float64")
        values = values[numpy.isfinite(values)]
        if values.size:
            if vmin is None:
                vmin = float(numpy.nanmin(values))
            if vmax is None:
                vmax = float(numpy.nanmax(values))

    n_rows = len(placements)
    n_cols = len(xanthan_values)

    cell_height = (A4_TEXT_WIDTH_IN / n_cols) * 0.5
    fig_height = cell_height * n_rows + 1.5

    fig, axes = matplotlib.pyplot.subplots(
        n_rows,
        n_cols,
        figsize=(A4_TEXT_WIDTH_IN, fig_height),
        squeeze=False,
    )

    img_handle = None

    for i, placement in enumerate(placements):
        for j, xan in enumerate(xanthan_values):
            ax = axes[i, j]

            sub = per_setting[
                (per_setting[placement_col] == placement) & (per_setting[xanthan_col] == xan)
            ]

            # Pivot into aer (rows) x rpm (cols)
            grid = sub.pivot_table(
                index=aer_col,
                columns=rpm_col,
                values=metric_col,
                aggfunc="first",
            ).reindex(index=aer_values, columns=rpm_values)

            arr = grid.to_numpy(dtype="float64")

            # Heatmap: origin lower so smallest aeration at bottom.
            img_handle = ax.imshow(
                arr,
                origin="lower",
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                interpolation="nearest",
            )

            # Ticks and labels
            ax.set_xticks(range(len(rpm_values)))
            ax.set_xticklabels(
                [str(int(x)) if float(x).is_integer() else str(x) for x in rpm_values],
                rotation=45,
                ha="right",
            )
            ax.set_yticks(range(len(aer_values)))
            ax.set_yticklabels(
                [str(int(y)) if float(y).is_integer() else str(y) for y in aer_values]
            )

            if i == n_rows - 1:
                ax.set_xlabel("rpm")
            else:
                ax.set_xlabel("")
            if j == 0:
                ax.set_ylabel("aeration (l/min)")
            else:
                ax.set_ylabel("")

            if i == 0:
                ax.set_title(f"{xan}")
            if j == n_cols - 1:
                # Put placement label on the rightmost column, readable in grids
                ax.text(
                    1.04,
                    0.5,
                    f"Position {placement}",
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="left",
                )

    # Colorbar
    if img_handle is not None:
        cbar = fig.colorbar(img_handle, ax=axes.ravel().tolist(), shrink=0.92, pad=0.02)
        cbar.set_label(f"{metric_col} ({reducer} across frames)")

    if title is None:
        title = f"{metric_col} grid ({reducer} across frames)"
    fig.suptitle(title, y=0.995)
    fig.tight_layout()

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if fname is None:
        safe_metric = metric_col.replace("/", "_")
        fname = f"grid_{safe_metric}_frames_{reducer}.png"
    outpath = outdir / fname

    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    matplotlib.pyplot.close(fig)
    return outpath


def setting_comparison_stem(settings: list[str], *, prefix: str = "", suffix: str = "") -> str:
    """
    Build the output file stem for a setting-comparison figure from the settings plotted.

    ``plot_settings_comparison`` names only the parameter that *varies* in its legend, so
    the held-fixed levels appear nowhere in the figure itself.  Deriving the stem from the
    same list that is plotted puts them in the filename and keeps the two from drifting:
    an agitation sweep records the aeration rate it was measured at, and an aeration sweep
    records the stirrer speed.

    Exactly one of agitation rate and aeration rate must vary across *settings*; the other
    one and the xanthan concentration are the fixed levels written into the stem.

    Parameters
    ----------
    settings : list of str
        Reactor-setting strings exactly as passed to ``plot_settings_comparison``, e.g.
        ``["75 rpm 55 lmin 000 xanthan", ..., "150 rpm 55 lmin 000 xanthan"]``.
    prefix : str, optional
        Fragment placed before the stem, e.g. ``"holdup_area_"`` to mark the figures
        showing the derived gas-holdup and interfacial-area metrics.
    suffix : str, optional
        Fragment placed after the stem, e.g. ``"_all_aeration"`` to mark the full
        five-setpoint aeration series against the four-setpoint reference layout.

    Returns
    -------
    str
        ``"{prefix}{swept}_at_{fixed}_{xanthan}_xanthan{suffix}"``, for example
        ``"rpm_at_55_lmin_000_xanthan"`` or
        ``"holdup_area_lmin_at_100_rpm_025_xanthan_all_aeration"``.  Contains no ``"."``,
        so the caller can append an extension with ``Path.with_suffix``.

    Raises
    ------
    ValueError
        If fewer than two settings are given, if the xanthan concentration is not held
        constant, or if the number of varying parameters is not exactly one.

    Examples
    --------
    >>> setting_comparison_stem(["75 rpm 55 lmin 000 xanthan", "150 rpm 55 lmin 000 xanthan"])
    'rpm_at_55_lmin_000_xanthan'
    >>> setting_comparison_stem(
    ...     ["100 rpm 45 lmin 025 xanthan", "100 rpm 90 lmin 025 xanthan"],
    ...     prefix="holdup_area_",
    ... )
    'holdup_area_lmin_at_100_rpm_025_xanthan'
    """
    if len(settings) < 2:
        raise ValueError(f"need at least two settings to define a sweep, got {settings!r}")

    parsed = [parse_setting(s) for s in settings]  # ("75 rpm", "55 lmin", "000 xanthan")
    rpm_levels = {rpm for rpm, _, _ in parsed}
    lmin_levels = {lmin for _, lmin, _ in parsed}
    xanthan_levels = {xanthan for _, _, xanthan in parsed}

    if len(xanthan_levels) != 1:
        raise ValueError(
            f"xanthan concentration must be held constant within one figure, "
            f"got {sorted(xanthan_levels)}"
        )

    varying = [
        name for name, levels in (("rpm", rpm_levels), ("lmin", lmin_levels)) if len(levels) > 1
    ]
    if len(varying) != 1:
        raise ValueError(
            f"exactly one of agitation and aeration rate must vary, {len(varying)} do "
            f"({varying or 'none'}); the stem cannot say what is on the x axis"
        )

    swept = varying[0]
    fixed = next(iter(lmin_levels if swept == "rpm" else rpm_levels))
    xanthan = next(iter(xanthan_levels))

    stem = f"{prefix}{swept}_at_{fixed}_{xanthan}{suffix}".replace(" ", "_")
    if "." in stem:
        # Callers pass the stem to Path.with_suffix, which would truncate at the dot.
        raise ValueError(f"figure stem {stem!r} contains '.', which would break the extension")
    return stem


def plot_settings_comparison(
    df: pandas.DataFrame,
    settings: list[str],
    metrics: list[str],
    y_labels: list[str],
    placements: Optional[list[str]] = None,
    *,
    n_eff_lookup: Optional[dict] = None,
    band_lookup: Optional[dict] = None,
    estimand: str = "frame_mean",
    figsize: tuple[float, float] = (A4_TEXT_WIDTH_IN, 5),
    dpi: int = 150,
    capsize: int = 4,
    dodge: float = 0.5,
    legend_ncol: Optional[int] = None,
    outpath: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> matplotlib.pyplot.Figure:
    """
    Plot a condition estimate ±95% CI for one or more metrics across reactor placements,
    with one line per reactor setting.

    Typically used to compare how a single varied parameter (e.g. agitation
    speed or aeration rate) affects bubble metrics at each endoscope position.

    The legend automatically shows only the parameter(s) that vary across
    the supplied *settings* (e.g. only rpm values when aeration and xanthan
    are held constant).

    Parameters
    ----------
    df : pandas.DataFrame
        Frame-level DataFrame containing at least the columns
        ``"placement"``, ``"reactor_setting"``, and every column listed in
        *metrics*.
    settings : list of str
        Reactor-setting strings to plot (one line each), e.g.
        ``["75 rpm 55 lmin 000 xanthan", "100 rpm 55 lmin 000 xanthan"]``.
    metrics : list of str
        Column names to visualise, one subplot per metric.
    y_labels : list of str
        Y-axis labels corresponding to each entry in *metrics*.  Supports
        LaTeX math strings (e.g. ``r"$\\overline{d}$ [mm]"``).
    placements : list of str, optional
        Ordered list of placement identifiers to show on the x-axis.
        Defaults to all unique values found in ``df["placement"]``, sorted.
    n_eff_lookup : dict, optional
        Mapping ``(metric, setting, placement) -> N_eff`` giving the effective
        autocorrelation-adjusted sample size for each error bar. When supplied,
        the 95% CI uses ``N_eff`` in place of the raw frame count ``n`` (clamped to
        ``(1, n]``). Built from ``temporal_independence_full_grid.csv``,
        summing each stream's N_eff over replicates. If ``None`` (default) the naive
        ``sqrt(n)`` CI is used, i.e. every frame treated as independent.
    band_lookup : dict, optional
        Mapping ``(metric, setting, placement) -> (lower, upper)`` giving a *systematic*
        interval to draw behind the confidence interval, as a wider translucent bar
        without caps. Intended for the prolate-oblate depth band on gas holdup and
        observed-volume interfacial area density, whose value is bracketed by the two admissible
        completions of the silhouette. Endpoints are absolute values in the metric's own
        units and must bracket the plotted mean (``ValueError`` otherwise). Entries
        missing from the mapping are not drawn, so metrics that have no band -- bubble
        count, mean diameter -- can share the same call.

        The two intervals are deliberately *not* combined: the confidence interval is
        sampling uncertainty and shrinks as frames accumulate, while the band is a
        bounded systematic range from an axis that is never observed and does not shrink
        at all. Feed this from ``data/public/headline_numbers_by_condition.csv``. It is
        NOT the per-frame ``*_band_pct`` column, a different quantity that diverges from
        it by up to a factor of three.
    estimand : {"frame_mean", "condition"}, optional
        Point-estimate policy. ``"frame_mean"`` preserves the generic arithmetic mean of
        the supplied per-frame metric. ``"condition"`` applies
        :func:`klarity.metrics.condition_metric_estimand`: registered population and ratio
        metrics are formed from summed physical numerators and denominators, while additive
        local quantities give equal weight to every valid observed frame volume. Its CI is
        a delete-one-frame jackknife widened by ``N_eff``. Public figures use
        ``"condition"``; ``"frame_mean"`` remains available for exploratory typical-frame
        plots.
    figsize : tuple of float, optional
        Figure size ``(width, height)`` in inches.  Default ``(A4_TEXT_WIDTH_IN, 5)``.
    dpi : int, optional
        Resolution used when saving (and for the in-memory figure).
        Default ``150``.
    capsize : int, optional
        Cap width for the error bars.  Default ``4``.
    dodge : float, optional
        Horizontal spread of the settings within each placement, in x-axis units where
        adjacent placements are 1.0 apart. Settings are placed symmetrically about the
        tick, so the group spans ``dodge`` and the tick keeps marking the placement
        itself. Default ``0.5``, which leaves half a placement's width of clear space
        between neighbouring groups.

        Without it every setting is drawn at the same x and the markers and their two
        intervals overlap into an unreadable stack -- the more so with five aeration
        setpoints and a band. Set ``0`` for the un-dodged layout.
    legend_ncol : int, optional
        Number of columns in the shared legend.  Defaults to
        ``len(settings)`` so all entries sit in a single row.
    outpath : str or Path, optional
        If given, the figure is saved to this path (format inferred from the
        extension, e.g. ``.png`` or ``.svg``).  The parent directory is
        created automatically.
    show : bool, optional
        Call ``matplotlib.pyplot.show()`` after building the figure.
        Default ``True``.

    Returns
    -------
    matplotlib.pyplot.Figure
        The created figure object.

    Examples
    --------
    >>> fig = plot_settings_comparison(
    ...     df,
    ...     settings=[
    ...         "75 rpm 55 lmin 000 xanthan",
    ...         "100 rpm 55 lmin 000 xanthan",
    ...         "125 rpm 55 lmin 000 xanthan",
    ...         "150 rpm 55 lmin 000 xanthan",
    ...     ],
    ...     metrics=["mean_diameter_mm", "n_bubbles_total"],
    ...     y_labels=[r"$\\overline{d}$ [mm]", "Number of bubbles per frame [-]"],
    ...     outpath="setting_comparison/rpm_at_55_lmin_000_xanthan.png",
    ... )
    """
    import math as _math

    _XANTHAN_MAP = {"000": "0.00", "025": "0.25", "0125": "0.125"}

    # --- Detect which parameters vary across settings ---
    def _find_varying(settings: list[str]) -> set[str]:
        rpms, lmins, xanthans = set(), set(), set()
        for s in settings:
            parts = s.split()
            rpms.add(parts[0])
            lmins.add(parts[2])
            xanthans.add(parts[4])
        varying = set()
        if len(rpms) > 1:
            varying.add("rpm")
        if len(lmins) > 1:
            varying.add("lmin")
        if len(xanthans) > 1:
            varying.add("xanthan")
        return varying

    varying = _find_varying(settings)

    def _format_setting(s: str) -> str:
        parts = s.split()
        rpm = parts[0]
        lmin = parts[2]
        xanthan = _XANTHAN_MAP.get(parts[4], parts[4])
        # Show only the parameter(s) that change; fall back to full label
        pieces = []
        if "rpm" in varying:
            pieces.append(f"{rpm} min$^{{-1}}$")
        if "lmin" in varying:
            pieces.append(f"{lmin} L min$^{{-1}}$")
        if "xanthan" in varying:
            pieces.append(f"{xanthan} wt%")
        if pieces:
            return ", ".join(pieces)
        return f"{rpm} min$^{{-1}}$, {lmin} L min$^{{-1}}$, {xanthan} wt%"

    if len(metrics) != len(y_labels):
        raise ValueError("`metrics` and `y_labels` must have the same length.")
    if estimand not in {"frame_mean", "condition"}:
        raise ValueError("`estimand` must be 'frame_mean' or 'condition'.")

    if placements is None:
        placements = sorted(df["placement"].unique())

    if legend_ncol is None:
        legend_ncol = len(settings)

    x = numpy.arange(len(placements))
    x_labels = [p.replace("placement_", "Position ") for p in placements]

    # Horizontal offset per setting, symmetric about the placement tick, so overlapping
    # markers and intervals separate without the tick ceasing to mark the placement.
    if len(settings) > 1 and dodge:
        x_offsets = numpy.linspace(-dodge / 2.0, dodge / 2.0, len(settings))
    else:
        x_offsets = numpy.zeros(len(settings))

    fig, axes = matplotlib.pyplot.subplots(len(metrics), 1, figsize=figsize, sharex=True, dpi=dpi)
    # Ensure axes is always a list, even for a single metric.
    if len(metrics) == 1:
        axes = [axes]

    drew_band = False
    for metric, y_label, ax in zip(metrics, y_labels, axes):
        # A band_lookup that carries entries for this metric but matches none of the
        # plotted points means the keys disagree -- a setting string formatted differently
        # ("45.0 lmin" vs "45 lmin"), or placements that never line up. That silently
        # produces a band-free figure of exactly the kind this parameter exists to prevent,
        # so it is caught below rather than shipped.
        band_expected = band_lookup is not None and any(k[0] == metric for k in band_lookup)
        band_matched = False
        for (setting, color), x_offset in zip(zip(settings, color_cycle), x_offsets):
            x_series = x + x_offset
            means, ci_lows, ci_highs = [], [], []
            band_lows, band_highs = [], []
            for placement in placements:
                group = df[(df["placement"] == placement) & (df["reactor_setting"] == setting)]
                data = group[metric].dropna()
                n = len(group) if estimand == "condition" else len(data)
                # 95% CI of the selected estimand. Frames are temporally autocorrelated,
                # so the naive independent-frame interval is too narrow. When an
                # N_eff lookup is supplied (per plot-metric, setting, placement; summed over
                # replicates), use the effective sample size instead -- clamped to (1, n]
                # so we never claim more independence than frames observed. The condition
                # estimand uses a ratio jackknife; frame_mean uses its ordinary standard
                # error. A supplied lookup must contain every requested condition.
                denom = float(n)
                if n_eff_lookup is not None:
                    key = (metric, setting, placement)
                    if key not in n_eff_lookup:
                        raise KeyError(f"N_eff is missing for {key!r}")
                    n_eff = float(n_eff_lookup[key])
                    if not numpy.isfinite(n_eff) or n_eff <= 0:
                        raise ValueError(f"N_eff must be finite and positive for {key!r}")
                    denom = min(n_eff, float(n))
                if estimand == "condition" and n > 0:
                    mean = metric_utils.condition_metric_estimand(group, metric)
                    standard_error = metric_utils.condition_metric_standard_error(
                        group, metric, n_eff=denom
                    )
                    ci = 1.96 * standard_error if n > 1 else 0.0
                else:
                    mean = data.mean() if n > 0 else float("nan")
                    ci = 1.96 * data.std() / _math.sqrt(denom) if n > 1 else 0.0
                means.append(mean)
                ci_lows.append(mean - ci)
                ci_highs.append(mean + ci)

                # Systematic depth-model interval, if one is supplied for this metric.
                # Absolute endpoints, not offsets; missing entries stay NaN and are simply
                # not drawn (the count and diameter metrics have no band at all).
                lo, hi = float("nan"), float("nan")
                if band_lookup is not None:
                    entry = band_lookup.get((metric, setting, placement))
                    if entry is not None:
                        lo, hi = (float(v) for v in entry)
                band_lows.append(lo)
                band_highs.append(hi)

            means_arr = numpy.array(means)
            ci_lows_arr = numpy.array(ci_lows)
            ci_highs_arr = numpy.array(ci_highs)
            y_err = numpy.array([means_arr - ci_lows_arr, ci_highs_arr - means_arr])

            # Band first, so the sampling interval reads on top of it.
            band_lows_arr = numpy.array(band_lows)
            band_highs_arr = numpy.array(band_highs)
            finite = numpy.isfinite(band_lows_arr) & numpy.isfinite(band_highs_arr)
            if finite.any():
                band_err = numpy.array([means_arr - band_lows_arr, band_highs_arr - means_arr])
                # The band must bracket the plotted point: it is built from the same
                # condition-level summed contributions as the plotted midpoint.
                # A negative offset means the band and the plotted series were built from
                # different quantities, which would be silently misleading -- so fail loudly
                # rather than draw a bar pointing the wrong way.
                scale = numpy.nanmax(numpy.abs(means_arr[finite]))
                if numpy.nanmin(band_err[:, finite]) < -_BAND_REL_TOL * scale:
                    raise ValueError(
                        f"band_lookup interval does not bracket the plotted mean for "
                        f"metric {metric!r}, setting {setting!r}; the band is centred on a "
                        f"different quantity than the series being plotted."
                    )
                ax.errorbar(
                    x_series[finite],
                    means_arr[finite],
                    yerr=numpy.clip(band_err[:, finite], 0.0, None),
                    fmt="none",
                    ecolor=color,
                    elinewidth=4,
                    alpha=0.30,
                    capsize=0,
                    zorder=1,
                )
                drew_band = True
                band_matched = True

            ax.errorbar(
                x_series,
                means_arr,
                yerr=y_err,
                fmt="o",
                capsize=capsize,
                color=color,
                label=_format_setting(setting),
                zorder=2,
            )

        if band_expected and not band_matched:
            raise ValueError(
                f"band_lookup has entries for metric {metric!r} but none matched the "
                f"plotted points, so no band would be drawn. Check that the lookup's "
                f"setting and placement keys match those being plotted, e.g. "
                f"{(metric, settings[0], placements[0])!r}."
            )

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha="right", color=Colors.gray)
        ax.set_ylabel(y_label)

    # When both intervals are present the figure has to say which is which, otherwise the
    # wide pale bar reads as a bigger error bar. Kept as a separate legend on the first
    # axes so the shared setting legend below stays a single row.
    if drew_band:
        interval_key = [
            matplotlib.lines.Line2D(
                [], [], color=Colors.gray, lw=4, alpha=0.30, label="prolate–oblate band"
            ),
            matplotlib.lines.Line2D([], [], color=Colors.gray, lw=1.5, label="95% CI"),
        ]
        axes[0].add_artist(
            axes[0].legend(handles=interval_key, loc="upper left", frameon=False, fontsize=8)
        )

    # Shared legend below the figure (single row by default).
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=legend_ncol,
        bbox_to_anchor=(0.5, -0.08),
        frameon=False,
    )

    fig.tight_layout()

    if outpath is not None:
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "svg", "pdf"):
            dest = outpath.with_suffix(f".{ext}")
            fig.savefig(dest, bbox_inches="tight")
            print(f"Saved: {dest}")

    if show:
        matplotlib.pyplot.show()

    return fig
