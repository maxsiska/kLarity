from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional, Union

import matplotlib
import matplotlib.pyplot
import matplotlib.ticker
import numpy
import pandas

from klarity import metrics
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
    "review-blend", ["#FFEBFE", "#52004E"]
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

METRIC_SPECS = {
    "mean_diameter_mm": dict(
        title="Mean bubble diameter",
        cbar="Mean bubble diameter [mm]",
        robust=False,
        vmin=None,
        vmax=None,
        annotation_decimals=2,
    ),
    "epsilon_obs": dict(
        title="Local gas holdup (observed volume)",
        cbar=r"$\varepsilon$ = V$_\mathrm{gas}$ / V$_\mathrm{obs}$",
        robust=True,
        vmin=None,
        vmax=None,
        annotation_decimals=3,
    ),
    "a_obs_m2_m3": dict(
        title="Local specific interfacial area (observed volume)",
        cbar=r"Interfacial area [m$^{2}$ m$^{-3}$]",
        robust=True,
        vmin=None,
        vmax=None,
        annotation_decimals=1,
    ),
    "frac_sphere_volume": dict(
        title="Sphere fraction (volume-weighted)",
        cbar=r"Volume fraction of spherical bubbles",
        robust=False,
        vmin=0.0,
        vmax=1.0,
        annotation_decimals=2,
    ),
    "frac_sphere_surface": dict(
        title="Sphere fraction (surface-weighted)",
        cbar=r"A$_\mathrm{sphere}$ / A$_\mathrm{total}$",
        robust=False,
        vmin=0.0,
        vmax=1.0,
        annotation_decimals=2,
    ),
    "frac_sphere_count": dict(
        title="Sphere fraction (count-based)",
        cbar=r"N$_\mathrm{sphere}$ / N$_\mathrm{total}$",
        robust=False,
        vmin=0.0,
        vmax=1.0,
        annotation_decimals=2,
    ),
    "n_bubbles_per_mL": dict(
        title="Bubble number density",
        cbar="bubbles/mL",
        robust=True,
        vmin=None,
        vmax=None,
        annotation_decimals=2,
    ),
    "a_specific_m2_m3": dict(
        title="Specific interfacial area (bubble surface / bubble volume)",
        cbar=r"a$_\mathrm{specific}$ = A$_\mathrm{bubble}$ / V$_\mathrm{bubble}$ [m$^{2}$ m$^{-3}$]",
        robust=True,
        vmin=None,
        vmax=None,
        annotate=False,
    ),
    "a_L_m2_m3": dict(
        title="Specific interfacial area",
        cbar=r"Specific interfacial area [m$^{2}$ m$^{-3}$]",
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
):
    """
    Fixes included:
      - Histograms have distinct colors per column (viscosity label).
      - Mean dashed line is ALWAYS red.
      - Xanthan values correctly assigned in settings-mode (labels derived from setting).
      - Binning respects xlim_cap via xlim_eff.
      - Column headers shown as wt%.
      - No 'Replicate' titles stamped on every subplot.
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
        "000": "0.00",
        "0125": "0.125",
        "025": "0.25",
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

    for ax_row in axes:
        for ax in ax_row:
            ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
            ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.25))
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
    # NEW: Viscosity axis
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

    New parameter:
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
                        str(pv_map[x][v]) if v in pv_map[x] else str(v) for v in rpm_levels
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
            df = metrics.enrich_with_setting_info(df, setting_col=setting_col)

    # Collapse frames -> one value per (placement, setting).
    reduce_fn = metrics._resolve_reducer(reducer)
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


def plot_settings_comparison(
    df: pandas.DataFrame,
    settings: list[str],
    metrics: list[str],
    y_labels: list[str],
    placements: Optional[list[str]] = None,
    *,
    figsize: tuple[float, float] = (A4_TEXT_WIDTH_IN, 5),
    dpi: int = 150,
    capsize: int = 4,
    legend_ncol: Optional[int] = None,
    outpath: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> matplotlib.pyplot.Figure:
    """
    Plot mean ±95% CI for one or more metrics across reactor placements,
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
    figsize : tuple of float, optional
        Figure size ``(width, height)`` in inches.  Default ``(A4_TEXT_WIDTH_IN, 5)``.
    dpi : int, optional
        Resolution used when saving (and for the in-memory figure).
        Default ``150``.
    capsize : int, optional
        Cap width for the error bars.  Default ``4``.
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
    ...     outpath="setting_comparison/rpm_000_xanthan.png",
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

    if placements is None:
        placements = sorted(df["placement"].unique())

    if legend_ncol is None:
        legend_ncol = len(settings)

    x = numpy.arange(len(placements))
    x_labels = [p.replace("placement_", "Position ") for p in placements]

    fig, axes = matplotlib.pyplot.subplots(len(metrics), 1, figsize=figsize, sharex=True, dpi=dpi)
    # Ensure axes is always a list, even for a single metric.
    if len(metrics) == 1:
        axes = [axes]

    for metric, y_label, ax in zip(metrics, y_labels, axes):
        for setting, color in zip(settings, color_cycle):
            means, ci_lows, ci_highs = [], [], []
            for placement in placements:
                data = df[(df["placement"] == placement) & (df["reactor_setting"] == setting)][
                    metric
                ].dropna()

                n = len(data)
                mean = data.mean() if n > 0 else float("nan")
                ci = 1.96 * data.std() / _math.sqrt(n) if n > 1 else 0.0
                means.append(mean)
                ci_lows.append(mean - ci)
                ci_highs.append(mean + ci)

            means_arr = numpy.array(means)
            ci_lows_arr = numpy.array(ci_lows)
            ci_highs_arr = numpy.array(ci_highs)
            y_err = numpy.array([means_arr - ci_lows_arr, ci_highs_arr - means_arr])

            ax.errorbar(
                x,
                means_arr,
                yerr=y_err,
                marker="o",
                fmt=".",
                capsize=capsize,
                color=color,
                label=_format_setting(setting),
            )

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha="right", color=Colors.gray)
        ax.set_ylabel(y_label)

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
