"""
Aggregation and derived-metric utilities for bubble-segmentation outputs.

This module sits “downstream” of image processing / segmentation. It assumes you have a
bubble-level DataFrame (one row per detected bubble) with geometry-derived quantities
(e.g., equivalent diameters, volumes, surface areas) and with sufficient metadata to
group by placement/setting and (optionally) by image-level indices.

Primary use cases
-----------------
1) Geometry-ratio analysis:
   - Fraction of bubbles (count-based) treated as spheres vs ellipsoids.
   - Fraction of total gas volume attributed to sphere-like bubbles.
   - Fraction of total interfacial area attributed to sphere-like bubbles.

2) Gas-holdup / interfacial-area-density metrics normalized by observed control volume:
   - epsilon_obs: (sum of bubble volumes) / (observed volume)
   - a_obs: (sum of bubble surface areas) / (observed volume)

3) Legacy / convenience helpers for image-level aggregation:
   - per-image bubble counts, areas, Sauter diameter, confidence summaries.

Notes on conventions
--------------------
- “Sphere-like” includes both `sphere_label` (default: "sphere") and its fallback class
  `f"{sphere_label}_fallback"`. Fallback is treated as sphere for ratio computations.
- Volumes are in mm^3, areas in mm^2, diameters in mm unless explicitly stated.
- Some functions are optimized to avoid creating full-length temporary arrays, which can
  otherwise crash notebooks for large datasets.
"""

from collections import defaultdict
from typing import Callable, Union

import numpy
import pandas

from klarity.parsing import parse_setting


def enrich_with_setting_info(df: pandas.DataFrame, setting_col: str):
    """
    Parse human-readable setting strings into numeric rpm/aeration plus categorical xanthan label.

    This is a convenience helper for plotting. It expects `parse_setting` to return three strings,
    typically:
        ("<rpm> rpm", "<aer> lmin", "<xanthan> xanthan" or similar)

    Parameters
    ----------
    df:
        DataFrame containing a column with setting identifiers.
    setting_col:
        Name of the column holding the setting string.

    Returns
    -------
    pandas.DataFrame
        The input `df` with three additional columns:
          - "rpm_val"   : float, extracted from the first token of the rpm string
          - "aer_val"   : float, extracted from the first token of the aeration string
          - "xanthan"   : str, the xanthan descriptor returned by parse_setting

    Notes
    -----
    - This function mutates `df` in place and returns it for convenience.
    - Parsing rules are intentionally simple and depend on consistent formatting upstream.
    """
    rpm_val, aer_val, xan = [], [], []

    for s in df[setting_col]:
        r, a, x = parse_setting(str(s))
        rpm_val.append(float(r.split()[0]))
        aer_val.append(float(a.split()[0]))
        xan.append(x)

    df["rpm_val"] = rpm_val
    df["aer_val"] = aer_val
    df["xanthan"] = xan
    return df


def collect_setting_accumulators(
    bubble_level_df: pandas.DataFrame,
    *,
    placement_level: str,
    setting_level: str,
    geom_col: str,
    sphere_label: str = "sphere",
    volume_col: str = "bubble_volume_mm3",
    surface_col: str = "bubble_surface_area_mm2",
    diameter_col: str = "equivalent_diameter_mm",
):
    """
    Memory-stable per-(placement, setting) aggregation for volume/area/diameter and sphere fractions.

    This function computes totals on the full DataFrame and sphere-only sums on a filtered subset.
    The design avoids allocating full-length masked temporary arrays for each metric, which is a
    common cause of kernel crashes for large bubble-level tables.

    Parameters
    ----------
    bubble_level_df:
        Bubble-level table (one row per bubble). Should contain:
          - `geom_col` : geometry/model label per bubble
          - `volume_col`, `surface_col`, `diameter_col` : per-bubble numeric metrics
        Grouping keys must be available either as index levels or columns.
    placement_level, setting_level:
        Names of the placement and setting grouping keys.
        If both are index levels, grouping is done with `groupby(level=...)` to avoid reset_index.
        Otherwise, `reset_index()` is used as a fallback (higher memory cost).
    geom_col:
        Column name with geometry labels ("sphere", "sphere_fallback", "asym_ellipsoid", ...).
    sphere_label:
        Base sphere label. Sphere-like is {sphere_label, f"{sphere_label}_fallback"}.
    volume_col, surface_col, diameter_col:
        Column names for bubble volume [mm^3], surface area [mm^2], and equivalent diameter [mm].

    Returns
    -------
    dict
        Mapping (placement, setting) -> accumulator dict with:
          - V_total_mm3, V_sphere_mm3
          - A_total_mm2, A_sphere_mm2
          - d_sum_mm, d_count
          - n_total, n_sphere

    Notes
    -----
    - Groups with no sphere-like bubbles are returned with sphere sums/counts = 0.
    - Diameter statistics are returned as sum/count to enable a stable mean downstream.
    """
    sphere_like = {sphere_label, f"{sphere_label}_fallback"}

    # Use index levels directly if possible (avoid reset_index allocation)
    keys_are_index = (
        placement_level in bubble_level_df.index.names
        and setting_level in bubble_level_df.index.names
    )

    if keys_are_index:
        df = bubble_level_df
        group_keys = [placement_level, setting_level]
        g_all = df.groupby(level=group_keys, observed=True, sort=False)
        # sphere subset
        mask = df[geom_col].isin(sphere_like)
        df_s = df.loc[mask]
        g_s = df_s.groupby(level=group_keys, observed=True, sort=False)
    else:
        df = bubble_level_df.reset_index()  # unavoidable if keys are not index levels
        group_keys = [placement_level, setting_level]
        g_all = df.groupby(group_keys, observed=True, sort=False)
        mask = df[geom_col].isin(sphere_like)
        df_s = df.loc[mask]
        g_s = df_s.groupby(group_keys, observed=True, sort=False)

    # Aggregate totals (no new columns)
    all_agg = g_all.agg(
        V_total_mm3=(volume_col, "sum"),
        A_total_mm2=(surface_col, "sum"),
        d_sum_mm=(diameter_col, "sum"),
        d_count=(diameter_col, "count"),
        n_total=(geom_col, "size"),
    )

    # Aggregate sphere-only (on smaller df_s)
    s_agg = g_s.agg(
        V_sphere_mm3=(volume_col, "sum"),
        A_sphere_mm2=(surface_col, "sum"),
        n_sphere=(geom_col, "size"),
    )

    # Join (align on group index); missing sphere groups -> 0
    out = all_agg.join(s_agg, how="left").fillna(
        {"V_sphere_mm3": 0.0, "A_sphere_mm2": 0.0, "n_sphere": 0}
    )

    # Return dict keyed by (placement, setting)
    acc: dict = defaultdict(dict)
    for key, row in out.iterrows():
        acc[key] = {
            "V_total_mm3": float(row["V_total_mm3"]) if pandas.notna(row["V_total_mm3"]) else 0.0,
            "V_sphere_mm3": (
                float(row["V_sphere_mm3"]) if pandas.notna(row["V_sphere_mm3"]) else 0.0
            ),
            "A_total_mm2": float(row["A_total_mm2"]) if pandas.notna(row["A_total_mm2"]) else 0.0,
            "A_sphere_mm2": (
                float(row["A_sphere_mm2"]) if pandas.notna(row["A_sphere_mm2"]) else 0.0
            ),
            "d_sum_mm": float(row["d_sum_mm"]) if pandas.notna(row["d_sum_mm"]) else 0.0,
            "d_count": int(row["d_count"]),
            "n_total": int(row["n_total"]),
            "n_sphere": int(row["n_sphere"]),
        }

    return acc


def accumulators_to_frame(acc: dict, *, placement_name: str, setting_name: str) -> pandas.DataFrame:
    rows = []
    for (placement, setting), v in acc.items():
        rows.append(
            {
                placement_name: placement,
                setting_name: setting,
                **v,
                "mean_diameter_mm": (
                    (v["d_sum_mm"] / v["d_count"]) if v["d_count"] > 0 else numpy.nan
                ),
                "frac_sphere_count": (
                    (v["n_sphere"] / v["n_total"]) if v["n_total"] > 0 else numpy.nan
                ),
                "frac_sphere_volume": (
                    (v["V_sphere_mm3"] / v["V_total_mm3"]) if v["V_total_mm3"] > 0 else numpy.nan
                ),
                "frac_sphere_surface": (
                    (v["A_sphere_mm2"] / v["A_total_mm2"]) if v["A_total_mm2"] > 0 else numpy.nan
                ),
            }
        )
    return pandas.DataFrame(rows)


def add_observed_volume_metrics(
    agg: pandas.DataFrame,
    *,
    geometry_module,
    depth_mm: float | None = None,
    V_total_col: str = "V_total_mm3",
    A_total_col: str = "A_total_mm2",
    n_col: str = "n_total",
) -> pandas.DataFrame:
    """
    Add observed-control-volume-normalized metrics (gas holdup, interfacial area density, counts).

    The "observed volume" refers to the control volume represented by the camera field of view
    and an assumed depth (optical path / effective thickness). The geometry module is expected
    to implement:
        geometry_module.get_observed_volume_mm3(depth_mm: float | None) -> float

    Parameters
    ----------
    agg:
        Aggregated per-(placement, setting) DataFrame containing at minimum:
        - V_total_col: total gas volume per group [mm^3]
        - A_total_col: total interfacial area per group [mm^2]
        - n_col      : bubble count per group
    geometry_module:
        Module or object with `get_observed_volume_mm3(...)`.
    depth_mm:
        Optional depth override; forwarded to geometry_module. If None, the geometry module
        should use its own default.
    V_total_col, A_total_col, n_col:
        Column names in `agg` for totals and counts.

    Returns
    -------
    pandas.DataFrame
        Copy of `agg` with added columns:
        - V_obs_mm3, V_obs_mL
        - epsilon_obs       : V_total / V_obs (gas holdup in the observed control volume)
        - a_obs_mm_inv      : A_total / V_obs [mm^-1]
        - a_obs_m2_m3       : interfacial area density [m^2/m^3] (converted from mm^-1)
        - n_per_mL          : bubble count per observed mL
        - a_specific_mm_inv : A_total / V_total [mm^-1] (specific interfacial area)
        - a_specific_m2_m3  : specific interfacial area [m^2/m^3]
        - V_liquid_mm3, V_liquid_mL : liquid volume in observed control volume (V_obs - V_gas)
        - a_L_mm_inv        : A_total / V_liquid [mm^-1]
        - a_L_m2_m3         : liquid-volume-based interfacial area density [m^2/m^3]
    """

    out = agg.copy()
    V_obs_mm3 = float(geometry_module.get_observed_volume_mm3(depth_mm=depth_mm))

    out["V_obs_mm3"] = V_obs_mm3
    out["V_obs_mL"] = V_obs_mm3 / 1000.0  # 1 mL = 1000 mm^3

    out["epsilon_obs"] = out[V_total_col] / V_obs_mm3  # gas holdup in observed volume
    out["a_obs_mm_inv"] = out[A_total_col] / V_obs_mm3  # [mm^2] / [mm^3] = [mm^-1]
    out["a_obs_m2_m3"] = out["a_obs_mm_inv"] * 1000.0  # [mm^-1] * 1000 = [m^2/m^3]
    out["n_per_mL"] = out[n_col] / (V_obs_mm3 / 1000.0)  # count / mL

    # Specific interfacial area: bubble surface area / bubble volume
    # [mm^2] / [mm^3] = [mm^-1],  * 1000 -> [m^2/m^3]
    out["a_specific_mm_inv"] = numpy.where(
        out[V_total_col] > 0,
        out[A_total_col] / out[V_total_col],
        numpy.nan,
    )
    out["a_specific_m2_m3"] = out["a_specific_mm_inv"] * 1000.0

    # Liquid-volume-based interfacial area density:
    #   a_L = A_total / V_liquid,  where V_liquid = V_obs - V_gas
    # [mm^2] / [mm^3] = [mm^-1],  * 1000 -> [m^2/m^3]
    V_liquid_mm3 = V_obs_mm3 - out[V_total_col]
    out["V_liquid_mm3"] = V_liquid_mm3
    out["V_liquid_mL"] = V_liquid_mm3 / 1000.0  # 1 mL = 1000 mm^3
    out["a_L_mm_inv"] = numpy.where(
        V_liquid_mm3 > 0,
        out[A_total_col] / V_liquid_mm3,
        numpy.nan,
    )
    out["a_L_m2_m3"] = out["a_L_mm_inv"] * 1000.0

    return out


def compute_frame_metrics_from_bubbles(
    bubble_level_df: pandas.DataFrame,
    *,
    placement_level: str,
    setting_level: str,
    geom_col: str = "model_used",
    sphere_label: str = "sphere",
    volume_col: str = "bubble_volume_mm3",
    surface_col: str = "bubble_surface_area_mm2",
    diameter_mm_col: str = "equivalent_diameter_mm",
    replicate_level: str = "replicate",
    burst_level: str = "burst_index",
    image_level: str = "image_number_in_burst",
) -> pandas.DataFrame:
    """
    Compute physically meaningful per-frame metrics from the bubble-level DataFrame.

    One row corresponds to one image (frame). Metrics include:
      - total gas volume and interfacial area
      - bubble counts
      - mean bubble diameter
      - sphere-like fractions (count / volume / surface)

    Geometry logic is preserved by aggregating *after* bubble-level geometry selection.
    """
    sphere_like_labels = {sphere_label, f"{sphere_label}_fallback"}

    frame_id_levels = [
        placement_level,
        setting_level,
        replicate_level,
        burst_level,
        image_level,
    ]

    # ------------------------------------------------------------
    # Determine whether grouping keys are index levels (cheap path)
    # ------------------------------------------------------------
    frame_keys_are_index_levels = all(key in bubble_level_df.index.names for key in frame_id_levels)

    if frame_keys_are_index_levels:
        bubbles = bubble_level_df  # alias only, no copy

        bubbles_by_frame = bubbles.groupby(level=frame_id_levels, observed=True, sort=False)

        sphere_bubbles = bubbles.loc[bubbles[geom_col].isin(sphere_like_labels)]
        sphere_bubbles_by_frame = sphere_bubbles.groupby(
            level=frame_id_levels, observed=True, sort=False
        )
    else:
        # unavoidable allocation if grouping keys are not index levels
        bubbles = bubble_level_df.reset_index()

        bubbles_by_frame = bubbles.groupby(frame_id_levels, observed=True, sort=False)

        sphere_bubbles = bubbles.loc[bubbles[geom_col].isin(sphere_like_labels)]
        sphere_bubbles_by_frame = sphere_bubbles.groupby(frame_id_levels, observed=True, sort=False)

    # ------------------------------------------------------------
    # Aggregate all bubbles per frame
    # ------------------------------------------------------------
    frame_totals = bubbles_by_frame.agg(
        V_total_mm3=(volume_col, "sum"),
        A_total_mm2=(surface_col, "sum"),
        diameter_sum_mm=(diameter_mm_col, "sum"),
        diameter_count=(diameter_mm_col, "count"),
        n_bubbles_total=(geom_col, "size"),
    )

    # ------------------------------------------------------------
    # Aggregate sphere-like bubbles per frame
    # ------------------------------------------------------------
    frame_sphere_totals = sphere_bubbles_by_frame.agg(
        V_sphere_mm3=(volume_col, "sum"),
        A_sphere_mm2=(surface_col, "sum"),
        n_bubbles_sphere=(geom_col, "size"),
    )

    # ------------------------------------------------------------
    # Merge totals and sphere-only contributions
    # ------------------------------------------------------------
    frame_metrics = frame_totals.join(frame_sphere_totals, how="left").fillna(
        {
            "V_sphere_mm3": 0.0,
            "A_sphere_mm2": 0.0,
            "n_bubbles_sphere": 0,
        }
    )

    # ------------------------------------------------------------
    # Derived per-frame quantities
    # ------------------------------------------------------------
    frame_metrics["mean_diameter_mm"] = (
        frame_metrics["diameter_sum_mm"] / frame_metrics["diameter_count"]
    )

    frame_metrics["frac_sphere_count"] = (
        frame_metrics["n_bubbles_sphere"] / frame_metrics["n_bubbles_total"]
    )

    frame_metrics["frac_sphere_volume"] = (
        frame_metrics["V_sphere_mm3"] / frame_metrics["V_total_mm3"]
    )

    frame_metrics["frac_sphere_surface"] = (
        frame_metrics["A_sphere_mm2"] / frame_metrics["A_total_mm2"]
    )

    return frame_metrics.reset_index()


def add_observed_volume_metrics_per_frame(
    frame_metrics_df: pandas.DataFrame,
    *,
    geometry_module,
    depth_mm: float | None = None,
    volume_col: str = "V_total_mm3",
    surface_col: str = "A_total_mm2",
    count_col: str = "n_bubbles_total",
) -> pandas.DataFrame:
    """
    Normalize per-frame metrics by the observed control volume.

    Adds gas holdup, interfacial area density (control-volume and liquid-volume based),
    specific interfacial area, and bubble count density for each frame.
    """
    frame_metrics = frame_metrics_df.copy()

    observed_volume_mm3 = float(geometry_module.get_observed_volume_mm3(depth_mm=depth_mm))

    frame_metrics["V_obs_mm3"] = observed_volume_mm3
    frame_metrics["V_obs_mL"] = observed_volume_mm3 / 1000.0  # 1 mL = 1000 mm^3

    frame_metrics["epsilon_obs"] = frame_metrics[volume_col] / observed_volume_mm3  # gas holdup [-]

    frame_metrics["a_obs_mm_inv"] = (
        frame_metrics[surface_col] / observed_volume_mm3
    )  # [mm^2] / [mm^3] = [mm^-1]
    frame_metrics["a_obs_m2_m3"] = (
        frame_metrics["a_obs_mm_inv"] * 1000.0
    )  # [mm^-1] * 1000 = [m^2/m^3]

    frame_metrics["n_bubbles_per_mL"] = frame_metrics[count_col] / (
        observed_volume_mm3 / 1000.0
    )  # count / mL

    # Specific interfacial area: bubble surface area / bubble volume
    # [mm^2] / [mm^3] = [mm^-1],  * 1000 -> [m^2/m^3]
    frame_metrics["a_specific_mm_inv"] = numpy.where(
        frame_metrics[volume_col] > 0,
        frame_metrics[surface_col] / frame_metrics[volume_col],
        numpy.nan,
    )
    frame_metrics["a_specific_m2_m3"] = frame_metrics["a_specific_mm_inv"] * 1000.0

    # Liquid-volume-based interfacial area density:
    #   a_L = A_total / V_liquid,  where V_liquid = V_obs - V_gas
    # [mm^2] / [mm^3] = [mm^-1],  * 1000 -> [m^2/m^3]
    V_liquid_mm3 = observed_volume_mm3 - frame_metrics[volume_col]
    frame_metrics["V_liquid_mm3"] = V_liquid_mm3
    frame_metrics["V_liquid_mL"] = V_liquid_mm3 / 1000.0  # 1 mL = 1000 mm^3
    frame_metrics["a_L_mm_inv"] = numpy.where(
        V_liquid_mm3 > 0,
        frame_metrics[surface_col] / V_liquid_mm3,
        numpy.nan,
    )
    frame_metrics["a_L_m2_m3"] = frame_metrics["a_L_mm_inv"] * 1000.0

    return frame_metrics


def _resolve_reducer(
    reducer: Union[str, Callable[[pandas.Series], float]],
) -> Callable[[pandas.Series], float]:
    """
    Map reducer specification to a callable. Supports common robust summaries.
    """
    if callable(reducer):
        return reducer

    if reducer == "mean":
        return lambda x: float(numpy.nanmean(x.to_numpy()))
    if reducer == "median":
        return lambda x: float(numpy.nanmedian(x.to_numpy()))
    if reducer == "q05":
        return lambda x: float(numpy.nanquantile(x.to_numpy(), 0.05))
    if reducer == "q25":
        return lambda x: float(numpy.nanquantile(x.to_numpy(), 0.25))
    if reducer == "q75":
        return lambda x: float(numpy.nanquantile(x.to_numpy(), 0.75))
    if reducer == "q95":
        return lambda x: float(numpy.nanquantile(x.to_numpy(), 0.95))
    if reducer == "std":
        return lambda x: float(numpy.nanstd(x.to_numpy(), ddof=1))
    if reducer == "cv":

        def cv(x: pandas.Series) -> float:
            arr = x.to_numpy(dtype="float64")
            m = numpy.nanmean(arr)
            s = numpy.nanstd(arr, ddof=1)
            return float(s / m) if numpy.isfinite(m) and m != 0 else numpy.nan

        return cv

    raise ValueError(
        f"Unknown reducer '{reducer}'. Use mean/median/q05/q25/q75/q95/std/cv or a callable."
    )


def aggregate_frames_for_grid(
    frame_df: pandas.DataFrame,
    metric_col: str,
    reducer: Union[str, Callable] = "mean",
    *,
    placement_col: str = "placement",
    xanthan_col: str = "xanthan",
    rpm_val_col: str = "rpm_val",
    aer_val_col: str = "aer_val",
) -> tuple[pandas.DataFrame, str]:
    group_cols = [placement_col, xanthan_col, rpm_val_col, aer_val_col]

    # ensure numeric axes (prevents lexicographic sorting issues)
    work = frame_df.copy(deep=False)
    work[rpm_val_col] = pandas.to_numeric(work[rpm_val_col], errors="coerce")
    work[aer_val_col] = pandas.to_numeric(work[aer_val_col], errors="coerce")
    work[metric_col] = pandas.to_numeric(work[metric_col], errors="coerce")

    out_col = f"{metric_col}_{reducer}" if isinstance(reducer, str) else f"{metric_col}_agg"

    agg = (
        work.dropna(subset=group_cols + [metric_col])
        .groupby(group_cols, observed=True)[metric_col]
        .agg(reducer)
        .reset_index()
        .rename(columns={metric_col: out_col})
    )
    return agg, out_col


def compute_burst_uncertainty(df):
    """
    Compute the average prediction confidence (or uncertainty) per burst.
    Lower confidence may indicate bursts where the model struggled.
    """
    burst_confidence = (
        df.groupby(["reactor_setting", "burst_index"])["confidence"].mean().reset_index()
    )
    return burst_confidence


def compute_hdi(data, credibility_mass=0.95):
    """
    Compute the 95% Highest Density Interval (HDI) for a given dataset.
    Returns: (hdi_min, hdi_max, hdi_width)
    """
    data = numpy.sort(data)
    n_data = len(data)
    interval_idx_inc = int(numpy.floor(credibility_mass * n_data))
    if interval_idx_inc < 1 or n_data < 2:
        return float("nan"), float("nan"), float("nan")
    intervals = [(data[i], data[i + interval_idx_inc]) for i in range(n_data - interval_idx_inc)]
    hdi_min, hdi_max = min(intervals, key=lambda x: x[1] - x[0])
    hdi_width = hdi_max - hdi_min
    return hdi_min, hdi_max, hdi_width
