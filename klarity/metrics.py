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

3) Convenience helpers for image-level aggregation:
   - per-image bubble counts, areas, diameters, and confidence summaries.

Notes on conventions
--------------------
- “Sphere-like” includes both `sphere_label` (default: "sphere") and its fallback class
  `f"{sphere_label}_fallback"`. Fallback is treated as sphere for ratio computations.
- Volumes are in mm^3, areas in mm^2, diameters in mm unless explicitly stated.
- Some functions are optimized to avoid creating full-length temporary arrays, which can
  otherwise crash notebooks for large datasets.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, NamedTuple, Union

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

    # Aggregate totals using the existing schema.
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
        - a_specific_mm_inv : A_total / V_total [mm^-1] (gas-volume-specific area)
        - a_specific_m2_m3  : gas-volume-specific area [m^2/m^3]
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


_SPHERE_LIKE_LABELS = ("sphere", "sphere_fallback")

# Columns required (beyond geom/volume/surface) to reconstruct the prolate/oblate band.
_BAND_SOURCE_COLUMNS = (
    "volume_mm3_prolate",
    "volume_mm3_oblate",
    "surface_area_mm2_prolate",
    "surface_area_mm2_oblate",
)


class SpheroidBand(NamedTuple):
    """Per-bubble prolate/oblate band, with volume and area intervals kept separate.

    ``v_lower``/``v_upper`` and ``a_lower``/``a_upper`` are true intervals: each is the
    elementwise min/max over the two depth models of that quantity alone. They are the
    endpoints to report for gas holdup and for interfacial area respectively.

    Volume and surface-area order are evaluated independently.
    """

    v_lower: numpy.ndarray
    v_upper: numpy.ndarray
    a_lower: numpy.ndarray
    a_upper: numpy.ndarray


def spheroid_band_arrays(
    bubbles: pandas.DataFrame,
    *,
    geom_col: str = "model_used",
    sphere_like=_SPHERE_LIKE_LABELS,
    volume_col: str = "bubble_volume_mm3",
    surface_col: str = "bubble_surface_area_mm2",
    volume_prolate_col: str = "volume_mm3_prolate",
    volume_oblate_col: str = "volume_mm3_oblate",
    surface_prolate_col: str = "surface_area_mm2_prolate",
    surface_oblate_col: str = "surface_area_mm2_oblate",
    classification_respecting: bool = True,
) -> SpheroidBand:
    """Per-bubble prolate/oblate band (see :class:`SpheroidBand`).

    The out-of-plane depth axis is unobserved, so a bubble's volume/surface depends on the
    assumed spheroid of revolution. Two conventions are supported:

    ``classification_respecting=True`` (default) mirrors ``parsing._spheroid_band``
    (verified bit-exact against the stored ``volume_mm3_lower``/``_upper`` columns):

    - sphere-classified bubbles have no depth ambiguity → lower == upper == chosen value;
    - ellipsoid-classified bubbles take the volume min/max of the prolate and oblate models
      for the VOLUME interval, and independently the surface min/max for the AREA interval
      (ties → prolate is the lower member).

    ``classification_respecting=False`` drops the sphere/ellipsoid threshold entirely and
    applies the band to EVERY bubble that has a two-half-ellipse fit, regardless of
    ``model_used``. The depth axis is unobserved for a near-spherical bubble just as it is
    for a deformed one, so this reports the shape assumption instead of hiding it behind a
    classification cutoff.

    Either way, bubbles with no stored spheroid (``sphere_fallback``: the ellipse fit failed,
    so prolate/oblate are NaN) fall back to the chosen sphere value, which is their only
    measurement.

    Returns a :class:`SpheroidBand` of numpy arrays aligned to ``bubbles``.
    """
    v_pro = bubbles[volume_prolate_col].to_numpy()
    v_ob = bubbles[volume_oblate_col].to_numpy()
    s_pro = bubbles[surface_prolate_col].to_numpy()
    s_ob = bubbles[surface_oblate_col].to_numpy()
    v_ch = bubbles[volume_col].to_numpy()
    s_ch = bubbles[surface_col].to_numpy()

    # A band needs BOTH models complete: two volumes AND their two surfaces. Deciding on
    # the volumes alone would admit a row whose surfaces are missing and silently sum it
    # as zero area downstream.
    complete = (
        numpy.isfinite(v_pro) & numpy.isfinite(v_ob) & numpy.isfinite(s_pro) & numpy.isfinite(s_ob)
    )
    if "band_status" in bubbles.columns:
        status = bubbles["band_status"].astype("string")
        sphere_degenerate = status.isin(("sphere_degenerate", "sphere_fallback")).to_numpy()
        invalid = status.isin(("one_model_only", "invalid")).to_numpy()
    else:
        sphere_degenerate = bubbles[geom_col].isin(sphere_like).to_numpy()
        invalid = ~complete & ~sphere_degenerate

    if classification_respecting:
        use_chosen = sphere_degenerate
        use_two_models = complete & ~sphere_degenerate
    else:
        sphere_fallback = bubbles[geom_col].eq("sphere_fallback").to_numpy()
        use_chosen = sphere_fallback
        use_two_models = complete & ~sphere_fallback
    invalid = invalid | ~(use_chosen | use_two_models)

    prolate_is_lower = v_pro <= v_ob  # ties → prolate is the lower member (matches parsing)
    v_lower = numpy.where(
        invalid,
        numpy.nan,
        numpy.where(use_chosen, v_ch, numpy.where(prolate_is_lower, v_pro, v_ob)),
    )
    v_upper = numpy.where(
        invalid,
        numpy.nan,
        numpy.where(use_chosen, v_ch, numpy.where(prolate_is_lower, v_ob, v_pro)),
    )

    # Area interval: sorted on AREA, independently of the volume order. Using the
    # volume order here would emit a_lower > a_upper whenever the two orders disagree.
    prolate_area_is_lower = s_pro <= s_ob
    a_lower = numpy.where(
        invalid,
        numpy.nan,
        numpy.where(use_chosen, s_ch, numpy.where(prolate_area_is_lower, s_pro, s_ob)),
    )
    a_upper = numpy.where(
        invalid,
        numpy.nan,
        numpy.where(use_chosen, s_ch, numpy.where(prolate_area_is_lower, s_ob, s_pro)),
    )

    return SpheroidBand(v_lower, v_upper, a_lower, a_upper)


def volume_equivalent_diameter_mm(
    volume_mm3: "numpy.ndarray | pandas.Series",
) -> numpy.ndarray:
    """Diameter of the sphere with the same volume: ``d_V = (6V/pi)^(1/3)`` [mm].

    The purely projected alternative ``d_A = 2*sqrt(A_mask/pi)`` is stored as
    ``d_mm_sphere``. This helper keeps reported diameter consistent with the corresponding
    volume estimate.
    """
    v = numpy.asarray(volume_mm3, dtype=float)
    with numpy.errstate(invalid="ignore"):
        return numpy.where(v > 0.0, (6.0 * v / numpy.pi) ** (1.0 / 3.0), numpy.nan)


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
    classification_respecting_band: bool = True,
) -> pandas.DataFrame:
    """
    Compute physically meaningful per-frame metrics from the bubble-level DataFrame.

    One row corresponds to one image (frame). Metrics include:
      - total gas volume and interfacial area
      - bubble counts
      - mean bubble diameter
      - sphere-like fractions (count / volume / surface)

    Geometry logic is preserved by aggregating *after* bubble-level geometry selection.

    ``classification_respecting_band`` selects the prolate/oblate band convention; see
    :func:`spheroid_band_arrays`. ``False`` applies the band to every bubble, which is the
    selected reporting convention.
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
    frame_totals = bubbles_by_frame[[volume_col, surface_col, diameter_mm_col]].sum(min_count=1)
    frame_totals = frame_totals.rename(
        columns={
            volume_col: "V_total_mm3",
            surface_col: "A_total_mm2",
            diameter_mm_col: "diameter_sum_mm",
        }
    )
    frame_totals["diameter_count"] = bubbles_by_frame[diameter_mm_col].count()
    frame_totals["n_bubbles_total"] = bubbles_by_frame.size()

    # ------------------------------------------------------------
    # Aggregate sphere-like bubbles per frame
    # ------------------------------------------------------------
    frame_sphere_totals = sphere_bubbles_by_frame[[volume_col, surface_col]].sum(min_count=1)
    frame_sphere_totals = frame_sphere_totals.rename(
        columns={volume_col: "V_sphere_mm3", surface_col: "A_sphere_mm2"}
    )
    frame_sphere_totals["n_bubbles_sphere"] = sphere_bubbles_by_frame.size()

    # ------------------------------------------------------------
    # Merge totals and sphere-only contributions
    # ------------------------------------------------------------
    frame_metrics = frame_totals.join(frame_sphere_totals, how="left")
    no_sphere = frame_metrics["n_bubbles_sphere"].isna()
    frame_metrics.loc[no_sphere, ["V_sphere_mm3", "A_sphere_mm2"]] = 0.0
    frame_metrics.loc[no_sphere, "n_bubbles_sphere"] = 0

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

    # ------------------------------------------------------------
    # Prolate/oblate spheroid band per frame (only if the source columns are present).
    # Sums the per-bubble classification-respecting band so downstream holdup / interfacial
    # area can be reported as prolate–oblate limits + midpoint rather than a single model.
    # ------------------------------------------------------------
    band_source = {geom_col, volume_col, surface_col, *_BAND_SOURCE_COLUMNS}
    if band_source.issubset(bubbles.columns):
        band_arrays = spheroid_band_arrays(
            bubbles,
            geom_col=geom_col,
            sphere_like=sphere_like_labels,
            volume_col=volume_col,
            surface_col=surface_col,
            classification_respecting=classification_respecting_band,
        )
        band = pandas.DataFrame(
            {
                "V_total_lower_mm3": band_arrays.v_lower,
                "V_total_upper_mm3": band_arrays.v_upper,
                "A_total_lower_mm2": band_arrays.a_lower,
                "A_total_upper_mm2": band_arrays.a_upper,
            }
        )
        # min_count=1 so a frame whose band quantities are all missing stays NaN instead of
        # being reported as a physically impossible zero gas volume / zero interfacial area.
        if frame_keys_are_index_levels:
            band.index = bubbles.index
            band_totals = band.groupby(level=frame_id_levels, observed=True, sort=False).sum(
                min_count=1
            )
        else:
            for key in frame_id_levels:
                band[key] = bubbles[key].to_numpy()
            band_totals = band.groupby(frame_id_levels, observed=True, sort=False).sum(min_count=1)
        frame_metrics = frame_metrics.join(band_totals)
        frame_metrics["V_total_mid_mm3"] = 0.5 * (
            frame_metrics["V_total_lower_mm3"] + frame_metrics["V_total_upper_mm3"]
        )
        frame_metrics["A_total_mid_mm2"] = 0.5 * (
            frame_metrics["A_total_lower_mm2"] + frame_metrics["A_total_upper_mm2"]
        )

    return frame_metrics.reset_index()


def attach_frame_census(
    frame_metrics_df: pandas.DataFrame,
    census_df: pandas.DataFrame,
) -> pandas.DataFrame:
    """Add every successfully processed frame to a bubble-derived metric table.

    Frames with no retained bubbles receive zero additive totals. Frames that are blank,
    unreadable, failed, or unresolved remain in the census artifact but are not scientific
    observations and therefore do not enter this table.
    """
    keys = [
        "placement",
        "reactor_setting",
        "replicate",
        "burst_index",
        "image_number_in_burst",
    ]
    required = {*keys, "frame_status", "image_filename"}
    missing = required.difference(census_df.columns)
    if missing:
        raise KeyError(f"frame census missing required columns: {sorted(missing)}")

    valid = census_df[
        census_df["frame_status"].isin(("processed_with_detections", "processed_zero_detections"))
    ].copy()
    frame_metrics = frame_metrics_df.copy()
    for column in ("placement", "reactor_setting", "replicate"):
        valid[column] = valid[column].astype("string").str.split().str.join(" ")
        frame_metrics[column] = frame_metrics[column].astype("string").str.split().str.join(" ")
    if valid.duplicated(keys).any():
        duplicates = valid.loc[valid.duplicated(keys, keep=False), keys]
        raise ValueError(f"frame census has duplicate scientific frame keys:\n{duplicates.head()}")

    metadata = [*keys, "image_filename", "frame_status", "mean_intensity", "raw_detection_count"]
    out = valid[metadata].merge(frame_metrics, on=keys, how="left", validate="one_to_one")

    additive = [
        "V_total_mm3",
        "A_total_mm2",
        "diameter_sum_mm",
        "diameter_count",
        "n_bubbles_total",
        "V_sphere_mm3",
        "A_sphere_mm2",
        "n_bubbles_sphere",
        "V_total_lower_mm3",
        "V_total_mid_mm3",
        "V_total_upper_mm3",
        "A_total_lower_mm2",
        "A_total_mid_mm2",
        "A_total_upper_mm2",
    ]
    no_retained = out["n_bubbles_total"].isna()
    for column in additive:
        if column in out.columns:
            out.loc[no_retained, column] = 0.0
    return out


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
    gas-volume-specific interfacial area, and bubble count density for each frame.
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

    # ------------------------------------------------------------
    # Prolate/oblate band on gas holdup and interfacial-area density (present only if the
    # per-frame spheroid band was computed). V_obs is a constant, so the relative band on
    # epsilon/a equals that on total volume/surface; ``band_pct`` is the ± relative
    # half-width about the midpoint = 100·(upper−lower)/(upper+lower).
    # ------------------------------------------------------------
    if "V_total_lower_mm3" in frame_metrics.columns:
        frame_metrics["epsilon_obs_lower"] = (
            frame_metrics["V_total_lower_mm3"] / observed_volume_mm3
        )
        frame_metrics["epsilon_obs_upper"] = (
            frame_metrics["V_total_upper_mm3"] / observed_volume_mm3
        )
        frame_metrics["epsilon_obs_mid"] = frame_metrics["V_total_mid_mm3"] / observed_volume_mm3
        eps_sum = frame_metrics["epsilon_obs_upper"] + frame_metrics["epsilon_obs_lower"]
        frame_metrics["epsilon_obs_band_pct"] = numpy.where(
            eps_sum > 0,
            100.0
            * (frame_metrics["epsilon_obs_upper"] - frame_metrics["epsilon_obs_lower"])
            / eps_sum,
            numpy.nan,
        )

        frame_metrics["a_obs_m2_m3_lower"] = (
            frame_metrics["A_total_lower_mm2"] / observed_volume_mm3 * 1000.0
        )
        frame_metrics["a_obs_m2_m3_upper"] = (
            frame_metrics["A_total_upper_mm2"] / observed_volume_mm3 * 1000.0
        )
        frame_metrics["a_obs_m2_m3_mid"] = (
            frame_metrics["A_total_mid_mm2"] / observed_volume_mm3 * 1000.0
        )
        a_sum = frame_metrics["a_obs_m2_m3_upper"] + frame_metrics["a_obs_m2_m3_lower"]
        frame_metrics["a_obs_m2_m3_band_pct"] = numpy.where(
            a_sum > 0,
            100.0
            * (frame_metrics["a_obs_m2_m3_upper"] - frame_metrics["a_obs_m2_m3_lower"])
            / a_sum,
            numpy.nan,
        )

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


@dataclass(frozen=True)
class _RatioEstimand:
    """Definition of a condition metric formed from summed frame contributions."""

    numerator_col: str
    denominator_col: str
    scale: float
    population: str


# Published condition-level estimands. Additive local quantities are written as ratios too:
# summing V_gas or A over valid frames and dividing by the summed observed volume gives every
# equal-duration frame equal weight. Population quantities use their physical denominator,
# avoiding a mean of local ratios whose value depends on how bubbles happen to be partitioned
# among frames.
_CONDITION_RATIO_ESTIMANDS: dict[str, _RatioEstimand] = {
    "mean_diameter_mm": _RatioEstimand(
        "diameter_sum_mm", "diameter_count", 1.0, "pooled retained-bubble population"
    ),
    "n_bubbles_per_mL": _RatioEstimand(
        "n_bubbles_total", "V_obs_mL", 1.0, "mean over valid observed frame volumes"
    ),
    "epsilon_obs": _RatioEstimand(
        "V_total_mm3", "V_obs_mm3", 1.0, "mean over valid observed frame volumes"
    ),
    "a_obs_m2_m3": _RatioEstimand(
        "A_total_mm2", "V_obs_mm3", 1000.0, "mean over valid observed frame volumes"
    ),
    "a_specific_m2_m3": _RatioEstimand(
        "A_total_mm2", "V_total_mm3", 1000.0, "pooled retained gas volume"
    ),
    "a_L_m2_m3": _RatioEstimand(
        "A_total_mm2", "V_liquid_mm3", 1000.0, "pooled observed liquid volume"
    ),
}

for _suffix in ("lower", "mid", "upper"):
    _CONDITION_RATIO_ESTIMANDS[f"epsilon_obs_{_suffix}"] = _RatioEstimand(
        f"V_total_{_suffix}_mm3",
        "V_obs_mm3",
        1.0,
        "mean over valid observed frame volumes",
    )
    _CONDITION_RATIO_ESTIMANDS[f"a_obs_m2_m3_{_suffix}"] = _RatioEstimand(
        f"A_total_{_suffix}_mm2",
        "V_obs_mm3",
        1000.0,
        "mean over valid observed frame volumes",
    )

_CONDITION_BAND_ENDPOINTS = {
    "epsilon_obs_band_pct": ("epsilon_obs_lower", "epsilon_obs_upper"),
    "a_obs_m2_m3_band_pct": ("a_obs_m2_m3_lower", "a_obs_m2_m3_upper"),
}


def condition_metric_population(metric_col: str) -> str:
    """Human-readable population/denominator represented by a condition metric."""
    if metric_col in _CONDITION_BAND_ENDPOINTS:
        return "condition-level systematic half-width"
    spec = _CONDITION_RATIO_ESTIMANDS.get(metric_col)
    return spec.population if spec is not None else "typical valid analyzed frame"


def condition_metric_estimand(frame_df: pandas.DataFrame, metric_col: str) -> float:
    """Return the published condition-level estimate for ``metric_col``.

    Valid, analyzed frames are the sampling units. Camera-failure/blank frames must be
    excluded before this function is called; an illuminated analyzed frame with zero
    retained detections must remain present with zero totals.

    Ratio metrics are computed from summed numerators and denominators:

    - mean diameter: ``sum(diameter_sum) / sum(diameter_count)``;
    - gas holdup: ``sum(V_gas) / sum(V_observed)``;
    - observed-volume area: ``1000 * sum(A) / sum(V_observed)``;
    - gas-specific area: ``1000 * sum(A) / sum(V_gas)``;
    - liquid-volume area: ``1000 * sum(A) / sum(V_liquid)``;

    Metrics without a registered physical numerator/denominator retain the typical-frame
    estimand and are averaged arithmetically. This includes ``n_bubbles_total``.

    Raises
    ------
    KeyError
        If a registered estimand's physical contribution columns are absent.
    """
    band_endpoints = _CONDITION_BAND_ENDPOINTS.get(metric_col)
    if band_endpoints is not None:
        lower = condition_metric_estimand(frame_df, band_endpoints[0])
        upper = condition_metric_estimand(frame_df, band_endpoints[1])
        endpoint_sum = lower + upper
        if not numpy.isfinite(endpoint_sum) or endpoint_sum <= 0:
            return float("nan")
        # Relative half-width about the midpoint, in percent.
        return float(100.0 * (upper - lower) / endpoint_sum)

    spec = _CONDITION_RATIO_ESTIMANDS.get(metric_col)
    if spec is None:
        if metric_col not in frame_df.columns:
            raise KeyError(f"frame data missing metric column {metric_col!r}")
        values = pandas.to_numeric(frame_df[metric_col], errors="coerce").to_numpy(dtype=float)
        return float(numpy.nanmean(values)) if numpy.isfinite(values).any() else float("nan")

    missing = [
        col for col in (spec.numerator_col, spec.denominator_col) if col not in frame_df.columns
    ]
    if missing:
        raise KeyError(f"frame data missing contribution columns for {metric_col!r}: {missing}")

    numerator = pandas.to_numeric(frame_df[spec.numerator_col], errors="coerce").to_numpy(
        dtype=float
    )
    denominator = pandas.to_numeric(frame_df[spec.denominator_col], errors="coerce").to_numpy(
        dtype=float
    )
    finite = numpy.isfinite(numerator) & numpy.isfinite(denominator)
    if not finite.any():
        return float("nan")
    denominator_sum = float(denominator[finite].sum())
    if denominator_sum <= 0:
        return float("nan")
    return float(spec.scale * numerator[finite].sum() / denominator_sum)


def condition_metric_standard_error(
    frame_df: pandas.DataFrame,
    metric_col: str,
    *,
    n_eff: float | None = None,
) -> float:
    """Approximate standard error of a condition estimand across valid frames.

    Registered ratios use an efficient delete-one-frame jackknife, so their uncertainty is
    centered on the same ratio-of-sums estimate that is plotted. If ``n_eff`` is supplied,
    the independent-frame jackknife error is widened by ``sqrt(n / n_eff)`` to retain the
    configured temporal-autocorrelation adjustment. Unregistered metrics use the ordinary
    standard error of their per-frame values with the same ``n_eff`` substitution.
    """
    spec = _CONDITION_RATIO_ESTIMANDS.get(metric_col)
    if spec is None:
        if metric_col not in frame_df.columns:
            raise KeyError(f"frame data missing metric column {metric_col!r}")
        values = pandas.to_numeric(frame_df[metric_col], errors="coerce").to_numpy(dtype=float)
        values = values[numpy.isfinite(values)]
        n = len(values)
        if n < 2:
            return 0.0
        effective_n = float(n) if n_eff is None else min(max(float(n_eff), 1.0), float(n))
        return float(numpy.std(values, ddof=1) / numpy.sqrt(effective_n))

    missing = [
        col for col in (spec.numerator_col, spec.denominator_col) if col not in frame_df.columns
    ]
    if missing:
        raise KeyError(f"frame data missing contribution columns for {metric_col!r}: {missing}")

    numerator = pandas.to_numeric(frame_df[spec.numerator_col], errors="coerce").to_numpy(
        dtype=float
    )
    denominator = pandas.to_numeric(frame_df[spec.denominator_col], errors="coerce").to_numpy(
        dtype=float
    )
    finite = numpy.isfinite(numerator) & numpy.isfinite(denominator)
    numerator = numerator[finite]
    denominator = denominator[finite]
    n = len(numerator)
    if n < 2:
        return 0.0

    numerator_sum = float(numerator.sum())
    denominator_sum = float(denominator.sum())
    leave_denominator = denominator_sum - denominator
    valid_leave = leave_denominator > 0
    if denominator_sum <= 0 or valid_leave.sum() != n:
        return float("nan")

    leave_one = spec.scale * (numerator_sum - numerator) / leave_denominator
    leave_mean = float(leave_one.mean())
    jackknife_se = numpy.sqrt((n - 1.0) / n * numpy.square(leave_one - leave_mean).sum())
    effective_n = float(n) if n_eff is None else min(max(float(n_eff), 1.0), float(n))
    return float(jackknife_se * numpy.sqrt(n / effective_n))


def aggregate_frames_for_grid(
    frame_df: pandas.DataFrame,
    metric_col: str,
    reducer: Union[str, Callable] = "condition",
    *,
    placement_col: str = "placement",
    xanthan_col: str = "xanthan",
    rpm_val_col: str = "rpm_val",
    aer_val_col: str = "aer_val",
) -> tuple[pandas.DataFrame, str]:
    """Collapse valid frames for a condition-level grid.

    ``reducer="condition"`` (the public-figure default) applies
    :func:`condition_metric_estimand`, using ratios of summed physical contributions for
    the audited metrics. Other reducers remain available for explicitly exploratory
    typical-frame summaries such as a median or upper quantile.
    """
    group_cols = [placement_col, xanthan_col, rpm_val_col, aer_val_col]

    # ensure numeric axes (prevents lexicographic sorting issues)
    work = frame_df.copy(deep=False)
    work[rpm_val_col] = pandas.to_numeric(work[rpm_val_col], errors="coerce")
    work[aer_val_col] = pandas.to_numeric(work[aer_val_col], errors="coerce")
    if metric_col in work.columns:
        work[metric_col] = pandas.to_numeric(work[metric_col], errors="coerce")

    out_col = f"{metric_col}_{reducer}" if isinstance(reducer, str) else f"{metric_col}_agg"

    if reducer == "condition":
        rows = []
        grouped = work.dropna(subset=group_cols).groupby(group_cols, observed=True, sort=False)
        for key, group in grouped:
            keys = key if isinstance(key, tuple) else (key,)
            rows.append(
                {
                    **dict(zip(group_cols, keys)),
                    out_col: condition_metric_estimand(group, metric_col),
                }
            )
        return pandas.DataFrame(rows, columns=[*group_cols, out_col]), out_col

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
