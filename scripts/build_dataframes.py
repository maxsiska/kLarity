#!/usr/bin/env python3
"""
Build bubble-level and frame-level DataFrames from processed Parquet files.

Run this script after process_images.py to produce the two analysis tables that all
plotting notebooks depend on.

The bubble-level Parquet is trimmed to the analysis-relevant columns (_BUBBLE_KEEP_COLUMNS)
so the full ~99M-row dataset fits in workstation RAM; the complete 50-column record stays
in the Parquet files. Parquet is streamed and concatenated with unified categorical dtypes
to keep memory bounded.

Usage:
    python scripts/build_dataframes.py            # rebuild if stale
    python scripts/build_dataframes.py --force    # always rebuild

    # Read from a non-default directory:
    python scripts/build_dataframes.py --parquet-dir /path/to/output --force
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from klarity import geometry, io, metrics, parsing

# Rename Parquet columns → names expected by metrics.py
_COLUMN_RENAMES = {
    "d_mm_chosen": "equivalent_diameter_mm",
    "volume_mm3_chosen": "bubble_volume_mm3",
    "surface_area_mm2_chosen": "bubble_surface_area_mm2",
}

# Columns kept in the bubble-level table, of the ~50 written per Parquet file. The full
# record stays in the source Parquet files; the analysis table is trimmed to
# what the analysis actually reads so all ~99M rows fit in RAM on a workstation. A full
# 50-column table of this dataset is ~40 GB in memory and does not fit in 17 GB.
# Raw Parquet names here; the three *_chosen columns are renamed via _COLUMN_RENAMES.
_BUBBLE_KEEP_COLUMNS = [
    # MultiIndex levels
    "placement",
    "reactor_setting",
    "replicate",
    "burst_index",
    "image_number_in_burst",
    "image_filename",
    "bubble_index",
    # chosen geometry (the model that model_used names) → renamed to metrics.py names
    "d_mm_chosen",
    "volume_mm3_chosen",
    "surface_area_mm2_chosen",
    # per-bubble model classification
    "model_used",
    # Stored spheroid values. metrics.spheroid_band_arrays builds the volume interval and
    # the independently sorted area interval from these four columns.
    "volume_mm3_prolate",
    "volume_mm3_oblate",
    "surface_area_mm2_prolate",
    "surface_area_mm2_oblate",
    # sphere-model counterfactual, and the purely projected (area-equivalent) diameter
    # d_A = 2*sqrt(A_mask/pi) -- the depth-model-free size reference for the methods text
    "d_mm_sphere",
    "volume_mm3_sphere",
    # In-plane aspect ratio: reported quantities are classification-independent. Keeping
    # this column lets the sphere-vs-ellipsoid split be constructed
    # downstream at ANY diameter/aspect cutoff (with d_mm_sphere) without re-reading Parquet.
    "aspect_ratio",
    # QA / screening (mask_area also drives the zero-area filter, then is kept for census)
    "solidity",
    "confidence",
    "mask_area",
    # Border-truncation filter inputs; dropped after filtering (see _load_bubbles_trimmed)
    "border_contact_px",
    "equivalent_diameter",
]

# Filter-only columns: needed to apply drop_border_clipped, then discarded to save memory.
# The full record stays in the Parquet files.
_FILTER_ONLY_COLUMNS = ["border_contact_px", "equivalent_diameter"]

# Columns accepted when present but not required by the core schema. Requesting a missing
# column from pyarrow raises, so these are resolved against each file's actual schema.
_BUBBLE_OPTIONAL_COLUMNS = ["band_status", "measurement_status", "edge_status"]

# Categorical columns whose per-file category dictionaries are disjoint (each Parquet holds
# one placement/setting/replicate). pandas.concat up-casts categoricals with mismatched
# categories to object (string) — which would blow past RAM — so they are re-cast to a
# shared CategoricalDtype before concatenation.
_CATEGORICAL_COLUMNS = [
    "placement",
    "reactor_setting",
    "replicate",
    "model_used",
]


def _columns_for(parquet_path: Path) -> "list[str]":
    """Required columns plus whichever optional ones this file actually carries."""
    available = set(pq.ParquetFile(parquet_path).schema.names)
    missing = [c for c in _BUBBLE_KEEP_COLUMNS if c not in available]
    if missing:
        raise KeyError(f"{parquet_path.name} is missing required columns: {missing}")
    return _BUBBLE_KEEP_COLUMNS + [c for c in _BUBBLE_OPTIONAL_COLUMNS if c in available]


def _load_bubbles_trimmed(
    parquet_dir: Path, keep_zero_area_masks: bool, keep_border_clipped: bool
) -> "tuple[pd.DataFrame, int, int, pd.DataFrame]":
    """Stream the per-rep Parquet files into one trimmed bubble-level DataFrame.

    Reads only ``_BUBBLE_KEEP_COLUMNS`` (of ~50) from each file, applies the two row
    filters per file (zero-area phantom masks and severe border clips), and unifies
    the categorical dtypes across files so the final ``concat`` keeps them as compact
    integer codes rather than up-casting to object strings. Peak memory is bounded by the
    trimmed data, not the full 50-column record.

    Returns
    -------
    (combined, n_raw, n_clipped) : the concatenated trimmed frame, the raw row count before
        any filter, and the number of rows removed by the border-contact policy.
    """
    files = sorted(parquet_dir.glob("*.parquet"))
    frames: "list[pd.DataFrame]" = []
    universe: "dict[str, set]" = {c: set() for c in _CATEGORICAL_COLUMNS}
    n_raw = 0
    n_clipped = 0
    filter_summaries: "list[pd.DataFrame]" = []
    for f in files:
        d = pd.read_parquet(f, columns=_columns_for(f))
        n_raw += len(d)
        measurable = pd.to_numeric(d["mask_area"], errors="coerce") > 0
        ratio = pd.to_numeric(d["border_contact_px"], errors="coerce") / pd.to_numeric(
            d["equivalent_diameter"], errors="coerce"
        )
        severe = measurable & ratio.gt(parsing.CLIP_RATIO_THRESHOLD)
        moderate = (
            measurable & pd.to_numeric(d["border_contact_px"], errors="coerce").gt(0) & ~severe
        )
        group_keys = [
            "placement",
            "reactor_setting",
            "replicate",
            "burst_index",
            "image_number_in_burst",
            "image_filename",
        ]
        decisions = d[group_keys].copy()
        decisions["raw_detection_count"] = 1
        decisions["zero_area_count"] = (~measurable).astype("int64")
        decisions["moderate_border_count"] = moderate.astype("int64")
        decisions["severe_border_count"] = severe.astype("int64")
        decisions["retained_detection_count"] = (measurable & ~severe).astype("int64")
        filter_summaries.append(
            decisions.groupby(group_keys, observed=True, sort=False).sum().reset_index()
        )
        if not keep_zero_area_masks:
            d = parsing.drop_zero_area_masks(d)
        if not keep_border_clipped:
            n_before_clip = len(d)
            d = parsing.drop_border_clipped(d)
            n_clipped += n_before_clip - len(d)
        d = d.drop(columns=_FILTER_ONLY_COLUMNS, errors="ignore")
        for c in _CATEGORICAL_COLUMNS:
            universe[c].update(map(str, d[c].unique().tolist()))
        frames.append(d)

    # Shared category dictionaries → concat preserves categorical (cheap codes).
    dtype_map = {c: pd.CategoricalDtype(sorted(universe[c])) for c in _CATEGORICAL_COLUMNS}
    frames = [d.astype(dtype_map) for d in frames]
    combined = pd.concat(frames, ignore_index=True)
    filter_summary = pd.concat(filter_summaries, ignore_index=True)
    return combined, n_raw, n_clipped, filter_summary


def _newest_parquet_mtime(parquet_dir: Path) -> float:
    files = list(parquet_dir.glob("*.parquet"))
    return max(f.stat().st_mtime for f in files) if files else 0.0


def is_stale(parquet_dir: Path) -> bool:
    """Return True if either analysis artifact is missing or older than the source."""
    newest = _newest_parquet_mtime(parquet_dir)
    for artifact in (config.BUBBLE_LEVEL_PARQUET, config.FRAME_LEVEL_PKL):
        if not artifact.exists() or artifact.stat().st_mtime < newest:
            return True
    return False


def build(
    parquet_dir: Path,
    force: bool = False,
    keep_zero_area_masks: bool = False,
    keep_border_clipped: bool = False,
    classification_respecting_band: bool = False,
    frame_census_path: Path | None = None,
) -> None:
    parquet_files = list(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"No Parquet files found in {parquet_dir}")
        sys.exit(1)

    if not force and not is_stale(parquet_dir):
        print("DataFrames are up to date. Use --force to rebuild anyway.")
        return

    print(f"Found {len(parquet_files)} Parquet files in {parquet_dir}\n")

    # ── 1. Stream Parquet → trimmed bubble frame (see _BUBBLE_KEEP_COLUMNS) ──────
    # Only the analysis-relevant columns are read, so all ~99M rows fit in RAM; the full
    # 50-column record stays in the Parquet files. Zero-area phantom masks carry NaN
    # volume/diameter and would inflate raw counts / number density by ~1% (up to ~5% at
    # low shear / high viscosity); they are dropped here unless --keep-zero-area-masks.
    # The published border-contact policy retains ratios <= 1 unchanged and excludes ratios
    # > 1; it never reconstructs off-frame geometry. --keep-border-clipped disables that
    # exclusion for sensitivity analysis. See parsing.drop_border_clipped for the rationale.
    print("Loading Parquet files (trimmed columns)...")
    bubble_df, n_raw, n_clipped, filter_summary = _load_bubbles_trimmed(
        parquet_dir, keep_zero_area_masks, keep_border_clipped
    )
    config.FRAME_FILTER_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    filter_summary.to_parquet(config.FRAME_FILTER_SUMMARY, index=False)
    dropped = n_raw - len(bubble_df)
    pct = 100.0 * dropped / n_raw if n_raw else 0.0
    print(
        f"  {n_raw:,} raw detections; dropped {dropped:,} ({pct:.2f}%)  →  "
        f"{len(bubble_df):,} measurable bubbles"
    )
    print(
        f"    zero-area phantom masks : {'kept' if keep_zero_area_masks else dropped - n_clipped}"
    )
    print(f"    border ratio > 1.0      : {'kept' if keep_border_clipped else n_clipped}")

    # ── 2. Rename to standard column names ────────────────────────────────────
    bubble_df = bubble_df.rename(
        columns={k: v for k, v in _COLUMN_RENAMES.items() if k in bubble_df.columns}
    )

    # ── 2b. Reported size = volume-equivalent diameter of the spheroid-band midpoint ──
    # The stored d_mm_chosen is classification-dependent: the area-equivalent circle
    # diameter for sphere-classified bubbles, the volume-equivalent diameter for
    # ellipsoid-classified ones. Reporting it would leave the sphere/ellipsoid threshold
    # embedded in the bubble-size distribution. Computing d from the same midpoint volume
    # that feeds epsilon makes
    # the size distribution and holdup mutually consistent. Where no spheroid was
    # fitted (sphere_fallback) the chosen sphere value is kept -- it is the only
    # measurement available. d_mm_sphere retains the purely projected diameter d_A.
    if not classification_respecting_band:
        band_cols = ("volume_mm3_prolate", "volume_mm3_oblate")
        if all(c in bubble_df.columns for c in band_cols):
            v_mid = 0.5 * (bubble_df["volume_mm3_prolate"] + bubble_df["volume_mm3_oblate"])
            d_v = metrics.volume_equivalent_diameter_mm(v_mid)
            n_fallback = int(np.isnan(d_v).sum())
            bubble_df["equivalent_diameter_mm"] = np.where(
                np.isnan(d_v), bubble_df["equivalent_diameter_mm"], d_v
            )
            print(
                f"  equivalent_diameter_mm computed from the spheroid-band midpoint "
                f"({n_fallback:,} rows kept at the sphere value)"
            )

    # ── 3. Set MultiIndex ─────────────────────────────────────────────────────
    missing = [c for c in parsing.index_levels if c not in bubble_df.columns]
    if missing:
        print(f"ERROR: Missing index columns: {missing}")
        sys.exit(1)
    bubble_df = bubble_df.set_index(list(parsing.index_levels))

    # ── 4. Save bubble-level DataFrame ────────────────────────────────────────
    # The complete artifact exceeds 4 GiB. Parquet avoids the oversized pickle write that
    # fails on some Windows/network filesystems, and the atomic helper prevents an
    # interrupted build from leaving a truncated destination behind.
    io.write_dataframe_atomic(bubble_df, config.BUBBLE_LEVEL_PARQUET)
    size_mb = config.BUBBLE_LEVEL_PARQUET.stat().st_size / 1e6
    print(f"  Saved bubble_level_df  →  {config.BUBBLE_LEVEL_PARQUET}  " f"({size_mb:.1f} MB)\n")

    # ── 5. Compute frame-level metrics ────────────────────────────────────────
    print("Computing frame-level metrics...")
    frame_df = metrics.compute_frame_metrics_from_bubbles(
        bubble_df,
        placement_level="placement",
        setting_level="reactor_setting",
        classification_respecting_band=classification_respecting_band,
    )
    if frame_census_path is not None:
        if not frame_census_path.exists():
            raise FileNotFoundError(f"frame census does not exist: {frame_census_path}")
        frame_census = pd.read_parquet(frame_census_path)
        frame_df = metrics.attach_frame_census(frame_df, frame_census)
    frame_df = metrics.add_observed_volume_metrics_per_frame(frame_df, geometry_module=geometry)
    frame_df = metrics.enrich_with_setting_info(frame_df, setting_col="reactor_setting")
    print(f"  {len(frame_df):,} frames computed")

    # ── 6. Save frame-level DataFrame ─────────────────────────────────────────
    io.write_dataframe_atomic(frame_df, config.FRAME_LEVEL_PKL)
    size_mb = config.FRAME_LEVEL_PKL.stat().st_size / 1e6
    print(f"  Saved frame_level_df   →  {config.FRAME_LEVEL_PKL}  ({size_mb:.1f} MB)\n")

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build bubble- and frame-level DataFrames.")
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=config.OUTPUT_DIR,
        help=f"Directory containing Parquet files (default: {config.OUTPUT_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if pkl files are already up to date",
    )
    parser.add_argument(
        "--keep-zero-area-masks",
        action="store_true",
        help="Keep zero-area phantom detections (mask_area==0). Default drops them so raw "
        "counts match the volume/diameter metrics; use this to reproduce the raw/published build.",
    )
    parser.add_argument(
        "--keep-border-clipped",
        action="store_true",
        help="Keep detections whose border_contact_px/equivalent_diameter ratio exceeds 1.0. "
        "By default those rows are excluded; ratios <= 1.0 are always retained unchanged.",
    )
    parser.add_argument(
        "--classification-respecting-band",
        action="store_true",
        help="Apply the prolate/oblate band only to ellipsoid-classified bubbles, and keep "
        "the stored classification-dependent diameter. Default applies the band to every "
        "bubble, removing the sphere/ellipsoid threshold from the reported quantities.",
    )
    parser.add_argument(
        "--frame-census",
        type=Path,
        default=config.FRAME_CENSUS_PARQUET,
        help="Frame census used to retain successful zero-detection frames.",
    )
    parser.add_argument(
        "--without-frame-census",
        action="store_true",
        help="Build only from bubble rows. Intended for diagnostics, not reported results.",
    )
    args = parser.parse_args()

    if not args.parquet_dir.exists():
        print(f"Error: Parquet directory does not exist: {args.parquet_dir}")
        sys.exit(1)

    build(
        args.parquet_dir,
        force=args.force,
        keep_zero_area_masks=args.keep_zero_area_masks,
        keep_border_clipped=args.keep_border_clipped,
        classification_respecting_band=args.classification_respecting_band,
        frame_census_path=None if args.without_frame_census else args.frame_census,
    )


if __name__ == "__main__":
    main()
