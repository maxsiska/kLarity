#!/usr/bin/env python3
"""
Build bubble-level and frame-level DataFrames from processed Parquet files.

Run this script after process_images.py to produce the two pkl files that all
plotting notebooks depend on.

Usage:
    python scripts/build_dataframes.py            # rebuild if stale
    python scripts/build_dataframes.py --force    # always rebuild
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from klarity import geometry, metrics, parsing

# Rename Parquet columns → names expected by metrics.py
_COLUMN_RENAMES = {
    "d_mm_chosen": "equivalent_diameter_mm",
    "volume_mm3_chosen": "bubble_volume_mm3",
    "surface_area_mm2_chosen": "bubble_surface_area_mm2",
}


def _newest_parquet_mtime(parquet_dir: Path) -> float:
    files = list(parquet_dir.glob("*.parquet"))
    return max(f.stat().st_mtime for f in files) if files else 0.0


def is_stale(parquet_dir: Path) -> bool:
    """Return True if either pkl is missing or older than the newest Parquet."""
    newest = _newest_parquet_mtime(parquet_dir)
    for pkl in (config.BUBBLE_LEVEL_PKL, config.FRAME_LEVEL_PKL):
        if not pkl.exists() or pkl.stat().st_mtime < newest:
            return True
    return False


def build(parquet_dir: Path, force: bool = False) -> None:
    parquet_files = list(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"No Parquet files found in {parquet_dir}")
        sys.exit(1)

    if not force and not is_stale(parquet_dir):
        print("DataFrames are up to date. Use --force to rebuild anyway.")
        return

    print(f"Found {len(parquet_files)} Parquet files in {parquet_dir}\n")

    # ── 1. Load all Parquets ───────────────────────────────────────────────────
    print("Loading Parquet files...")
    bubble_df = parsing.load_all_data_parquet(parquet_dir, set_index=False)
    print(f"  {len(bubble_df):,} bubbles loaded")

    # ── 2. Rename to standard column names ────────────────────────────────────
    bubble_df = bubble_df.rename(
        columns={k: v for k, v in _COLUMN_RENAMES.items() if k in bubble_df.columns}
    )

    # ── 3. Set MultiIndex ─────────────────────────────────────────────────────
    missing = [c for c in parsing.index_levels if c not in bubble_df.columns]
    if missing:
        print(f"ERROR: Missing index columns: {missing}")
        sys.exit(1)
    bubble_df = bubble_df.set_index(list(parsing.index_levels))

    # ── 4. Save bubble-level DataFrame ────────────────────────────────────────
    config.BUBBLE_LEVEL_PKL.parent.mkdir(parents=True, exist_ok=True)
    bubble_df.to_pickle(config.BUBBLE_LEVEL_PKL)
    size_mb = config.BUBBLE_LEVEL_PKL.stat().st_size / 1e6
    print(f"  Saved bubble_level_df  →  {config.BUBBLE_LEVEL_PKL}  ({size_mb:.1f} MB)\n")

    # ── 5. Compute frame-level metrics ────────────────────────────────────────
    print("Computing frame-level metrics...")
    frame_df = metrics.compute_frame_metrics_from_bubbles(
        bubble_df,
        placement_level="placement",
        setting_level="reactor_setting",
    )
    frame_df = metrics.add_observed_volume_metrics_per_frame(frame_df, geometry_module=geometry)
    frame_df = metrics.enrich_with_setting_info(frame_df, setting_col="reactor_setting")
    print(f"  {len(frame_df):,} frames computed")

    # ── 6. Save frame-level DataFrame ─────────────────────────────────────────
    frame_df.to_pickle(config.FRAME_LEVEL_PKL)
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
    args = parser.parse_args()

    if not args.parquet_dir.exists():
        print(f"Error: Parquet directory does not exist: {args.parquet_dir}")
        sys.exit(1)

    build(args.parquet_dir, force=args.force)


if __name__ == "__main__":
    main()
