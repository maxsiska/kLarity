"""
Utilities for checking whether derived DataFrames are up to date.
"""

from pathlib import Path


def check_dataframes_stale() -> None:
    """
    Print a warning if bubble_level_df.pkl or frame_level_df.pkl are older
    than the newest Parquet file in the output directory.

    Call this at the top of any plotting notebook before loading pkl files.
    """
    import config

    parquet_files = list(Path(config.OUTPUT_DIR).glob("*.parquet"))
    if not parquet_files:
        print("WARNING: No Parquet files found in output directory. Run process_images.py first.")
        return

    newest_parquet = max(f.stat().st_mtime for f in parquet_files)

    stale = []
    for pkl in (config.BUBBLE_LEVEL_PKL, config.FRAME_LEVEL_PKL):
        if not pkl.exists():
            stale.append(f"  - {pkl.name} is missing")
        elif pkl.stat().st_mtime < newest_parquet:
            stale.append(f"  - {pkl.name} is older than the newest Parquet")

    if stale:
        print("=" * 60)
        print("WARNING: DataFrames are out of date.")
        print("Run:  python scripts/build_dataframes.py")
        print("Reason:")
        for s in stale:
            print(s)
        print("=" * 60)
