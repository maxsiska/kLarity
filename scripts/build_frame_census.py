#!/usr/bin/env python3
"""Build the acquired-frame inventory used by frame-level analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from klarity.census import build_frame_census, summarize_census


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=config.IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR)
    parser.add_argument("--census", type=Path, default=config.FRAME_CENSUS_PARQUET)
    parser.add_argument("--summary", type=Path, default=config.FRAME_CENSUS_SUMMARY)
    parser.add_argument("--failure-log", type=Path, action="append", default=[])
    parser.add_argument("--limit", type=int, help="Process only this many frames for a smoke test")
    args = parser.parse_args()

    census = build_frame_census(
        args.image_root,
        args.output_dir,
        blank_mean_threshold=config.PROCESSING_CONFIG.blank_mean_threshold,
        failure_logs=args.failure_log,
        limit=args.limit,
    )
    args.census.parent.mkdir(parents=True, exist_ok=True)
    census.to_parquet(args.census, index=False)
    summary = summarize_census(census)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary, index=False)
    print(f"Wrote {len(census):,} frame records to {args.census}")
    print(summary.groupby("frame_status", observed=True)["frame_count"].sum().to_string())


if __name__ == "__main__":
    main()
