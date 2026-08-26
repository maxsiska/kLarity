#!/usr/bin/env python3
"""Build the public condition-level table from the canonical frame table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from klarity import metrics

GROUP_COLUMNS = ["xanthan", "rpm_val", "aer_val", "placement"]
OUTPUT_COLUMNS = [
    "n_frames",
    "n_bubbles",
    "epsilon_obs_lower",
    "epsilon_obs_mid",
    "epsilon_obs_upper",
    "a_obs_m2_m3_lower",
    "a_obs_m2_m3_mid",
    "a_obs_m2_m3_upper",
    "mean_diameter_mm",
    "a_specific_m2_m3",
    "a_L_m2_m3",
]


def summarize_condition(group: pd.DataFrame) -> pd.Series:
    """Compute the published estimands for one operating condition and placement."""
    values: dict[str, float] = {
        "n_frames": float(len(group)),
        "n_bubbles": float(group["n_bubbles_total"].sum()),
    }
    for metric in (
        "epsilon_obs_lower",
        "epsilon_obs_mid",
        "epsilon_obs_upper",
        "a_obs_m2_m3_lower",
        "a_obs_m2_m3_mid",
        "a_obs_m2_m3_upper",
        "mean_diameter_mm",
        "a_specific_m2_m3",
        "a_L_m2_m3",
    ):
        values[metric] = metrics.condition_metric_estimand(group, metric)
    return pd.Series(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, default=config.FRAME_LEVEL_PKL)
    parser.add_argument(
        "--output",
        type=Path,
        default=config.ROOT / "data" / "public" / "headline_numbers_by_condition.csv",
    )
    args = parser.parse_args()

    frames = pd.read_pickle(args.frames)
    missing = [column for column in GROUP_COLUMNS if column not in frames.columns]
    if missing:
        raise KeyError(f"frame table is missing grouping columns: {missing}")
    table = (
        frames.groupby(GROUP_COLUMNS, observed=True)
        .apply(summarize_condition, include_groups=False)
        .reset_index()
    )
    table = table[[*GROUP_COLUMNS, *OUTPUT_COLUMNS]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f"Wrote {len(table):,} condition rows to {args.output}")


if __name__ == "__main__":
    main()
