#!/usr/bin/env python3
"""Build effective-sample-size tables from the canonical frame table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

FRAME_DT_MS = {"000": 7.6923, "0125": 100.0, "025": 100.0}
METRICS = {
    "count": ("n_bubbles_total", 0.0),
    "mean_diameter_mm": ("mean_diameter_mm", np.nan),
    "epsilon_obs_mid": ("epsilon_obs_mid", np.nan),
    "a_obs_m2_m3_mid": ("a_obs_m2_m3_mid", np.nan),
}


def burst_acf(series: np.ndarray, n_bursts: int, frames_per_burst: int) -> np.ndarray:
    """Return a NaN-aware autocorrelation function averaged within bursts."""
    values = series.reshape(n_bursts, frames_per_burst) - np.nanmean(series)
    variance = np.nanmean(values * values)
    max_lag = max(1, frames_per_burst // 2)
    if not np.isfinite(variance) or variance <= 0:
        return np.full(max_lag + 1, np.nan)
    rho = np.empty(max_lag + 1)
    for lag in range(max_lag + 1):
        width = frames_per_burst - lag
        rho[lag] = (
            np.nanmean(values[:, :width] * values[:, lag:]) / variance if width > 0 else np.nan
        )
    return rho


def integrated_autocorrelation_time(rho: np.ndarray) -> float:
    """Sokal-windowed integrated autocorrelation time, bounded below by one."""
    tau = 1.0
    for lag in range(1, len(rho)):
        if not np.isfinite(rho[lag]):
            break
        tau += 2.0 * rho[lag]
        if lag >= 5.0 * tau:
            break
    return max(tau, 1.0)


def analyze(series: np.ndarray, n_bursts: int, frames_per_burst: int) -> dict[str, float]:
    """Compute lag-one correlation, autocorrelation time, and effective sample size."""
    n = int(np.isfinite(series).sum())
    rho = burst_acf(series, n_bursts, frames_per_burst)
    tau = integrated_autocorrelation_time(rho)
    rho1 = float(rho[1]) if len(rho) > 1 and np.isfinite(rho[1]) else float("nan")
    n_eff_ar1 = (
        n * (1.0 - rho1) / (1.0 + rho1) if np.isfinite(rho1) and -1.0 < rho1 < 1.0 else np.nan
    )
    return {
        "N": float(n),
        "rho1": rho1,
        "tau_int": tau,
        "N_eff": n / tau,
        "N_eff_ar1": n_eff_ar1,
    }


def stream_series(
    stream: pd.DataFrame, column: str, fill_value: float
) -> tuple[int, int, np.ndarray]:
    """Place one stream on its dense within-burst frame grid."""
    n_bursts = int(stream["burst_index"].nunique())
    frames_per_burst = int(stream["image_number_in_burst"].max())
    burst_rank = stream["burst_index"].rank(method="dense").astype(int) - 1
    frame = burst_rank * frames_per_burst + stream["image_number_in_burst"]
    grid = pd.RangeIndex(1, n_bursts * frames_per_burst + 1)
    series = (
        stream.assign(_frame=frame)
        .groupby("_frame", observed=True)[column]
        .mean()
        .reindex(grid, fill_value=fill_value)
        .to_numpy(float)
    )
    return n_bursts, frames_per_burst, series


def build_tables(frames: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the count/diameter and holdup/area temporal tables."""
    records: list[dict[str, object]] = []
    keys = ["placement", "reactor_setting", "replicate"]
    for (placement, _setting, replicate), stream in frames.groupby(keys, observed=True):
        stream = stream.sort_values(["burst_index", "image_number_in_burst"])
        xanthan = str(stream["xanthan"].iloc[0]).split()[0]
        metadata = {
            "position": int(str(placement).removeprefix("placement_")),
            "rpm": int(stream["rpm_val"].iloc[0]),
            "lmin": int(stream["aer_val"].iloc[0]),
            "xanthan": xanthan,
            "rep": str(replicate).strip(),
            "fluid": "water" if xanthan == "000" else "xanthan",
        }
        dt_ms = FRAME_DT_MS[xanthan]
        for metric, (column, fill_value) in METRICS.items():
            n_bursts, frames_per_burst, series = stream_series(stream, column, fill_value)
            result = analyze(series, n_bursts, frames_per_burst)
            records.append(
                {
                    **metadata,
                    "n_bursts": n_bursts,
                    "metric": metric,
                    "dt_ms": dt_ms,
                    "tau_s": result["tau_int"] * dt_ms / 1000.0,
                    **result,
                }
            )
    table = pd.DataFrame(records)
    return (
        table[table["metric"].isin(("count", "mean_diameter_mm"))].reset_index(drop=True),
        table[table["metric"].isin(("epsilon_obs_mid", "a_obs_m2_m3_mid"))].reset_index(drop=True),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, default=config.FRAME_LEVEL_PKL)
    parser.add_argument("--output-dir", type=Path, default=config.ROOT / "data" / "public")
    args = parser.parse_args()
    frames = pd.read_pickle(args.frames)
    full_grid, holdup = build_tables(frames)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    full_grid.to_csv(args.output_dir / "temporal_independence_full_grid.csv", index=False)
    holdup.to_csv(args.output_dir / "temporal_independence_holdup.csv", index=False)
    print(f"Wrote {len(full_grid):,} count/diameter rows and {len(holdup):,} holdup/area rows")


if __name__ == "__main__":
    main()
