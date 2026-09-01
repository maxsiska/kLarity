import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "build_temporal_independence", ROOT / "scripts" / "build_temporal_independence.py"
)
TEMPORAL = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(TEMPORAL)


def test_integrated_autocorrelation_time_is_bounded_and_windowed():
    assert TEMPORAL.integrated_autocorrelation_time(np.array([1.0, -0.75])) == 1.0
    assert TEMPORAL.integrated_autocorrelation_time(np.array([1.0, 0.5, 0.0])) == pytest.approx(2.0)


def test_stream_series_places_missing_count_frame_at_zero():
    stream = pd.DataFrame(
        {
            "burst_index": [1, 1],
            "image_number_in_burst": [1, 3],
            "n_bubbles_total": [2.0, 4.0],
        }
    )
    n_bursts, frames_per_burst, series = TEMPORAL.stream_series(stream, "n_bubbles_total", 0.0)
    assert (n_bursts, frames_per_burst) == (1, 3)
    np.testing.assert_array_equal(series, [2.0, 0.0, 4.0])


def test_analyze_uses_only_finite_observations():
    result = TEMPORAL.analyze(np.array([1.0, np.nan, 2.0, 1.0]), 1, 4)
    assert result["N"] == 3.0
    assert 1.0 <= result["N_eff"] <= result["N"]
