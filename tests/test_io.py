import pandas as pd
import pytest

from klarity import io


def test_atomic_parquet_round_trip_preserves_multiindex_and_categories(tmp_path):
    index = pd.MultiIndex.from_tuples(
        [
            ("placement_1", "100 rpm 45 lmin 000 xanthan", "rep_1", 1, 1),
            ("placement_1", "100 rpm 45 lmin 000 xanthan", "rep_1", 1, 2),
        ],
        names=[
            "placement",
            "reactor_setting",
            "replicate",
            "burst_index",
            "image_number_in_burst",
        ],
    )
    expected = pd.DataFrame(
        {
            "model_used": pd.Categorical(["sphere", "ellipsoid"]),
            "equivalent_diameter_mm": [0.75, 1.25],
        },
        index=index,
    )
    destination = tmp_path / "bubble_level_df.parquet"

    io.write_dataframe_atomic(expected, destination, parquet_row_group_rows=1)
    actual = io.read_dataframe(destination)

    pd.testing.assert_frame_equal(actual, expected)
    assert not list(tmp_path.glob("*.tmp"))


def test_failed_atomic_write_preserves_previous_artifact(tmp_path, monkeypatch):
    destination = tmp_path / "frame_level_df.pkl"
    previous = pd.DataFrame({"value": [1]})
    io.write_dataframe_atomic(previous, destination)

    def fail_after_partial_write(self, path, **kwargs):
        path.write_bytes(b"incomplete")
        raise OSError(22, "Invalid argument")

    monkeypatch.setattr(pd.DataFrame, "to_pickle", fail_after_partial_write)

    with pytest.raises(OSError, match="Invalid argument"):
        io.write_dataframe_atomic(pd.DataFrame({"value": [2]}), destination)

    pd.testing.assert_frame_equal(io.read_dataframe(destination), previous)
    assert not list(tmp_path.glob("*.tmp"))


def test_truncated_pickle_has_actionable_error(tmp_path):
    path = tmp_path / "bubble_level_df.pkl"
    path.write_bytes(b"not a complete pickle")

    with pytest.raises(RuntimeError, match="build_dataframes.py --force"):
        io.read_dataframe(path)
