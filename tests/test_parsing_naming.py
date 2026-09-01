"""Output naming convention: normalization, resume, and retry."""

from pathlib import Path

import pytest

from klarity.parsing import (
    check_if_processed,
    existing_parquet_paths,
    normalize_metadata_value,
    parquet_stem,
)

PLACEMENT = "placement_3"
SETTING = "100 rpm 45 lmin 000 xanthan"
CANONICAL = "placement_3_100_rpm_45_lmin_000_xanthan_rep_1"
ALTERNATE = "placement_3_100_rpm_45_lmin_000_xanthan__rep_1"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" rep_1", "rep_1"),  # the ten real folders
        ("rep_1 ", "rep_1"),
        ("  rep_2  ", "rep_2"),
        ("rep_1", "rep_1"),  # already clean: unchanged
        ("100 rpm 45 lmin 000 xanthan", "100 rpm 45 lmin 000 xanthan"),
        ("100  rpm   45 lmin", "100 rpm 45 lmin"),  # internal runs collapse to one space
    ],
)
def test_normalize_metadata_value(raw, expected):
    assert normalize_metadata_value(raw) == expected


def test_setting_keeps_single_spaces():
    """parse_setting splits the reactor setting on spaces, so they must survive."""
    from klarity.parsing import parse_setting

    assert parse_setting(normalize_metadata_value(f"  {SETTING} ")) == (
        "100 rpm",
        "45 lmin",
        "000 xanthan",
    )


def test_parquet_stem_is_identical_for_raw_and_clean_replicate():
    """The leading space must not produce a different file from the clean name."""
    assert parquet_stem(PLACEMENT, SETTING, " rep_1") == CANONICAL
    assert parquet_stem(PLACEMENT, SETTING, "rep_1") == CANONICAL


def test_alternate_double_underscore_file_counts_as_processed(tmp_path: Path):
    """Resume accepts a stream name containing an extra underscore."""
    (tmp_path / f"{ALTERNATE}.parquet").touch()
    assert check_if_processed(PLACEMENT, SETTING, "rep_1", tmp_path) is True
    assert check_if_processed(PLACEMENT, SETTING, " rep_1", tmp_path) is True


def test_canonical_file_counts_as_processed(tmp_path: Path):
    (tmp_path / f"{CANONICAL}.parquet").touch()
    assert check_if_processed(PLACEMENT, SETTING, " rep_1", tmp_path) is True


def test_unprocessed_stream_is_not_claimed(tmp_path: Path):
    assert check_if_processed(PLACEMENT, SETTING, "rep_1", tmp_path) is False
    # a different stream's file must not satisfy this one
    (tmp_path / "placement_3_100_rpm_45_lmin_000_xanthan_rep_2.parquet").touch()
    assert check_if_processed(PLACEMENT, SETTING, "rep_1", tmp_path) is False


def test_missing_output_dir_is_not_processed(tmp_path: Path):
    assert check_if_processed(PLACEMENT, SETTING, "rep_1", tmp_path / "nope") is False


def test_retry_finds_both_copies_so_none_is_left_behind(tmp_path: Path):
    """--retry-failures deletes what this returns; a survivor would double-count.

    build_dataframes globs *.parquet, so two accepted spellings would load a stream twice.
    """
    (tmp_path / f"{ALTERNATE}.parquet").touch()
    (tmp_path / f"{CANONICAL}.parquet").touch()
    found = existing_parquet_paths(PLACEMENT, SETTING, "rep_1", tmp_path)
    assert {Path(p).name for p in found} == {f"{ALTERNATE}.parquet", f"{CANONICAL}.parquet"}


def test_retry_finds_alternate_copy_from_cleaned_metadata(tmp_path: Path):
    """The retry path reads the replicate back from a log, so only the clean name exists."""
    (tmp_path / f"{ALTERNATE}.parquet").touch()
    found = existing_parquet_paths(PLACEMENT, SETTING, "rep_1", tmp_path)
    assert [Path(p).name for p in found] == [f"{ALTERNATE}.parquet"]
