"""Synthetic tests for the processed-dataset manifest builder."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location(
    "build_dataset_manifest", Path("scripts/build_dataset_manifest.py")
)
assert SPEC is not None and SPEC.loader is not None
manifest_builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manifest_builder)


def test_manifest_records_files_schema_units_and_accounting(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True)

    bubbles = pd.DataFrame(
        {
            "volume_mm3_prolate": [1.0, 2.0],
            "surface_area_mm2_oblate": [3.0, 4.0],
            "centroid_x": [10.0, 20.0],
            "confidence": [0.8, 0.9],
        }
    )
    bubbles.iloc[:1].to_parquet(output_dir / "stream_a.parquet", index=False)
    bubbles.iloc[1:].to_parquet(output_dir / "stream_b.parquet", index=False)

    census_path = metadata_dir / "frame_census.parquet"
    pd.DataFrame(
        {
            "frame_status": [
                "processed_with_detections",
                "processed_zero_detections",
                "blank",
            ]
        }
    ).to_parquet(census_path, index=False)
    census_summary_path = metadata_dir / "frame_census_summary.csv"
    pd.DataFrame({"frame_status": ["blank"], "frame_count": [1]}).to_csv(
        census_summary_path, index=False
    )
    filter_summary_path = metadata_dir / "frame_filter_summary.parquet"
    pd.DataFrame(
        {
            "raw_detection_count": [5],
            "zero_area_count": [1],
            "moderate_border_count": [2],
            "severe_border_count": [1],
            "retained_detection_count": [3],
        }
    ).to_parquet(filter_summary_path, index=False)
    source_artifact = tmp_path / "images.zip"
    source_artifact.write_bytes(b"synthetic input")

    manifest = manifest_builder.build_manifest(
        output_dir,
        census_path,
        census_summary_path,
        filter_summary_path,
        [source_artifact],
    )

    assert manifest["dataset"] == {"directory": "output", "files": 2, "rows": 2}
    assert [record["rows"] for record in manifest["files"]] == [1, 1]
    assert all(len(record["sha256"]) == 64 for record in manifest["files"])
    units = {field["name"]: field["unit"] for field in manifest["schema"]}
    assert units == {
        "volume_mm3_prolate": "mm^3",
        "surface_area_mm2_oblate": "mm^2",
        "centroid_x": "px",
        "confidence": "1",
    }
    assert {field["name"]: field["types"] for field in manifest["schema"]} == {
        "volume_mm3_prolate": ["double"],
        "surface_area_mm2_oblate": ["double"],
        "centroid_x": ["double"],
        "confidence": ["double"],
    }
    assert len(manifest["schema_variants"]) == 1
    assert manifest["schema_variants"][0]["file_count"] == 2
    assert {record["schema_variant"] for record in manifest["files"]} == {"schema_1"}
    assert manifest["frame_status_counts"] == {
        "processed_with_detections": 1,
        "processed_zero_detections": 1,
        "blank": 1,
    }
    assert manifest["filter_decision_counts"] == {
        "raw_detection_count": 5,
        "zero_area_count": 1,
        "moderate_border_count": 2,
        "severe_border_count": 1,
        "retained_detection_count": 3,
    }
    assert manifest["source"]["input_artifacts"][0]["file"] == "images.zip"
    assert len(manifest["source"]["input_artifacts"][0]["sha256"]) == 64


def test_manifest_rejects_inconsistent_parquet_schemas(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    pd.DataFrame({"volume_mm3_prolate": [1.0]}).to_parquet(
        output_dir / "stream_a.parquet", index=False
    )
    pd.DataFrame({"volume_mm3_prolate": ["one"]}).to_parquet(
        output_dir / "stream_b.parquet", index=False
    )

    with pytest.raises(
        ValueError, match="incompatible types for volume_mm3_prolate: double, string"
    ):
        manifest_builder.build_manifest(output_dir, None)


def test_manifest_accepts_compatible_unsigned_integer_widths(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    pd.DataFrame({"bubble_index": pd.Series([1], dtype="uint8")}).to_parquet(
        output_dir / "stream_a.parquet", index=False
    )
    pd.DataFrame({"bubble_index": pd.Series([300], dtype="uint16")}).to_parquet(
        output_dir / "stream_b.parquet", index=False
    )

    manifest = manifest_builder.build_manifest(output_dir, None)

    assert manifest["schema"][0]["types"] == ["uint16", "uint8"]
    assert len(manifest["schema_variants"]) == 2
