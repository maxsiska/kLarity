#!/usr/bin/env python3
"""Create the neutral provenance manifest for a processed dataset."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=config.ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _artifact_name(path: Path, output_dir: Path) -> str:
    try:
        return path.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        return path.name


def _artifact_record(path: Path, output_dir: Path) -> dict:
    if path.suffix == ".parquet":
        rows = pq.ParquetFile(path).metadata.num_rows
    elif path.suffix == ".csv":
        with path.open() as source:
            rows = max(sum(1 for _ in source) - 1, 0)
    else:
        rows = None
    return {
        "file": _artifact_name(path, output_dir),
        "rows": rows,
        "sha256": sha256(path),
    }


def _column_unit(name: str) -> str | None:
    if name.endswith("_mm3") or "_mm3_" in name:
        return "mm^3"
    if name.endswith("_mm2") or "_mm2_" in name:
        return "mm^2"
    if name.endswith("_mm") or "_mm_" in name:
        return "mm"
    if name.endswith("_px") or name.startswith(("bbox_", "centroid_")):
        return "px"
    if name == "mask_area":
        return "px^2"
    if name == "equivalent_diameter":
        return "px"
    if name.endswith("_deg"):
        return "degree"
    if name in {"score", "confidence", "solidity", "aspect_ratio", "aspect_delta"}:
        return "1"
    return None


def _schema_signature(schema: pyarrow.Schema) -> tuple[tuple[str, str, bool], ...]:
    """Return the physical Arrow fields, excluding file-specific pandas metadata."""
    return tuple((field.name, str(field.type), field.nullable) for field in schema)


def _types_are_compatible(types: set[pyarrow.DataType]) -> bool:
    """Return whether physical Arrow types represent one compatible logical column."""
    if len(types) <= 1:
        return True
    if all(pyarrow.types.is_unsigned_integer(value) for value in types):
        return True
    if all(pyarrow.types.is_signed_integer(value) for value in types):
        return True
    if all(pyarrow.types.is_dictionary(value) for value in types):
        dictionaries = list(types)
        return all(
            value.value_type == dictionaries[0].value_type
            and value.ordered == dictionaries[0].ordered
            and pyarrow.types.is_integer(value.index_type)
            for value in dictionaries
        )
    return False


def build_manifest(
    output_dir: Path,
    census_path: Path | None,
    census_summary_path: Path | None = None,
    filter_summary_path: Path | None = None,
    source_artifacts: list[Path] | None = None,
) -> dict:
    files = sorted(output_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no Parquet files found in {output_dir}")

    schema = pq.ParquetFile(files[0]).schema_arrow
    column_names = schema.names
    column_nullability = {field.name: field.nullable for field in schema}
    column_types: dict[str, set[pyarrow.DataType]] = {field.name: {field.type} for field in schema}
    schema_variants: dict[tuple[tuple[str, str, bool], ...], dict[str, Any]] = {}
    file_records = []
    total_rows = 0
    for path in files:
        parquet_file = pq.ParquetFile(path)
        metadata = parquet_file.metadata
        file_schema = parquet_file.schema_arrow
        if file_schema.names != column_names:
            raise ValueError(f"schema differs: {path.name}")
        for field in file_schema:
            if field.nullable != column_nullability[field.name]:
                raise ValueError(f"schema differs: {path.name}")
            column_types[field.name].add(field.type)
        signature = _schema_signature(file_schema)
        if signature not in schema_variants:
            schema_variants[signature] = {
                "id": f"schema_{len(schema_variants) + 1}",
                "file_count": 0,
                "fields": [
                    {"name": name, "type": field_type, "nullable": nullable}
                    for name, field_type, nullable in signature
                ],
            }
        variant = schema_variants[signature]
        variant["file_count"] = int(variant["file_count"]) + 1
        rows = metadata.num_rows
        total_rows += rows
        file_records.append(
            {
                "file": path.name,
                "rows": rows,
                "sha256": sha256(path),
                "schema_variant": variant["id"],
            }
        )

    for name, types in column_types.items():
        if not _types_are_compatible(types):
            observed = ", ".join(sorted(str(value) for value in types))
            raise ValueError(f"incompatible types for {name}: {observed}")

    frame_status_counts: dict[str, int] = {}
    census_record = None
    if census_path is not None:
        census = pd.read_parquet(census_path)
        frame_status_counts = {
            str(key): int(value)
            for key, value in census["frame_status"].value_counts(dropna=False).items()
        }
        census_record = _artifact_record(census_path, output_dir)

    filter_decision_counts: dict[str, int] = {}
    if filter_summary_path is not None:
        count_columns = [
            "raw_detection_count",
            "zero_area_count",
            "moderate_border_count",
            "severe_border_count",
            "retained_detection_count",
        ]
        filters = pd.read_parquet(filter_summary_path, columns=count_columns)
        filter_decision_counts = {
            column: int(pd.to_numeric(filters[column], errors="coerce").sum())
            for column in count_columns
        }

    metadata_paths = [
        path for path in (census_path, census_summary_path, filter_summary_path) if path is not None
    ]

    processing = config.PROCESSING_CONFIG
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {"directory": "output", "files": len(files), "rows": total_rows},
        "source": {
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_dirty": bool(git_value("status", "--porcelain")),
            "model_sha256": sha256(config.MODEL_PATH) if config.MODEL_PATH.exists() else None,
            "configuration_sha256": {
                "config.py": sha256(config.ROOT / "config.py"),
                "klarity/processing_config.py": sha256(
                    config.ROOT / "klarity" / "processing_config.py"
                ),
            },
            "input_artifacts": [
                _artifact_record(path, output_dir) for path in (source_artifacts or [])
            ],
        },
        "processing_parameters": {
            "confidence": processing.confidence,
            "iou": processing.iou,
            "mask_threshold": processing.mask_threshold,
            "large_mask_confidence": processing.large_mask_confidence,
            "large_mask_fraction": processing.large_mask_fraction,
            "blank_mean_threshold": processing.blank_mean_threshold,
            "geometry_mode": processing.geometry_mode,
            "sphere_aspect_tolerance": processing.sphere_aspect_tolerance,
        },
        "geometry": {
            "axes": "a is the full major axis; b1 and b2 are in-plane semi-minor axes",
            "prolate_volume_mm3": "(pi*a/3)*(b1^2+b2^2)",
            "prolate_surface": "smooth spheroid with b_eq=sqrt((b1^2+b2^2)/2)",
            "oblate_volume_mm3": "(pi*a^2/6)*(b1+b2)",
            "intervals": "volume and surface-area endpoints are ordered independently",
            "derivation": "geometry columns are deterministic functions of stored measured axes",
        },
        "band_status": {
            "two_models": "both spheroid models are finite",
            "sphere_degenerate": "sphere-classified zero-width interval",
            "sphere_fallback": "ellipse fit unavailable; sphere is the only measurement",
            "one_model_only": "one spheroid model is incomplete",
            "invalid": "no usable geometry interval",
        },
        "measurement_status": {
            "measurable": "positive finite mask area",
            "zero_area": "mask area is non-positive or non-finite",
        },
        "edge_status": {
            "not_touching": "silhouette does not touch the image boundary",
            "moderate_retained": "boundary chord to equivalent-diameter ratio is at most one",
            "severe_excluded": "boundary chord to equivalent-diameter ratio exceeds one",
        },
        "schema": [
            {
                "name": field.name,
                "types": sorted(str(value) for value in column_types[field.name]),
                "nullable": field.nullable,
                "unit": _column_unit(field.name),
            }
            for field in schema
        ],
        "schema_variants": list(schema_variants.values()),
        "frame_status_counts": frame_status_counts,
        "filter_decision_counts": filter_decision_counts,
        "census": census_record,
        "metadata_files": [_artifact_record(path, output_dir) for path in metadata_paths],
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "opencv": cv2.__version__,
            "pyarrow": pyarrow.__version__,
            "scipy": importlib.metadata.version("scipy"),
            "torch": importlib.metadata.version("torch"),
            "ultralytics": importlib.metadata.version("ultralytics"),
        },
        "files": file_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR)
    parser.add_argument("--census", type=Path, default=config.FRAME_CENSUS_PARQUET)
    parser.add_argument("--census-summary", type=Path, default=config.FRAME_CENSUS_SUMMARY)
    parser.add_argument("--filter-summary", type=Path, default=config.FRAME_FILTER_SUMMARY)
    parser.add_argument(
        "--source-artifact",
        type=Path,
        action="append",
        default=[],
        help="Input artifact to hash; repeat for each raw archive or source table.",
    )
    parser.add_argument("--without-census", action="store_true")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest or args.output_dir / "dataset_manifest.json"
    manifest = build_manifest(
        args.output_dir,
        None if args.without_census else args.census,
        None if args.without_census else args.census_summary,
        None if args.without_census else args.filter_summary,
        args.source_artifact,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
