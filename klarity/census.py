"""Frame inventory and reconciliation for processed image collections."""

from __future__ import annotations

import json
import re
import zipfile
from collections import Counter
from collections.abc import Iterator
from itertools import islice
from pathlib import Path, PurePosixPath
from typing import TypeAlias

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
PROCESSED_STATUSES = {"processed_with_detections", "processed_zero_detections"}
FrameSource: TypeAlias = Path | tuple[Path, str]
FrameRecord: TypeAlias = tuple[str, str, str, str, FrameSource]
_REPLICATE_DIR = re.compile(r"^rep[_ ]?(\d+)$", re.IGNORECASE)
_FILENAME_METADATA = re.compile(
    r"(?P<xanthan>\d+(?:\.\d+)?)\s+xanthan\s+"
    r"(?P<rpm>\d+)\s+rpm\s+(?P<lmin>\d+)\s+lmin\s+rep\s*(?P<replicate>\d+)",
    re.IGNORECASE,
)


def _normalize(value: str) -> str:
    return " ".join(str(value).split())


def _parquet_stem(placement: str, setting: str, replicate: str) -> str:
    return "_".join(_normalize(value) for value in (placement, setting, replicate)).replace(
        " ", "_"
    )


def _existing_parquet_paths(output_dir: Path, key: tuple[str, str, str]) -> list[Path]:
    target = re.sub(r"_+", "_", _parquet_stem(*key))
    return sorted(
        path for path in output_dir.glob("*.parquet") if re.sub(r"_+", "_", path.stem) == target
    )


def _natural_key(value: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def _metadata_from_path(parts: tuple[str, ...]) -> tuple[str, str]:
    setting = _normalize(parts[-3])
    replicate = _normalize(parts[-2])
    if _REPLICATE_DIR.fullmatch(replicate):
        return setting, replicate.replace(" ", "_")

    match = _FILENAME_METADATA.search(parts[-1])
    if match is None:
        return setting, replicate
    xanthan = match.group("xanthan").replace(".", "")
    setting = f"{match.group('rpm')} rpm {match.group('lmin')} lmin {xanthan} xanthan"
    return setting, f"rep_{match.group('replicate')}"


def _zip_frames(image_root: Path) -> Iterator[FrameRecord]:
    for archive in sorted(image_root.glob("images_pos_*.zip"), key=lambda p: _natural_key(p.name)):
        match = re.search(r"pos_(\d+)", archive.stem)
        if match is None:
            continue
        placement = f"placement_{match.group(1)}"
        with zipfile.ZipFile(archive) as source:
            members = [
                name
                for name in source.namelist()
                if not name.endswith("/") and PurePosixPath(name).suffix.lower() in IMAGE_SUFFIXES
            ]
            members.sort(key=_natural_key)
            for name in members:
                parts = PurePosixPath(name).parts
                if len(parts) < 3:
                    continue
                setting, replicate = _metadata_from_path(parts)
                yield placement, setting, replicate, parts[-1], (archive, name)


def _directory_frames(image_root: Path) -> Iterator[FrameRecord]:
    for image_path in sorted(image_root.rglob("*"), key=lambda p: _natural_key(str(p))):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative = image_path.relative_to(image_root)
        if len(relative.parts) < 4:
            continue
        placement, setting, replicate = relative.parts[-4:-1]
        yield (
            _normalize(placement),
            _normalize(setting),
            _normalize(replicate),
            image_path.name,
            image_path,
        )


def iter_frames(image_root: Path) -> Iterator[FrameRecord]:
    """Yield normalized frame metadata and a lazy archive/directory source."""
    archives = list(image_root.glob("images_pos_*.zip"))
    yield from (_zip_frames(image_root) if archives else _directory_frames(image_root))


def _read_frame_source(
    source: FrameSource,
    open_archives: dict[Path, zipfile.ZipFile],
) -> bytes:
    if isinstance(source, Path):
        return source.read_bytes()
    archive_path, member = source
    archive = open_archives.get(archive_path)
    if archive is None:
        archive = zipfile.ZipFile(archive_path)
        open_archives[archive_path] = archive
    return archive.read(member)


def _processed_frames(
    output_dir: Path,
    restrict_keys: set[tuple[str, str, str]] | None = None,
) -> tuple[dict[tuple[str, str, str], Counter], set]:
    frames: dict[tuple[str, str, str], Counter] = {}
    completed: set[tuple[str, str, str]] = set()
    if restrict_keys is None:
        parquet_paths = sorted(output_dir.glob("*.parquet"))
    else:
        parquet_paths = sorted(
            {
                Path(path)
                for key in restrict_keys
                for path in _existing_parquet_paths(output_dir, key)
            }
        )
    for parquet_path in parquet_paths:
        available = set(pq.ParquetFile(parquet_path).schema.names)
        image_col = "image_filename" if "image_filename" in available else "image"
        metadata = pd.read_parquet(
            parquet_path,
            columns=["placement", "reactor_setting", "replicate", image_col],
        )
        if metadata.empty:
            continue
        key = (
            _normalize(metadata.iloc[0]["placement"]),
            _normalize(metadata.iloc[0]["reactor_setting"]),
            _normalize(metadata.iloc[0]["replicate"]),
        )
        completed.add(key)
        frames[key] = Counter(metadata[image_col].astype(str))
    return frames, completed


def _failed_frames(paths: list[Path]) -> set[tuple[str, str, str, str]]:
    failed: set[tuple[str, str, str, str]] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open(errors="replace") as source:
            for line in source:
                try:
                    record = json.loads(line)
                    failed.add(
                        (
                            _normalize(record["placement"]),
                            _normalize(record["reactor_setting"]),
                            _normalize(record["replicate"]),
                            str(record["image_filename"]),
                        )
                    )
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
    return failed


def build_frame_census(
    image_root: Path,
    output_dir: Path,
    *,
    blank_mean_threshold: float,
    failure_logs: list[Path] | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Classify each acquired frame without running model inference."""
    if limit is None:
        frame_records: Iterator[FrameRecord] = iter_frames(image_root)
        restrict_keys = None
    else:
        limited_records = list(islice(iter_frames(image_root), limit))
        frame_records = iter(limited_records)
        restrict_keys = {
            (placement, setting, replicate)
            for placement, setting, replicate, _, _ in limited_records
        }
    processed, completed = _processed_frames(output_dir, restrict_keys)
    failed = _failed_frames(failure_logs or [])
    rows: list[dict[str, object]] = []
    positions: Counter = Counter()
    raw_positions: Counter = Counter()
    open_archives: dict[Path, zipfile.ZipFile] = {}
    seen_frames: set[tuple[str, str, str, str]] = set()

    try:
        for index, (placement, setting, replicate, filename, source) in enumerate(frame_records):
            if limit is not None and index >= limit:
                break
            key = (placement, setting, replicate)
            frame_identity = (*key, filename)
            if frame_identity in seen_frames:
                continue
            seen_frames.add(frame_identity)
            raw_positions[key] += 1
            detection_count = processed.get(key, Counter()).get(filename, 0)
            mean_intensity = float("nan")
            if detection_count:
                status = "processed_with_detections"
                valid_frame = True
            else:
                encoded = _read_frame_source(source, open_archives)
                array = np.frombuffer(encoded, dtype=np.uint8)
                image = cv2.imdecode(array, cv2.IMREAD_COLOR)
                mean_intensity = float(image.mean()) if image is not None else float("nan")
                if image is None:
                    status = "unreadable"
                    valid_frame = False
                elif mean_intensity < blank_mean_threshold:
                    status = "blank"
                    valid_frame = False
                elif (*key, filename) in failed:
                    status = "inference_failed"
                    valid_frame = True
                elif key in completed:
                    status = "processed_zero_detections"
                    valid_frame = True
                else:
                    status = "unresolved"
                    valid_frame = True

            if valid_frame:
                valid_position = positions[key]
                positions[key] += 1
                if "000 xanthan" in setting.lower():
                    burst_index = valid_position // 50 + 1
                    image_number = valid_position % 50 + 1
                else:
                    burst_index = 0
                    image_number = valid_position + 1
            else:
                burst_index = pd.NA
                image_number = pd.NA

            rows.append(
                {
                    "placement": placement,
                    "reactor_setting": setting,
                    "replicate": replicate,
                    "image_filename": filename,
                    "raw_ordinal": raw_positions[key],
                    "burst_index": burst_index,
                    "image_number_in_burst": image_number,
                    "mean_intensity": mean_intensity,
                    "raw_detection_count": detection_count,
                    "frame_status": status,
                }
            )
    finally:
        for archive in open_archives.values():
            archive.close()

    census = pd.DataFrame(rows)
    if not census.empty:
        census["burst_index"] = census["burst_index"].astype("Int64")
        census["image_number_in_burst"] = census["image_number_in_burst"].astype("Int64")
    return census


def summarize_census(census: pd.DataFrame) -> pd.DataFrame:
    """Count frame statuses for each experimental stream."""
    return (
        census.groupby(
            ["placement", "reactor_setting", "replicate", "frame_status"],
            observed=True,
            dropna=False,
        )
        .size()
        .rename("frame_count")
        .reset_index()
    )
