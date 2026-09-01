import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from klarity import metrics
from klarity.census import build_frame_census


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.full((8, 8, 3), value, dtype=np.uint8))


def test_census_classifies_frames_without_inference(tmp_path: Path):
    image_root = tmp_path / "images"
    replicate = image_root / "placement_1" / "100 rpm 45 lmin 000 xanthan" / "rep_1"
    _write_image(replicate / "frame1.jpg", 100)
    _write_image(replicate / "frame2.jpg", 100)
    _write_image(replicate / "frame3.jpg", 0)
    (replicate / "frame4.jpg").write_bytes(b"not an image")

    output = tmp_path / "output"
    output.mkdir()
    pd.DataFrame(
        {
            "placement": ["placement_1"],
            "reactor_setting": ["100 rpm 45 lmin 000 xanthan"],
            "replicate": ["rep_1"],
            "image_filename": ["frame1.jpg"],
        }
    ).to_parquet(output / "stream.parquet", index=False)

    census = build_frame_census(image_root, output, blank_mean_threshold=15.0)
    statuses = dict(zip(census["image_filename"], census["frame_status"]))
    assert statuses == {
        "frame1.jpg": "processed_with_detections",
        "frame2.jpg": "processed_zero_detections",
        "frame3.jpg": "blank",
        "frame4.jpg": "unreadable",
    }


def test_failure_status_takes_precedence_over_zero_detection(tmp_path: Path):
    image_root = tmp_path / "images"
    replicate = image_root / "placement_1" / "setting" / "rep_1"
    _write_image(replicate / "frame.jpg", 100)
    output = tmp_path / "output"
    output.mkdir()
    pd.DataFrame(
        {
            "placement": ["placement_1"],
            "reactor_setting": ["setting"],
            "replicate": ["rep_1"],
            "image_filename": ["another.jpg"],
        }
    ).to_parquet(output / "stream.parquet", index=False)
    failure_log = tmp_path / "failed_images.jsonl"
    failure_log.write_text(
        json.dumps(
            {
                "placement": "placement_1",
                "reactor_setting": "setting",
                "replicate": "rep_1",
                "image_filename": "frame.jpg",
            }
        )
        + "\n"
    )
    census = build_frame_census(
        image_root,
        output,
        blank_mean_threshold=15.0,
        failure_logs=[failure_log],
    )
    assert census.loc[0, "frame_status"] == "inference_failed"


def test_census_recovers_metadata_from_a_nested_archive_filename(tmp_path: Path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    filename = "0.125 xanthan 75 rpm 80 lmin rep 2 place 4.frame.jpg"
    ok, encoded = cv2.imencode(".jpg", np.full((8, 8, 3), 100, dtype=np.uint8))
    assert ok
    with zipfile.ZipFile(image_root / "images_pos_4.zip", "w") as archive:
        archive.writestr(f"75 rpm 80 lmin 0125 xanthan/rep_2/{filename}", encoded.tobytes())
        archive.writestr(
            f"75 rpm 70 lmin 0125 xanthan/rep_2/75 rpm 80 lmin rep 2/{filename}",
            encoded.tobytes(),
        )

    output = tmp_path / "output"
    output.mkdir()
    pd.DataFrame(
        {
            "placement": ["placement_4"],
            "reactor_setting": ["75 rpm 80 lmin 0125 xanthan"],
            "replicate": ["rep_2"],
            "image_filename": [filename],
        }
    ).to_parquet(output / "stream.parquet", index=False)

    census = build_frame_census(image_root, output, blank_mean_threshold=15.0)
    assert len(census) == 1
    assert census.loc[0, "reactor_setting"] == "75 rpm 80 lmin 0125 xanthan"
    assert census.loc[0, "replicate"] == "rep_2"
    assert census.loc[0, "frame_status"] == "processed_with_detections"


def test_attach_frame_census_retains_successful_zero_frame():
    computed = pd.DataFrame(
        {
            "placement": ["placement_1"],
            "reactor_setting": ["setting"],
            "replicate": ["rep_1"],
            "burst_index": [0],
            "image_number_in_burst": [1],
            "V_total_mm3": [2.0],
            "A_total_mm2": [3.0],
            "diameter_sum_mm": [1.0],
            "diameter_count": [1],
            "n_bubbles_total": [1],
        }
    )
    census = pd.DataFrame(
        {
            "placement": ["placement_1", "placement_1", "placement_1"],
            "reactor_setting": ["setting"] * 3,
            "replicate": ["rep_1"] * 3,
            "burst_index": [0, 0, pd.NA],
            "image_number_in_burst": [1, 2, pd.NA],
            "image_filename": ["one.jpg", "two.jpg", "blank.jpg"],
            "frame_status": [
                "processed_with_detections",
                "processed_zero_detections",
                "blank",
            ],
            "mean_intensity": [100.0, 100.0, 0.0],
            "raw_detection_count": [1, 0, 0],
        }
    )
    out = metrics.attach_frame_census(computed, census)
    assert out["image_filename"].tolist() == ["one.jpg", "two.jpg"]
    assert out["n_bubbles_total"].tolist() == [1.0, 0.0]
    assert out["V_total_mm3"].tolist() == [2.0, 0.0]


def test_attach_frame_census_normalizes_metadata_whitespace():
    computed = pd.DataFrame(
        {
            "placement": ["placement_1"],
            "reactor_setting": ["setting"],
            "replicate": [" rep_1"],
            "burst_index": [0],
            "image_number_in_burst": [1],
            "V_total_mm3": [2.0],
            "A_total_mm2": [3.0],
            "diameter_sum_mm": [1.0],
            "diameter_count": [1],
            "n_bubbles_total": [1],
        }
    )
    census = pd.DataFrame(
        {
            "placement": ["placement_1"],
            "reactor_setting": ["setting"],
            "replicate": ["rep_1"],
            "burst_index": [0],
            "image_number_in_burst": [1],
            "image_filename": ["one.jpg"],
            "frame_status": ["processed_with_detections"],
            "mean_intensity": [100.0],
            "raw_detection_count": [1],
        }
    )
    out = metrics.attach_frame_census(computed, census)
    assert out.loc[0, "V_total_mm3"] == 2.0
