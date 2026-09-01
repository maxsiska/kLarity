"""
Unit tests for the parallel driver's per-replicate worker logic
(``scripts/process_images_parallel.py``), focused on the failure/retry path that the
local end-to-end smoke run cannot trigger deterministically.

The driver is a script (imports config + klarity at module scope), so it is loaded by
file path with the repo root on sys.path. ``_run_one_image`` and ``save_to_parquet`` are
monkeypatched so the test needs no model, GPU, or Parquet engine — it exercises only the
orchestration: blank-frame skip, per-image failure capture, retry-once recovery of a
transient failure, and the summary counts.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "process_images_parallel", ROOT / "scripts" / "process_images_parallel.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_all_processing_entry_points_share_canonical_parameters():
    import config
    from klarity import parsing
    from scripts import process_images

    parallel = _load_driver()
    processing = config.PROCESSING_CONFIG
    assert parsing.DEFAULT_CONF == process_images.CONF == parallel.CONF == processing.confidence
    assert parsing.DEFAULT_IOU == process_images.IOU == parallel.IOU == processing.iou
    assert (
        parsing.DEFAULT_CONF_LARGE
        == process_images.CONF_LARGE
        == parallel.CONF_LARGE
        == processing.large_mask_confidence
    )
    assert (
        parsing.DEFAULT_BLANK_MEAN_THRESH
        == process_images.BLANK_FRAME_MEAN_THRESH
        == parallel.BLANK_THR
        == processing.blank_mean_threshold
    )

    notebook = json.loads((ROOT / "notebooks" / "process_images.ipynb").read_text())
    source = "\n".join(
        (
            "".join(cell.get("source", []))
            if isinstance(cell.get("source"), list)
            else cell.get("source", "")
        )
        for cell in notebook["cells"]
    )
    for attribute in (
        "confidence",
        "iou",
        "mask_threshold",
        "large_mask_confidence",
        "large_mask_fraction",
        "blank_mean_threshold",
        "geometry_mode",
        "sphere_aspect_tolerance",
    ):
        assert f"processing.{attribute}" in source


@pytest.fixture
def drv():
    return _load_driver()


def _write_frame(path, value):
    cv2.imwrite(str(path), np.full((80, 100, 3), value, dtype=np.uint8))


def test_retry_recovers_transient_failure_and_logs_residual(drv, tmp_path, monkeypatch):
    """A frame that fails once then succeeds must be recovered by the retry-once pass;
    a frame that always fails must be logged; a black frame must be skipped."""
    rep_dir = tmp_path / "placement_1" / "100 rpm 45 lmin 025 xanthan" / "rep_1"
    rep_dir.mkdir(parents=True)
    _write_frame(rep_dir / "f000.jpg", 180)  # ok
    _write_frame(rep_dir / "f001.jpg", 180)  # transient: fails once, then succeeds
    _write_frame(rep_dir / "f002.jpg", 180)  # always fails
    _write_frame(rep_dir / "f003.jpg", 0)  # black -> skipped

    monkeypatch.setattr(drv, "_DEVICE", "cpu")
    monkeypatch.setattr(drv, "_FAIL_PATH", str(tmp_path / "fail.jsonl"))
    monkeypatch.setattr(drv, "OUTPUT_DIR", str(tmp_path / "out"))
    saved = {}
    monkeypatch.setattr(
        drv.parsing, "save_to_parquet", lambda data, *a, **k: saved.update(n=len(data))
    )

    calls = {}

    def fake_run_one_image(path, overlay_dir):
        name = os.path.basename(path)
        calls[name] = calls.get(name, 0) + 1
        if name == "f001.jpg" and calls[name] == 1:
            raise RuntimeError("transient CUDA OOM")
        if name == "f002.jpg":
            raise ValueError("degenerate mask")
        return [{"score": 0.9}]

    monkeypatch.setattr(drv, "_run_one_image", fake_run_one_image)

    rep = {
        "placement": "placement_1",
        "setting": "100 rpm 45 lmin 025 xanthan",
        "replicate": "rep_1",
        "path": str(rep_dir),
    }
    summary = drv._process_replicate(rep)

    assert summary["blank"] == 1  # the black frame
    assert summary["unreadable"] == 0
    assert summary["failed"] == 1  # only f002 survives as a failure
    assert summary["bubbles"] == 2  # f000 + recovered f001
    assert summary["frames"] == 2
    assert calls["f001.jpg"] == 2  # failed once, retried once
    assert saved["n"] == 2  # only the recovered bubbles are written

    # residual failure is durably logged with its identifying metadata
    lines = Path(drv._FAIL_PATH).read_text().strip().splitlines()
    assert len(lines) == 1
    assert "f002.jpg" in lines[0]
    assert "degenerate mask" in lines[0]


def test_no_retry_when_all_frames_succeed(drv, tmp_path, monkeypatch):
    """Clean replicate: no failure file is written and every frame is counted."""
    rep_dir = tmp_path / "placement_1" / "100 rpm 45 lmin 000 xanthan" / "rep_1"
    rep_dir.mkdir(parents=True)
    for i in range(4):
        _write_frame(rep_dir / f"f{i:03d}.jpg", 200)

    monkeypatch.setattr(drv, "_DEVICE", "cpu")
    monkeypatch.setattr(drv, "_FAIL_PATH", str(tmp_path / "fail.jsonl"))
    monkeypatch.setattr(drv, "OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(drv.parsing, "save_to_parquet", lambda *a, **k: None)
    monkeypatch.setattr(drv, "_run_one_image", lambda path, overlay_dir: [{"score": 0.5}])

    summary = drv._process_replicate(
        {
            "placement": "placement_1",
            "setting": "100 rpm 45 lmin 000 xanthan",
            "replicate": "rep_1",
            "path": str(rep_dir),
        }
    )
    assert summary["failed"] == 0
    assert summary["blank"] == 0
    assert summary["frames"] == 4
    assert summary["bubbles"] == 4
    assert not Path(drv._FAIL_PATH).exists()


def test_auto_select_picks_free_card_and_caps_workers(drv, monkeypatch):
    """Auto selection must skip a saturated GPU, prefer the emptiest card, deal slots
    round-robin (most-free first), and cap the worker count to the fitted VRAM."""
    # cuda:0 nearly full, cuda:1 ~4 GiB free, cuda:2 ~40 GiB free.
    monkeypatch.setattr(
        drv,
        "_query_gpu_memory",
        lambda: [(0, 50, 40000), (1, 4000, 40000), (2, 40000, 40000)],
    )
    # per_worker_mb=2000, min_free_mb=2000 -> cuda:1 fits 2, cuda:2 fits 20; want 16.
    assignment, n, err = drv._auto_select_devices(16, min_free_mb=2000, per_worker_mb=2000)
    assert err is None
    assert n == 16 and len(assignment) == 16
    assert "cuda:0" not in assignment  # saturated card excluded
    # emptiest card is dealt first each round; cuda:1's capacity (2) is exhausted early.
    assert assignment[0] == "cuda:2"
    assert assignment[1] == "cuda:1"
    assert assignment.count("cuda:1") == 2  # capped at free // per_worker_mb
    assert assignment.count("cuda:2") == 14


def test_auto_select_errors_when_all_cards_saturated(drv, monkeypatch):
    """When no card clears the free-VRAM floor, selection returns an error (caller exits)."""
    monkeypatch.setattr(drv, "_query_gpu_memory", lambda: [(0, 100, 40000), (1, 50, 40000)])
    assignment, n, err = drv._auto_select_devices(16, min_free_mb=2000, per_worker_mb=1800)
    assert assignment is None and n == 0
    assert "no CUDA device" in err


def test_auto_select_errors_when_nvidia_smi_unavailable(drv, monkeypatch):
    """No nvidia-smi (e.g. a CPU/Mac host) is a clean error, not a crash."""
    monkeypatch.setattr(drv, "_query_gpu_memory", lambda: [])
    assignment, n, err = drv._auto_select_devices(4, min_free_mb=2000, per_worker_mb=1800)
    assert assignment is None and n == 0
    assert "nvidia-smi" in err
