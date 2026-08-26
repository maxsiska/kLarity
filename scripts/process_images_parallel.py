#!/usr/bin/env python3
"""
Parallel batch driver for image processing.

Design: a fixed pool of worker processes, each pinned to a
GPU (round-robin over ``DEVICES``) and holding ONE YOLO model, pulls whole replicates
from a queue submitted *slowest-first* (densest fluid first) so long replicates start
early and do not form a tail. Each worker runs ``klarity.parsing.process_image`` per frame.

The bottleneck is the per-bubble geometry loop (CPU), not inference, so parallelism is
across PROCESSES with one math-library thread each (see the env caps at the top), and a
few models feed many CPU cores. GPU capacity is resolved at launch.

Robustness:
  * blank/black acquisition frames are skipped (parsing.is_blank_frame);
  * per-image failures are captured to a durable per-worker JSONL and retried ONCE
    (after ``torch.cuda.empty_cache``) before the replicate's Parquet is written;
  * Parquet is written atomically (parsing.save_to_parquet -> tmp + os.replace), so a
    killed/OOM-ed worker never leaves a truncated file;
  * resume-safe: replicates whose Parquet already exists are skipped
    (parsing.check_if_processed).

Config comes from config.py (paths, inference parameters, and blank threshold). Runtime knobs
via CLI or env: ``--devices``/KLARITY_DEVICES (e.g. "cuda:3" or "cuda:0,cuda:3"),
``--workers``/KLARITY_NUM_WORKERS. Run from the project root:

    python scripts/process_images_parallel.py --devices cuda:3 --workers 16
    python scripts/process_images_parallel.py --retry-failures   # retry failed replicates
    python scripts/process_images_parallel.py --list             # dry run: list, no work
"""

from __future__ import annotations

# Single-thread the native math libraries BEFORE numpy/torch/cv2 are imported, so N
# worker processes do not each spawn ``cores`` BLAS/OpenMP threads and thrash the CPU.
# Geometry is parallelised across processes (one core per worker), not within a worker.
import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import argparse
import hashlib
import json
import multiprocessing as mp
import re
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

import cv2
from tqdm import tqdm

import config
from klarity import parsing

# ---------------------------------------------------------------------------
# Configuration (config.py, overridable per invocation via CLI / env)
# ---------------------------------------------------------------------------
# Paths default to config.py but may be overridden per-run via env (which, unlike a
# Python-level monkeypatch, propagates to spawn-started workers that re-import config).
IMAGE_ROOT_DIR = os.environ.get("KLARITY_IMAGE_DIR", str(config.IMAGE_DIR))
OUTPUT_DIR = os.environ.get("KLARITY_OUTPUT_DIR", str(config.OUTPUT_DIR))
MODEL_PATH = str(config.MODEL_PATH)
OVERLAYS_PATH = (
    os.environ.get("KLARITY_OVERLAYS_DIR", str(config.OVERLAYS_DIR) if config.OVERLAYS_DIR else "")
    or None
)
# Overlays default off for batch processing; override per run via the environment.
OVERLAY_MODE = os.environ.get("KLARITY_OVERLAY_MODE", getattr(config, "OVERLAY_MODE", "none"))
PROCESSING = config.PROCESSING_CONFIG
CONF = PROCESSING.confidence
IOU = PROCESSING.iou
MASK_THR = PROCESSING.mask_threshold
CONF_LARGE = PROCESSING.large_mask_confidence
LARGE_FRAC = PROCESSING.large_mask_fraction
BLANK_THR = PROCESSING.blank_mean_threshold

FAILURES_FILENAME = "failed_images.jsonl"


def _parse_devices(raw):
    """Return a list of torch device strings to round-robin workers over.

    ``raw`` (CLI/env) wins; otherwise derive from config.DEVICE which may be an int,
    list[int], "mps", or "cpu". The literal ``"auto"`` is passed through as a sentinel and
    resolved later by ``_auto_select_devices`` (it needs the requested worker count).
    """
    if raw:
        return [d.strip() for d in str(raw).split(",") if d.strip()]
    dev = config.DEVICE
    if isinstance(dev, bool):  # guard: bool is an int subclass
        dev = int(dev)
    if isinstance(dev, int):
        return [f"cuda:{dev}"]
    if isinstance(dev, (list, tuple)):
        return [f"cuda:{int(d)}" for d in dev]
    return [str(dev)]  # "mps" | "cpu" | "auto"


def _query_gpu_memory():
    """Return ``[(index, free_MiB, total_MiB), ...]`` from nvidia-smi, or ``[]``.

    Used to pick the least-loaded card(s) at launch on the shared production cluster,
    where other tenants' jobs (vLLM/Airflow) come and go and a hardcoded GPU can be full.
    """
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception:
        return []
    if out.returncode != 0:
        return []
    stats = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[0].isdigit():
            try:
                stats.append((int(parts[0]), int(float(parts[1])), int(float(parts[2]))))
            except ValueError:
                continue
    return stats


def _format_gpu_table(stats):
    """Human-readable free/total memory table for the run log and error messages."""
    lines = ["    GPU   free_MiB   total_MiB"]
    for idx, free, total in stats:
        lines.append(f"    {idx:>3}   {free:>8}   {total:>9}")
    return "\n".join(lines)


def _auto_select_devices(want_workers, min_free_mb, per_worker_mb):
    """Pick the least-loaded CUDA device(s) with room, capping workers to fit memory.

    Returns ``(assignment, n_workers, error)``. ``assignment`` is a capacity-weighted,
    round-robin device list of length ``n_workers`` so that worker rank ``r`` pins to
    ``assignment[r]`` (``_init_worker`` indexes ``devices[rank % len]``) without
    oversubscribing any card: each GPU appears at most ``free // per_worker_mb`` times, and
    slots are dealt round-robin over the qualifying GPUs (most-free first) so load spreads
    across cards. On failure returns ``(None, 0, message)``.
    """
    stats = _query_gpu_memory()
    if not stats:
        return None, 0, "nvidia-smi unavailable or returned no GPUs"
    qualifying = [(idx, free) for idx, free, _ in stats if free >= min_free_mb]
    if not qualifying:
        return (
            None,
            0,
            f"no CUDA device has >= {min_free_mb} MiB free:\n{_format_gpu_table(stats)}",
        )
    order = [idx for idx, _ in sorted(qualifying, key=lambda t: -t[1])]
    capacity = {idx: max(1, free // per_worker_mb) for idx, free in qualifying}
    assignment = []
    while sum(capacity.values()) > 0 and len(assignment) < want_workers:
        for idx in order:
            if capacity[idx] > 0:
                assignment.append(f"cuda:{idx}")
                capacity[idx] -= 1
                if len(assignment) >= want_workers:
                    break
    return assignment, len(assignment), None


# ---------------------------------------------------------------------------
# Per-worker globals (set once in _init_worker; reused across replicate tasks)
# ---------------------------------------------------------------------------
_MODEL = None
_DEVICE = None
_FAIL_PATH = None


def _init_worker(devices, model_path, output_dir, counter, lock):
    """Load one model per worker, pinned to a device by the worker's launch order.

    ``counter``/``lock`` are Manager proxies (picklable under the spawn start method) used
    to hand each worker a distinct rank so devices round-robin evenly.
    """
    global _MODEL, _DEVICE, _FAIL_PATH
    import torch

    torch.set_num_threads(1)
    cv2.setNumThreads(1)

    with lock:
        rank = counter.value
        counter.value = rank + 1

    _DEVICE = devices[rank % len(devices)]
    # Load with a short retry: on the shared cluster another tenant can momentarily spike a
    # card's memory exactly as we call model.to(device). A worker that dies here takes the
    # whole ProcessPool down (BrokenProcessPool), so absorb transient CUDA OOM before giving
    # up. A genuinely full card still fails after the retries (auto-selection avoids those).
    last_exc = None
    for attempt in range(3):
        try:
            _MODEL = parsing.load_yolo_model(model_path, device=_DEVICE)
            last_exc = None
            break
        except Exception as exc:  # noqa: BLE001 - re-raised below if all attempts fail
            last_exc = exc
            if _DEVICE.startswith("cuda"):
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            time.sleep(2.0 + 3.0 * attempt)
    if last_exc is not None:
        raise last_exc
    _FAIL_PATH = os.path.join(output_dir, f"_failures_worker_{os.getpid()}.jsonl")

    reserved = ""
    if _DEVICE.startswith("cuda"):
        try:
            reserved = f", torch reserved {torch.cuda.memory_reserved(_DEVICE) / 1e9:.2f} GB"
        except Exception:
            pass
    print(f"[worker {rank}] pid {os.getpid()} model on {_DEVICE}{reserved}", flush=True)


def should_generate_overlay(image_index, total_images, mode):
    """Whether to render an overlay for image ``image_index`` under ``mode``."""
    if mode == "all":
        return True
    if mode == "none":
        return False
    if mode == "every_10th":
        return (image_index + 1) % 10 == 0
    if mode == "every_5th":
        return (image_index + 1) % 5 == 0
    if mode == "first_only":
        return image_index == 0
    if mode == "first_last":
        return image_index == 0 or image_index == total_images - 1
    return False


def _fail_record(placement, setting, replicate, image, path, err_type, msg, tb=""):
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "placement": placement,
        "reactor_setting": setting,
        "replicate": replicate,
        "image_filename": image,
        "image_path": path,
        "error_type": err_type,
        "error": str(msg)[:500],
        "traceback": tb[-2000:] if tb else "",
    }


def _run_one_image(path, overlay_dir):
    """Run the canonical per-image pipeline. Raises on failure."""
    from klarity.geometry import pixel_size_mm as ps

    return parsing.process_image(
        path,
        _MODEL,
        conf=CONF,
        iou=IOU,
        binarize_thr=MASK_THR,
        overlay_dir=overlay_dir,
        save_masks_overlay=False,
        save_fit_overlay=bool(overlay_dir),
        pixel_size_mm=ps,
        geom_mode=PROCESSING.geometry_mode,
        sphere_if_aspect_tol=PROCESSING.sphere_aspect_tolerance,
        show_axes=False,
        conf_large=CONF_LARGE,
        large_frac=LARGE_FRAC,
    )


def _process_replicate(rep):
    """Worker task: process one replicate end to end; write its Parquet atomically.

    Returns a small summary dict (no bubble data crosses the process boundary). Blank
    frames are skipped; per-image failures are captured, retried once, and the residual
    are appended to this worker's failure JSONL.
    """
    placement = rep["placement"]
    setting = rep["setting"]
    replicate = rep["replicate"]
    replicate_path = rep["path"]

    images = sorted(
        [f for f in os.listdir(replicate_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))],
        key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)],
    )
    zero_xanthan = parsing._is_zero_xanthan(setting)

    overlay_base = None
    if OVERLAYS_PATH and OVERLAY_MODE != "none":
        overlay_base = os.path.join(OVERLAYS_PATH, placement, setting, replicate)
        os.makedirs(overlay_base, exist_ok=True)

    def burst_meta(pos):
        # Burst indexing follows POSITION among non-blank frames (deterministic, retry-safe).
        # 000-xanthan is acquired in 50-frame bursts; other fluids are continuous.
        if zero_xanthan:
            return (pos // 50) + 1, (pos % 50) + 1
        return 0, pos + 1

    def attach(bubbles, name, pos, has_overlay, out):
        b_index, num_in_burst = burst_meta(pos)
        for b in bubbles:
            b["placement"] = placement
            b["reactor_setting"] = setting
            b["replicate"] = replicate
            b["image_filename"] = name
            b["burst_index"] = b_index
            b["image_number_in_burst"] = num_in_burst
            b["has_overlay"] = has_overlay
            out.append(b)

    bubble_data = []
    failures = []
    to_retry = []
    valid_pos = 0
    blank_skipped = 0
    unreadable = 0

    # ---- first pass ----
    for idx, name in enumerate(images):
        path = os.path.join(replicate_path, name)
        im0 = cv2.imread(path)
        if im0 is None:
            unreadable += 1
            failures.append(
                _fail_record(
                    placement,
                    setting,
                    replicate,
                    name,
                    path,
                    "unreadable",
                    "cv2.imread returned None",
                )
            )
            continue
        if parsing.is_blank_frame(im0, BLANK_THR):
            blank_skipped += 1
            continue
        pos = valid_pos  # this frame's position among non-blank frames
        valid_pos += 1
        has_overlay = should_generate_overlay(idx, len(images), OVERLAY_MODE)
        overlay_dir = overlay_base if has_overlay else None
        try:
            attach(_run_one_image(path, overlay_dir), name, pos, has_overlay, bubble_data)
        except Exception:
            to_retry.append((name, path, pos, has_overlay, overlay_dir))

    # ---- retry-once (handles transient CUDA OOM under GPU contention) ----
    if to_retry:
        try:
            import torch

            if _DEVICE.startswith("cuda"):
                torch.cuda.empty_cache()
        except Exception:
            pass
        for name, path, pos, has_overlay, overlay_dir in to_retry:
            try:
                attach(_run_one_image(path, overlay_dir), name, pos, has_overlay, bubble_data)
            except Exception as exc:
                failures.append(
                    _fail_record(
                        placement,
                        setting,
                        replicate,
                        name,
                        path,
                        type(exc).__name__,
                        exc,
                        traceback.format_exc(),
                    )
                )

    # ---- durable failure log (per-worker file, merged by main at the end) ----
    if failures and _FAIL_PATH:
        with open(_FAIL_PATH, "a") as f:
            for rec in failures:
                f.write(json.dumps(rec) + "\n")

    n_frames = len({b["image_filename"] for b in bubble_data})
    if bubble_data:
        parsing.save_to_parquet(bubble_data, placement, setting, replicate, OUTPUT_DIR)

    return {
        "rep": f"{placement}/{setting}/{replicate}",
        "images": len(images),
        "frames": n_frames,
        "bubbles": len(bubble_data),
        "blank": blank_skipped,
        "unreadable": unreadable,
        "failed": len(failures),
    }


# ---------------------------------------------------------------------------
# Discovery / scheduling / manifest
# ---------------------------------------------------------------------------
def discover_replicates(only_reps=None):
    """List replicates under IMAGE_ROOT_DIR that still need processing (resume-safe).

    ``only_reps`` (set of (placement, setting, replicate)) restricts to those, ignoring
    the already-processed check -- used by --retry-failures.
    """
    reps = []
    # Paths use the on-disk names; the key, the stored metadata and the output filename
    # all use the normalized form, so the (placement, setting, replicate) tuple written to
    # the failure log matches the one rebuilt here on --retry-failures.
    for placement_dir in sorted(os.listdir(IMAGE_ROOT_DIR)):
        p_path = os.path.join(IMAGE_ROOT_DIR, placement_dir)
        if not os.path.isdir(p_path):
            continue
        placement = parsing.normalize_metadata_value(placement_dir)
        for setting_dir in sorted(os.listdir(p_path)):
            s_path = os.path.join(p_path, setting_dir)
            if not os.path.isdir(s_path):
                continue
            setting = parsing.normalize_metadata_value(setting_dir)
            for replicate_dir in sorted(os.listdir(s_path)):
                r_path = os.path.join(s_path, replicate_dir)
                if not os.path.isdir(r_path):
                    continue
                replicate = parsing.normalize_metadata_value(replicate_dir)
                key = (placement, setting, replicate)
                if only_reps is not None:
                    if key not in only_reps:
                        continue
                elif parsing.check_if_processed(placement, setting, replicate, OUTPUT_DIR):
                    continue
                reps.append(
                    {
                        "placement": placement,
                        "setting": setting,
                        "replicate": replicate,
                        "path": r_path,
                    }
                )
    return reps


def cost_key(rep):
    """Longest-processing-time-first: densest fluid first so slow replicates start early.

    The geometry loop cost scales with bubble count, which is highest for the most
    viscous/dense fluid. Order: non-water before water, higher xanthan % first.
    """
    setting = rep["setting"]
    water = parsing._is_zero_xanthan(setting)
    m = re.search(r"(\d+)\s*xanthan", setting)
    xanthan = int(m.group(1)) if m else (0 if water else 999)
    return (1 if water else 0, -xanthan)


def _git(*args):
    try:
        return subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            timeout=10,
        ).stdout.strip()
    except Exception:
        return "unavailable"


def _repo_version():
    """Return (git_commit, git_branch, git_dirty) for the manifest.

    Uses git when a checkout is present; when the code was shipped as an scp bundle
    (no .git dir) it falls back to BUILD_INFO.json written at bundle-build time, so the
    manifest still records exactly which commit produced the outputs.
    """
    commit = _git("rev-parse", "HEAD")
    if commit and commit != "unavailable":
        return (
            commit,
            _git("rev-parse", "--abbrev-ref", "HEAD"),
            bool(_git("status", "--porcelain")),
        )
    info = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "BUILD_INFO.json"
    )
    try:
        with open(info) as f:
            bi = json.load(f)
        return (
            bi.get("git_commit", "unavailable"),
            bi.get("git_branch", "unavailable"),
            bool(bi.get("git_dirty", True)),
        )
    except Exception:
        return "unavailable", "unavailable", False


def write_manifest(devices, num_workers, n_reps):
    """Record everything needed to reproduce this run next to the Parquet outputs."""
    import numpy as np
    import pandas
    import pyarrow
    import scipy
    import torch
    import ultralytics

    model_hash = "unavailable"
    try:
        with open(MODEL_PATH, "rb") as f:
            model_hash = hashlib.sha256(f.read()).hexdigest()
    except Exception:
        pass

    commit, branch, dirty = _repo_version()
    entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "driver": "process_images_parallel.py",
        "git_commit": commit,
        "git_branch": branch,
        "git_dirty": dirty,
        "model_path": MODEL_PATH,
        "model_sha256": model_hash,
        "image_root": IMAGE_ROOT_DIR,
        "conf": CONF,
        "iou": IOU,
        "mask_binarize_thr": MASK_THR,
        "conf_large": CONF_LARGE,
        "large_frac": LARGE_FRAC,
        "size_gate_active": CONF_LARGE > CONF,
        "blank_frame_mean_thresh": BLANK_THR,
        "overlay_mode": OVERLAY_MODE,
        "devices": devices,
        "num_workers": num_workers,
        "replicates_this_run": n_reps,
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "pandas": pandas.__version__,
            "pyarrow": pyarrow.__version__,
            "scipy": scipy.__version__,
        },
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "processing_manifest.json")
    entries = []
    if os.path.exists(path):
        try:
            with open(path) as f:
                entries = json.load(f)
        except Exception:
            entries = []
    entries.append(entry)
    with open(path, "w") as f:
        json.dump(entries, f, indent=2)
    dirty = " DIRTY" if entry["git_dirty"] else ""
    print(
        f"Manifest -> {path} (git {entry['git_commit'][:9]}{dirty}, size_gate_active={entry['size_gate_active']})"
    )


def merge_failure_logs():
    """Concatenate per-worker failure JSONLs into failed_images.jsonl; return the count."""
    merged = os.path.join(OUTPUT_DIR, FAILURES_FILENAME)
    parts = [
        os.path.join(OUTPUT_DIR, f)
        for f in os.listdir(OUTPUT_DIR)
        if f.startswith("_failures_worker_") and f.endswith(".jsonl")
    ]
    n = 0
    if parts:
        with open(merged, "a") as out:
            for part in parts:
                with open(part) as f:
                    for line in f:
                        out.write(line)
                        n += 1
                os.remove(part)
    return n


def load_failed_replicates():
    """Read failed_images.jsonl into replicate identifiers for retry."""
    path = os.path.join(OUTPUT_DIR, FAILURES_FILENAME)
    reps = set()
    if not os.path.exists(path):
        return reps
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
                reps.add((r["placement"], r["reactor_setting"], r["replicate"]))
            except Exception:
                continue
    return reps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_pool(reps, devices, num_workers, fastest_first=False):
    """Process ``reps`` across a spawn-based process pool; return aggregate totals."""
    reps.sort(key=cost_key, reverse=fastest_first)
    n_workers = max(1, min(num_workers, len(reps)))
    print(f"Launching {n_workers} worker(s) over devices {devices} for {len(reps)} replicate(s)")
    write_manifest(devices, n_workers, len(reps))

    # spawn is CUDA-safe (fork after CUDA init corrupts the context); Manager proxies are
    # picklable under spawn so the rank counter survives into each worker.
    ctx = mp.get_context("spawn")
    mgr = ctx.Manager()
    counter = mgr.Value("i", 0)
    lock = mgr.Lock()

    totals = {
        "frames": 0,
        "bubbles": 0,
        "blank": 0,
        "unreadable": 0,
        "failed": 0,
        "reps_done": 0,
        "reps_failed": 0,
    }
    t0 = time.time()
    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(devices, MODEL_PATH, OUTPUT_DIR, counter, lock),
    ) as ex:
        futures = {ex.submit(_process_replicate, r): r for r in reps}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Replicates"):
            rep = futures[fut]
            try:
                s = fut.result()
                for k in ("frames", "bubbles", "blank", "unreadable", "failed"):
                    totals[k] += s[k]
                totals["reps_done"] += 1
                if s["blank"] or s["failed"] or s["unreadable"]:
                    tqdm.write(
                        f"  {s['rep']}: {s['bubbles']:,} bubbles / {s['frames']} frames"
                        f" | blank {s['blank']} | unreadable {s['unreadable']} | failed {s['failed']}"
                    )
            except Exception as exc:
                totals["reps_failed"] += 1
                tqdm.write(
                    f"  REPLICATE FAILED {rep['placement']}/{rep['setting']}/{rep['replicate']}: "
                    f"{type(exc).__name__}: {exc}"
                )

    n_failed = merge_failure_logs()
    dt = (time.time() - t0) / 60.0
    print(
        f"\nDone in {dt:.1f} min | reps_done={totals['reps_done']} reps_failed={totals['reps_failed']}"
        f" | frames={totals['frames']:,} bubbles={totals['bubbles']:,}"
        f" | blank_skipped={totals['blank']:,} unreadable={totals['unreadable']}"
        f" | failed_images={n_failed}"
    )
    if n_failed or totals["reps_failed"]:
        print(
            f"  Failures logged to {os.path.join(OUTPUT_DIR, FAILURES_FILENAME)}"
            f" -- retry with:  python scripts/process_images_parallel.py --retry-failures"
        )
    return totals


def main():
    ap = argparse.ArgumentParser(description="Parallel image-processing driver.")
    ap.add_argument(
        "--devices",
        default=os.environ.get("KLARITY_DEVICES"),
        help="Comma list of torch devices, e.g. 'cuda:3' or 'cuda:0,cuda:3'.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("KLARITY_NUM_WORKERS", str(min(4, os.cpu_count() or 1)))),
        help="Number of worker processes.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N replicates (after slowest-first sort) -- dry runs.",
    )
    ap.add_argument(
        "--list", action="store_true", help="List replicates that would be processed, then exit."
    )
    ap.add_argument(
        "--retry-failures",
        action="store_true",
        help="Retry replicates listed in failed_images.jsonl (replaces their Parquet).",
    )
    ap.add_argument(
        "--fastest-first",
        action="store_true",
        help="Order fastest (water) replicates first instead of slowest-first.",
    )
    ap.add_argument(
        "--min-free-mb",
        type=int,
        default=2000,
        help="With --devices auto: minimum free VRAM (MiB) for a GPU to be used at all.",
    )
    ap.add_argument(
        "--per-worker-mb",
        type=int,
        default=1800,
        help="With --devices auto: VRAM (MiB) budgeted per worker; caps workers per GPU to free//this.",
    )
    args = ap.parse_args()

    raw_devices = _parse_devices(args.devices)
    is_auto = len(raw_devices) == 1 and raw_devices[0].lower() == "auto"

    print("=" * 72)
    print("PARALLEL BUBBLE IMAGE PROCESSING")
    print(f"  images : {IMAGE_ROOT_DIR}")
    print(f"  output : {OUTPUT_DIR}")
    print(f"  model  : {MODEL_PATH}")
    print(
        f"  params : conf={CONF} iou={IOU} conf_large={CONF_LARGE} large_frac={LARGE_FRAC}"
        f" (size_gate_active={CONF_LARGE > CONF}) blank_thr={BLANK_THR}"
    )
    print(f"  devices: {raw_devices} | workers: {args.workers}")
    print("=" * 72)

    if args.retry_failures:
        only = load_failed_replicates()
        if not only:
            print("No failures logged -- nothing to retry.")
            return
        print(f"Retry mode: {len(only)} replicate(s) had failed images; replacing their Parquet.")
        for placement, setting, replicate in only:
            # Remove every accepted spelling so retry cannot leave duplicate streams.
            for pq in parsing.existing_parquet_paths(placement, setting, replicate, OUTPUT_DIR):
                os.remove(pq)
        # start a clean failure log for the retry pass
        fpath = os.path.join(OUTPUT_DIR, FAILURES_FILENAME)
        if os.path.exists(fpath):
            os.replace(fpath, fpath + ".prev")
        reps = discover_replicates(only_reps=only)
    else:
        reps = discover_replicates()

    if args.limit is not None:
        reps.sort(key=cost_key, reverse=args.fastest_first)
        reps = reps[: args.limit]

    if not reps:
        print("All replicates already processed -- nothing to do.")
        return

    if args.list:
        reps.sort(key=cost_key)
        print(f"{len(reps)} replicate(s) would be processed (slowest-first):")
        for r in reps:
            print(f"  {r['placement']} / {r['setting']} / {r['replicate']}")
        return

    # Resolve devices only now that we know work exists. With --devices auto we probe
    # nvidia-smi and pin workers to the least-loaded card(s), capping the worker count to
    # what their free VRAM holds; a saturated cluster fails loudly (exit 2) rather than
    # OOM-killing the pool mid-run.
    if is_auto:
        stats = _query_gpu_memory()
        if stats:
            print("GPU memory (nvidia-smi):")
            print(_format_gpu_table(stats))
        devices, workers, err = _auto_select_devices(
            args.workers, args.min_free_mb, args.per_worker_mb
        )
        if err:
            print(f"ERROR: auto device selection failed: {err}")
            print(
                "  Wait for a GPU to free up, lower --min-free-mb, or pass"
                " --devices cuda:N explicitly."
            )
            sys.exit(2)
        from collections import Counter

        summary = ", ".join(f"{dev}x{cnt}" for dev, cnt in sorted(Counter(devices).items()))
        print(f"Auto-selected {workers} worker(s): {summary}")
    else:
        devices, workers = raw_devices, args.workers

    run_pool(reps, devices, workers, fastest_first=args.fastest_first)


if __name__ == "__main__":
    main()
