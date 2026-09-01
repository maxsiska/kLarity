#!/usr/bin/env python3
"""
Batched image processing pipeline.

Run from the project root with:
    python scripts/process_images.py

All paths and parameters are set in config.py at the project root.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Direct execution sets sys.path[0] to scripts/, not the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
from tqdm import tqdm

import config
from klarity import parsing

IMAGE_ROOT_DIR = config.IMAGE_DIR
OUTPUT_DIR = config.OUTPUT_DIR
MODEL_PATH = config.MODEL_PATH
OVERLAYS_PATH = config.OVERLAYS_DIR
PROCESSING = config.PROCESSING_CONFIG
CONF = PROCESSING.confidence
IOU = PROCESSING.iou
MASK_THR = PROCESSING.mask_threshold
CONF_LARGE = PROCESSING.large_mask_confidence
LARGE_FRAC = PROCESSING.large_mask_fraction
BLANK_FRAME_MEAN_THRESH = PROCESSING.blank_mean_threshold
OVERLAY_MODE = config.OVERLAY_MODE
DEVICE = config.DEVICE


def write_processing_manifest(output_dir):
    """Write a reproducibility manifest next to the Parquet outputs.

    Records everything needed to reproduce this processing run: code version
    (git commit + dirty flag), model file hash, inference parameters, and key
    package versions. Each run start appends an entry so partially processed
    directories stay traceable.
    """
    import ultralytics

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

    model_hash = "unavailable"
    try:
        with open(MODEL_PATH, "rb") as f:
            model_hash = hashlib.sha256(f.read()).hexdigest()
    except Exception:
        pass

    entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "model_path": str(MODEL_PATH),
        "model_sha256": model_hash,
        "image_root": str(IMAGE_ROOT_DIR),
        "conf": CONF,
        "iou": IOU,
        "mask_binarize_thr": MASK_THR,
        "conf_large": CONF_LARGE,
        "large_frac": LARGE_FRAC,
        "size_gate_active": CONF_LARGE > CONF,
        "overlay_mode": OVERLAY_MODE,
        "device": str(DEVICE),
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
    }
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "processing_manifest.json")
    entries = []
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path) as f:
                entries = json.load(f)
        except Exception:
            entries = []
    entries.append(entry)
    with open(manifest_path, "w") as f:
        json.dump(entries, f, indent=2)
    print(
        f"Manifest -> {manifest_path} (git {entry['git_commit'][:9]}"
        f"{' DIRTY' if entry['git_dirty'] else ''})"
    )


def should_generate_overlay(image_index, total_images, mode):
    """Determine if overlay should be generated."""
    if mode == "all":
        return True
    elif mode == "none":
        return False
    elif mode == "every_10th":
        return (image_index + 1) % 10 == 0
    elif mode == "every_5th":
        return (image_index + 1) % 5 == 0
    elif mode == "first_only":
        return image_index == 0
    elif mode == "first_last":
        return image_index == 0 or image_index == total_images - 1
    else:
        return False


def process_replicate_single(
    placement,
    setting,
    replicate,
    replicate_path,
    model,
    conf,
    iou,
    binarize_thr,
    overlay_root,
    overlay_mode,
    device,
):
    """Process a single replicate - one image at a time (no batching)."""
    from klarity.geometry import pixel_size_mm as default_pixel_size_mm

    bubble_data = []

    # Get all images
    images = sorted(
        [
            img
            for img in os.listdir(replicate_path)
            if img.lower().endswith((".jpg", ".png", ".jpeg"))
        ],
        key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)],
    )

    if not images:
        return bubble_data

    ps = default_pixel_size_mm
    zero_xanthan = parsing._is_zero_xanthan(setting)
    valid_image_count = 0
    blank_skipped = 0

    # Prepare overlay directory
    base_overlay_dir = None
    if overlay_root and overlay_mode != "none":
        base_overlay_dir = os.path.join(overlay_root, placement, setting, replicate)
        os.makedirs(base_overlay_dir, exist_ok=True)

    overlay_count = sum(
        1 for i in range(len(images)) if should_generate_overlay(i, len(images), overlay_mode)
    )

    # Progress bar
    desc = f"{placement}/{setting}/{replicate} (overlays: {overlay_count}/{len(images)})"
    pbar = tqdm(total=len(images), desc=desc, leave=False, position=1)

    # Process images one at a time
    for img_idx, img_name in enumerate(images):
        img_path = os.path.join(replicate_path, img_name)

        # Load and validate image
        im0 = cv2.imread(img_path)
        if im0 is None:
            # Unreadable frame: skip without counting it as a valid image so the
            # 50-frame burst indexing stays aligned.
            pbar.update(1)
            continue
        if parsing.is_blank_frame(im0, BLANK_FRAME_MEAN_THRESH):
            # Blank/black acquisition frame (illumination off, ~one per burst): no
            # analyzable content, and the model would emit a spurious full-frame mask.
            # Skip without counting it as a valid image so burst indexing stays aligned.
            blank_skipped += 1
            pbar.update(1)
            continue

        generate_overlay = should_generate_overlay(img_idx, len(images), overlay_mode)
        overlay_dir = base_overlay_dir if generate_overlay else None

        try:
            # Process single image
            bubbles = parsing.process_image(
                img_path,
                model,
                conf=conf,
                iou=iou,
                binarize_thr=binarize_thr,
                overlay_dir=overlay_dir,
                save_masks_overlay=False,
                save_fit_overlay=True if overlay_dir else False,
                pixel_size_mm=ps,
                geom_mode=PROCESSING.geometry_mode,
                sphere_if_aspect_tol=PROCESSING.sphere_aspect_tolerance,
                show_axes=False,
                conf_large=CONF_LARGE,
                large_frac=LARGE_FRAC,
            )

            # Add metadata
            if zero_xanthan:
                burst_index = (valid_image_count // 50) + 1
                image_number_in_burst = (valid_image_count % 50) + 1
            else:
                burst_index = 0
                image_number_in_burst = valid_image_count + 1

            for bubble in bubbles:
                bubble["placement"] = placement
                bubble["reactor_setting"] = setting
                bubble["replicate"] = replicate
                bubble["image_filename"] = img_name
                bubble["burst_index"] = burst_index
                bubble["image_number_in_burst"] = image_number_in_burst
                bubble["has_overlay"] = generate_overlay
                bubble_data.append(bubble)

            valid_image_count += 1

        except Exception as e:
            print(f"\nWARNING Error processing {img_name}: {e}")

        pbar.update(1)

        # Clear cache periodically
        if img_idx % 50 == 0:
            if isinstance(device, str) and "," in device:
                for gpu_id in device.split(","):
                    with torch.cuda.device(int(gpu_id)):
                        torch.cuda.empty_cache()
            elif device not in ("cpu", "mps"):
                torch.cuda.empty_cache()

    pbar.close()
    if blank_skipped:
        print(
            f"  Skipped {blank_skipped} blank/black frame(s) in {placement}/{setting}/{replicate}"
        )
    return bubble_data


def process_all_settings():
    """Main processing function."""

    # Find replicates to process
    replicates_to_process = []
    placements = [
        p for p in os.listdir(IMAGE_ROOT_DIR) if os.path.isdir(os.path.join(IMAGE_ROOT_DIR, p))
    ]

    # Paths use the on-disk names; the stored metadata and output filenames use the
    # normalized form (parsing.normalize_metadata_value).
    for placement_dir in placements:
        placement_path = os.path.join(IMAGE_ROOT_DIR, placement_dir)
        placement = parsing.normalize_metadata_value(placement_dir)
        for setting_dir in os.listdir(placement_path):
            setting_path = os.path.join(placement_path, setting_dir)
            if not os.path.isdir(setting_path):
                continue
            setting = parsing.normalize_metadata_value(setting_dir)
            for replicate_dir in os.listdir(setting_path):
                replicate_path = os.path.join(setting_path, replicate_dir)
                if not os.path.isdir(replicate_path):
                    continue
                replicate = parsing.normalize_metadata_value(replicate_dir)
                if not parsing.check_if_processed(placement, setting, replicate, OUTPUT_DIR):
                    replicates_to_process.append(
                        {
                            "placement": placement,
                            "setting": setting,
                            "replicate": replicate,
                            "path": replicate_path,
                        }
                    )

    print(f"\n{'='*70}")
    print("PROCESSING CONFIGURATION")
    print(f"{'='*70}")
    print(f"Found {len(replicates_to_process)} replicates to process")
    print("Processing mode: Single image at a time (no batching)")
    print(f"Overlay mode: {OVERLAY_MODE}")
    print(f"{'='*70}\n")

    if len(replicates_to_process) == 0:
        print("OK All replicates already processed!")
        return

    # Record the run configuration before any processing starts
    write_processing_manifest(OUTPUT_DIR)

    # Resolve the device before constructing the model.
    if DEVICE == "auto":
        if torch.cuda.is_available():
            device = "0"
        else:
            device = "cpu"
    elif isinstance(DEVICE, list):
        # Convert list to comma-separated string for YOLO
        device = ",".join(str(d) for d in DEVICE)
    elif isinstance(DEVICE, int):
        device = str(DEVICE)
    else:
        device = DEVICE

    print("Loading model...")
    model_device = device if device in ("cpu", "mps") else f"cuda:{device.split(',')[0]}"
    model = parsing.load_yolo_model(MODEL_PATH, device=model_device)

    if device == "cpu":
        print("OK Using CPU")
    elif device == "mps":
        print("OK Using Apple MPS")
    else:
        # Get first GPU from device string
        first_gpu = device.split(",")[0]
        if "," in device:
            # Multi-GPU
            gpu_list = [int(g) for g in device.split(",")]
            print(f"OK Multi-GPU mode: Using GPUs {gpu_list}")
            for gpu_id in gpu_list:
                if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
                    gpu_name = torch.cuda.get_device_name(gpu_id)
                    gpu_mem = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9
                    print(f"  GPU {gpu_id}: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            # Single GPU
            gpu_id = int(first_gpu)
            gpu_name = torch.cuda.get_device_name(gpu_id)
            gpu_mem = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9
            print(f"OK Using GPU {gpu_id}: {gpu_name} ({gpu_mem:.1f} GB)")

    print("OK Ready to process\n")

    # Process replicates with HARDCODED batch size
    with tqdm(total=len(replicates_to_process), desc="Replicates", position=0) as outer_pbar:
        for rep in replicates_to_process:
            try:
                bubble_data = process_replicate_single(
                    placement=rep["placement"],
                    setting=rep["setting"],
                    replicate=rep["replicate"],
                    replicate_path=rep["path"],
                    model=model,
                    conf=CONF,
                    iou=IOU,
                    binarize_thr=MASK_THR,
                    overlay_root=OVERLAYS_PATH,
                    overlay_mode=OVERLAY_MODE,
                    device=device,
                )
            except RuntimeError as e:
                print(
                    f"\nERROR Error processing {rep['placement']}/{rep['setting']}/{rep['replicate']}"
                )
                print(f"   {e}")
                print("   Skipping this replicate...")
                outer_pbar.update(1)
                continue

            # Save Parquet
            if bubble_data:
                parsing.save_to_parquet(
                    bubble_data,
                    rep["placement"],
                    rep["setting"],
                    rep["replicate"],
                    OUTPUT_DIR,
                )

            outer_pbar.update(1)

    print(f"\n{'='*70}")
    print("OK Processing complete!")
    print(f"  Results: {OUTPUT_DIR}")
    if OVERLAYS_PATH:
        print(f"  Overlays: {OVERLAYS_PATH}")
    print(f"{'='*70}\n")


def main():
    """Entry point."""
    print("\n" + "=" * 70)
    print("BATCHED BUBBLE IMAGE PROCESSING")
    print("=" * 70)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"CUDA devices: {num_gpus}")
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {gpu_name} ({gpu_mem:.1f} GB)")

    print("=" * 70)

    start_time = time.time()

    try:
        process_all_settings()
    except KeyboardInterrupt:
        print("\n\nWARNING Interrupted by user")
        print("OK Progress saved - rerun to resume from where you left off")
    except Exception as e:
        print(f"\n\nERROR Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - start_time
    print(f"Total time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
