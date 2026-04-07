"""
Model evaluation script.

Tunes detection parameters on the validation set, then evaluates the
held-out test set and writes benchmark metrics and visualisations.

Run from the project root with:
    python scripts/evaluate_model.py

All paths are configured in config.py.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from klarity import parsing
from klarity.geometry import pixel_size_mm

TUNING_DIR = config.EVAL_DATASET_DIR / "valid" / "images"
TUNING_ANNOTATIONS = config.EVAL_DATASET_DIR / "valid" / "labels"
TEST_DIR = config.EVAL_DATASET_DIR / "test" / "images"
TEST_ANNOTATIONS = config.EVAL_DATASET_DIR / "test" / "labels"

MODELS = {
    "v2": config.MODEL_PATH,
}

# Tuned parameters (skips validation-set search when a model appears here)
DEFAULT_PARAMS = {
    "v2": {
        "conf": config.CONF,
        "iou": config.IOU,
        "mask_thr": config.MASK_THR,
        "geom_mode": "hybrid",
    },
}

BENCHMARK_OUTPUT_DIR = config.BENCHMARK_OUT_DIR
BENCHMARK_OVERLAYS_DIR = BENCHMARK_OUTPUT_DIR / "overlays"
BENCHMARK_METRICS_DIR = BENCHMARK_OUTPUT_DIR / "metrics"

for dir_path in [BENCHMARK_OUTPUT_DIR, BENCHMARK_OVERLAYS_DIR, BENCHMARK_METRICS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)


# ============================================================
# Ground Truth Utilities
# ============================================================


def parse_image_metadata(filename):
    """
    Parse image filename to extract placement, rpm, lmin, and xanthan concentration.

    Returns:
        dict with 'placement', 'rpm', 'lmin', and 'xanthan' keys
    """
    import re

    name = Path(filename).stem

    # Placement
    placement_match = re.search(r"place(?:ment)?[_-]?(\d+)", name, re.IGNORECASE)
    placement = f"place_{placement_match.group(1)}" if placement_match else "unknown"

    # Xanthan
    xanthan = "unknown"
    xanthan_match = re.search(r"(\d+)-(\d+)-xanthan", name, re.IGNORECASE)
    if xanthan_match:
        whole = xanthan_match.group(1)
        decimal = xanthan_match.group(2)
        xanthan = f"{whole}.{decimal}"
        try:
            xanthan_float = float(xanthan)
            if xanthan_float == 0.0:
                xanthan = "0.00"
            elif xanthan_float == 0.125:
                xanthan = "0.125"
            elif xanthan_float == 0.25:
                xanthan = "0.25"
            else:
                xanthan = f"{xanthan_float:.3f}"
        except ValueError:
            xanthan = "unknown"

    # RPM and L/min
    rpm = None
    lmin = None
    rpm_match = re.search(r"(\d+)[_-]?rpm", name, re.IGNORECASE)
    lmin_match = re.search(r"(\d+)[_-]?l[_-]?min", name, re.IGNORECASE)
    if rpm_match:
        rpm = int(rpm_match.group(1))
    if lmin_match:
        lmin = int(lmin_match.group(1))

    return {
        "placement": placement,
        "rpm": rpm,
        "lmin": lmin,
        "xanthan": xanthan,
        "filename": filename,
    }


def load_yolo_annotations(label_path, img_width, img_height):
    """Load YOLO segmentation format annotations from Roboflow."""
    annotations = []

    if not os.path.exists(label_path):
        return annotations

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue

            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]

            polygon = []
            for i in range(0, len(coords), 2):
                if i + 1 < len(coords):
                    x_norm = coords[i]
                    y_norm = coords[i + 1]
                    polygon.append((x_norm * img_width, y_norm * img_height))

            if len(polygon) < 3:
                continue

            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]

            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)

            annotations.append(
                {
                    "class_id": class_id,
                    "bbox": [x1, y1, x2, y2],
                    "x_center": (x1 + x2) / 2,
                    "y_center": (y1 + y2) / 2,
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "polygon": polygon,
                }
            )

    return annotations


def calculate_iou(bbox1, bbox2):
    """Calculate IoU between two bounding boxes in xyxy format."""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i < x1_i or y2_i < y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def match_predictions_to_gt(predictions, ground_truth, iou_threshold=0.5):
    """Match predictions to ground truth annotations using IoU."""
    gt_matched = [False] * len(ground_truth)
    pred_matched = [False] * len(predictions)
    matches = []

    for pred_idx, pred in enumerate(predictions):
        pred_bbox = [pred["bbox_x1"], pred["bbox_y1"], pred["bbox_x2"], pred["bbox_y2"]]

        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(ground_truth):
            if gt_matched[gt_idx]:
                continue
            iou = calculate_iou(pred_bbox, gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            matches.append(
                {
                    "pred_idx": pred_idx,
                    "gt_idx": best_gt_idx,
                    "iou": best_iou,
                    "pred": pred,
                    "gt": ground_truth[best_gt_idx],
                }
            )
            pred_matched[pred_idx] = True
            gt_matched[best_gt_idx] = True

    return {
        "matches": matches,
        "true_positives": len(matches),
        "false_positives": sum(1 for m in pred_matched if not m),
        "false_negatives": sum(1 for m in gt_matched if not m),
        "pred_matched": pred_matched,
        "gt_matched": gt_matched,
    }


# ============================================================
# Model Evaluation
# ============================================================


def evaluate_model_with_params(
    model, images, params, annotations_dir, overlay_dir=None, save_overlays=False
):
    """
    Evaluate a model on a set of images with given parameters.

    Parameters
    ----------
    model : YOLO model
    images : list of Path
        Images to evaluate on.
    params : dict
        Detection parameters (conf, iou, mask_thr, geom_mode).
    annotations_dir : Path or str
        Directory containing the YOLO label .txt files for this split.
    overlay_dir : str or None
        If provided together with save_overlays=True, save fit overlays here.
    save_overlays : bool
        Whether to write overlay images.
    """
    conf = params["conf"]
    iou = params["iou"]
    mask_thr = params["mask_thr"]
    geom_mode = params["geom_mode"]
    annotations_dir = Path(annotations_dir)

    all_metrics = []
    detection_stats = {
        "total_tp": 0,
        "total_fp": 0,
        "total_fn": 0,
        "total_gt": 0,
        "total_pred": 0,
    }

    size_bins = {
        "tiny": (0, 10),
        "small": (10, 50),
        "medium": (50, 150),
        "large": (150, 500),
        "xlarge": (500, 9999),
    }

    size_stats = {size: {"tp": 0, "fn": 0, "gt": 0} for size in size_bins}
    placement_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "gt": 0})
    xanthan_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "gt": 0})

    for img_path in images:
        img_path = Path(img_path)

        metadata = parse_image_metadata(img_path.name)
        placement = metadata["placement"]
        rpm = metadata["rpm"]
        lmin = metadata["lmin"]
        xanthan = metadata["xanthan"]

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        # Load ground truth from the correct annotations directory
        label_path = annotations_dir / (img_path.stem + ".txt")
        gt_annotations = load_yolo_annotations(str(label_path), img_w, img_h)

        # Run inference
        bubbles = parsing.process_image(
            str(img_path),
            model,
            conf=conf,
            iou=iou,
            binarize_thr=mask_thr,
            overlay_dir=str(overlay_dir) if (overlay_dir and save_overlays) else None,
            save_masks_overlay=False,
            save_fit_overlay=save_overlays,
            pixel_size_mm=pixel_size_mm,
            geom_mode=geom_mode,
            show_axes=False,
        )

        matching_result = match_predictions_to_gt(bubbles, gt_annotations, iou_threshold=0.5)

        tp = matching_result["true_positives"]
        fp = matching_result["false_positives"]
        fn = matching_result["false_negatives"]

        detection_stats["total_tp"] += tp
        detection_stats["total_fp"] += fp
        detection_stats["total_fn"] += fn
        detection_stats["total_gt"] += len(gt_annotations)
        detection_stats["total_pred"] += len(bubbles)

        placement_stats[placement]["tp"] += tp
        placement_stats[placement]["fp"] += fp
        placement_stats[placement]["fn"] += fn
        placement_stats[placement]["gt"] += len(gt_annotations)

        xanthan_stats[xanthan]["tp"] += tp
        xanthan_stats[xanthan]["fp"] += fp
        xanthan_stats[xanthan]["fn"] += fn
        xanthan_stats[xanthan]["gt"] += len(gt_annotations)

        # Per-image size stats
        img_size_stats = {size: {"tp": 0, "fn": 0, "gt": 0} for size in size_bins}

        for match in matching_result["matches"]:
            gt = match["gt"]
            gt_eq_diameter = 2 * np.sqrt((gt["width"] * gt["height"]) / np.pi)
            for size_name, (min_d, max_d) in size_bins.items():
                if min_d <= gt_eq_diameter < max_d:
                    size_stats[size_name]["tp"] += 1
                    size_stats[size_name]["gt"] += 1
                    img_size_stats[size_name]["tp"] += 1
                    img_size_stats[size_name]["gt"] += 1
                    break

        for i, is_matched in enumerate(matching_result["gt_matched"]):
            if not is_matched:
                gt = gt_annotations[i]
                gt_eq_diameter = 2 * np.sqrt((gt["width"] * gt["height"]) / np.pi)
                for size_name, (min_d, max_d) in size_bins.items():
                    if min_d <= gt_eq_diameter < max_d:
                        size_stats[size_name]["fn"] += 1
                        size_stats[size_name]["gt"] += 1
                        img_size_stats[size_name]["fn"] += 1
                        img_size_stats[size_name]["gt"] += 1
                        break

        img_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        img_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        img_f1 = (
            2 * img_precision * img_recall / (img_precision + img_recall)
            if (img_precision + img_recall) > 0
            else 0.0
        )

        img_metrics = {
            "image": img_path.name,
            "placement": placement,
            "rpm": rpm,
            "lmin": lmin,
            "xanthan": xanthan,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "gt_count": len(gt_annotations),
            "pred_count": len(bubbles),
            "precision": img_precision,
            "recall": img_recall,
            "f1_score": img_f1,
        }
        for size_name in ["tiny", "small", "medium", "large", "xlarge"]:
            s = img_size_stats[size_name]
            img_metrics[f"{size_name}_tp"] = s["tp"]
            img_metrics[f"{size_name}_fn"] = s["fn"]
            img_metrics[f"{size_name}_gt"] = s["gt"]
            img_metrics[f"{size_name}_recall"] = s["tp"] / s["gt"] if s["gt"] > 0 else None

        all_metrics.append(img_metrics)

    # Overall
    tp = detection_stats["total_tp"]
    fp = detection_stats["total_fp"]
    fn = detection_stats["total_fn"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Size metrics
    size_metrics = {}
    for size_name, stats in size_stats.items():
        size_metrics[size_name] = {
            "recall": stats["tp"] / stats["gt"] if stats["gt"] > 0 else 0.0,
            "tp": stats["tp"],
            "fn": stats["fn"],
            "gt": stats["gt"],
        }

    # Placement metrics
    placement_metrics = {}
    for placement, stats in placement_stats.items():
        if stats["gt"] > 0:
            placement_metrics[placement] = {
                "recall": stats["tp"] / stats["gt"],
                "precision": (
                    stats["tp"] / (stats["tp"] + stats["fp"])
                    if (stats["tp"] + stats["fp"]) > 0
                    else 0
                ),
                "tp": stats["tp"],
                "fp": stats["fp"],
                "fn": stats["fn"],
                "gt": stats["gt"],
            }

    # Xanthan metrics
    xanthan_metrics = {}
    for xanthan, stats in xanthan_stats.items():
        if stats["gt"] > 0:
            xanthan_metrics[xanthan] = {
                "recall": stats["tp"] / stats["gt"],
                "precision": (
                    stats["tp"] / (stats["tp"] + stats["fp"])
                    if (stats["tp"] + stats["fp"]) > 0
                    else 0
                ),
                "tp": stats["tp"],
                "fp": stats["fp"],
                "fn": stats["fn"],
                "gt": stats["gt"],
            }

    return {
        "params": params,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "total_tp": tp,
        "total_fp": fp,
        "total_fn": fn,
        "total_gt": detection_stats["total_gt"],
        "total_pred": detection_stats["total_pred"],
        "per_image_metrics": all_metrics,
        "size_metrics": size_metrics,
        "placement_metrics": placement_metrics,
        "xanthan_metrics": xanthan_metrics,
    }


def quick_tune_model(model, tuning_images, model_name):
    """
    Quick parameter tuning on the VALIDATION (tuning) set.

    Returns best_params, tuning_result (on validation), and tuning_history.
    """
    print(f"\n{'=' * 70}")
    print(f"TUNING MODEL: {model_name}  (on validation set, {len(tuning_images)} images)")
    print("=" * 70)

    tuning_history = []

    # ── Verification step ────────────────────────────────────
    print("\n>> Verifying max_det fix...")
    test_sample = tuning_images[:5]
    total_detections = 0
    total_gt = 0

    verification_overlay_dir = Path(BENCHMARK_OVERLAYS_DIR) / f"{model_name}_verification"
    verification_overlay_dir.mkdir(exist_ok=True, parents=True)

    print(f"   Saving verification overlays to: {verification_overlay_dir}")
    print(f"\n   {'Image':<30} {'Detected':<12} {'GT Count':<12} {'% Detected':<12}")
    print(f"   {'-'*66}")

    for img_path in test_sample:
        bubbles = parsing.process_image(
            str(img_path),
            model,
            conf=0.20,
            iou=0.80,
            binarize_thr=0.40,
            overlay_dir=str(verification_overlay_dir),
            save_masks_overlay=False,
            save_fit_overlay=True,
            pixel_size_mm=pixel_size_mm,
            geom_mode="hybrid",
            show_axes=False,
        )

        img = cv2.imread(str(img_path))
        img_h, img_w = img.shape[:2]
        label_path = Path(TUNING_ANNOTATIONS) / (img_path.stem + ".txt")
        gt_annotations = load_yolo_annotations(str(label_path), img_w, img_h)

        detected_count = len(bubbles)
        gt_count = len(gt_annotations)
        percent_detected = (detected_count / gt_count * 100) if gt_count > 0 else 0

        total_detections += detected_count
        total_gt += gt_count

        print(
            f"   {img_path.name:<30} {detected_count:<12} {gt_count:<12} {percent_detected:<12.1f}%"
        )

    avg_detections = total_detections / len(test_sample)
    avg_gt = total_gt / len(test_sample)
    overall_percent = (total_detections / total_gt * 100) if total_gt > 0 else 0

    print(f"   {'-'*66}")
    print(f"   {'AVERAGE':<30} {avg_detections:<12.1f} {avg_gt:<12.1f} {overall_percent:<12.1f}%")
    print(f"   {'TOTAL':<30} {total_detections:<12} {total_gt:<12}")

    print("\n   Key insights:")
    if avg_detections > 290 and avg_gt > 400:
        print("   OK Detecting >290 bubbles/image with GT showing >400 bubbles/image")
        print("   OK max_det fix appears to be working!")
    elif avg_detections < 305 and avg_gt > 350:
        print("   WARNING WARNING: Detecting ~300 bubbles but GT shows significantly more!")
        print("   WARNING This strongly suggests max_det=300 limit is still active")
        print("   WARNING Check that you're using the modified parsing.py with max_det=2500")
    elif overall_percent > 85:
        print("   OK Detecting >85% of GT bubbles - good detection rate!")
    else:
        print(f"   WARNING Only detecting {overall_percent:.1f}% of GT bubbles")
        print("   This could be due to: conf threshold too high, max_det limit, or model issues")

    print(f"\n   Review overlays in: {verification_overlay_dir}")

    # ── Stage 1: Tune CONF ───────────────────────────────────
    print("\nStage 1: Tuning CONF...")
    conf_values = [0.10, 0.12, 0.15, 0.18, 0.20, 0.22]
    best_conf_result = None
    best_conf_f1 = 0

    for conf in conf_values:
        params = {"conf": conf, "iou": 0.80, "mask_thr": 0.40, "geom_mode": "hybrid"}
        result = evaluate_model_with_params(model, tuning_images, params, TUNING_ANNOTATIONS)
        print(
            f"  CONF={conf:.2f}: F1={result['f1_score']:.3f}, P={result['precision']:.3f}, R={result['recall']:.3f}"
        )

        tuning_history.append(
            {
                "stage": "conf",
                "conf": conf,
                "iou": 0.80,
                "mask_thr": 0.40,
                "f1_score": result["f1_score"],
                "precision": result["precision"],
                "recall": result["recall"],
                "tp": result["total_tp"],
                "fp": result["total_fp"],
                "fn": result["total_fn"],
            }
        )

        if result["f1_score"] > best_conf_f1:
            best_conf_f1 = result["f1_score"]
            best_conf_result = result

    best_conf = best_conf_result["params"]["conf"]
    print(f"OK Best CONF: {best_conf} (F1={best_conf_f1:.3f})")

    # ── Stage 2: Tune IOU ────────────────────────────────────
    print("\nStage 2: Tuning IOU...")
    iou_values = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    best_iou_result = None
    best_iou_f1 = 0

    for iou in iou_values:
        params = {"conf": best_conf, "iou": iou, "mask_thr": 0.40, "geom_mode": "hybrid"}
        result = evaluate_model_with_params(model, tuning_images, params, TUNING_ANNOTATIONS)
        print(
            f"  IOU={iou:.2f}: F1={result['f1_score']:.3f}, P={result['precision']:.3f}, R={result['recall']:.3f}"
        )

        tuning_history.append(
            {
                "stage": "iou",
                "conf": best_conf,
                "iou": iou,
                "mask_thr": 0.40,
                "f1_score": result["f1_score"],
                "precision": result["precision"],
                "recall": result["recall"],
                "tp": result["total_tp"],
                "fp": result["total_fp"],
                "fn": result["total_fn"],
            }
        )

        if result["f1_score"] > best_iou_f1:
            best_iou_f1 = result["f1_score"]
            best_iou_result = result

    best_iou = best_iou_result["params"]["iou"]
    print(f"OK Best IOU: {best_iou} (F1={best_iou_f1:.3f})")

    # MASK_THR: fixed default
    mask_thr = 0.30
    print(f"\n>> Using fixed MASK_THR: {mask_thr}")
    print("   (MASK_THR doesn't affect detection metrics, only mask pixel accuracy)")

    best_params = {
        "conf": best_conf,
        "iou": best_iou,
        "mask_thr": mask_thr,
        "geom_mode": "hybrid",
    }

    # Final validation-set result with best params (for reference)
    tuning_result = evaluate_model_with_params(
        model, tuning_images, best_params, TUNING_ANNOTATIONS
    )

    print(
        f"\nOK {model_name} VALIDATION: F1={tuning_result['f1_score']:.3f}, "
        f"P={tuning_result['precision']:.3f}, R={tuning_result['recall']:.3f}"
    )

    _print_breakdown(tuning_result)

    return best_params, tuning_result, tuning_history


def _print_breakdown(result):
    """Print size / placement / xanthan breakdown for a result."""

    print("\n   Performance by bubble size:")
    print(f"   {'Size':<12} {'Recall':<10} {'TP':<8} {'FN':<8} {'GT':<8}")
    print(f"   {'-'*48}")
    for size_name in ["tiny", "small", "medium", "large", "xlarge"]:
        sm = result["size_metrics"].get(size_name, {})
        if sm.get("gt", 0) > 0:
            print(
                f"   {size_name.capitalize():<12} {sm['recall']:<10.3f} {sm['tp']:<8} {sm['fn']:<8} {sm['gt']:<8}"
            )

    print("\n   Performance by camera placement:")
    print(f"   {'Placement':<15} {'Recall':<10} {'Precision':<10} {'F1':<10} {'GT':<8}")
    print(f"   {'-'*58}")
    for placement in sorted(result.get("placement_metrics", {})):
        pm = result["placement_metrics"][placement]
        if pm["gt"] > 0:
            p_f1 = (
                2 * pm["precision"] * pm["recall"] / (pm["precision"] + pm["recall"])
                if (pm["precision"] + pm["recall"]) > 0
                else 0
            )
            print(
                f"   {placement:<15} {pm['recall']:<10.3f} {pm['precision']:<10.3f} {p_f1:<10.3f} {pm['gt']:<8}"
            )

    print("\n   Performance by xanthan concentration:")
    print(f"   {'Xanthan':<15} {'Recall':<10} {'Precision':<10} {'F1':<10} {'GT':<8}")
    print(f"   {'-'*58}")
    for xanthan in ["0.00", "0.125", "0.25", "unknown"]:
        if xanthan in result.get("xanthan_metrics", {}):
            xm = result["xanthan_metrics"][xanthan]
            if xm["gt"] > 0:
                x_f1 = (
                    2 * xm["precision"] * xm["recall"] / (xm["precision"] + xm["recall"])
                    if (xm["precision"] + xm["recall"]) > 0
                    else 0
                )
                print(
                    f"   {xanthan:<15} {xm['recall']:<10.3f} {xm['precision']:<10.3f} {x_f1:<10.3f} {xm['gt']:<8}"
                )


# ============================================================
# Visualization
# ============================================================


def create_benchmark_plots(results_dict, split_label="test"):
    """Create benchmark plots for multiple models."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Model Benchmark — {split_label} set", fontsize=16, fontweight="bold", y=1.01)

    model_names = list(results_dict.keys())
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # Plot 1: F1 Score
    f1_scores = [results_dict[name]["test_result"]["f1_score"] for name in model_names]
    axes[0, 0].bar(
        model_names, f1_scores, color=colors[: len(model_names)], alpha=0.7, edgecolor="black"
    )
    axes[0, 0].set_ylabel("F1 Score", fontsize=12)
    axes[0, 0].set_title("F1 Score Benchmark", fontsize=14, fontweight="bold")
    axes[0, 0].set_ylim([0, 1.0])
    axes[0, 0].grid(True, alpha=0.3, axis="y")
    for i, (name, score) in enumerate(zip(model_names, f1_scores)):
        axes[0, 0].text(
            i, score + 0.02, f"{score:.3f}", ha="center", fontsize=11, fontweight="bold"
        )

    # Plot 2: Precision vs Recall
    for i, name in enumerate(model_names):
        result = results_dict[name]["test_result"]
        axes[0, 1].scatter(
            result["recall"],
            result["precision"],
            s=200,
            c=colors[i],
            label=name,
            alpha=0.7,
            edgecolors="black",
            linewidths=2,
        )
    axes[0, 1].set_xlabel("Recall", fontsize=12)
    axes[0, 1].set_ylabel("Precision", fontsize=12)
    axes[0, 1].set_title("Precision vs Recall", fontsize=14, fontweight="bold")
    axes[0, 1].set_xlim([0, 1.0])
    axes[0, 1].set_ylim([0, 1.0])
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Detection counts
    x = np.arange(len(model_names))
    width = 0.25
    tp_counts = [results_dict[name]["test_result"]["total_tp"] for name in model_names]
    fp_counts = [results_dict[name]["test_result"]["total_fp"] for name in model_names]
    fn_counts = [results_dict[name]["test_result"]["total_fn"] for name in model_names]

    axes[1, 0].bar(x - width, tp_counts, width, label="True Positives", color="green", alpha=0.7)
    axes[1, 0].bar(x, fp_counts, width, label="False Positives", color="red", alpha=0.7)
    axes[1, 0].bar(x + width, fn_counts, width, label="False Negatives", color="orange", alpha=0.7)
    axes[1, 0].set_ylabel("Count", fontsize=12)
    axes[1, 0].set_title("Detection Counts", fontsize=14, fontweight="bold")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(model_names)
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    # Plot 4: Recall by bubble size
    size_order = ["tiny", "small", "medium", "large", "xlarge"]
    size_labels = [
        "Tiny\n(<50px)",
        "Small\n(50-100px)",
        "Medium\n(100-200px)",
        "Large\n(200-500px)",
        "X-Large\n(>500px)",
    ]
    x_sizes = np.arange(len(size_order))
    width_size = 0.35

    for i, name in enumerate(model_names):
        size_metrics = results_dict[name]["test_result"].get("size_metrics", {})
        recalls = [size_metrics.get(size, {}).get("recall", 0) for size in size_order]
        axes[0, 2].bar(
            x_sizes + i * width_size,
            recalls,
            width_size,
            label=name,
            color=colors[i],
            alpha=0.7,
            edgecolor="black",
        )

    axes[0, 2].set_ylabel("Recall", fontsize=12)
    axes[0, 2].set_xlabel("Bubble Size", fontsize=12)
    axes[0, 2].set_title("Recall by Bubble Size", fontsize=14, fontweight="bold")
    axes[0, 2].set_xticks(x_sizes + width_size * (len(model_names) - 1) / 2)
    axes[0, 2].set_xticklabels(size_labels, fontsize=9)
    axes[0, 2].set_ylim([0, 1.0])
    axes[0, 2].legend(fontsize=9)
    axes[0, 2].grid(True, alpha=0.3, axis="y")
    axes[0, 2].axhline(y=0.8, color="red", linestyle="--", linewidth=1, alpha=0.5)

    # Plot 5: Parameters summary
    axes[1, 1].axis("off")
    summary_text = "Best Parameters by Model\n\n"
    for name in model_names:
        params = results_dict[name]["best_params"]
        result = results_dict[name]["test_result"]
        summary_text += f"{name}:\n"
        summary_text += f"  CONF: {params['conf']:.2f}\n"
        summary_text += f"  IOU: {params['iou']:.2f}\n"
        summary_text += f"  MASK_THR: {params['mask_thr']:.2f}\n"
        summary_text += f"  F1: {result['f1_score']:.3f}  (test)\n"
        summary_text += f"  P:  {result['precision']:.3f}\n"
        summary_text += f"  R:  {result['recall']:.3f}\n\n"
    axes[1, 1].text(
        0.1,
        0.5,
        summary_text,
        fontsize=10,
        verticalalignment="center",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.5),
    )
    axes[1, 1].set_title("Best Parameters Summary", fontsize=14, fontweight="bold")

    # Plot 6: Size distribution
    axes[1, 2].axis("off")
    size_table_text = "Bubble Count by Size (test set)\n\n"
    for name in model_names:
        size_metrics = results_dict[name]["test_result"].get("size_metrics", {})
        size_table_text += f"{name}:\n"
        for size in size_order:
            sm = size_metrics.get(size, {})
            if sm.get("gt", 0) > 0:
                size_table_text += f"  {size.capitalize()}: {sm['gt']} bubbles\n"
        size_table_text += "\n"
    axes[1, 2].text(
        0.1,
        0.5,
        size_table_text,
        fontsize=10,
        verticalalignment="center",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.5),
    )
    axes[1, 2].set_title("Size Distribution", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(Path(BENCHMARK_METRICS_DIR) / "model_benchmark.png", dpi=300, bbox_inches="tight")
    print(f"\nOK Benchmark plots saved to {BENCHMARK_METRICS_DIR}/model_benchmark.png")


# ============================================================
# Main
# ============================================================


def run_model_benchmark(force_tune=False):
    """
    Main benchmark workflow.

    Parameters
    ----------
    force_tune : bool
        If True, always run parameter tuning on the validation set even when
        DEFAULT_PARAMS are available for a model.  Default is False (use
        stored defaults and skip tuning).
    """

    print("=" * 70)
    print("MODEL BENCHMARK")
    print("  Testing on: held-out test set")
    print("=" * 70)

    # Load image lists
    tuning_images = sorted(
        list(Path(TUNING_DIR).glob("*.jpg")) + list(Path(TUNING_DIR).glob("*.png"))
    )
    test_images = sorted(list(Path(TEST_DIR).glob("*.jpg")) + list(Path(TEST_DIR).glob("*.png")))

    print(f"\nFound {len(tuning_images)} validation (tuning) images")
    print(f"Found {len(test_images)} test images")

    if len(test_images) == 0:
        print("ERROR: No test images found!")
        return

    results_dict = {}

    # ── Step 1: Get parameters for each model ────────────────
    for model_name, model_path in MODELS.items():
        print(f"\nLoading {model_name}: {model_path}")

        if not model_path.exists():
            print(f"WARNING: Model not found at {model_path}, skipping...")
            continue

        model = parsing.load_yolo_model(str(model_path))

        if not force_tune and model_name in DEFAULT_PARAMS:
            # Use stored defaults — skip tuning
            best_params = DEFAULT_PARAMS[model_name]
            print(f"  Using stored default params: {best_params}")
            print("  (pass force_tune=True to re-tune on the validation set)")
            tuning_result = None
            tuning_history = []
        else:
            # Tune on validation set
            if len(tuning_images) == 0:
                print("ERROR: No validation images found for tuning!")
                continue
            print(f"  Tuning on validation set ({len(tuning_images)} images)...")
            best_params, tuning_result, tuning_history = quick_tune_model(
                model, tuning_images, model_name
            )

        results_dict[model_name] = {
            "model_path": str(model_path),
            "best_params": best_params,
            "tuning_result": tuning_result,
            "model": model,
            "tuning_history": tuning_history,
        }

    # ── Step 2: Evaluate on the held-out TEST set ────────────
    print("\n" + "=" * 70)
    print("EVALUATING ON HELD-OUT TEST SET")
    print("=" * 70)

    for model_name, data in results_dict.items():
        best_params = data["best_params"]
        model = data["model"]

        print(
            f"\n{model_name}: CONF={best_params['conf']}, IOU={best_params['iou']}, "
            f"MASK_THR={best_params['mask_thr']}"
        )

        overlay_dir = Path(BENCHMARK_OVERLAYS_DIR) / model_name
        overlay_dir.mkdir(exist_ok=True, parents=True)

        test_result = evaluate_model_with_params(
            model,
            test_images,
            best_params,
            TEST_ANNOTATIONS,
            overlay_dir=overlay_dir,
            save_overlays=True,
        )

        data["test_result"] = test_result

        print(
            f"  TEST  F1={test_result['f1_score']:.3f}, "
            f"P={test_result['precision']:.3f}, R={test_result['recall']:.3f}"
        )

        if data["tuning_result"] is not None:
            print(
                f"  VALID F1={data['tuning_result']['f1_score']:.3f}, "
                f"P={data['tuning_result']['precision']:.3f}, R={data['tuning_result']['recall']:.3f}"
            )
            delta_f1 = test_result["f1_score"] - data["tuning_result"]["f1_score"]
            print(f"  ΔF1 (test − valid) = {delta_f1:+.3f}")

        _print_breakdown(test_result)

        print(f"\n  OK Saved {len(test_images)} overlays to {overlay_dir}")

    # ── Step 3: Save results ─────────────────────────────────
    print("\n" + "=" * 70)
    print("SAVING BENCHMARK RESULTS")
    print("=" * 70)

    detailed_results = []

    for model_name, data in results_dict.items():
        params = data["best_params"]

        for split_name, result_key in [("validation", "tuning_result"), ("test", "test_result")]:
            result = data[result_key]
            if result is None:
                continue
            for img_metrics in result["per_image_metrics"]:
                detailed_results.append(
                    {
                        "model": model_name,
                        "split": split_name,
                        "conf": params["conf"],
                        "iou": params["iou"],
                        "mask_thr": params["mask_thr"],
                        **img_metrics,
                    }
                )

    df_detailed = pd.DataFrame(detailed_results)
    csv_path = Path(BENCHMARK_METRICS_DIR) / "detailed_results.csv"
    df_detailed.to_csv(csv_path, index=False)
    print(f"OK Detailed results saved to {csv_path}")

    # Save tuning history
    tuning_records = []
    for model_name, data in results_dict.items():
        for record in data["tuning_history"]:
            tuning_records.append({"model": model_name, **record})

    df_tuning = pd.DataFrame(tuning_records)
    tuning_csv_path = Path(BENCHMARK_METRICS_DIR) / "tuning_history.csv"
    df_tuning.to_csv(tuning_csv_path, index=False)
    print(f"OK Tuning history saved to {tuning_csv_path}")

    # Summary JSON
    summary = {}
    for model_name, data in results_dict.items():
        test_r = data["test_result"]
        val_r = data["tuning_result"]
        summary_entry = {
            "model_path": data["model_path"],
            "best_params": data["best_params"],
            "test_metrics": {
                "f1_score": float(test_r["f1_score"]),
                "precision": float(test_r["precision"]),
                "recall": float(test_r["recall"]),
                "total_tp": int(test_r["total_tp"]),
                "total_fp": int(test_r["total_fp"]),
                "total_fn": int(test_r["total_fn"]),
                "total_gt": int(test_r["total_gt"]),
                "total_pred": int(test_r["total_pred"]),
            },
            "overlay_directory": str(BENCHMARK_OVERLAYS_DIR / model_name),
        }
        if val_r is not None:
            summary_entry["validation_metrics"] = {
                "f1_score": float(val_r["f1_score"]),
                "precision": float(val_r["precision"]),
                "recall": float(val_r["recall"]),
            }
        summary[model_name] = summary_entry

    with open(Path(BENCHMARK_METRICS_DIR) / "benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Create benchmark plots (based on test-set metrics)
    create_benchmark_plots(results_dict, split_label="test")

    # ── Final table ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL BENCHMARK (test set)")
    print("=" * 70)

    print(
        f"\n{'Model':<10} {'F1':<8} {'Precision':<10} {'Recall':<10} {'TP':<8} {'FP':<8} {'FN':<8} {'(val F1)':<10}"
    )
    print("-" * 74)
    for model_name in results_dict:
        test_r = results_dict[model_name]["test_result"]
        val_r = results_dict[model_name]["tuning_result"]
        val_f1_str = f"({val_r['f1_score']:.3f})" if val_r is not None else "(n/a)"
        print(
            f"{model_name:<10} {test_r['f1_score']:<8.3f} {test_r['precision']:<10.3f} "
            f"{test_r['recall']:<10.3f} {test_r['total_tp']:<8} {test_r['total_fp']:<8} "
            f"{test_r['total_fn']:<8} {val_f1_str}"
        )

    best_model = max(results_dict.keys(), key=lambda k: results_dict[k]["test_result"]["f1_score"])
    print("\n" + "=" * 70)
    print(
        f"🏆 BEST MODEL: {best_model} (test F1={results_dict[best_model]['test_result']['f1_score']:.3f})"
    )
    print("=" * 70)

    print(f"\nRecommended parameters for {best_model}:")
    for key, value in results_dict[best_model]["best_params"].items():
        print(f"  {key.upper()}: {value}")

    print(f"\nResults saved to: {BENCHMARK_METRICS_DIR}/")
    print("Overlays saved to:")
    for model_name in results_dict:
        print(f"  {model_name}: {BENCHMARK_OVERLAYS_DIR}/{model_name}/")

    return results_dict


if __name__ == "__main__":
    results = run_model_benchmark()
