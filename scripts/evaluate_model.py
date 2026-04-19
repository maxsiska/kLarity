"""
Model evaluation script.

Tunes detection parameters on the validation set, then evaluates the
held-out test set and writes benchmark metrics and visualisations.

Run from the project root with:
    python scripts/evaluate_model.py

All paths are configured in config.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

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


def rasterize_polygon(polygon, img_height, img_width):
    """Rasterize a polygon (list of (x, y) tuples) into a binary mask."""
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def calculate_mask_iou(mask1, mask2):
    """Calculate IoU between two boolean masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(intersection) / float(union) if union > 0 else 0.0


def get_predicted_masks(model, img_path, conf, iou, mask_thr):
    """
    Run YOLO inference and return a list of dicts with 'mask', 'score', 'bbox'
    for each detection.  Uses the same binarize + resize logic as parsing.py.
    """
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return [], 0, 0
    H, W = img_bgr.shape[:2]

    results = model.predict(
        img_bgr,
        conf=conf,
        iou=iou,
        max_det=2500,
        verbose=False,
    )
    det = results[0] if results else None

    out = []
    if det is None or det.masks is None:
        return out, H, W

    masks_np = det.masks.data.detach().cpu().numpy()
    confs = det.boxes.conf.detach().cpu().numpy() if det.boxes is not None else None
    boxes_xyxy = det.boxes.xyxy.detach().cpu().numpy() if det.boxes is not None else None

    for i, m in enumerate(masks_np):
        mb = (m > mask_thr).astype(np.uint8)
        mask = cv2.resize(mb, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        score = float(confs[i]) if confs is not None and i < len(confs) else 1.0
        bbox = None
        if boxes_xyxy is not None and i < len(boxes_xyxy):
            x1, y1, x2, y2 = boxes_xyxy[i].tolist()
            bbox = [float(x1), float(y1), float(x2), float(y2)]
        out.append({"mask": mask, "score": score, "bbox": bbox})

    return out, H, W


def compute_instance_ap(
    predictions,
    ground_truths,
    iou_fn,
    iou_threshold=0.5,
):
    """
    Compute Average Precision at a single IoU threshold (COCO-style 101-pt interpolation).

    This is the generic engine used by both mask AP and box AP.

    Parameters
    ----------
    predictions : list
        Each element is a dict with at least 'score' (float) and whatever
        the iou_fn needs (e.g. 'mask' for mask IoU, 'bbox' for box IoU).
    ground_truths : list
        Each element is a ground-truth item passed to iou_fn.
    iou_fn : callable(pred, gt) -> float
        Returns the IoU between a prediction and a ground truth.
    iou_threshold : float
        IoU threshold for a true positive.

    Returns
    -------
    float
        AP at the given IoU threshold (0.0–1.0).
    """
    if len(ground_truths) == 0 and len(predictions) == 0:
        return 1.0  # no GT, no preds → perfect
    if len(ground_truths) == 0:
        return 0.0  # all preds are FP
    if len(predictions) == 0:
        return 0.0  # all GT are FN

    # Sort predictions by descending confidence
    sorted_preds = sorted(predictions, key=lambda d: d["score"], reverse=True)

    gt_matched = [False] * len(ground_truths)
    tp = np.zeros(len(sorted_preds))
    fp = np.zeros(len(sorted_preds))

    for i, pred in enumerate(sorted_preds):
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(ground_truths):
            if gt_matched[gt_idx]:
                continue
            cur_iou = iou_fn(pred, gt)
            if cur_iou > best_iou:
                best_iou = cur_iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp[i] = 1
            gt_matched[best_gt_idx] = True
        else:
            fp[i] = 1

    # Cumulative sums → precision / recall at each detection
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / len(ground_truths)
    precision = tp_cum / (tp_cum + fp_cum)

    # 101-point interpolated AP (COCO style)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([1.0], precision, [0.0]))

    # Make precision monotonically decreasing (right to left)
    for j in range(len(precision) - 2, -1, -1):
        precision[j] = max(precision[j], precision[j + 1])

    # Sample at 101 recall thresholds
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p_at_r = precision[recall >= t]
        ap += p_at_r.max() if len(p_at_r) > 0 else 0.0
    ap /= 101.0

    return float(ap)


# ── Vectorised IoU matrix builders ───────────────────────────

_MASK_SCALE = 256  # downscale masks to this resolution for IoU computation


def compute_iou_matrix_mask(pred_dets, gt_masks):
    """
    Return (n_pred, n_gt) float32 mask-IoU matrix.

    Masks are downscaled to _MASK_SCALE × _MASK_SCALE before the matrix
    multiply, keeping memory and compute manageable regardless of image size.
    IoU values are accurate to within ~1% of the full-resolution result.
    """
    n_pred = len(pred_dets)
    n_gt = len(gt_masks)
    if n_pred == 0 or n_gt == 0:
        return np.zeros((n_pred, n_gt), dtype=np.float32)

    s = _MASK_SCALE
    pred_flat = np.stack(
        [
            cv2.resize(d["mask"].astype(np.uint8), (s, s), interpolation=cv2.INTER_NEAREST).ravel()
            for d in pred_dets
        ]
    ).astype(
        np.float32
    )  # (n_pred, s²)

    gt_flat = np.stack(
        [
            cv2.resize(m.astype(np.uint8), (s, s), interpolation=cv2.INTER_NEAREST).ravel()
            for m in gt_masks
        ]
    ).astype(
        np.float32
    )  # (n_gt, s²)

    intersection = pred_flat @ gt_flat.T  # (n_pred, n_gt)
    pred_areas = pred_flat.sum(axis=1, keepdims=True)
    gt_areas = gt_flat.sum(axis=1, keepdims=True)
    union = pred_areas + gt_areas.T - intersection
    return np.where(union > 0, intersection / union, 0.0).astype(np.float32)


def compute_iou_matrix_box(pred_dets, gt_boxes):
    """Return (n_pred, n_gt) float32 box-IoU matrix via vectorised numpy ops."""
    n_pred = len(pred_dets)
    n_gt = len(gt_boxes)
    if n_pred == 0 or n_gt == 0:
        return np.zeros((n_pred, n_gt), dtype=np.float32)

    pb = np.array([d["bbox"] for d in pred_dets], dtype=np.float32)  # (n_pred, 4)
    gb = np.array([g["bbox"] for g in gt_boxes], dtype=np.float32)  # (n_gt, 4)

    x1 = np.maximum(pb[:, 0:1], gb[:, 0])
    y1 = np.maximum(pb[:, 1:2], gb[:, 1])
    x2 = np.minimum(pb[:, 2:3], gb[:, 2])
    y2 = np.minimum(pb[:, 3:4], gb[:, 3])

    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    pred_areas = (pb[:, 2] - pb[:, 0]) * (pb[:, 3] - pb[:, 1])
    gt_areas = (gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1])
    union = pred_areas[:, None] + gt_areas[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def compute_instance_ap_from_matrix(iou_matrix, scores, n_gt, iou_threshold=0.5):
    """
    Compute AP at a single IoU threshold from a pre-computed (n_pred, n_gt) IoU matrix.

    Avoids recomputing IoU for each threshold — call once per image to build
    the matrix, then call this function once per threshold.
    """
    n_pred = len(scores)
    if n_gt == 0 and n_pred == 0:
        return 1.0
    if n_gt == 0 or n_pred == 0:
        return 0.0

    sorted_idx = np.argsort(-np.asarray(scores))
    iou_sorted = iou_matrix[sorted_idx]  # (n_pred, n_gt), sorted by descending score

    gt_matched = np.zeros(n_gt, dtype=bool)
    tp = np.zeros(n_pred)
    fp = np.zeros(n_pred)

    for i in range(n_pred):
        row = iou_sorted[i].copy()
        row[gt_matched] = -1.0  # mask already-matched GTs
        best_gt = int(np.argmax(row))
        if row[best_gt] >= iou_threshold:
            tp[i] = 1
            gt_matched[best_gt] = True
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / n_gt
    precision = tp_cum / (tp_cum + fp_cum)

    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([1.0], precision, [0.0]))
    for j in range(len(precision) - 2, -1, -1):
        precision[j] = max(precision[j], precision[j + 1])

    ap = (
        sum(
            (precision[recall >= t].max() if (recall >= t).any() else 0.0)
            for t in np.linspace(0, 1, 101)
        )
        / 101.0
    )
    return float(ap)


def _dataset_ap_image_aware(iou_matrices, scores_list, n_gt_list, iou_threshold):
    """
    Dataset-level AP with COCO-correct image-aware matching.

    Predictions from all images are pooled and sorted by confidence score, but
    each prediction can only be matched to ground truths from its own image.
    This avoids cross-image mask IoU computation while implementing the standard
    COCO pooled-AP protocol correctly.
    """
    all_entries = [
        (score, img_idx, pred_idx)
        for img_idx, scores in enumerate(scores_list)
        for pred_idx, score in enumerate(scores)
    ]
    if not all_entries:
        return 0.0

    total_gt = sum(n_gt_list)
    if total_gt == 0:
        return 0.0

    all_entries.sort(key=lambda x: -x[0])

    gt_matched = [np.zeros(n, dtype=bool) for n in n_gt_list]
    n_total = len(all_entries)
    tp = np.zeros(n_total)
    fp = np.zeros(n_total)

    for i, (_, img_idx, pred_idx) in enumerate(all_entries):
        mat = iou_matrices[img_idx]
        if mat.shape[1] == 0:
            fp[i] = 1
            continue
        row = mat[pred_idx].copy()
        row[gt_matched[img_idx]] = -1.0
        best_gt = int(np.argmax(row))
        if row[best_gt] >= iou_threshold:
            tp[i] = 1
            gt_matched[img_idx][best_gt] = True
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / total_gt
    precision = tp_cum / (tp_cum + fp_cum)

    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([1.0], precision, [0.0]))
    for j in range(len(precision) - 2, -1, -1):
        precision[j] = max(precision[j], precision[j + 1])

    ap = (
        sum(
            (precision[recall >= t].max() if (recall >= t).any() else 0.0)
            for t in np.linspace(0, 1, 101)
        )
        / 101.0
    )
    return float(ap)


def compute_ap_for_dataset(
    model,
    images,
    annotations_dir,
    params,
):
    """
    Compute mask AP and box AP (AP50, AP75, mAP50:95) over a full image set.

    Strategy
    --------
    Phase 1 — per image: run inference and build vectorised IoU matrices
    (mask and box) once per image. Re-use them across all 10 IoU thresholds
    to compute per-image AP values.

    Phase 2 — dataset level: use the cached per-image IoU matrices with
    image-aware pooling (COCO-correct: predictions are ranked globally by
    confidence but can only match GTs from their own image).  This avoids any
    cross-image mask computation.

    Returns
    -------
    dict with keys:
        mask_ap50, mask_ap75, mask_map50_95  : dataset-level mask AP
        box_ap50,  box_ap75,  box_map50_95   : dataset-level box AP
        mean_per_image_mask_ap50             : mean of per-image mask AP50
        mean_per_image_box_ap50              : mean of per-image box AP50
        per_image_mask_ap50                  : list of per-image mask AP50
        per_image_box_ap50                   : list of per-image box AP50
        per_threshold_mask_aps               : dict IoU -> mean per-image mask AP
        per_threshold_box_aps                : dict IoU -> mean per-image box AP
    """
    conf = params["conf"]
    iou = params["iou"]
    mask_thr = params["mask_thr"]
    annotations_dir = Path(annotations_dir)

    iou_thresholds = np.arange(0.50, 1.0, 0.05)  # [0.50, 0.55, ..., 0.95]

    per_threshold_mask_aps = {round(t, 2): [] for t in iou_thresholds}
    per_threshold_box_aps = {round(t, 2): [] for t in iou_thresholds}

    # Per-image IoU matrices cached for dataset-level AP
    all_mask_matrices = []
    all_box_matrices = []
    all_scores_list = []
    all_n_gt_mask = []
    all_n_gt_box = []

    n = len(images)
    print(f"\n  Phase 1/2 — inference + IoU matrix computation ({n} images)")
    print(f"  {'Img':<8} {'Name':<52} {'preds':<7} {'gt':<7} {'mAP50 mask':<14} {'mAP50 box'}")
    print(f"  {'-'*100}")

    for idx, img_path in enumerate(images):
        img_path = Path(img_path)
        label_path = annotations_dir / (img_path.stem + ".txt")

        pred_dets, img_h, img_w = get_predicted_masks(model, img_path, conf, iou, mask_thr)
        if img_h == 0:
            print(f"  {idx+1}/{n:<6} {img_path.name:<52} (skipped — could not read image)")
            continue

        gt_annotations = load_yolo_annotations(str(label_path), img_w, img_h)
        gt_masks = [rasterize_polygon(gt["polygon"], img_h, img_w) for gt in gt_annotations]
        gt_boxes = [{"bbox": gt["bbox"]} for gt in gt_annotations]

        scores = np.array([d["score"] for d in pred_dets]) if pred_dets else np.array([])

        # Build IoU matrices once per image
        mask_iou_mat = compute_iou_matrix_mask(pred_dets, gt_masks)
        box_iou_mat = compute_iou_matrix_box(pred_dets, gt_boxes)

        # Per-image AP at each threshold — reuse the matrices
        for t in iou_thresholds:
            t_key = round(t, 2)
            per_threshold_mask_aps[t_key].append(
                compute_instance_ap_from_matrix(mask_iou_mat, scores, len(gt_masks), t)
            )
            per_threshold_box_aps[t_key].append(
                compute_instance_ap_from_matrix(box_iou_mat, scores, len(gt_boxes), t)
            )

        # Cache for dataset-level phase
        all_mask_matrices.append(mask_iou_mat)
        all_box_matrices.append(box_iou_mat)
        all_scores_list.append(scores)
        all_n_gt_mask.append(len(gt_masks))
        all_n_gt_box.append(len(gt_boxes))

        img_mask_ap50 = per_threshold_mask_aps[0.5][-1]
        img_box_ap50 = per_threshold_box_aps[0.5][-1]
        print(
            f"  {idx+1}/{n:<6} {img_path.name:<52} {len(pred_dets):<7} {len(gt_annotations):<7} "
            f"{img_mask_ap50:<14.3f} {img_box_ap50:.3f}"
        )

    print(f"  {'-'*100}")

    # ── Phase 2: dataset-level AP (image-aware pooling) ──────

    n_thresh = len(iou_thresholds)
    total_calls = n_thresh * 2  # mask + box, no duplicate AP50/AP75 calls
    call_idx = [0]

    def _pooled_ap(mat_list, scores_list, n_gt_list, threshold, label):
        call_idx[0] += 1
        total_preds = sum(len(s) for s in scores_list)
        total_gt = sum(n_gt_list)
        print(
            f"  [{call_idx[0]:>2}/{total_calls}] {label} @ IoU={threshold:.2f}"
            f"  ({total_preds} preds / {total_gt} GT, image-aware) ...",
            end=" ",
            flush=True,
        )
        ap = _dataset_ap_image_aware(mat_list, scores_list, n_gt_list, threshold)
        print(f"AP={ap:.3f}")
        return ap

    print(f"\n  Phase 2/2 — dataset-level AP ({total_calls} computations)")
    print(f"  {'-'*70}")

    mask_ap_vals = {
        round(t, 2): _pooled_ap(all_mask_matrices, all_scores_list, all_n_gt_mask, t, "Mask AP")
        for t in iou_thresholds
    }
    box_ap_vals = {
        round(t, 2): _pooled_ap(all_box_matrices, all_scores_list, all_n_gt_box, t, "Box  AP")
        for t in iou_thresholds
    }

    mask_ap50 = mask_ap_vals[0.5]
    mask_ap75 = mask_ap_vals[0.75]
    mask_map50_95 = float(np.mean(list(mask_ap_vals.values())))

    box_ap50 = box_ap_vals[0.5]
    box_ap75 = box_ap_vals[0.75]
    box_map50_95 = float(np.mean(list(box_ap_vals.values())))

    per_image_mask_ap50 = per_threshold_mask_aps.get(0.5, [])
    per_image_box_ap50 = per_threshold_box_aps.get(0.5, [])

    # Per-image mean AP at every IoU threshold (used to be called
    # 'per_threshold_*_aps'; renamed for clarity now that dataset-level
    # per-threshold APs are also returned).
    per_image_mean_mask_aps = {
        k: float(np.mean(v)) if v else 0.0 for k, v in per_threshold_mask_aps.items()
    }
    per_image_mean_box_aps = {
        k: float(np.mean(v)) if v else 0.0 for k, v in per_threshold_box_aps.items()
    }

    # Dataset-level AP at every IoU threshold (already computed in
    # mask_ap_vals / box_ap_vals above; previously only 0.50, 0.75 and
    # the mean were exposed).
    dataset_mask_aps = {float(k): float(v) for k, v in mask_ap_vals.items()}
    dataset_box_aps = {float(k): float(v) for k, v in box_ap_vals.items()}

    return {
        # Mask metrics
        "mask_ap50": mask_ap50,
        "mask_ap75": mask_ap75,
        "mask_map50_95": mask_map50_95,
        "mean_per_image_mask_ap50": float(np.mean(per_image_mask_ap50))
        if per_image_mask_ap50
        else 0.0,
        "per_image_mask_ap50": per_image_mask_ap50,
        "per_image_mean_mask_aps": per_image_mean_mask_aps,
        "dataset_mask_aps": dataset_mask_aps,
        # Kept for backward compatibility with older summary files:
        "per_threshold_mask_aps": per_image_mean_mask_aps,
        # Box metrics
        "box_ap50": box_ap50,
        "box_ap75": box_ap75,
        "box_map50_95": box_map50_95,
        "mean_per_image_box_ap50": float(np.mean(per_image_box_ap50))
        if per_image_box_ap50
        else 0.0,
        "per_image_box_ap50": per_image_box_ap50,
        "per_image_mean_box_aps": per_image_mean_box_aps,
        "dataset_box_aps": dataset_box_aps,
        # Kept for backward compatibility with older summary files:
        "per_threshold_box_aps": per_image_mean_box_aps,
    }


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

    # ── Step 2b: Compute mask & box AP metrics on the TEST set ─
    print("\n" + "=" * 70)
    print("COMPUTING AP METRICS — MASK & BOX (test set)")
    print("=" * 70)

    for model_name, data in results_dict.items():
        best_params = data["best_params"]
        model = data["model"]

        print(f"\n  {model_name}: computing AP50, AP75, mAP50:95 (mask & box) ...")
        ap_results = compute_ap_for_dataset(
            model,
            test_images,
            TEST_ANNOTATIONS,
            best_params,
        )

        data["ap_results"] = ap_results

        print(f"\n  {'Metric':<20} {'Mask':<10} {'Box':<10}")
        print(f"  {'-'*40}")
        print(f"  {'AP50':<20} {ap_results['mask_ap50']:<10.3f} {ap_results['box_ap50']:<10.3f}")
        print(f"  {'AP75':<20} {ap_results['mask_ap75']:<10.3f} {ap_results['box_ap75']:<10.3f}")
        print(
            f"  {'mAP50:95':<20} {ap_results['mask_map50_95']:<10.3f} {ap_results['box_map50_95']:<10.3f}"
        )

        # Per-threshold breakdown (both views)
        print("\n  AP by IoU threshold:")
        print(
            f"  {'IoU':<6} "
            f"{'Mask (ds)':<11} {'Mask (per-img)':<16} "
            f"{'Box (ds)':<10} {'Box (per-img)':<15}"
        )
        print(f"  {'-'*60}")
        mask_ds = ap_results.get("dataset_mask_aps", {})
        box_ds = ap_results.get("dataset_box_aps", {})
        mask_pi = ap_results.get(
            "per_image_mean_mask_aps", ap_results.get("per_threshold_mask_aps", {})
        )
        box_pi = ap_results.get(
            "per_image_mean_box_aps", ap_results.get("per_threshold_box_aps", {})
        )
        for t_key in sorted(mask_pi):
            print(
                f"  {float(t_key):<6.2f} "
                f"{mask_ds.get(float(t_key), float('nan')):<11.3f} "
                f"{mask_pi[t_key]:<16.3f} "
                f"{box_ds.get(float(t_key), float('nan')):<10.3f} "
                f"{box_pi[t_key]:<15.3f}"
            )

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
        # AP metrics (mask + box, if computed)
        ap = data.get("ap_results")
        if ap is not None:
            summary_entry["mask_ap_metrics"] = {
                "mask_ap50": float(ap["mask_ap50"]),
                "mask_ap75": float(ap["mask_ap75"]),
                "mask_map50_95": float(ap["mask_map50_95"]),
                "mean_per_image_mask_ap50": float(ap["mean_per_image_mask_ap50"]),
                "per_image_mean_mask_aps": ap.get(
                    "per_image_mean_mask_aps", ap.get("per_threshold_mask_aps", {})
                ),
                "dataset_mask_aps": ap.get("dataset_mask_aps", {}),
            }
            summary_entry["box_ap_metrics"] = {
                "box_ap50": float(ap["box_ap50"]),
                "box_ap75": float(ap["box_ap75"]),
                "box_map50_95": float(ap["box_map50_95"]),
                "mean_per_image_box_ap50": float(ap["mean_per_image_box_ap50"]),
                "per_image_mean_box_aps": ap.get(
                    "per_image_mean_box_aps", ap.get("per_threshold_box_aps", {})
                ),
                "dataset_box_aps": ap.get("dataset_box_aps", {}),
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
        f"\n{'Model':<10} {'F1':<8} {'Prec':<8} {'Rec':<8} "
        f"{'mAP50':<8} {'mAP50:95':<10} "
        f"{'bAP50':<8} {'bAP50:95':<10} "
        f"{'TP':<8} {'FP':<8} {'FN':<8} {'(val F1)':<10}"
    )
    print("-" * 118)
    for model_name in results_dict:
        test_r = results_dict[model_name]["test_result"]
        val_r = results_dict[model_name]["tuning_result"]
        ap = results_dict[model_name].get("ap_results")
        val_f1_str = f"({val_r['f1_score']:.3f})" if val_r is not None else "(n/a)"
        m_ap50 = f"{ap['mask_ap50']:.3f}" if ap else "n/a"
        m_map = f"{ap['mask_map50_95']:.3f}" if ap else "n/a"
        b_ap50 = f"{ap['box_ap50']:.3f}" if ap else "n/a"
        b_map = f"{ap['box_map50_95']:.3f}" if ap else "n/a"
        print(
            f"{model_name:<10} {test_r['f1_score']:<8.3f} {test_r['precision']:<8.3f} "
            f"{test_r['recall']:<8.3f} {m_ap50:<8} {m_map:<10} "
            f"{b_ap50:<8} {b_map:<10} "
            f"{test_r['total_tp']:<8} {test_r['total_fp']:<8} "
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
