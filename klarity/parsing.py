import math
import os
import re
import typing
from pathlib import Path

import cv2
import numpy
import pandas
import torch
from tqdm.notebook import tqdm
from ultralytics import YOLO

from klarity.geometry import pixel_size_mm

cv2.setNumThreads(8)


# ============================================================
# Global defaults for detection/segmentation thresholds
# ============================================================
DEFAULT_CONF: float = 0.12
DEFAULT_IOU: float = 0.30
DEFAULT_MASK_THR: float = 0.30  # binarization threshold for seg masks

# ============================================================
# Overlay visualization color legend (when show_axes=True)
# ============================================================
# BLUE     = Fitted ellipsoid halves (front and rear)
# RED      = Principal axis (major axis from PCA)
# GREEN    = Perpendicular split line (dashed, through center)
# YELLOW   = Front quarter-point region points & measurement line
# MAGENTA  = Rear quarter-point region points & measurement line
# ORANGE   = Points that determined max perpendicular distance
# WHITE    = Bubble center point
# GRAY     = All contour points
# ============================================================

# ============================================================
# File naming / pixel size
# ============================================================

FNAME_RE = re.compile(
    r"^(?P<placement>placement_\d+)_"  # e.g. placement_2
    r"(?P<setting>.+?)_"  # e.g. 75_rpm_55_lmin
    r"(?P<replicate>rep_\d+)\.csv$"  # e.g. rep_1.csv
)

index_levels: typing.Tuple[str, ...] = (
    "placement",
    "reactor_setting",
    "replicate",
    "burst_index",
    "image_number_in_burst",
)


def parse_setting(setting: str) -> typing.Tuple[str, str, str]:
    """
    Parse a reactor setting string like '100 rpm 55 lmin 0125 xanthan'
    into its three components: ('100 rpm', '55 lmin', '0125 xanthan').
    """
    parts = setting.split()
    i_rpm = parts.index("rpm")
    i_lmin = parts.index("lmin")
    rpm = " ".join(parts[: i_rpm + 1])
    aeration = " ".join(parts[i_rpm + 1 : i_lmin + 1])
    xanthan = " ".join(parts[i_lmin + 1 :])
    return rpm, aeration, xanthan


# ============================================================
# Geometry selection modes
# ============================================================
BubbleGeomMode = ("sphere_only", "ellipsoid_only", "hybrid")

# ============================================================
# Processing Pipeline
# ============================================================


def process_all_settings(
    image_root_dir,
    model,
    output_dir,
    *,
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    binarize_thr: float = DEFAULT_MASK_THR,
    overlays_root: typing.Optional[str] = None,
    save_masks_overlay: bool = False,
    save_fit_overlay: bool = False,
    pixel_size_mm_override: typing.Optional[float] = None,
    geom_mode: str = "ellipsoid_only",
    sphere_if_aspect_tol: typing.Optional[float] = 0.10,
    show_axes: bool = False,
):
    """
    Loop through placement folders, reactor settings, and replicates to process images.
    The outer progress bar tracks replicate folders while the inner progress bar inside process_replicate
    tracks image-level progress.
    """
    replicates_to_process = []  # List of tuples: (placement, setting, replicate, replicate_path)
    placements = [
        p for p in os.listdir(image_root_dir) if os.path.isdir(os.path.join(image_root_dir, p))
    ]

    for placement in placements:
        placement_path = os.path.join(image_root_dir, placement)
        for setting in os.listdir(placement_path):
            setting_path = os.path.join(placement_path, setting)
            if not os.path.isdir(setting_path):
                continue
            for replicate in os.listdir(setting_path):
                replicate_path = os.path.join(setting_path, replicate)
                if not os.path.isdir(replicate_path):
                    continue
                if not check_if_processed(placement, setting, replicate, output_dir):
                    replicates_to_process.append((placement, setting, replicate, replicate_path))

    # Outer progress bar: tracking replicate folders.
    with tqdm(
        total=len(replicates_to_process),
        desc="Processing Replicates",
        dynamic_ncols=True,
        mininterval=0.1,
    ) as pbar_overall:
        for placement, setting, replicate, replicate_path in replicates_to_process:
            conf_local = conf

            bubble_data = process_replicate(
                placement,
                setting,
                replicate,
                replicate_path,
                model,
                conf=conf_local,
                iou=iou,
                binarize_thr=binarize_thr,
                overlay_root=overlays_root,
                save_masks_overlay=save_masks_overlay,
                save_fit_overlay=save_fit_overlay,
                pixel_size_mm_override=(
                    pixel_size_mm_override if pixel_size_mm_override is not None else pixel_size_mm
                ),
                geom_mode=geom_mode,
                sphere_if_aspect_tol=sphere_if_aspect_tol,
                show_axes=show_axes,
            )
            save_to_csv(bubble_data, placement, setting, replicate, output_dir)
            pbar_overall.update(1)


def check_if_processed(placement, setting, replicate, output_dir):
    """Check if a CSV already exists for a given placement, reactor setting, and replicate."""
    sanitized_name = f"{placement}_{setting}_{replicate}".replace(" ", "_")
    output_file = os.path.join(output_dir, f"{sanitized_name}.csv")
    return os.path.exists(output_file)


def process_replicate(
    placement,
    setting,
    replicate,
    replicate_path,
    model,
    *,
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    binarize_thr: float = DEFAULT_MASK_THR,
    overlay_root: typing.Optional[str] = None,
    save_masks_overlay: bool = False,
    save_fit_overlay: bool = False,
    pixel_size_mm_override: typing.Optional[float] = None,
    geom_mode: str = "hybrid",
    sphere_if_aspect_tol: typing.Optional[float] = 0.10,
    show_axes: bool = False,
):
    """Process all images in a replicate folder with an inner progress bar."""
    bubble_data = []
    # Filter only image files (ensuring the progress bar length is correct)
    images = sorted(
        [
            img
            for img in os.listdir(replicate_path)
            if img.lower().endswith((".jpg", ".png", ".jpeg"))
        ],
        key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)],  # natural sort
    )
    valid_image_count = 0
    zero_xanthan = _is_zero_xanthan(setting)
    ps = pixel_size_mm_override if pixel_size_mm_override is not None else pixel_size_mm

    # Inner progress bar: tracking images in the current replicate folder
    with tqdm(
        total=len(images),
        desc=f"Processing {placement}/{setting}/{replicate}",
        leave=True,
        dynamic_ncols=True,
        mininterval=0.1,
    ) as pbar:
        for image_name in images:
            image_path = os.path.join(replicate_path, image_name)
            im0 = cv2.imread(image_path)
            # Skip images that fail to load or are entirely zero
            if im0 is None or numpy.all(im0 == 0):
                pbar.update(1)
                continue

            # Determine overlay folder when requested
            overlay_dir = None
            if overlay_root is not None and (save_masks_overlay or save_fit_overlay):
                overlay_dir = os.path.join(overlay_root, placement, setting, replicate)

            # Process the image and extract bubble information
            bubbles = process_image(
                image_path,
                model,
                conf=conf,
                iou=iou,
                binarize_thr=binarize_thr,
                overlay_dir=overlay_dir,
                save_masks_overlay=save_masks_overlay,
                save_fit_overlay=save_fit_overlay,
                pixel_size_mm=ps,
                geom_mode=geom_mode,
                sphere_if_aspect_tol=sphere_if_aspect_tol,
                show_axes=show_axes,
            )

            if zero_xanthan:
                # 000 xanthan --> apply burst logic
                burst_index = (valid_image_count // 50) + 1
                image_number_in_burst = (valid_image_count % 50) + 1
            else:
                burst_index = 0
                image_number_in_burst = valid_image_count + 1

            for bubble in bubbles:
                bubble["placement"] = placement
                bubble["reactor_setting"] = setting
                bubble["replicate"] = replicate
                bubble["image"] = image_name
                bubble["burst_index"] = burst_index
                bubble["image_number_in_burst"] = image_number_in_burst
                bubble_data.append(bubble)

            valid_image_count += 1
            pbar.set_description(
                f"{placement}/{setting}/{replicate}: {valid_image_count}/{len(images)} processed"
            )
            pbar.update(1)
    return bubble_data


def process_image(
    image_path: str | Path,
    model,
    *,
    yolo_result=None,
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    binarize_thr: float = DEFAULT_MASK_THR,
    overlay_dir: typing.Optional[str] = None,
    save_masks_overlay: bool = False,
    save_fit_overlay: bool = False,
    pixel_size_mm: typing.Optional[float] = None,
    geom_mode: str = "hybrid",  # "sphere_only" | "ellipsoid_only" | "hybrid"
    sphere_size_thresh: typing.Optional[int] = 100,  # threshold at 100 px for eq_d_px
    sphere_if_aspect_tol: typing.Optional[float] = 0.10,  # only if geom_mode == "hybrid"
    show_axes: bool = False,  # show axes, lines, and measurement points on ellipsoid overlays
) -> list[dict]:
    """
    Segment one image and compute per-bubble measurements in ORIGINAL pixel space,
    with optional physics in mm if `pixel_size_mm` is provided.

    Coordinate systems / units (important):
    - All geometry (boxes, centroids, areas) is in the ORIGINAL image pixel space (W, H).
    - Fields with `_mm` are physical units derived using `pixel_size_mm`.
    - `img_w` / `img_h` (original) are stored per row for drift-free downstream overlays.

    Outputs (one dict per bubble) – superset of legacy + new:
      - Localization / legacy:
          bbox_x1..y2 (px), centroid_x/centroid_y (px),
          mask_area (px^2), equivalent_diameter (px), score/confidence, img_w/img_h
      - Ellipse/ellipsoid diagnostics (pixels/mm):
          cx_px, cy_px, angle_deg, a_mm, b1_mm, b2_mm
      - Sphere & ellipsoid physics (mm / mm^2 / mm^3), if pixel_size_mm is known:
          d_mm_sphere, volume_mm3_sphere, surface_area_mm2_sphere
          volume_mm3_ellipsoid, surface_area_mm2_ellipsoid
      - Hybrid decision:
          aspect_ratio, aspect_delta, aspect_method,
          d_mm_chosen, volume_mm3_chosen, surface_area_mm2_chosen,
          model_used in {"sphere", "asym_ellipsoid", "sphere_fallback", "unknown"}

    Parameters
    ----------
    image_path : str | Path
        Path to the ORIGINAL image file (this is the geometry reference).
    model : Ultralytics YOLO model
        Loaded YOLOv8 segmentation model.
    conf : float, optional
        Confidence threshold forwarded to detection.
    iou : float, optional
        IoU threshold for NMS during prediction (Ultralytics `predict(..., iou=...)`).
    binarize_thr : float, optional
        Threshold applied to mask logits/probabilities before resizing.
    overlay_dir : Optional[str], optional
        Directory to save visual overlays. If None, no files are written.
    save_masks_overlay : bool, optional
        If True, writes a semi-transparent mask overlay.
    save_fit_overlay : bool, optional
        If True, writes sphere/ellipsoid overlays derived from measurements.
    pixel_size_mm : Optional[float], optional
        Pixel size in mm/pixel. If provided, mm-based metrics are computed.
    geom_mode : str, optional
        "sphere_only", "ellipsoid_only", or "hybrid" (choose model per instance).
    sphere_if_aspect_tol : Optional[float], optional
        Tolerance for near-spherical decision in "hybrid" mode.

    Returns
    -------
    list[dict]
        One dictionary per detected bubble with the fields described above.
        Empty list if no detections or image load failed.
    """
    image_path = Path(image_path)
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return []
    H, W = img_bgr.shape[:2]

    # Decide pixel size to use (explicit arg overrides any global)
    ps = pixel_size_mm if pixel_size_mm is not None else globals().get("pixel_size_mm", None)

    # Predict masks + boxes in ORIGINAL pixel space.
    # If a precomputed Ultralytics `Results` object is provided (e.g. from batched inference),
    # we MUST reuse it to avoid a second forward pass (critical for MPS memory stability).
    if yolo_result is not None:
        dets = yolo_dets_from_result(
            yolo_result,
            img_h=H,
            img_w=W,
            binarize_thr=binarize_thr,
            conf_small=conf,
        )
    else:
        dets = yolo_segment_image(
            model,
            img_bgr,
            conf=conf,
            iou=iou,
            binarize_thr=binarize_thr,
        )

    # Prepare overlays (draw in RGB space for saving as JPEG/PNG)
    mask_overlay = fit_overlay = None
    if overlay_dir:
        Path(overlay_dir).mkdir(parents=True, exist_ok=True)
        rgb = img_bgr[:, :, ::-1].copy()
        mask_overlay = rgb.copy()
        fit_overlay = rgb.copy()

    rows: list[dict] = []

    for j, d in enumerate(dets):
        mask = d["mask"].astype(bool)
        score = float(d["score"])
        bbox = d.get("bbox", None)

        # Optional: raw mask overlay (semi-transparent)
        if save_masks_overlay and overlay_dir and mask_overlay is not None:
            mask_overlay = _draw_mask_overlay(mask_overlay, mask, alpha=0.28)

        # Basic mask-based metrics (in ORIGINAL pixels)
        area_px = float(mask.sum())
        eq_d_px = 2.0 * math.sqrt(area_px / math.pi) if area_px > 0 else float("nan")

        # Robust centroid via moments (fallback to mean of nonzeros)
        m = cv2.moments(mask.astype(numpy.uint8), binaryImage=True)
        if m["m00"] > 0:
            centroid_x = m["m10"] / m["m00"]
            centroid_y = m["m01"] / m["m00"]
        else:
            ys, xs = numpy.nonzero(mask)
            centroid_x = float(xs.mean()) if xs.size else float("nan")
            centroid_y = float(ys.mean()) if ys.size else float("nan")

        # If bbox missing, compute a conservative one from mask (avoids NaNs downstream)
        bbox_x1 = bbox_y1 = bbox_x2 = bbox_y2 = float("nan")
        if bbox is not None:
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
        else:
            ys, xs = numpy.where(mask)
            if xs.size and ys.size:
                x, y, w_box, h_box = cv2.boundingRect(numpy.column_stack((xs, ys)))
                bbox_x1, bbox_y1 = float(x), float(y)
                bbox_x2, bbox_y2 = float(x + w_box), float(y + h_box)

        # Near-spherical test (shape-based) to drive hybrid choice
        near, r_aspect, delta_aspect, used_method, (dx, dy) = is_near_spherical_from_mask(
            mask, tol=(sphere_if_aspect_tol or 0.10), method="ellipse"
        )

        size_thresh = sphere_size_thresh is not None and eq_d_px <= sphere_size_thresh

        # Defaults for physics / fit outputs
        d_mm_sph = V_sph = S_sph = float("nan")
        a_mm = b1_mm = b2_mm = V_ell = S_ell = float("nan")
        cx_fit = cy_fit = ang_deg = float("nan")

        # Sphere metrics (if pixel size known)
        if ps is not None and area_px > 0:
            d_mm_sph, V_sph, S_sph = sphere_metrics_from_mask(mask, ps)

        # Ellipsoid metrics (if pixel size known) - USING IMPROVED QUARTER-POINT METHOD
        est = None
        if ps is not None:
            m8 = mask.astype(numpy.uint8) * 255
            # Only pass fit_overlay for debug visualization if show_axes=True
            debug_img = (
                fit_overlay
                if (save_fit_overlay and fit_overlay is not None and show_axes)
                else None
            )
            est = estimate_a_b1_b2_split_fit(m8, ps, debug_overlay=debug_img)
            if est is not None:
                cx_fit, cy_fit, ang_deg, a_mm, b1_mm, b2_mm = est
                V_ell, S_ell = volume_surface_from_abi(a_mm, b1_mm, b2_mm)

        # Choose model + draw overlay (center = mask centroid for stability)
        model_used = "unknown"
        V_ch = S_ch = float("nan")
        d_mm_ch = float("nan")

        def _draw_circle_if(img_rgb: numpy.ndarray) -> numpy.ndarray:
            """Draw sphere overlay in pixel space if we have mm diameter + pixel size."""
            if save_fit_overlay and overlay_dir and ps is not None and not math.isnan(d_mm_sph):
                r_px = (d_mm_sph / 2.0) / ps
                return draw_circle_overlay(img_rgb, centroid_x, centroid_y, r_px, thickness=2)
            return img_rgb

        if geom_mode == "sphere_only":
            model_used = "sphere"
            V_ch, S_ch = V_sph, S_sph
            d_mm_ch = d_mm_sph
            if fit_overlay is not None:
                fit_overlay = _draw_circle_if(fit_overlay)

        elif geom_mode == "ellipsoid_only":
            model_used = "asym_ellipsoid" if est is not None else "sphere_fallback"
            if est is not None:
                V_ch, S_ch = V_ell, S_ell
                if V_ch > 0.0:
                    d_mm_ch = (6.0 * V_ch / math.pi) ** (1.0 / 3.0)
                if save_fit_overlay and overlay_dir and ps is not None and fit_overlay is not None:
                    a_px, b1_px, b2_px = a_mm / ps, b1_mm / ps, b2_mm / ps
                    fit_overlay = draw_asymmetric_ellipsoid_overlay(
                        fit_overlay,
                        cx_fit,
                        cy_fit,
                        ang_deg,
                        a_px,
                        b1_px,
                        b2_px,
                        thickness=2,
                        show_axes=show_axes,
                    )
            elif ps is not None and not math.isnan(d_mm_sph):
                V_ch, S_ch = V_sph, S_sph
                d_mm_ch = d_mm_sph
                if fit_overlay is not None:
                    fit_overlay = _draw_circle_if(fit_overlay)

        else:  # "hybrid"
            if size_thresh or near and not math.isnan(d_mm_sph):
                model_used = "sphere"
                V_ch, S_ch = V_sph, S_sph
                d_mm_ch = d_mm_sph
                if fit_overlay is not None:
                    fit_overlay = _draw_circle_if(fit_overlay)
            else:
                model_used = "asym_ellipsoid" if est is not None else "sphere_fallback"
                if est is not None:
                    V_ch, S_ch = V_ell, S_ell
                    if V_ch > 0.0:
                        d_mm_ch = (6.0 * V_ch / math.pi) ** (1.0 / 3.0)
                    if (
                        save_fit_overlay
                        and overlay_dir
                        and ps is not None
                        and fit_overlay is not None
                    ):
                        a_px, b1_px, b2_px = a_mm / ps, b1_mm / ps, b2_mm / ps
                        fit_overlay = draw_asymmetric_ellipsoid_overlay(
                            fit_overlay,
                            cx_fit,
                            cy_fit,
                            ang_deg,
                            a_px,
                            b1_px,
                            b2_px,
                            thickness=2,
                            show_axes=show_axes,
                        )
                elif ps is not None and not math.isnan(d_mm_sph):
                    V_ch, S_ch = V_sph, S_sph
                    d_mm_ch = d_mm_sph
                    if fit_overlay is not None:
                        fit_overlay = _draw_circle_if(fit_overlay)

        # Assemble row (keep legacy names for compatibility)
        rows.append(
            {
                "image_path": str(image_path),
                "image": Path(image_path).name,
                "img_w": W,
                "img_h": H,
                "bubble_index": j,
                "score": score,
                "confidence": score,  # legacy alias
                "conf_thresh": float(conf),
                "iou_thresh": float(iou),
                "mask_binarize_thr": float(binarize_thr),
                # bbox (px)
                "bbox_x1": bbox_x1,
                "bbox_y1": bbox_y1,
                "bbox_x2": bbox_x2,
                "bbox_y2": bbox_y2,
                # mask-derived legacy fields (px / px^2)
                "mask_area": area_px,  # px^2
                "equivalent_diameter": eq_d_px,  # px
                "centroid_x": centroid_x,
                "centroid_y": centroid_y,
                # ellipse diagnostics (fit)
                "cx_px": cx_fit,
                "cy_px": cy_fit,
                "angle_deg": ang_deg,
                "a_mm": a_mm,
                "b1_mm": b1_mm,
                "b2_mm": b2_mm,
                # sphere (if ps provided)
                "d_mm_sphere": d_mm_sph,
                "volume_mm3_sphere": V_sph,
                "surface_area_mm2_sphere": S_sph,
                # ellipsoid (if ps provided)
                "volume_mm3_ellipsoid": V_ell,
                "surface_area_mm2_ellipsoid": S_ell,
                # aspect / decision
                "aspect_ratio": r_aspect,
                "aspect_delta": delta_aspect,
                "aspect_method": used_method,
                # chosen model
                "d_mm_chosen": d_mm_ch,
                "volume_mm3_chosen": V_ch,
                "surface_area_mm2_chosen": S_ch,
                "model_used": model_used,
            }
        )

    # Save overlays (if requested)
    if overlay_dir:
        if save_masks_overlay and mask_overlay is not None:
            cv2.imwrite(
                str(Path(overlay_dir) / f"{image_path.stem}_mask_overlay.jpg"),
                mask_overlay[:, :, ::-1],
            )
        if save_fit_overlay and fit_overlay is not None:
            cv2.imwrite(
                str(Path(overlay_dir) / f"{image_path.stem}_fit_overlay.jpg"),
                fit_overlay[:, :, ::-1],
            )

    return rows


def yolo_segment_image(
    model,
    img_bgr: numpy.ndarray,
    *,
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    binarize_thr: float = DEFAULT_MASK_THR,
    conf_large: float = 0.60,
    large_frac: float = 0.08,
) -> list[dict]:
    """
    Run a YOLOv8-seg model on a single BGR image and return one dict per instance.

    Coordinate systems / units:
    - Boxes are returned in the ORIGINAL image coordinate system (pixels).
      (Ultralytics maps xyxy back from the network's letterboxed tensor.)
    - Masks are resized back to the ORIGINAL image size (H, W) and binarized.

    Parameters
    ----------
    model : Ultralytics YOLO model
        A loaded YOLOv8 segmentation model (e.g., YOLO('...pt')).
    img_bgr : numpy.ndarray
        Input image in BGR (H, W, 3). Must be the ORIGINAL you plan to draw/measure on.
    conf : float, optional
        Confidence threshold forwarded to Ultralytics during prediction.
    iou : float, optional
        IoU threshold for NMS in Ultralytics prediction.
    binarize_thr : float, optional
        Threshold applied to mask logits/probabilities before resizing.

    Returns
    -------
    list[dict]
        Each dict contains:
            {
              'mask': bool(H, W)         # original geometry
              'score': float,            # confidence
              'bbox': (x1, y1, x2, y2)   # original pixels; None if unavailable
            }
        If no instances are detected, returns [].
    """
    # Ultralytics expects RGB; we give it a view without copying bytes.
    results = model.predict(
        source=img_bgr[..., ::-1], conf=conf, iou=iou, max_det=2500, verbose=False
    )
    det = results[0]
    out: list[dict] = []

    # No instances → return empty list early
    if det.masks is None:
        return out

    # Apply size-aware filtering IN-PLACE on the Ultralytics result
    H_img, W_img = img_bgr.shape[:2]
    _size_aware_filter_result(
        det,
        img_h=H_img,
        img_w=W_img,
        conf_small=conf,
    )

    # Original geometry (height, width) that masks/boxes must be mapped to
    H, W = det.masks.orig_shape

    # Boxes come already in original-pixel xyxy; confidences too.
    boxes_xyxy = det.boxes.xyxy.cpu().numpy() if det.boxes is not None else None
    confs = det.boxes.conf.cpu().numpy() if hasattr(det.boxes, "conf") else None

    # Defensive: ensure 1:1 instance counts if both are present
    masks_np = det.masks.data.cpu().numpy()  # (N, H_in, W_in) at network space
    if boxes_xyxy is not None and len(boxes_xyxy) != len(masks_np):
        # If this ever triggers, something upstream is off. We still proceed per-index safely.
        pass

    for i, m in enumerate(masks_np):
        # 1) Binarize in network space (avoid soft edges changing area)
        mb = (m > binarize_thr).astype(numpy.uint8)

        # 2) Resize the binary mask back to ORIGINAL image geometry
        mask = cv2.resize(mb, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

        # 3) Confidence / bbox (guard against length mismatches)
        score = float(confs[i]) if confs is not None and i < len(confs) else 1.0
        bbox = None
        if boxes_xyxy is not None and i < len(boxes_xyxy):
            x1, y1, x2, y2 = boxes_xyxy[i].tolist()
            bbox = (float(x1), float(y1), float(x2), float(y2))

        out.append({"mask": mask, "score": score, "bbox": bbox})

    return out


def yolo_dets_from_result(
    det,
    *,
    img_h: int,
    img_w: int,
    binarize_thr: float = DEFAULT_MASK_THR,
    conf_small: float = DEFAULT_CONF,
) -> list[dict]:
    """Convert a precomputed Ultralytics ``Results`` object into our internal det list.

    This is intentionally parallel to :func:`yolo_segment_image`, but **does not**
    execute a model forward pass. It is designed for batched inference workflows,
    where the caller already has ``det = model.predict([...])[k]``.

    Parameters
    ----------
    det
        A single Ultralytics ``Results`` object (i.e., one element from ``model.predict``).
    img_h, img_w
        Original image geometry. Used to validate/override mask geometry.
    binarize_thr
        Threshold applied to mask probabilities/logits before resizing.
    conf_small
        Confidence threshold for the size-aware filter.

    Returns
    -------
    list[dict]
        Same schema as :func:`yolo_segment_image`.
    """
    out: list[dict] = []

    if det is None or det.masks is None:
        return out

    # Apply the same size-aware filtering as in the single-image path.
    _size_aware_filter_result(det, img_h=img_h, img_w=img_w, conf_small=conf_small)

    # Original geometry (height, width) that masks/boxes must be mapped to.
    # Ultralytics reports orig_shape as (H, W). We keep a defensive override.
    H, W = det.masks.orig_shape
    if (H != img_h) or (W != img_w):
        H, W = int(img_h), int(img_w)

    boxes_xyxy = det.boxes.xyxy.detach().cpu().numpy() if det.boxes is not None else None
    confs = (
        det.boxes.conf.detach().cpu().numpy()
        if (det.boxes is not None and hasattr(det.boxes, "conf"))
        else None
    )

    masks_t = det.masks.data  # (N, H_in, W_in) in model space; torch.Tensor
    if masks_t is None:
        return out
    masks_np = masks_t.detach().cpu().numpy()

    for i, m in enumerate(masks_np):
        mb = (m > binarize_thr).astype(numpy.uint8)
        mask = cv2.resize(mb, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

        score = float(confs[i]) if confs is not None and i < len(confs) else 1.0
        bbox = None
        if boxes_xyxy is not None and i < len(boxes_xyxy):
            x1, y1, x2, y2 = boxes_xyxy[i].tolist()
            bbox = (float(x1), float(y1), float(x2), float(y2))

        out.append({"mask": mask, "score": score, "bbox": bbox})

    return out


# ============================================================
# Process Replicate Helpers
# ============================================================


def _is_zero_xanthan(setting: str) -> bool:
    return "000 xanthan" in setting.lower()


def _draw_mask_overlay(
    rgb: numpy.ndarray, mask: numpy.ndarray, alpha: float = 0.28
) -> numpy.ndarray:
    out = rgb.copy()
    base_bgr = out[:, :, ::-1]
    layer = base_bgr.copy()
    m = mask > 0
    layer[m] = (0, 255, 255)  # aqua in BGR
    blended = cv2.addWeighted(base_bgr, 1.0, layer, alpha, 0)
    return blended[:, :, ::-1]


def draw_circle_overlay(
    rgb: numpy.ndarray, cx: float, cy: float, r_px: float, thickness: int = 2
) -> numpy.ndarray:
    """Draw a circle (spherical choice) on RGB image."""
    out = rgb.copy()
    cv2.circle(
        out, (int(round(cx)), int(round(cy))), int(round(r_px)), (0, 255, 0), thickness
    )  # green
    return out


def _draw_ellipsoid_axes(
    out: numpy.ndarray,
    cx: float,
    cy: float,
    ang_deg: float,
    A: float,
    b1_px: float,
    b2_px: float,
    thickness: int,
) -> numpy.ndarray:
    """Draw principal axis, dashed perpendicular, and quarter-point measurement lines."""
    ang_rad = numpy.deg2rad(ang_deg)
    u_x, u_y = numpy.cos(ang_rad), numpy.sin(ang_rad)
    v_x, v_y = -u_y, u_x

    # Principal axis — red
    axis_length = A * 1.2
    cv2.line(
        out,
        (int(round(cx - axis_length * u_x)), int(round(cy - axis_length * u_y))),
        (int(round(cx + axis_length * u_x)), int(round(cy + axis_length * u_y))),
        (255, 0, 0),
        max(1, thickness - 1),
    )

    # Perpendicular split line — green, dashed
    perp_length = max(b1_px, b2_px) * 1.5
    p1 = (cx + perp_length * v_x, cy + perp_length * v_y)
    p2 = (cx - perp_length * v_x, cy - perp_length * v_y)
    for i in range(0, 10, 2):
        t1, t2 = i / 10, (i + 1) / 10
        seg1 = (int(round(p2[0] + t1 * (p1[0] - p2[0]))), int(round(p2[1] + t1 * (p1[1] - p2[1]))))
        seg2 = (int(round(p2[0] + t2 * (p1[0] - p2[0]))), int(round(p2[1] + t2 * (p1[1] - p2[1]))))
        cv2.line(out, seg1, seg2, (0, 255, 0), max(1, thickness - 1))

    # Quarter-point markers and measurement lines — cyan
    for sign, b_px in ((1.0, b1_px), (-1.0, b2_px)):
        qx = cx + sign * (A * 0.5) * u_x
        qy = cy + sign * (A * 0.5) * u_y
        cv2.circle(out, (int(round(qx)), int(round(qy))), 4, (0, 255, 255), -1)
        cv2.line(
            out,
            (int(round(qx + b_px * v_x)), int(round(qy + b_px * v_y))),
            (int(round(qx - b_px * v_x)), int(round(qy - b_px * v_y))),
            (255, 255, 0),
            max(1, thickness - 1),
        )

    # Center point — white
    cv2.circle(out, (int(round(cx)), int(round(cy))), 3, (255, 255, 255), -1)
    return out


def draw_asymmetric_ellipsoid_overlay(
    rgb: numpy.ndarray,
    cx: float,
    cy: float,
    ang_deg: float,
    a_px: float,
    b1_px: float,
    b2_px: float,
    thickness: int = 2,
    show_axes: bool = False,
) -> numpy.ndarray:
    """
    Draw an asymmetric ellipsoid outline on an RGB image.

    The ellipsoid is rendered as two half-ellipse polylines (front and rear).
    When *show_axes* is True, the principal axis, perpendicular split line, and
    quarter-point measurement indicators are drawn as well.
    """
    out = rgb.copy()
    A = 0.5 * a_px
    b1_px = min(b1_px, A * (1 - 1e-6))
    b2_px = min(b2_px, A * (1 - 1e-6))

    poly_front = _polyline_half_ellipse(cx, cy, ang_deg, A, b1_px, "front")
    poly_rear = _polyline_half_ellipse(cx, cy, ang_deg, A, b2_px, "rear")
    cv2.polylines(out, [poly_front], False, (255, 0, 0), thickness)
    cv2.polylines(out, [poly_rear], False, (255, 0, 0), thickness)

    if show_axes:
        out = _draw_ellipsoid_axes(out, cx, cy, ang_deg, A, b1_px, b2_px, thickness)

    return out


def _polyline_half_ellipse(
    cx: float, cy: float, ang_deg: float, A: float, B: float, side: str, num: int = 200
) -> numpy.ndarray:
    th = numpy.deg2rad(ang_deg)
    ct, st = numpy.cos(th), numpy.sin(th)
    t = numpy.linspace(-numpy.pi / 2, numpy.pi / 2, num)
    x = A * numpy.cos(t)
    y = B * numpy.sin(t)
    if side == "rear":
        x = -x
    X = cx + x * ct - y * st
    Y = cy + x * st + y * ct
    return numpy.stack([X, Y], axis=1).astype(numpy.int32)


# ======================================
# Utility
# ======================================


def load_all_data_parquet(
    parquet_dir: Path,
    columns: typing.Optional[typing.List[str]] = None,
    placements: typing.Optional[typing.List[str]] = None,
    settings: typing.Optional[typing.List[str]] = None,
    set_index: bool = True,
) -> pandas.DataFrame:
    """
    Load data from Parquet files with optional filtering and column selection.

    This is the NEW version optimized for Parquet format.
    Use this instead of load_all_data() when working with Parquet files.

    Parameters
    ----------
    parquet_dir : Path
        Directory containing Parquet files (e.g., "output_parquet/")
    columns : List[str], optional
        Specific columns to load. If None, loads all columns.
        Examples:
            - ["equivalent_diameter_mm", "bubble_volume_mm3"]
            - ["reactor_setting", "confidence", "d_mm_chosen"]
    placements : List[str], optional
        Filter to specific placements (e.g., ["placement_1", "placement_2"])
        If None, loads all placements.
    settings : List[str], optional
        Filter to specific reactor settings after loading.
    set_index : bool
        If True, sets MultiIndex using index_levels.
        If False, returns flat DataFrame (useful for filtering).

    Returns
    -------
    pandas.DataFrame
        Combined dataframe from all matching Parquet files

    Examples
    --------
    # Load everything (like old load_all_data)
    df = load_all_data_parquet(Path("output_parquet/"))

    # Load only specific columns (FAST, memory-efficient!)
    df = load_all_data_parquet(
        Path("output_parquet/"),
        columns=["reactor_setting", "equivalent_diameter_mm", "confidence"]
    )

    # Load only specific placements
    df = load_all_data_parquet(
        Path("output_parquet/"),
        placements=["placement_1", "placement_2"]
    )

    # Combine filters
    df = load_all_data_parquet(
        Path("output_parquet/"),
        columns=["equivalent_diameter_mm", "bubble_volume_mm3"],
        placements=["placement_1"]
    )
    """
    parquet_dir = Path(parquet_dir)

    if not parquet_dir.exists():
        raise FileNotFoundError(f"Parquet directory does not exist: {parquet_dir}")

    # Find all Parquet files
    parquet_files = list(parquet_dir.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found in {parquet_dir}")

    # Filter by placement if specified
    if placements:
        parquet_files = [f for f in parquet_files if any(p in f.stem for p in placements)]

    if not parquet_files:
        raise ValueError(f"No files match the placement filter: {placements}")

    # Load files
    dfs = []
    for parquet_file in parquet_files:
        df = pandas.read_parquet(parquet_file, columns=columns)
        dfs.append(df)

    # Combine
    combined = pandas.concat(dfs, ignore_index=True)

    # Filter by settings if specified
    if settings and "reactor_setting" in combined.columns:
        combined = combined[combined["reactor_setting"].isin(settings)]

    # Set index if requested
    if set_index:
        # Check if all index columns are present
        missing_cols = set(index_levels) - set(combined.columns)
        if missing_cols:
            raise ValueError(f"Cannot set index - missing columns: {missing_cols}")
        combined = combined.set_index(list(index_levels))

    return combined


def load_placement_parquet(
    parquet_dir: Path,
    placement: str,
    columns: typing.Optional[typing.List[str]] = None,
    set_index: bool = True,
) -> pandas.DataFrame:
    """
    Load data for a single placement from Parquet.

    This is faster than load_all_data_parquet when you only need one placement.

    Parameters
    ----------
    parquet_dir : Path
        Directory containing Parquet files
    placement : str
        Placement name (e.g., "placement_1")
    columns : List[str], optional
        Specific columns to load
    set_index : bool
        Whether to set MultiIndex

    Returns
    -------
    pandas.DataFrame
        Data for the specified placement

    Example
    -------
    # Load all data for placement_1
    df = load_placement_parquet(Path("output_parquet/"), "placement_1")

    # Load only specific columns
    df = load_placement_parquet(
        Path("output_parquet/"),
        "placement_1",
        columns=["equivalent_diameter_mm", "confidence"]
    )
    """
    parquet_file = Path(parquet_dir) / f"{placement}.parquet"

    if not parquet_file.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_file}")

    df = pandas.read_parquet(parquet_file, columns=columns)

    if set_index:
        missing_cols = set(index_levels) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Cannot set index - missing columns: {missing_cols}")
        df = df.set_index(list(index_levels))

    return df


def load_filtered_parquet(
    parquet_dir: Path,
    filters: typing.List,
    columns: typing.Optional[typing.List[str]] = None,
) -> pandas.DataFrame:
    """
    Load Parquet data with row-level filtering (very efficient!).

    Filters are applied DURING reading, so only matching rows are loaded.
    This is much faster than loading everything then filtering.

    Parameters
    ----------
    parquet_dir : Path
        Directory containing Parquet files
    filters : List
        PyArrow-style filters. Format: [(column, operator, value), ...]
        Operators: "==", "!=", ">", ">=", "<", "<=", "in", "not in"
    columns : List[str], optional
        Columns to load

    Returns
    -------
    pandas.DataFrame
        Filtered data

    Examples
    --------
    # Load only high-confidence bubbles
    df = load_filtered_parquet(
        Path("output_parquet/"),
        filters=[("confidence", ">", 0.8)]
    )

    # Load specific setting
    df = load_filtered_parquet(
        Path("output_parquet/"),
        filters=[("reactor_setting", "==", "100 rpm 55 lmin 000 xanthan")]
    )

    # Combine multiple filters
    df = load_filtered_parquet(
        Path("output_parquet/"),
        filters=[
            ("placement", "==", "placement_1"),
            ("confidence", ">", 0.7),
            ("bubble_volume_mm3", "<", 1000.0)
        ],
        columns=["reactor_setting", "bubble_volume_mm3"]
    )
    """
    parquet_files = list(Path(parquet_dir).glob("*.parquet"))

    dfs = []
    for f in parquet_files:
        df = pandas.read_parquet(f, columns=columns, filters=filters)
        if len(df) > 0:
            dfs.append(df)

    if not dfs:
        return pandas.DataFrame()

    return pandas.concat(dfs, ignore_index=True)


def load_yolo_model(model_path, device="mps"):
    """Load YOLO model for instance segmentation"""

    # is MPS available?
    if device == "mps" and not torch.backends.mps.is_available():
        print("WARNING MPS not available, falling back to CPU")
        device = "cpu"

    model = YOLO(model_path)
    model.to(device)
    print(f"Model loaded on: {device}")
    return model


def save_to_csv(data, placement, setting, replicate, output_dir):
    """Save extracted bubble data to a CSV file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    sanitized_name = f"{placement}_{setting}_{replicate}".replace(" ", "_")
    output_file = os.path.join(output_dir, f"{sanitized_name}.csv")
    df = pandas.DataFrame(data)
    df.to_csv(output_file, index=False)


# ============================================================
# Bubble geometry - IMPROVED WITH QUARTER-POINT MEASUREMENT
# ============================================================


def _draw_fit_debug(
    overlay: numpy.ndarray,
    pts: numpy.ndarray,
    cx: float,
    cy: float,
    u: numpy.ndarray,
    v: numpy.ndarray,
    a_px: float,
    proj_max: float,
    proj_min: float,
    front_region_mask: numpy.ndarray,
    rear_region_mask: numpy.ndarray,
    front_max_pt: typing.Optional[numpy.ndarray],
    front_max_dist: typing.Optional[float],
    rear_max_pt: typing.Optional[numpy.ndarray],
    rear_max_dist: typing.Optional[float],
) -> None:
    """Draw PCA fit diagnostics onto an RGB overlay in-place (used when show_axes=True)."""
    for pt in pts:
        cv2.circle(overlay, (int(pt[0]), int(pt[1])), 1, (200, 200, 200), -1)
    cv2.circle(overlay, (int(cx), int(cy)), 5, (255, 255, 255), -1)

    axis_len = a_px * 0.6
    cv2.line(
        overlay,
        (int(cx - axis_len * u[0]), int(cy - axis_len * u[1])),
        (int(cx + axis_len * u[0]), int(cy + axis_len * u[1])),
        (0, 0, 255),
        2,
    )

    perp_len = a_px * 0.4
    p1 = (int(cx + perp_len * v[0]), int(cy + perp_len * v[1]))
    p2 = (int(cx - perp_len * v[0]), int(cy - perp_len * v[1]))
    for i in range(0, 10, 2):
        t1, t2 = i / 10, (i + 1) / 10
        seg1 = (int(p2[0] + t1 * (p1[0] - p2[0])), int(p2[1] + t1 * (p1[1] - p2[1])))
        seg2 = (int(p2[0] + t2 * (p1[0] - p2[0])), int(p2[1] + t2 * (p1[1] - p2[1])))
        cv2.line(overlay, seg1, seg2, (0, 255, 0), 2)

    for pt in pts[front_region_mask]:
        cv2.circle(overlay, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)
    for pt in pts[rear_region_mask]:
        cv2.circle(overlay, (int(pt[0]), int(pt[1])), 2, (255, 0, 255), -1)

    front_qtr = numpy.array([cx, cy]) + (proj_max * 0.5) * u
    rear_qtr = numpy.array([cx, cy]) + (proj_min * 0.5) * u
    cv2.circle(overlay, (int(front_qtr[0]), int(front_qtr[1])), 4, (0, 255, 255), -1)
    cv2.circle(overlay, (int(rear_qtr[0]), int(rear_qtr[1])), 4, (255, 0, 255), -1)

    if front_max_pt is not None:
        fq = numpy.array([cx, cy]) + (proj_max * 0.5) * u
        cv2.line(
            overlay,
            (int(fq[0] + front_max_dist * v[0]), int(fq[1] + front_max_dist * v[1])),
            (int(fq[0] - front_max_dist * v[0]), int(fq[1] - front_max_dist * v[1])),
            (0, 255, 255),
            3,
        )
        cv2.circle(overlay, (int(front_max_pt[0]), int(front_max_pt[1])), 5, (0, 165, 255), 2)

    if rear_max_pt is not None:
        rq = numpy.array([cx, cy]) + (proj_min * 0.5) * u
        cv2.line(
            overlay,
            (int(rq[0] + rear_max_dist * v[0]), int(rq[1] + rear_max_dist * v[1])),
            (int(rq[0] - rear_max_dist * v[0]), int(rq[1] - rear_max_dist * v[1])),
            (255, 0, 255),
            3,
        )
        cv2.circle(overlay, (int(rear_max_pt[0]), int(rear_max_pt[1])), 5, (0, 165, 255), 2)


def estimate_a_b1_b2_split_fit(
    mask: numpy.ndarray,
    pixel_size_mm: float,
    debug_overlay: typing.Optional[numpy.ndarray] = None,
) -> typing.Optional[typing.Tuple[float, float, float, float, float, float]]:
    """
    Fit an asymmetric ellipsoid to a binary bubble mask using PCA + quarter-point measurement.

    Returns (cx, cy, angle_deg, a_mm, b1_mm, b2_mm) or None if fitting fails.

    The major axis length is measured tip-to-tip along the PCA principal axis.
    Minor semi-axes b1 (front) and b2 (rear) are measured at the 50 % quarter
    points along each half, which avoids over-estimating width for tapered bubbles.

    When *debug_overlay* is provided (an RGB image), PCA diagnostics and
    measurement regions are drawn onto it in-place.
    """
    m = (mask > 0).astype(numpy.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None

    cnt = max(cnts, key=cv2.contourArea)
    pts = cnt[:, 0, :].astype(float)
    if pts.shape[0] < 5:
        return None

    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    pts_centered = pts - numpy.array([cx, cy])
    cov_mat = numpy.cov(pts_centered.T)
    eigenvalues, eigenvectors = numpy.linalg.eig(cov_mat)
    principal_idx = numpy.argmax(eigenvalues)
    u = eigenvectors[:, principal_idx].real
    u = u / numpy.linalg.norm(u)
    v = numpy.array([-u[1], u[0]])
    angle_deg = float(numpy.rad2deg(numpy.arctan2(u[1], u[0])))

    projections = pts_centered @ u
    proj_max = projections.max()
    proj_min = projections.min()
    a_px = float(proj_max - proj_min)

    measurement_width = a_px * 0.15
    front_region_mask = numpy.abs(projections - proj_max * 0.5) < measurement_width
    rear_region_mask = numpy.abs(projections - proj_min * 0.5) < measurement_width

    def measure_in_region(region_mask):
        if region_mask.sum() == 0:
            return numpy.nan, None, None
        region_pts = pts_centered[region_mask]
        perp_distances = numpy.abs(region_pts @ v)
        max_idx = numpy.argmax(perp_distances)
        max_dist = perp_distances[max_idx]
        max_pt = pts[region_mask][max_idx]
        return 2.0 * float(max_dist), max_pt, max_dist

    minor_front_px, front_max_pt, front_max_dist = measure_in_region(front_region_mask)
    minor_rear_px, rear_max_pt, rear_max_dist = measure_in_region(rear_region_mask)

    if debug_overlay is not None:
        _draw_fit_debug(
            debug_overlay,
            pts,
            cx,
            cy,
            u,
            v,
            a_px,
            proj_max,
            proj_min,
            front_region_mask,
            rear_region_mask,
            front_max_pt,
            front_max_dist,
            rear_max_pt,
            rear_max_dist,
        )

    # Fallback for empty measurement regions: 75th-percentile on full half
    if numpy.isnan(minor_front_px) or numpy.isnan(minor_rear_px):
        front_mask = projections >= 0
        rear_mask = projections < 0
        if front_mask.sum() > 0:
            minor_front_px = 2.0 * float(
                numpy.percentile(numpy.abs(pts_centered[front_mask] @ v), 75)
            )
        if rear_mask.sum() > 0:
            minor_rear_px = 2.0 * float(
                numpy.percentile(numpy.abs(pts_centered[rear_mask] @ v), 75)
            )

    # Fallback: if regions are empty, use percentile method on full halves
    if numpy.isnan(minor_front_px) or numpy.isnan(minor_rear_px):
        front_mask = projections >= 0
        rear_mask = projections < 0

        if front_mask.sum() > 0:
            front_perp = numpy.abs(pts_centered[front_mask] @ v)
            # Use 75th percentile instead of max for robustness
            minor_front_px = 2.0 * float(numpy.percentile(front_perp, 75))

        if rear_mask.sum() > 0:
            rear_perp = numpy.abs(pts_centered[rear_mask] @ v)
            minor_rear_px = 2.0 * float(numpy.percentile(rear_perp, 75))

    if numpy.isnan(minor_front_px) or numpy.isnan(minor_rear_px):
        return None

    # Convert to mm (note: minor_front_px and minor_rear_px are DIAMETERS)
    a_mm = a_px * pixel_size_mm
    b1_mm = (minor_front_px / 2.0) * pixel_size_mm  # Semi-minor axis (front)
    b2_mm = (minor_rear_px / 2.0) * pixel_size_mm  # Semi-minor axis (rear)

    return float(cx), float(cy), float(angle_deg), float(a_mm), float(b1_mm), float(b2_mm)


def is_near_spherical_from_mask(
    mask: numpy.ndarray,
    tol: float = 0.10,
    method: str = "ellipse",  # "ellipse" | "minrect" | "aabb"
):
    """
    Decide 'near spherical' with an orientation-free test.
    Returns (near, ratio_r, delta, method_used, dims_used)
      - ratio_r = max(x,y)/min(x,y)  (>=1)
      - delta   = |x - y|/max(x,y)
      - dims_used = (x, y)
    """
    m8 = (mask > 0).astype(numpy.uint8)
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return False, float("nan"), float("nan"), method, (float("nan"), float("nan"))
    cnt = max(cnts, key=cv2.contourArea)

    x = y = float("nan")
    used = method

    if method == "ellipse":
        if len(cnt) >= 5:
            (cx, cy), (a1, a2), ang = cv2.fitEllipse(cnt)
            x, y = float(a1), float(a2)  # diameters
        else:
            used = "minrect"  # fallback
    if method == "minrect" or (method == "ellipse" and numpy.isnan(x)):
        rect = cv2.minAreaRect(cnt)  # ((cx,cy),(w,h),angle)
        (w, h) = rect[1]
        x, y = float(w), float(h)
        used = "minrect"
    if method == "aabb" or (numpy.isnan(x) or numpy.isnan(y)):
        x_, y_, w, h = cv2.boundingRect(cnt)  # axis-aligned
        x, y = float(w), float(h)
        used = "aabb"

    near, r, delta = _near_unity(x, y, tol)
    return near, r, delta, used, (x, y)


def _near_unity(x: float, y: float, tol: float) -> typing.Tuple[bool, float, float]:
    """Orientation-free check: r = max/min; delta = |x - y| / max."""
    if x <= 0 or y <= 0:
        return False, float("nan"), float("nan")
    r = max(x, y) / min(x, y)
    delta = abs(x - y) / max(x, y)
    return (r <= 1.0 + tol), r, delta


def sphere_metrics_from_mask(
    mask: numpy.ndarray, pixel_size_mm: float
) -> typing.Tuple[float, float, float]:
    """
    Sphere assumption from 2D area:
      - Compute equivalent diameter d from mask area.
      - Return (d_mm, V_mm3, S_mm2).
    """
    area_px = float(mask.sum())
    area_mm2 = area_px * (pixel_size_mm**2)
    if area_mm2 <= 0:
        return float("nan"), float("nan"), float("nan")
    d_mm = 2.0 * math.sqrt(area_mm2 / math.pi)
    V_mm3 = (math.pi / 6.0) * d_mm**3
    S_mm2 = math.pi * d_mm**2
    return d_mm, V_mm3, S_mm2


def _safe_arcsin(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return float(numpy.arcsin(x))


def volume_surface_from_abi(a_mm: float, b1_mm: float, b2_mm: float) -> typing.Tuple[float, float]:
    """Two half-prolate spheroids glued at equator: exact V, closed-form S."""
    A = 0.5 * a_mm
    V = (numpy.pi * a_mm / 6.0) * (b1_mm**2 + b2_mm**2)

    def S_half(B: float) -> float:
        if B <= 0 or A <= 0:
            return 0.0
        if B >= A * (1.0 - 1e-12):  # near-sphere
            term = 1.0 + (A / B)
        else:
            e = float(numpy.sqrt(1.0 - (B * B) / (A * A)))
            term = 1.0 + (A / (B * e)) * _safe_arcsin(e)
        return float(numpy.pi * (B**2) * term)

    S = S_half(b1_mm) + S_half(b2_mm)
    return float(V), float(S)


def _size_aware_filter_result(
    det,
    img_h: int,
    img_w: int,
    conf_small: float,
    conf_large: float = 0.60,
    large_frac: float = 0.08,
):
    """
    In-place filter for YOLO `result`:
      • small masks (area/img < big_frac)  → need ≥ conf_small
      • big   masks (area/img ≥ big_frac)  → need ≥ conf_large
    """
    if det.masks is None or det.boxes is None:
        return

    img_area = float(img_h * img_w)
    confs = det.boxes.conf.cpu().numpy()
    masks = det.masks.data
    keep = []

    for i in range(masks.shape[0]):
        area_px = float(masks[i].sum().item())
        frac = area_px / img_area
        need = conf_large if frac >= large_frac else conf_small
        if float(confs[i]) >= need:
            keep.append(i)

    if not keep:
        det.boxes = det.boxes[:0]
        det.masks.data = det.masks.data[:0]
    else:
        det.boxes = det.boxes[keep]
        det.masks.data = det.masks.data[keep]
