from __future__ import annotations

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

from klarity.ellipse_fits import fit_two_half_ellipse_lsq
from klarity.geometry import pixel_size_mm
from klarity.processing_config import DEFAULT_PROCESSING_CONFIG
from klarity.shape_models import volume_surface_oblate, volume_surface_prolate

cv2.setNumThreads(8)


# ============================================================
# Global defaults for detection/segmentation thresholds
# ============================================================
DEFAULT_CONF = DEFAULT_PROCESSING_CONFIG.confidence
DEFAULT_IOU = DEFAULT_PROCESSING_CONFIG.iou
DEFAULT_MASK_THR = DEFAULT_PROCESSING_CONFIG.mask_threshold
DEFAULT_CONF_LARGE = DEFAULT_PROCESSING_CONFIG.large_mask_confidence
DEFAULT_LARGE_FRAC = DEFAULT_PROCESSING_CONFIG.large_mask_fraction
# Whole-frame mean intensity below which a frame is treated as blank/black. Some
# acquisitions write one black (illumination-off) frame per 50-frame burst; see
# is_blank_frame for the rationale and the measured intensity separation.
DEFAULT_BLANK_MEAN_THRESH = DEFAULT_PROCESSING_CONFIG.blank_mean_threshold

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


def normalize_metadata_value(value: str) -> str:
    """Collapse whitespace in one filesystem-derived metadata value.

    Placement, reactor setting and replicate are all taken from directory names in the
    acquisition tree, and those names are not uniformly clean: ten replicate folders
    (placement_3 / 100 rpm / 000 xanthan, every aeration rate) are named " rep_1" and
    " rep_2" with a leading space. Untouched, that space reached two places at once --
    the ``replicate`` column written into the Parquet file, giving four distinct
    replicate labels (" rep_1", "rep_1", ...) where the experiment has two, and the
    output filename, where ``.replace(" ", "_")`` turned it into a doubled underscore
    (``..._000_xanthan__rep_1.parquet``) that filename-parsing regexes elsewhere had to
    special-case.

    Leading/trailing whitespace is stripped and internal runs are collapsed to a single
    space. The single-space form is preserved rather than removed because the reactor
    setting is space-separated by contract ("100 rpm 55 lmin 000 xanthan") and
    :func:`parse_setting` splits on it.
    """
    return " ".join(str(value).split())


def parquet_stem(placement: str, setting: str, replicate: str) -> str:
    """Canonical Parquet basename (no extension) for one placement/setting/replicate.

    Single source of truth for the output naming convention, so the writer
    (:func:`save_to_parquet`) and the resume check (:func:`check_if_processed`) cannot
    disagree about where a stream lives. Values are normalized first, so a raw directory
    name and its cleaned form map to the same file.

    >>> parquet_stem("placement_3", "100 rpm 45 lmin 000 xanthan", " rep_1")
    'placement_3_100_rpm_45_lmin_000_xanthan_rep_1'
    """
    parts = (normalize_metadata_value(v) for v in (placement, setting, replicate))
    return "_".join(parts).replace(" ", "_")


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
    geom_mode: str = DEFAULT_PROCESSING_CONFIG.geometry_mode,
    sphere_if_aspect_tol: typing.Optional[
        float
    ] = DEFAULT_PROCESSING_CONFIG.sphere_aspect_tolerance,
    conf_large: float = DEFAULT_CONF_LARGE,
    large_frac: float = DEFAULT_LARGE_FRAC,
    blank_mean_thresh: float = DEFAULT_BLANK_MEAN_THRESH,
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

    # Directory names are joined onto the filesystem paths verbatim, but the normalized
    # form is what gets stored in the Parquet columns and used for the filename -- see
    # normalize_metadata_value for the " rep_1" folders this exists to handle.
    for placement_dir in placements:
        placement_path = os.path.join(image_root_dir, placement_dir)
        placement = normalize_metadata_value(placement_dir)
        for setting_dir in os.listdir(placement_path):
            setting_path = os.path.join(placement_path, setting_dir)
            if not os.path.isdir(setting_path):
                continue
            setting = normalize_metadata_value(setting_dir)
            for replicate_dir in os.listdir(setting_path):
                replicate_path = os.path.join(setting_path, replicate_dir)
                if not os.path.isdir(replicate_path):
                    continue
                replicate = normalize_metadata_value(replicate_dir)
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
                conf_large=conf_large,
                large_frac=large_frac,
                blank_mean_thresh=blank_mean_thresh,
            )
            save_to_parquet(bubble_data, placement, setting, replicate, output_dir)
            pbar_overall.update(1)


def existing_parquet_paths(placement, setting, replicate, output_dir) -> typing.List[str]:
    """Every Parquet file in *output_dir* that holds this placement/setting/replicate.

    A name is matched if it equals the canonical stem once runs of underscores are
    collapsed. This accepts acquisition directory names containing incidental whitespace
    without requiring callers to retain the raw directory spelling.

    Returning the list rather than a bool is what keeps resume and retry consistent: a
    retry deletes *every* returned path before processing, so a stream can never end up
    represented by two spellings. ``build_dataframes`` globs
    ``*.parquet``, so a leftover duplicate would silently double-count that stream.
    """
    if not os.path.isdir(output_dir):
        return []
    # Deliberately no "canonical exists -> return early" shortcut: a directory can hold
    # BOTH names for one stream, and returning only the canonical one would leave the
    # alternate file for the retry to miss.
    target = re.sub(r"_+", "_", parquet_stem(placement, setting, replicate))
    return sorted(
        os.path.join(output_dir, name)
        for name in os.listdir(output_dir)
        if name.endswith(".parquet") and re.sub(r"_+", "_", name[: -len(".parquet")]) == target
    )


def check_if_processed(placement, setting, replicate, output_dir):
    """Check if a Parquet file already exists for a given placement, reactor setting, and replicate."""
    # Fast path for the common case; this runs once per stream in the discovery loop.
    canonical = os.path.join(output_dir, f"{parquet_stem(placement, setting, replicate)}.parquet")
    if os.path.exists(canonical):
        return True
    return bool(existing_parquet_paths(placement, setting, replicate, output_dir))


def is_blank_frame(img_bgr: numpy.ndarray, mean_thresh: float = DEFAULT_BLANK_MEAN_THRESH) -> bool:
    """Return True for a blank/black acquisition frame that carries no analyzable content.

    Some acquisitions (observed in the water / 000-xanthan settings) write one black
    frame per 50-frame burst with the illumination off. Such frames are all-zero apart
    from trace sensor speckle, so the exact-zero guard ``numpy.all(img == 0)`` misses
    them; the segmentation model then paints a single large low-confidence mask over the
    lit sensor area and yields a spurious full-frame "bubble" (~59 % of the image, an
    ~11 mm perfectly-round "sphere") that can dominate the gas-holdup estimate.

    Detection is by whole-frame mean intensity. The default threshold is deliberately
    below the illuminated-image range and can be overridden for another acquisition.

    Callers should handle an unreadable image (``cv2.imread`` returning ``None``)
    separately -- that is a load failure, not a blank frame -- so this expects a loaded
    BGR array.

    Parameters
    ----------
    img_bgr : numpy.ndarray
        Loaded BGR image (H, W, 3).
    mean_thresh : float
        Whole-frame mean-intensity threshold below which the frame is blank.
    """
    return float(img_bgr.mean()) < mean_thresh


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
    blank_mean_thresh: float = DEFAULT_BLANK_MEAN_THRESH,
    conf_large: float = DEFAULT_CONF_LARGE,
    large_frac: float = DEFAULT_LARGE_FRAC,
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
            # Skip images that fail to load or are blank/black acquisition frames
            # (illumination off; ~one per burst). Not counted as valid images so the
            # 50-frame burst indexing stays aligned. See is_blank_frame.
            if im0 is None or is_blank_frame(im0, blank_mean_thresh):
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
                conf_large=conf_large,
                large_frac=large_frac,
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
                bubble["image_filename"] = image_name
                bubble["burst_index"] = burst_index
                bubble["image_number_in_burst"] = image_number_in_burst
                bubble_data.append(bubble)

            valid_image_count += 1
            pbar.set_description(
                f"{placement}/{setting}/{replicate}: {valid_image_count}/{len(images)} processed"
            )
            pbar.update(1)
    return bubble_data


def _use_sphere_model(size_thresh: bool, near_spherical: bool, d_mm_sphere: float) -> bool:
    """Sphere-vs-ellipsoid gate for the ``hybrid`` geometry mode.

    A bubble is modelled as a sphere when it is small enough (``size_thresh``) OR its
    in-plane aspect ratio is within tolerance (``near_spherical``) -- but only when a
    valid sphere-equivalent diameter is available (guards against NaN sphere metrics for
    degenerate masks or unknown pixel size).

    The parentheses are load-bearing. Written inline as
    ``size_thresh or near_spherical and not isnan(d)`` Python binds it as
    ``size_thresh or (near_spherical and not isnan(d))`` (``and`` has higher precedence
    than ``or``), which would select the sphere model for a small bubble even when its
    sphere diameter is NaN, writing NaN into the chosen volume. Grouping the two
    sphere criteria before the NaN guard is the intended logic.
    """
    return (size_thresh or near_spherical) and not math.isnan(d_mm_sphere)


def mask_solidity(mask: numpy.ndarray) -> float:
    """Solidity of a binary mask: foreground pixel area / convex-hull area, in (0, 1].

    A clean single-bubble silhouette is convex to good approximation (solidity
    near 1); a mask that actually covers several overlapping bubbles has concave
    waists where the outlines meet, pulling solidity down. Stored per bubble so suspected merged
    detections can be screened DOWNSTREAM — no hard drop is baked into the
    pipeline.

    The hull is taken over all foreground contour points (multi-blob masks use
    the union), the area is the foreground pixel count (consistent with the
    ``mask_area`` column). Pixel-count area can slightly exceed the polygonal
    hull area for small convex shapes, so the ratio is capped at 1.0. Returns
    NaN for degenerate masks (< 3 contour points or zero hull area).
    """
    m = (mask > 0).astype(numpy.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return float("nan")
    pts = numpy.concatenate([c[:, 0, :] for c in cnts])
    if len(pts) < 3:
        return float("nan")
    hull = cv2.convexHull(pts)
    hull_area = float(cv2.contourArea(hull))
    if hull_area <= 0:
        return float("nan")
    return min(float(m.sum()) / hull_area, 1.0)


def _all_finite(*values: float) -> bool:
    """True when every value is a finite number (not NaN, not +/-inf)."""
    return all(math.isfinite(v) for v in values)


class BubbleBand(typing.NamedTuple):
    """Per-bubble prolate/oblate band for one bubble, volume and area kept separate.

    ``*_lower``/``*_upper`` are true intervals, each sorted on its own quantity.
    Volume and surface-area intervals are sorted independently. ``status`` records which
    case produced the row.
    """

    d_lower: float
    volume_lower: float
    surface_lower: float
    d_upper: float
    volume_upper: float
    surface_upper: float
    status: str


# Band row provenance. ``two_models`` is the normal ellipsoid case; ``sphere_degenerate``
# is a sphere-classified bubble (no depth ambiguity by construction); ``sphere_fallback``
# is a failed ellipse fit, so no spheroid exists; ``one_model_only`` means exactly one of
# the two models is complete and the band could not be formed from it; ``invalid`` means
# neither is usable. The last two must stay visible rather than silently collapsing to the
# sphere value, which is why the status is stored per bubble.
BAND_STATUS_TWO_MODELS = "two_models"
BAND_STATUS_SPHERE_DEGENERATE = "sphere_degenerate"
BAND_STATUS_SPHERE_FALLBACK = "sphere_fallback"
BAND_STATUS_ONE_MODEL_ONLY = "one_model_only"
BAND_STATUS_INVALID = "invalid"


def _band_status_for(band: BubbleBand, model_used: str) -> str:
    """Refine the band status with what the model label knows.

    ``_spheroid_band`` sees only ``as_sphere`` and cannot tell a bubble classified as a
    sphere (both spheroid models available, band degenerate by construction) from one
    whose two-half-ellipse fit failed (no spheroid exists at all). The distinction
    matters — the second is missing data, the first is not — so it is resolved here,
    where the chosen model label is known.
    """
    if band.status == BAND_STATUS_SPHERE_DEGENERATE and model_used == "sphere_fallback":
        return BAND_STATUS_SPHERE_FALLBACK
    return band.status


def _spheroid_band_unset() -> BubbleBand:
    """All-NaN band for a bubble whose geometry never resolved (no pixel size, no fit).

    Used as the initial value so a bubble that reaches row assembly without passing
    through any model branch is recorded as ``invalid`` rather than as a silent NaN of
    unknown provenance.
    """
    nan = float("nan")
    return BubbleBand(nan, nan, nan, nan, nan, nan, BAND_STATUS_INVALID)


def _spheroid_band(
    as_sphere: bool,
    d_mm_sphere: float,
    volume_sphere: float,
    surface_sphere: float,
    volume_prolate: float,
    surface_prolate: float,
    volume_oblate: float,
    surface_oblate: float,
) -> BubbleBand:
    """Classification-respecting spheroid model band for one bubble.

    The out-of-plane depth axis is unobserved, so a NON-spherical bubble's 3D
    volume depends on the assumed spheroid of revolution: prolate (depth =
    in-plane minor; ``V = pi*a/3 * (b1^2+b2^2)``) or oblate/Mikaelian (depth =
    in-plane major; ``V = pi*a^2/6 * (b1+b2)``). Both are stored so aggregates
    can be reported as a band; note the two models bound the true volume only
    for a bubble viewed edge-on (symmetry axis in the image plane) — for tilted
    bubbles they are the two spheroid limits, not strict bounds.

    Volume and area intervals are formed INDEPENDENTLY, each sorted on its own
    quantity, because a fore/aft-asymmetric fit can give the two quantities different
    model orderings.

    For a symmetric fit (b1 == b2) the ratio V_oblate/V_prolate = a/(2b) >= 1, so
    the volume-lower member is prolate; for a strongly asymmetric fit the general
    ratio a*(b1+b2) / (2*(b1^2+b2^2)) can drop below 1, hence min/max rather than a
    fixed assignment. ``*_chosen`` remains the oblate (Mikaelian) values.

    Diameters are volume-equivalent: ``d = (6V/pi)^(1/3)``, and therefore follow
    the VOLUME interval.
    """
    if as_sphere:
        return BubbleBand(
            d_mm_sphere,
            volume_sphere,
            surface_sphere,
            d_mm_sphere,
            volume_sphere,
            surface_sphere,
            BAND_STATUS_SPHERE_DEGENERATE,
        )

    # A band needs BOTH models complete: two volumes AND their two surfaces.
    pro_ok = _all_finite(volume_prolate, surface_prolate)
    ob_ok = _all_finite(volume_oblate, surface_oblate)
    if not (pro_ok and ob_ok):
        status = BAND_STATUS_ONE_MODEL_ONLY if (pro_ok or ob_ok) else BAND_STATUS_INVALID
        nan = float("nan")
        return BubbleBand(nan, nan, nan, nan, nan, nan, status)

    # Volume interval.
    if volume_prolate <= volume_oblate:  # ties -> prolate is the lower member
        v_lo, v_up = volume_prolate, volume_oblate
    else:
        v_lo, v_up = volume_oblate, volume_prolate

    # Area interval, sorted on area alone.
    s_lo = min(surface_prolate, surface_oblate)
    s_up = max(surface_prolate, surface_oblate)

    d_lower = (6.0 * v_lo / math.pi) ** (1.0 / 3.0) if v_lo > 0 else float("nan")
    d_upper = (6.0 * v_up / math.pi) ** (1.0 / 3.0) if v_up > 0 else float("nan")
    return BubbleBand(
        d_lower,
        v_lo,
        s_lo,
        d_upper,
        v_up,
        s_up,
        BAND_STATUS_TWO_MODELS,
    )


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
    geom_mode: str = DEFAULT_PROCESSING_CONFIG.geometry_mode,
    sphere_size_thresh: typing.Optional[int] = 100,  # threshold at 100 px for eq_d_px
    sphere_if_aspect_tol: typing.Optional[
        float
    ] = DEFAULT_PROCESSING_CONFIG.sphere_aspect_tolerance,
    show_axes: bool = False,  # show axes, lines, and measurement points on ellipsoid overlays
    conf_large: float = DEFAULT_CONF_LARGE,
    large_frac: float = DEFAULT_LARGE_FRAC,
) -> list[dict]:
    """
    Segment one image and compute per-bubble measurements in ORIGINAL pixel space,
    with optional physics in mm if `pixel_size_mm` is provided.

    Coordinate systems / units (important):
    - All geometry (boxes, centroids, areas) is in the ORIGINAL image pixel space (W, H).
    - Fields with `_mm` are physical units derived using `pixel_size_mm`.
    - `img_w` / `img_h` (original) are stored per row for drift-free downstream overlays.

    Outputs (one dictionary per bubble):
      - Localization:
          bbox_x1..y2 (px), centroid_x/centroid_y (px),
          mask_area (px^2), equivalent_diameter (px), score/confidence, img_w/img_h,
          solidity (mask area / convex-hull area; merge/truncation screen)
      - Ellipse/ellipsoid diagnostics (pixels/mm):
          cx_px, cy_px, angle_deg, a_mm, b1_mm, b2_mm
      - Sphere & ellipsoid physics (mm / mm^2 / mm^3), if pixel_size_mm is known:
          d_mm_sphere, volume_mm3_sphere, surface_area_mm2_sphere
          volume_mm3_ellipsoid, surface_area_mm2_ellipsoid
          volume_mm3_oblate, surface_area_mm2_oblate         (depth = in-plane major)
          volume_mm3_prolate, surface_area_mm2_prolate       (depth = in-plane minor)
      - Hybrid decision:
          aspect_ratio, aspect_delta, aspect_method,
          d_mm_chosen, volume_mm3_chosen, surface_area_mm2_chosen,
          model_used in {"sphere", "asym_ellipsoid", "sphere_fallback", "unknown"}
      - Classification-respecting spheroid band (see _spheroid_band; sphere-class
        bubbles have lower == upper == sphere; ellipsoid-class = elementwise
        volume min/max of the prolate/oblate pair — usually prolate/oblate in
        that order, swapped for strongly fore/aft-asymmetric fits; `_chosen`
        stays the oblate values):
          d_mm_lower, volume_mm3_lower, surface_area_mm2_lower,
          d_mm_upper, volume_mm3_upper, surface_area_mm2_upper

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
        Nominal mask-binarization threshold. Ultralytics ``Results.masks.data`` is
        already binary (the segmentation post-process thresholds the mask prototypes at
        0.5), so ``mask > binarize_thr`` is effectively a no-op for any value in (0, 1);
        it is retained as a defensive guard. Mask granularity is set by the model's mask
        resolution, not by this value.
    overlay_dir : Optional[str], optional
        Directory to save visual overlays. If None, no files are written.
    save_masks_overlay : bool, optional
        If True, writes a semi-transparent mask overlay.
    save_fit_overlay : bool, optional
        If True, writes the fitted ellipsoid for every successful in-plane fit. A
        circle is drawn only when the ellipsoid fit failed and the sphere fallback is
        the sole available measurement. Overlay geometry is independent of the legacy
        sphere/ellipsoid classification used by ``model_used`` and ``*_chosen``.
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
            conf_large=conf_large,
            large_frac=large_frac,
        )
    else:
        dets = yolo_segment_image(
            model,
            img_bgr,
            conf=conf,
            iou=iou,
            binarize_thr=binarize_thr,
            conf_large=conf_large,
            large_frac=large_frac,
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
        solidity = mask_solidity(mask)

        # Image-border clip length: a bubble cut by the frame has its rounded
        # outline replaced by a straight segment lying on the image edge. We
        # measure that chord as the extent of mask touching each border (columns
        # along top/bottom + rows along left/right), within a 1 px margin. 0 for
        # a fully-visible bubble; large for a heavily clipped one. The chord-to-
        # diameter ratio (border_contact_px / equivalent_diameter) is the
        # severity used downstream to drop partial edge bubbles.
        _em = 1  # edge margin (px)
        border_contact_px = int(
            mask[: _em + 1, :].any(axis=0).sum()
            + mask[H - _em - 1 :, :].any(axis=0).sum()
            + mask[:, : _em + 1].any(axis=1).sum()
            + mask[:, W - _em - 1 :].any(axis=1).sum()
        )
        # The published edge policy is applied downstream by drop_border_clipped:
        # retain a border-touching detection when border_contact_px / equivalent_diameter
        # is <= 1, and exclude it only when the ratio is > 1. Geometry is always measured
        # from the observed mask; no off-frame arc reconstruction or size substitution is
        # applied.

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
        V_pro = S_pro = float("nan")
        cx_fit = cy_fit = ang_deg = float("nan")

        # Sphere metrics (if pixel size known)
        if ps is not None and area_px > 0:
            d_mm_sph, V_sph, S_sph = sphere_metrics_from_mask(mask, ps)

        # Ellipsoid metrics (if pixel size known). In-plane fit: Mikaelian
        # two-half-ellipse (fore/aft asymmetric semi-minor axes), with a
        # cv2.fitEllipse fallback for the rare masks where it fails.
        est = None
        if ps is not None:
            est = estimate_a_b1_b2_ellipsoid(mask, ps)
            if est is not None:
                cx_fit, cy_fit, ang_deg, a_mm, b1_mm, b2_mm = est
                # The depth axis is unobserved, so BOTH spheroid revolutions of
                # the same 2D fit are computed and stored (see _spheroid_band):
                #   oblate (Mikaelian, depth = major): V = pi*a^2/6 * (b1+b2)
                #   prolate (depth = minor):           V = pi*a/3  * (b1^2+b2^2)
                # V_ell/S_ell keep the oblate values used by the `_ellipsoid` columns
                # and the `_chosen` path; the band stores the volume min/max.
                V_ell, S_ell = volume_surface_oblate(a_mm, b1_mm, b2_mm)
                V_ell, S_ell = float(V_ell), float(S_ell)
                V_pro, S_pro = volume_surface_prolate(a_mm, b1_mm, b2_mm)
                V_pro, S_pro = float(V_pro), float(S_pro)

        # Choose the legacy model used by the ``*_chosen`` output columns. Fit overlays
        # are drawn independently below: every successful in-plane fit is shown as an
        # ellipsoid, including masks classified as spheres by this compatibility path.
        model_used = "unknown"
        V_ch = S_ch = float("nan")
        d_mm_ch = float("nan")
        # classification-respecting band (volume min/max of prolate/oblate; see
        # _spheroid_band). Set alongside the chosen model in each branch below.
        band = _spheroid_band_unset()

        if geom_mode == "sphere_only":
            model_used = "sphere"
            V_ch, S_ch = V_sph, S_sph
            d_mm_ch = d_mm_sph
            band = _spheroid_band(True, d_mm_sph, V_sph, S_sph, V_pro, S_pro, V_ell, S_ell)

        elif geom_mode == "ellipsoid_only":
            model_used = "asym_ellipsoid" if est is not None else "sphere_fallback"
            if est is not None:
                V_ch, S_ch = V_ell, S_ell
                if V_ch > 0.0:
                    d_mm_ch = (6.0 * V_ch / math.pi) ** (1.0 / 3.0)
                band = _spheroid_band(False, d_mm_sph, V_sph, S_sph, V_pro, S_pro, V_ell, S_ell)
            elif ps is not None and not math.isnan(d_mm_sph):
                V_ch, S_ch = V_sph, S_sph
                d_mm_ch = d_mm_sph
                band = _spheroid_band(True, d_mm_sph, V_sph, S_sph, V_pro, S_pro, V_ell, S_ell)

        else:  # "hybrid"
            if _use_sphere_model(size_thresh, near, d_mm_sph):
                model_used = "sphere"
                V_ch, S_ch = V_sph, S_sph
                d_mm_ch = d_mm_sph
                band = _spheroid_band(True, d_mm_sph, V_sph, S_sph, V_pro, S_pro, V_ell, S_ell)
            else:
                model_used = "asym_ellipsoid" if est is not None else "sphere_fallback"
                if est is not None:
                    V_ch, S_ch = V_ell, S_ell
                    if V_ch > 0.0:
                        d_mm_ch = (6.0 * V_ch / math.pi) ** (1.0 / 3.0)
                    band = _spheroid_band(False, d_mm_sph, V_sph, S_sph, V_pro, S_pro, V_ell, S_ell)
                elif ps is not None and not math.isnan(d_mm_sph):
                    V_ch, S_ch = V_sph, S_sph
                    d_mm_ch = d_mm_sph
                    band = _spheroid_band(True, d_mm_sph, V_sph, S_sph, V_pro, S_pro, V_ell, S_ell)

        # The overlay is a fit diagnostic, not a visualization of the legacy hybrid
        # classification. Draw every available two-half-ellipse fit. Only a genuinely
        # failed ellipse fit uses the circle fallback, because in that case the
        # area-equivalent sphere is the sole available geometry.
        if save_fit_overlay and overlay_dir and ps is not None and fit_overlay is not None:
            if est is not None:
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
            elif not math.isnan(d_mm_sph):
                r_px = (d_mm_sph / 2.0) / ps
                fit_overlay = draw_circle_overlay(
                    fit_overlay, centroid_x, centroid_y, r_px, thickness=2
                )

        # Assemble the public row schema.
        if not math.isfinite(area_px) or area_px <= 0.0:
            measurement_status = "zero_area"
        else:
            measurement_status = "measurable"
        if border_contact_px <= 0.0:
            edge_status = "not_touching"
        elif math.isfinite(eq_d_px) and eq_d_px > 0.0 and border_contact_px / eq_d_px > 1.0:
            edge_status = "severe_excluded"
        else:
            edge_status = "moderate_retained"
        rows.append(
            {
                "image_path": str(image_path),
                "image": Path(image_path).name,
                "img_w": W,
                "img_h": H,
                "bubble_index": j,
                "measurement_status": measurement_status,
                "edge_status": edge_status,
                "score": score,
                "confidence": score,
                "conf_thresh": float(conf),
                "iou_thresh": float(iou),
                "mask_binarize_thr": float(binarize_thr),
                # bbox (px)
                "bbox_x1": bbox_x1,
                "bbox_y1": bbox_y1,
                "bbox_x2": bbox_x2,
                "bbox_y2": bbox_y2,
                # mask-derived fields (px / px^2)
                "mask_area": area_px,  # px^2
                "equivalent_diameter": eq_d_px,  # px
                # mask area / convex-hull area; < ~0.95 flags a possible merged
                # or truncated detection (screen downstream, no hard drop here)
                "solidity": solidity,
                "centroid_x": centroid_x,
                "centroid_y": centroid_y,
                # image-border clip length (px); 0 = fully visible. The clip
                # ratio border_contact_px / equivalent_diameter is the edge filter.
                "border_contact_px": border_contact_px,
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
                # ellipsoid (if ps provided). Both spheroid revolutions of the
                # same 2D fit are stored; `_ellipsoid` contains the oblate values.
                "volume_mm3_ellipsoid": V_ell,
                "surface_area_mm2_ellipsoid": S_ell,
                "volume_mm3_oblate": V_ell,
                "surface_area_mm2_oblate": S_ell,
                "volume_mm3_prolate": V_pro,
                "surface_area_mm2_prolate": S_pro,
                # aspect / decision
                "aspect_ratio": r_aspect,
                "aspect_delta": delta_aspect,
                "aspect_method": used_method,
                # chosen model (= oblate for ellipsoid-class bubbles — usually,
                # not always, the band's upper member; kept for compatibility)
                "d_mm_chosen": d_mm_ch,
                "volume_mm3_chosen": V_ch,
                "surface_area_mm2_chosen": S_ch,
                "model_used": model_used,
                # classification-respecting spheroid band (see _spheroid_band):
                # sphere-class -> lower == upper == sphere (no depth ambiguity);
                # ellipsoid-class -> volume min/max and, independently, area min/max of
                # the prolate/oblate pair.
                "d_mm_lower": band.d_lower,
                "volume_mm3_lower": band.volume_lower,
                "surface_area_mm2_lower": band.surface_lower,
                "d_mm_upper": band.d_upper,
                "volume_mm3_upper": band.volume_upper,
                "surface_area_mm2_upper": band.surface_upper,
                "band_status": _band_status_for(band, model_used),
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
    conf_large: float = DEFAULT_CONF_LARGE,
    large_frac: float = DEFAULT_LARGE_FRAC,
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
        Nominal mask-binarization threshold. Ultralytics ``Results.masks.data`` is
        already binary (the segmentation post-process thresholds the mask prototypes at
        0.5), so ``mask > binarize_thr`` is effectively a no-op for any value in (0, 1);
        it is retained as a defensive guard. Mask granularity is set by the model's mask
        resolution, not by this value.

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

    # Apply size-aware filtering IN-PLACE on the Ultralytics result. The gate measures
    # area fraction in the mask tensor's own (network) space, so no image size is passed.
    _size_aware_filter_result(
        det,
        conf_small=conf,
        conf_large=conf_large,
        large_frac=large_frac,
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
        # 1) Binarize in network space. masks.data is already 0/1 (thresholded at 0.5
        #    upstream), so this is a no-op guard for any binarize_thr in (0, 1).
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
    conf_large: float = DEFAULT_CONF_LARGE,
    large_frac: float = DEFAULT_LARGE_FRAC,
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
        Nominal mask-binarization threshold. Ultralytics ``Results.masks.data`` is
        already binary (post-process thresholds mask prototypes at 0.5), so
        ``mask > binarize_thr`` is a no-op for any value in (0, 1); kept as a guard.
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
    _size_aware_filter_result(
        det, conf_small=conf_small, conf_large=conf_large, large_frac=large_frac
    )

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


def drop_zero_area_masks(
    bubble_df: pandas.DataFrame, area_col: str = "mask_area"
) -> pandas.DataFrame:
    """Drop zero-area "phantom" detections from a bubble-level DataFrame.

    A YOLO detection can survive as a box + confidence while its segmentation mask
    thresholds (``MASK_THR``) to zero area. Such rows carry ``mask_area == 0`` and NaN
    volume/diameter (``model_used == "sphere_fallback"``); they are not physical bubbles.
    Volume, surface, and diameter aggregates already skip them (NaN-aware ``sum``/
    ``count``), but a raw ``size()`` bubble count would include them and inflate number
    density by ~1% (up to ~5% in low-shear / high-viscosity conditions) -- a condition-
    dependent bias. Removing them makes every downstream count consistent with the
    volume/diameter metrics. The source Parquet files retain every raw detection, so
    nothing is lost for traceability.

    Rows with NaN ``area_col`` are also dropped (non-measurable). Returns a filtered view;
    if ``area_col`` is absent the frame is returned unchanged.
    """
    if area_col not in bubble_df.columns:
        return bubble_df
    return bubble_df.loc[bubble_df[area_col] > 0]


# Published clip-severity threshold. A detection is excluded only when the straight chord
# that its mask lays on the image edge exceeds the mask's own area-equivalent diameter.
# Border contact alone is not an exclusion criterion: ratios <= 1 are retained without any
# geometric reconstruction or size substitution. The policy was selected from synthetic
# clipping against synthetic shapes with known geometry.
CLIP_RATIO_THRESHOLD = 1.0


def drop_border_clipped(
    bubble_df: pandas.DataFrame,
    *,
    border_col: str = "border_contact_px",
    diameter_px_col: str = "equivalent_diameter",
    threshold: float = CLIP_RATIO_THRESHOLD,
) -> pandas.DataFrame:
    """Apply the published border-contact exclusion policy.

    Severity is the dimensionless chord-to-diameter ratio
    ``border_contact_px / equivalent_diameter`` (both inputs are in pixels). Rows with a
    finite ratio strictly greater than ``threshold`` are excluded. Interior detections and
    border-touching detections with ratio <= ``threshold`` are retained with their original
    measured geometry. No off-frame boundary reconstruction, diameter adjustment, or
    containment weighting is applied.

    This is a size-biased deletion -- a large bubble is far more likely to touch the frame
    than a small one -- so the retained population under-represents the largest bubbles. The
    textbook remedy is Miles-Lantuejoul containment weighting ``w = 1/P(d)`` with
    ``P(d) = (W-d)(H-d)/(W*H)``, but it is not applied here: our largest detections are a
    substantial fraction of the 14.6 x 11.7 mm field of view, where ``P(d)`` collapses and
    ``1/P(d)`` becomes both huge and unstable (the weighted total is dominated by a handful
    of objects and depends entirely on where the weight is capped). The residual
    under-sampling of the very largest bubbles is reported as a stated limitation instead.

    Rows whose clip ratio is undefined -- NaN, or a non-positive equivalent diameter (a
    zero-area phantom mask) -- are kept, so this filter stays orthogonal to
    :func:`drop_zero_area_masks` and ``--keep-zero-area-masks`` really does keep them.
    ``threshold=1.0`` is the published policy; the argument remains configurable for
    sensitivity analyses. Returns a filtered view; if either column is absent the frame is
    returned unchanged.
    """
    if border_col not in bubble_df.columns or diameter_px_col not in bubble_df.columns:
        return bubble_df
    diameter = bubble_df[diameter_px_col]
    ratio = bubble_df[border_col].where(diameter > 0) / diameter.where(diameter > 0)
    return bubble_df.loc[~(ratio > threshold)]


def load_all_data_parquet(
    parquet_dir: Path,
    columns: typing.Optional[typing.List[str]] = None,
    placements: typing.Optional[typing.List[str]] = None,
    settings: typing.Optional[typing.List[str]] = None,
    set_index: bool = True,
) -> pandas.DataFrame:
    """
    Load data from Parquet files with optional filtering and column selection.

    This loader supports column and experiment filters for memory-efficient Parquet access.

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
    # Load every column
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


def load_yolo_model(model_path, device="auto"):
    """Load YOLO model for instance segmentation"""

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda:0"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
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
    output_file = os.path.join(output_dir, f"{parquet_stem(placement, setting, replicate)}.csv")
    df = pandas.DataFrame(data)
    df.to_csv(output_file, index=False)


# Columns written by process_image() that carry no analytical value.
_PARQUET_DROP_COLS = {
    "image_path",  # redundant: directory structure + image filename is sufficient
    "conf_thresh",  # constant per run, lives in config
    "iou_thresh",  # constant per run, lives in config
    "mask_binarize_thr",  # constant per run, lives in config
    "img_w",  # constant, lives in geometry.py
    "img_h",  # constant, lives in geometry.py
    "has_overlay",  # process metadata, not science
    "aspect_method",  # debug string
}


def _optimize_parquet_dtypes(df: pandas.DataFrame) -> pandas.DataFrame:
    """Downcast numeric columns and categorise low-cardinality strings."""
    for col in df.columns:
        dtype = df[col].dtype
        if dtype == "float64":
            df[col] = df[col].astype("float32")
        elif dtype == "int64":
            col_min, col_max = df[col].min(), df[col].max()
            if col_min >= 0:
                if col_max < 255:
                    df[col] = df[col].astype("uint8")
                elif col_max < 65_535:
                    df[col] = df[col].astype("uint16")
                elif col_max < 4_294_967_295:
                    df[col] = df[col].astype("uint32")
            else:
                if col_min > -128 and col_max < 127:
                    df[col] = df[col].astype("int8")
                elif col_min > -32_768 and col_max < 32_767:
                    df[col] = df[col].astype("int16")
                elif col_min > -2_147_483_648 and col_max < 2_147_483_647:
                    df[col] = df[col].astype("int32")
        elif dtype == "object":
            if df[col].nunique() / len(df[col]) < 0.5:
                df[col] = df[col].astype("category")
    return df


def save_to_parquet(data, placement, setting, replicate, output_dir, compression="zstd"):
    """Save extracted bubble data as a trimmed, typed Parquet file."""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{parquet_stem(placement, setting, replicate)}.parquet")
    df = pandas.DataFrame(data)
    df = df.drop(columns=[c for c in _PARQUET_DROP_COLS if c in df.columns])
    df = _optimize_parquet_dtypes(df)
    # Atomic write: a killed/OOM-ed worker must never leave a truncated Parquet that
    # check_if_processed() (existence-only) would treat as done on resume. Write to a
    # per-process temp file, then atomically rename onto the final path (same filesystem).
    tmp_file = f"{output_file}.tmp.{os.getpid()}"
    df.to_parquet(tmp_file, engine="pyarrow", compression=compression, index=False)
    os.replace(tmp_file, output_file)


# ============================================================
# Bubble geometry
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


def estimate_a_b1_b2_ellipsoid(
    mask: numpy.ndarray,
    pixel_size_mm: float,
) -> typing.Optional[typing.Tuple[float, float, float, float, float, float]]:
    """In-plane ellipsoid fit feeding the oblate volume model.

    Primary: Mikaelian two-half-ellipse least-squares fit (independent fore/aft
    semi-minor axes b1, b2). Fallback: symmetric ``cv2.fitEllipse`` (b1 = b2) for
    the rare masks where the two-half fit fails or returns implausible axes. If
    both fail, returns ``None`` (the caller then uses the sphere fallback).

    Returns ``(cx, cy, angle_deg, a_mm, b1_mm, b2_mm)`` where ``a_mm`` is the
    FULL major-axis length (tip to tip) and ``b1_mm``/``b2_mm`` are the front/
    rear SEMI-minor axes -- the exact convention ``volume_surface_oblate``
    expects. Note ``fit_two_half_ellipse_lsq`` reports a SEMI-major axis, so it
    is doubled here to the full major length.
    """
    m8 = (mask > 0).astype(numpy.uint8)
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if len(cnt) < 5:
        return None

    # Sanity bound: reject degenerate fits whose axes blow past the mask extent.
    _, _, w_box, h_box = cv2.boundingRect(cnt)
    max_axis_px = 1.5 * math.hypot(float(w_box), float(h_box))

    def _plausible(*axes_px: float) -> bool:
        return all(numpy.isfinite(v) and v > 0 for v in axes_px) and max(axes_px) <= max_axis_px

    # Primary: Mikaelian two-half-ellipse (A_semi_px is the SEMI-major axis).
    fit = fit_two_half_ellipse_lsq(m8)
    if fit is not None:
        cx, cy, angle_deg, a_semi_px, b1_px, b2_px = fit
        a_full_px = 2.0 * a_semi_px  # -> full major-axis length for the oblate model
        if _plausible(a_full_px, b1_px, b2_px):
            return (
                float(cx),
                float(cy),
                float(angle_deg),
                float(a_full_px * pixel_size_mm),  # FULL major axis
                float(b1_px * pixel_size_mm),  # front semi-minor
                float(b2_px * pixel_size_mm),  # rear semi-minor
            )

    # Fallback: symmetric cv2.fitEllipse (returns FULL axis lengths). Set the
    # orientation to the major axis so the asymmetry-aware overlay stays correct.
    (cx, cy), (axis1, axis2), angle_deg = cv2.fitEllipse(cnt)
    major_px, minor_px = max(axis1, axis2), min(axis1, axis2)
    angle_major = angle_deg if axis1 >= axis2 else angle_deg + 90.0
    b_semi_px = minor_px / 2.0  # semi-minor (front = rear for a symmetric fit)
    if _plausible(major_px, b_semi_px):
        return (
            float(cx),
            float(cy),
            float(angle_major),
            float(major_px * pixel_size_mm),  # FULL major axis
            float(b_semi_px * pixel_size_mm),
            float(b_semi_px * pixel_size_mm),
        )
    return None


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


def _size_gate_keep_indices(
    mask_areas_px: numpy.ndarray,
    mask_canvas_area_px: float,
    confs: numpy.ndarray,
    conf_small: float,
    conf_large: float,
    large_frac: float,
) -> list[int]:
    """Indices surviving the size-aware confidence gate.

    Large masks (area fraction >= ``large_frac``) must clear ``conf_large``; smaller
    masks only need ``conf_small`` (a large false positive is the costlier error).

    ``mask_areas_px`` and ``mask_canvas_area_px`` MUST be in the SAME coordinate space.
    The gate operates on ``det.masks.data``, which lives at the network/letterbox
    resolution, so both the per-instance foreground counts and the canvas area are taken
    there. The denominator is therefore the mask-tensor canvas area, not the source-image
    area.
    """
    if mask_canvas_area_px <= 0 or len(mask_areas_px) == 0:
        return list(range(len(mask_areas_px)))
    fracs = mask_areas_px / mask_canvas_area_px
    keep: list[int] = []
    for i in range(len(fracs)):
        need = conf_large if fracs[i] >= large_frac else conf_small
        if float(confs[i]) >= need:
            keep.append(i)
    return keep


def _size_aware_filter_result(
    det,
    conf_small: float,
    conf_large: float = DEFAULT_CONF_LARGE,
    large_frac: float = DEFAULT_LARGE_FRAC,
):
    """
    In-place size-aware confidence filter for an Ultralytics ``Results`` object:
      • small masks (area fraction < ``large_frac``) → need ≥ ``conf_small``
      • large masks (area fraction ≥ ``large_frac``) → need ≥ ``conf_large``

    The area fraction is measured in the mask tensor's own (network/letterbox) space; see
    :func:`_size_gate_keep_indices` for why the original image area must not be used.
    """
    if det.masks is None or det.boxes is None:
        return

    masks = det.masks.data
    confs = det.boxes.conf.cpu().numpy()
    # Canvas area in the SAME space as the per-instance foreground counts below.
    mask_canvas_area = float(masks.shape[1] * masks.shape[2]) if masks.ndim == 3 else 0.0
    mask_areas = numpy.array(
        [float(masks[i].sum().item()) for i in range(masks.shape[0])], dtype=float
    )
    keep = _size_gate_keep_indices(
        mask_areas, mask_canvas_area, confs, conf_small, conf_large, large_frac
    )

    if not keep:
        det.boxes = det.boxes[:0]
        det.masks.data = det.masks.data[:0]
    else:
        det.boxes = det.boxes[keep]
        det.masks.data = det.masks.data[keep]
