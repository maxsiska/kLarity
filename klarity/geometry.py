# ============================================================
# Image geometry
# ============================================================

FOV_WIDTH_MM = 14.6
FOV_HEIGHT_MM = 11.7
IMG_WIDTH_PX = 1280
IMG_HEIGHT_PX = 1024

OBS_DEPTH_MM = 40.0

# Pixel size calibration (example: sensor FoV 14.6x11.7 mm on 1280x1024 px)
px_size_x = FOV_WIDTH_MM / IMG_WIDTH_PX  # mm per pixel on the x-axis
px_size_y = FOV_HEIGHT_MM / IMG_HEIGHT_PX  # mm per pixel on the y-axis

pixel_size_mm = (px_size_x + px_size_y) / 2  # global fallback


def get_fov_mm() -> tuple[float, float]:
    """(width_mm, height_mm) of the imaged field of view."""
    return float(FOV_WIDTH_MM), float(FOV_HEIGHT_MM)


def get_image_shape_px() -> tuple[int, int]:
    """(width_px, height_px) for the calibration that defines px_size_*."""
    return int(IMG_WIDTH_PX), int(IMG_HEIGHT_PX)


def get_px_size_mm() -> tuple[float, float, float]:
    """(px_size_x_mm, px_size_y_mm, pixel_size_mm_global)."""
    return float(px_size_x), float(px_size_y), float(pixel_size_mm)


def get_observed_volume_mm3(depth_mm: float | None = None) -> float:
    """Observed control volume in mm^3, using FoV width x height x depth."""
    d = float(OBS_DEPTH_MM if depth_mm is None else depth_mm)
    return float(FOV_WIDTH_MM * FOV_HEIGHT_MM * d)
