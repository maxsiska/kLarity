"""Fit-overlay tests.

Overlays visualize the measured in-plane fit, independently of the legacy
sphere/ellipsoid classification retained in the raw processing output.
"""

from __future__ import annotations

import cv2
import numpy as np

from klarity import parsing


def _write_image(path) -> None:
    cv2.imwrite(str(path), np.full((80, 100, 3), 180, dtype=np.uint8))


def _disk_mask() -> np.ndarray:
    mask = np.zeros((80, 100), dtype=np.uint8)
    cv2.circle(mask, (50, 40), 15, 1, thickness=-1)
    return mask.astype(bool)


def test_sphere_classified_mask_draws_available_ellipsoid_fit(tmp_path, monkeypatch):
    """A legacy sphere label must not replace a successful fit with a circle."""
    image_path = tmp_path / "frame.png"
    _write_image(image_path)
    mask = _disk_mask()

    monkeypatch.setattr(
        parsing,
        "yolo_segment_image",
        lambda *args, **kwargs: [{"mask": mask, "score": 0.9, "bbox": None}],
    )
    monkeypatch.setattr(
        parsing,
        "estimate_a_b1_b2_ellipsoid",
        lambda *args, **kwargs: (50.0, 40.0, 0.0, 0.30, 0.15, 0.15),
    )

    drawn: list[str] = []

    def record_ellipsoid(rgb, *args, **kwargs):
        drawn.append("ellipsoid")
        return rgb

    def record_circle(rgb, *args, **kwargs):
        drawn.append("circle")
        return rgb

    monkeypatch.setattr(parsing, "draw_asymmetric_ellipsoid_overlay", record_ellipsoid)
    monkeypatch.setattr(parsing, "draw_circle_overlay", record_circle)

    rows = parsing.process_image(
        image_path,
        model=None,
        overlay_dir=tmp_path / "overlays",
        save_fit_overlay=True,
        pixel_size_mm=0.01,
        geom_mode="hybrid",
        sphere_size_thresh=1_000,
    )

    assert rows[0]["model_used"] == "sphere"
    assert drawn == ["ellipsoid"]


def test_failed_ellipsoid_fit_draws_sphere_fallback(tmp_path, monkeypatch):
    """A circle remains useful when no ellipsoid fit exists to visualize."""
    image_path = tmp_path / "frame.png"
    _write_image(image_path)
    mask = _disk_mask()

    monkeypatch.setattr(
        parsing,
        "yolo_segment_image",
        lambda *args, **kwargs: [{"mask": mask, "score": 0.9, "bbox": None}],
    )
    monkeypatch.setattr(parsing, "estimate_a_b1_b2_ellipsoid", lambda *args, **kwargs: None)

    drawn: list[str] = []

    def record_ellipsoid(rgb, *args, **kwargs):
        drawn.append("ellipsoid")
        return rgb

    def record_circle(rgb, *args, **kwargs):
        drawn.append("circle")
        return rgb

    monkeypatch.setattr(parsing, "draw_asymmetric_ellipsoid_overlay", record_ellipsoid)
    monkeypatch.setattr(parsing, "draw_circle_overlay", record_circle)

    parsing.process_image(
        image_path,
        model=None,
        overlay_dir=tmp_path / "overlays",
        save_fit_overlay=True,
        pixel_size_mm=0.01,
        geom_mode="ellipsoid_only",
    )

    assert drawn == ["circle"]
