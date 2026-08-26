"""Canonical scientific parameters for image processing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProcessingConfig:
    """Parameters that affect detections or measured bubble geometry."""

    confidence: float = 0.22
    iou: float = 0.35
    mask_threshold: float = 0.30
    large_mask_confidence: float = 0.22
    large_mask_fraction: float = 0.08
    blank_mean_threshold: float = 15.0
    geometry_mode: str = "hybrid"
    sphere_aspect_tolerance: float = 0.10

    def __post_init__(self) -> None:
        for name in (
            "confidence",
            "iou",
            "mask_threshold",
            "large_mask_confidence",
            "large_mask_fraction",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value!r}")
        if self.blank_mean_threshold < 0.0:
            raise ValueError("blank_mean_threshold must be non-negative")
        if self.geometry_mode not in {"sphere_only", "ellipsoid_only", "hybrid"}:
            raise ValueError(f"unsupported geometry_mode: {self.geometry_mode!r}")
        if self.sphere_aspect_tolerance < 0.0:
            raise ValueError("sphere_aspect_tolerance must be non-negative")


DEFAULT_PROCESSING_CONFIG = ProcessingConfig()
