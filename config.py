"""
Project configuration.

All paths are resolved relative to this file (the project root), so the
project works regardless of where scripts are invoked from.

Edit the values in each section to match your local setup before running
scripts or notebooks.
"""

import os
from pathlib import Path

from klarity.processing_config import DEFAULT_PROCESSING_CONFIG

ROOT = Path(__file__).parent


# ── Data ──────────────────────────────────────────────────────────────────────

# Raw images produced by the endoscope cameras
IMAGE_DIR = ROOT / "images"

# Per-replicate Parquet files written and consumed by the pipeline. An environment
# override is useful on clusters and for read-only downloaded datasets.
OUTPUT_DIR = Path(os.environ.get("KLARITY_OUTPUT_DIR", ROOT / "output"))
OUTPUT_METADATA_DIR = OUTPUT_DIR / "metadata"

# Pre-computed aggregates used by notebooks
BUBBLE_LEVEL_PKL = ROOT / "data" / "bubble_level_df.pkl"
FRAME_LEVEL_PKL = ROOT / "data" / "frame_level_df.pkl"
FRAME_CENSUS_PARQUET = OUTPUT_METADATA_DIR / "frame_census.parquet"
FRAME_CENSUS_SUMMARY = OUTPUT_METADATA_DIR / "frame_census_summary.csv"
FRAME_FILTER_SUMMARY = OUTPUT_METADATA_DIR / "frame_filter_summary.parquet"

# Reference spreadsheets
KLA_XLSX = ROOT / "data" / "kla_data_000_xanthan.xlsx"
RHEOLOGY_XLSX = ROOT / "data" / "xanthan_rheology.xlsx"


# ── Model ─────────────────────────────────────────────────────────────────────

MODEL_PATH = ROOT / "models" / "klarity-model.pt"

# Scientific processing parameters used by every entry point.
PROCESSING_CONFIG = DEFAULT_PROCESSING_CONFIG
CONF = PROCESSING_CONFIG.confidence
IOU = PROCESSING_CONFIG.iou
MASK_THR = PROCESSING_CONFIG.mask_threshold

# The size-aware gate is inactive when both confidence thresholds are equal.
CONF_LARGE = PROCESSING_CONFIG.large_mask_confidence
LARGE_FRAC = PROCESSING_CONFIG.large_mask_fraction


# ── Processing ────────────────────────────────────────────────────────────────

# Blank-frame rejection. Some acquisitions (observed in the 000-xanthan / water
# settings) write one black frame per 50-frame burst with the illumination off.
# These frames are all-zero apart from sensor speckle, so an exact-zero test misses
# them and the segmentation model paints a spurious full-frame mask that becomes a
# fake ~11 mm "bubble" dominating the gas-holdup estimate. Frames whose whole-frame
# mean intensity is below this threshold are skipped (not counted as valid frames).
# The threshold is deliberately far below the illuminated-image range.
BLANK_FRAME_MEAN_THRESH = PROCESSING_CONFIG.blank_mean_threshold

# Overlay images written alongside output CSVs (set to "none" to skip)
OVERLAYS_DIR = ROOT / "overlays"
OVERLAY_MODE = "every_10th"  # "all" | "every_10th" | "every_5th" | "first_only" | "none"

# GPU selection for process_images.py and train_model.py
# Examples: 0  |  [0, 1]  |  [1, 2, 3]  |  "cpu"  |  "mps"
# The parallel driver resolves available devices at launch.
DEVICE = "auto"


# ── Evaluation ────────────────────────────────────────────────────────────────

EVAL_DATASET_DIR = ROOT / "model_eval" / "KLarity-18"
BENCHMARK_OUT_DIR = ROOT / "model_eval" / "model_benchmark_output"


# ── Notebook outputs ──────────────────────────────────────────────────────────

SETTING_COMPARISON_DIR = ROOT / "setting_comparison"
HEATMAPS_DIR = ROOT / "heatmaps"
VISC_COMPARISON_DIR = ROOT / "visc_comparison"
