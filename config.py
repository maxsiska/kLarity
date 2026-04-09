"""
Project configuration.

All paths are resolved relative to this file (the project root), so the
project works regardless of where scripts are invoked from.

Edit the values in each section to match your local setup before running
scripts or notebooks.
"""

from pathlib import Path

ROOT = Path(__file__).parent


# ── Data ──────────────────────────────────────────────────────────────────────

# Raw images produced by the endoscope cameras
IMAGE_DIR = ROOT / "images"

# Per-replicate Parquet files written by process_images.py
OUTPUT_DIR = ROOT / "output"

# Pre-computed aggregates used by notebooks
BUBBLE_LEVEL_PKL = ROOT / "data" / "bubble_level_df.pkl"
FRAME_LEVEL_PKL = ROOT / "data" / "frame_level_df.pkl"

# Reference spreadsheets
KLA_XLSX = ROOT / "data" / "kla_data_000_xanthan.xlsx"
RHEOLOGY_XLSX = ROOT / "data" / "xanthan_rheology.xlsx"


# ── Model ─────────────────────────────────────────────────────────────────────

MODEL_PATH = ROOT / "models" / "klarity-model.pt"

# Inference parameters (tuned on the KLarity-18 validation set)
CONF = 0.22
IOU = 0.35
MASK_THR = 0.30


# ── Processing ────────────────────────────────────────────────────────────────

# Overlay images written alongside output CSVs (set to "none" to skip)
OVERLAYS_DIR = ROOT / "overlays"
OVERLAY_MODE = "every_10th"  # "all" | "every_10th" | "every_5th" | "first_only" | "none"

# GPU selection for process_images.py and train_model.py
# Examples: 0  |  [0, 1]  |  [1, 2, 3]  |  "cpu"
DEVICE = [1, 2, 3]


# ── Evaluation ────────────────────────────────────────────────────────────────

EVAL_DATASET_DIR = ROOT / "model_eval" / "KLarity-18"
BENCHMARK_OUT_DIR = ROOT / "model_eval" / "model_benchmark_output"


# ── Notebook outputs ──────────────────────────────────────────────────────────

SETTING_COMPARISON_DIR = ROOT / "setting_comparison"
HEATMAPS_DIR = ROOT / "heatmaps"
VISC_COMPARISON_DIR = ROOT / "visc_comparison"
