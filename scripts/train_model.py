"""
Model training script.

Run from the project root with:
    python scripts/train_model.py

The training dataset is expected at model_eval/KLarity-18/data.yaml.
Download it from Roboflow (see model_eval/README.md) before running.
All paths and device settings are configured in config.py.
"""

from ultralytics import YOLO

import config


def train():
    model = YOLO("yolo11x-seg.pt")

    model.train(
        data=str(config.EVAL_DATASET_DIR / "data.yaml"),
        epochs=2000,
        imgsz=1024,
        batch=18,
        lr0=0.01,
        cos_lr=True,
        device=config.DEVICE,
        amp=True,
        seed=20251028,
        save_period=100,
        patience=200,
        workers=8,
        project=str(config.ROOT / "training_results"),
        name="klarity_model",
        verbose=True,
    )


if __name__ == "__main__":
    train()
