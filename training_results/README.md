# Training Results

This directory is populated by `scripts/train_model.py`.

It contains YOLOv11 training outputs including loss curves, validation metrics, and the best model checkpoint. The trained weights are saved as `training_results/<run_name>/weights/best.pt` and should be copied to `models/klarity-model.pt` before running the detection pipeline.
