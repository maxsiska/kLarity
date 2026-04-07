# Models

Model weights are not included in this repository.

## Download

> **Zenodo:** [DOI placeholder — link will be added upon publication]

Place the downloaded file at:

```
models/klarity-model.pt
```

## Training your own model

To retrain from scratch using the annotated dataset:

1. Download the KLarity-18 dataset from Roboflow (see `model_eval/README.md`).
2. Place it at `model_eval/KLarity-18/`.
3. Run:

```bash
python scripts/train_model.py
```

Training configuration (epochs, batch size, device) can be edited at the top of that script.
