# Model Evaluation — KLarity-18

KLarity-18 is the annotated image dataset used to train and evaluate the kLarity bubble detection model.

## Dataset

The dataset is hosted on Roboflow Universe:

> https://universe.roboflow.com/klarity/klarity

**KLarity-18 (v18 systematic_tests-v4)**

## Download

Download the dataset from Roboflow and place it at:

```
model_eval/KLarity-18/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

## Reproducing the benchmark

To reproduce the test-set results reported in the paper:

```bash
python scripts/evaluate_model.py
```

Results (precision, recall, F1, mask AP, box AP) are written to `model_eval/model_benchmark_output/metrics/`.
