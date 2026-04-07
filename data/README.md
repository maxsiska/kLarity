# Data

The raw images and processed data files are not included in this repository due to their size.

## Download

Processed data (CSV/Parquet bubble detections, pickle aggregates, reference spreadsheets):
> **Zenodo:** [DOI placeholder — link will be added upon publication]

Raw images:
> **Roboflow:** [project URL placeholder — link will be added upon publication]

## Expected layout

After downloading, populate this directory as follows:

```text
data/
├── bubble_level_df.pkl     # Bubble-level detections (one row per bubble)
├── frame_level_df.pkl      # Frame-level aggregates with derived physical metrics
├── kla_data_000_xanthan.xlsx  # Reference kLa measurements (water)
└── xanthan_rheology.xlsx      # Xanthan gum rheology properties
```

Parquet files (used by scripts) live in a sibling `output_parquet/` directory:

```
output_parquet/
├── placement_1.parquet
├── placement_2.parquet
└── ...
```

## Reproducing processed data from raw images

If you have the raw images, run the processing pipeline instead of downloading the
pre-processed files:

```bash
python scripts/process_images.py
```

This writes per-replicate CSVs to `output/` and Parquet files to `output_parquet/`.
