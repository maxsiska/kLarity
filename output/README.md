# Output

This directory is populated by `scripts/process_images.py`.

It contains one CSV file per replicate recording with per-bubble detection results (bounding box, equivalent diameter, ellipse axes, confidence score). Parquet equivalents are written to a sibling `output_parquet/` directory and consumed by `scripts/build_dataframes.py`.
