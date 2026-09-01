# Processed bubble data

This directory contains one Parquet file per camera-placement, operating-condition, and
replicate combination. Download the processed dataset from
[Zenodo](https://doi.org/10.5281/zenodo.19582132) and extract it here, or populate the
directory with `scripts/process_images.py` or `scripts/process_images_parallel.py`.

The distributed archive includes `metadata/frame_census.parquet`,
`metadata/frame_census_summary.csv`, `metadata/frame_filter_summary.parquet`, and
`dataset_manifest.json`. The manifest records schema and units, processing parameters,
source-code and model hashes, software versions, file row counts, census totals, and
SHA-256 checksums.
