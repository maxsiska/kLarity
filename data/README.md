# Analysis data

`scripts/build_dataframes.py` creates the bubble- and frame-level analysis tables in this
directory. Large generated tables are not tracked by Git and are distributed with the
processed dataset linked from [`../output/README.md`](../output/README.md). Frame and filter
censuses live under `output/metadata/` beside the processed Parquet collection.

`public/` contains compact, frozen inputs required by the notebooks. The two small
spreadsheets in this directory contain the kLa and rheology reference measurements used by
the corresponding analyses.
