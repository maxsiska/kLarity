# Analysis data

`scripts/build_dataframes.py` creates the bubble- and frame-level analysis tables in this
directory. New builds store the >4 GiB bubble table as `bubble_level_df.parquet`; readers
retain a fallback for legacy `bubble_level_df.pkl` files. The smaller frame table remains
`frame_level_df.pkl`. Large generated tables are not tracked by Git and are distributed
with the processed dataset linked from [`../output/README.md`](../output/README.md). Frame
and filter censuses live under `output/metadata/` beside the processed Parquet collection.

`public/` contains compact, frozen inputs required by the notebooks. The two small
spreadsheets in this directory contain the kLa and rheology reference measurements used by
the corresponding analyses.
