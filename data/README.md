# Analysis data

`scripts/build_dataframes.py` creates the bubble- and frame-level analysis tables in this
directory. New builds store the >4 GiB bubble table as `bubble_level_df.parquet`; readers
retain a fallback for legacy `bubble_level_df.pkl` files. The smaller frame table remains
`frame_level_df.pkl`. Large generated tables are not tracked by Git and are distributed
with the processed dataset linked from [`../output/README.md`](../output/README.md). Frame
and filter censuses live under `output/metadata/` beside the processed Parquet collection.

`public/` contains compact, frozen inputs required by the plotting notebooks.

`notebooks/xanthan_rheology.ipynb` additionally requires `xanthan_rheology.xlsx`. Download
that reference workbook with the processed dataset from
[Zenodo](https://doi.org/10.5281/zenodo.19582132) and place it at
`data/xanthan_rheology.xlsx`. The workbook is deliberately not tracked by Git.
