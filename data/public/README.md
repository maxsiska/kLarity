# Public analysis inputs

These compact tables support the analysis notebooks without embedding values in notebook
source. Columns retain the units stated in their headers or in the corresponding notebook.

- `temporal_independence_full_grid.csv`: effective sample sizes for bubble count and mean diameter.
- `temporal_independence_holdup.csv`: effective sample sizes for gas holdup and interfacial area.
- `headline_numbers_by_condition.csv`: condition-level model intervals used by comparison plots.

The larger frame- and bubble-level tables are distributed with the processed dataset. Each
public dataset includes a manifest containing source-code, parameter, schema, row-count, and
file-checksum information.

Build these tables from the canonical frame table with
`scripts/build_headline_table.py` and `scripts/build_temporal_independence.py`.
