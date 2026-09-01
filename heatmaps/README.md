# Heatmaps

This directory is populated by `notebooks/create_heatmaps.ipynb`.

It contains spatial heatmap figures showing bubble density and size distributions across the
bioreactor cross-section for each operating condition and camera placement.

## Condition-level estimands

Public heatmaps use `metrics.aggregate_frames_for_grid(..., reducer="condition")`. Each
cell is one condition–placement estimate, not an unlabelled mean of local frame ratios.

| Quantity | Condition estimate |
|---|---|
| mean diameter | `sum(diameter_sum_mm) / sum(diameter_count)` |
| gas holdup | `sum(V_gas) / sum(V_observed)` |
| observed-volume area density | `1000 * sum(A) / sum(V_observed)` |
| gas-specific area | `1000 * sum(A) / sum(V_gas)` |
| liquid-volume area density | `1000 * sum(A) / sum(V_liquid)` |
The first row describes the pooled retained-bubble population. Gas holdup and
observed-volume area give equal weight to every valid, equal-volume analyzed frame. The
remaining ratios use the physical population named in their denominator. Camera-failure
black frames are excluded; an illuminated analyzed frame with zero retained detections
remains in the observed/liquid-volume denominators as a physical zero.
