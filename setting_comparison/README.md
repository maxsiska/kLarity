# Setting Comparison

This directory is populated by `notebooks/setting_comparison.ipynb`.

Each figure sweeps **one** operating parameter across the six endoscope positions with the
other two held fixed, showing the published condition estimand and its 95% confidence
interval per position. Mean diameter is pooled over retained bubbles; count is the mean per
valid analyzed frame; and gas holdup/interfacial area are ratios of summed physical
contributions. Ratio intervals use a delete-one-frame jackknife and are widened with the
available effective sample size `N_eff` because frames within a stream are temporally
autocorrelated. Camera-failure black frames are excluded;
valid analyzed frames with zero retained detections remain physical zeros.

## File naming

The legend of a setting-comparison figure names only the parameter that *varies*, so the
held-fixed levels are recorded in the file name instead:

```text
[holdup_area_]{swept}_at_{fixed}_{xanthan}_xanthan[_all_aeration].{png,svg,pdf}
```

| stem | x axis | held fixed |
|---|---|---|
| `rpm_at_55_lmin_000_xanthan` | agitation rate, 75–150 min⁻¹ | 55 L min⁻¹, 0.00 wt% xanthan |
| `lmin_at_100_rpm_025_xanthan` | aeration rate, four setpoints | 100 min⁻¹, 0.25 wt% xanthan |
| `holdup_area_rpm_at_90_lmin_0125_xanthan` | agitation rate | 90 L min⁻¹, 0.125 wt% xanthan |

* prefix `holdup_area_` — gas holdup and observed-volume interfacial area density (`epsilon_obs_mid`,
  `a_obs_m2_m3_mid`) instead of mean bubble diameter and bubble count. These figures carry
  **two** intervals per point that must not be conflated: a capped whisker for the 95% CI
  of the condition estimand, and a wider translucent bar for the prolate–oblate depth band. The CI
  is sampling uncertainty and shrinks as frames accumulate; the band is a systematic range
  from the unobserved depth axis and does not.
* suffix `_all_aeration` — the complete five-setpoint aeration series including
  80 L min⁻¹, as opposed to the four setpoints (45/55/70/90) of the manuscript reference
  figure. Same convention as the `*_all_aeration` heat map grids in `heatmaps/`.
* the xanthan token is the directory-level identifier, not a number: `000` = 0.00 wt%,
  `0125` = 0.125 wt%, `025` = 0.25 wt%.

Names are built by `klarity.viz.setting_comparison_stem` from the settings list that is
actually plotted, so a file name cannot claim a condition the figure does not show.

## Coverage

The factorial is measured in full — 4 agitation × 5 aeration × 3 xanthan = 60 settings —
and both interval lookups (`N_eff`, prolate–oblate band) cover all 360 condition–placements,
so every combination below carries the same CI and, where applicable, the same
depth band. Per metric pair:

| sweep | fixed levels | figures |
|---|---|---|
| agitation | each of 45/55/70/80/90 L min⁻¹ × 3 xanthan | 15 |
| aeration, five setpoints (`_all_aeration`) | each of 75/100/125/150 min⁻¹ × 3 xanthan | 12 |
| aeration, four setpoints | 100 min⁻¹ × 3 xanthan | 3 |

That is 30 stems per metric pair and 60 in total, each written as `.png`, `.svg` and
`.pdf`. The four-setpoint aeration layout exists **only** at 100 min⁻¹, because it is
there to reproduce the manuscript reference figure; every other stirrer speed has the
complete five-setpoint series and no four-setpoint counterpart.

Color assignment differs between the four- and five-setpoint aeration figures:
`viz.color_cycle` is applied in setting order, so 90 L min⁻¹ is pink in the four-setpoint
figures and eminence in the five-setpoint ones, with pink taken by 80 L min⁻¹. Do not
compare the two versions by color.
