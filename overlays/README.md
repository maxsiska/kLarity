# Overlays

This directory is populated by `scripts/process_images.py` when overlay saving is enabled.

It contains annotated images showing detected bubble masks and fitted ellipses overlaid on the original frames. The overlay frequency is controlled by `OVERLAY_MODE` in `config.py` (`"all"`, `"every_10th"`, `"every_5th"`, `"first_only"`, or `"none"`).
