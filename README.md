# kLarity

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19709975.svg)](https://doi.org/10.5281/zenodo.19709975)

**kLarity** detects, segments, and characterizes gas bubbles in endoscope images from
pilot-scale stirred bioreactors. It combines a YOLO instance-segmentation model with a
two-half-ellipse fit to estimate bubble size, gas volume, and interfacial area across
camera placements and operating conditions.

The repository accompanies *Machine Learning Driven Quantification of Local Bubble Dynamics across a Pilot Scale Stirred Bioreactor* by Emilie Overgaard
Willer, Maximilian Siska, Benjamin Petersen, Juliet J. Victoria, Eric von Lieres, and John M. Woodley.

## Installation

```bash
git clone https://github.com/maxsiska/kLarity.git
cd kLarity
pip install -e .
```

For CUDA inference, install the appropriate PyTorch build before installing kLarity. CPU
and Apple MPS are selected automatically when CUDA is unavailable.

## Workflow

Run commands from the repository root.

1. Process extracted image directories:

   ```bash
   python scripts/process_images.py
   # or
   python scripts/process_images_parallel.py --devices auto
   ```

   One Parquet file is written per replicate under `output/`. Set
   `KLARITY_OUTPUT_DIR=/path/to/output` to use another location.

2. Build the analysis tables:

   ```bash
   python scripts/build_frame_census.py
   python scripts/build_dataframes.py --force
   ```

   The frame inventory ensures that every usable input image contributes correctly to
   frame-based averages. It reads the images without running model inference.

3. Build the compact condition and temporal tables:

   ```bash
   python scripts/build_headline_table.py
   python scripts/build_temporal_independence.py
   ```

4. Run the notebooks in `notebooks/` as needed. They read paths from `config.py` and compact
   reference tables from `data/public/`.

5. Record the files and settings used for a processed dataset:

   ```bash
   python scripts/build_dataset_manifest.py
   ```

## Geometry and filtering

The split-ellipse fit supplies the in-plane major-axis length `a` and semi-minor axes `b1`
and `b2`. The unobserved depth is represented by two spheroid models:

- Prolate: `V = (pi*a/3)*(b1^2+b2^2)`. Surface area uses the smooth, volume-equivalent
  symmetric spheroid with radius `sqrt((b1^2+b2^2)/2)`.
- Oblate: `V = (pi*a^2/6)*(b1+b2)`. Surface area is the sum of the two half-oblate
  spheroids sharing an equatorial circle of radius `a/2`.

Volume and surface-area intervals are ordered independently. Their midpoint is the primary
reported estimate, while the interval represents sensitivity to the two depth assumptions.
Invalid or incomplete model pairs remain missing rather than being converted to zero.

Frames below the configured mean-intensity threshold are classified as blank. Zero-area
masks are not measurable bubbles. Border contacts with
`border_contact_px/equivalent_diameter > 1` are excluded; moderate contacts are retained
without off-frame reconstruction. The source Parquets retain the measurements needed to
inspect these decisions.

## Data and model weights

Large assets are distributed separately:

| Asset | Location |
|---|---|
| Raw images | [`images/README.md`](images/README.md) |
| Processed data | [`output/README.md`](output/README.md) |
| Analysis tables | [`data/README.md`](data/README.md) |
| Model weights | [`models/README.md`](models/README.md) |
| KLarity-18 annotations | [`model_eval/README.md`](model_eval/README.md) |

Each processed dataset includes `dataset_manifest.json` with schema, parameters,
software versions, source commit, row counts, and SHA-256 checksums.

## Development checks

```bash
python -m pytest
black --check config.py klarity scripts tests
ruff check config.py klarity scripts tests
mypy --explicit-package-bases config.py klarity scripts
```

## License

kLarity is licensed under the GNU Affero General Public License v3.0. See
[`LICENSE.md`](LICENSE.md).
