# Images

Raw endoscope images are not included in this repository due to their size.

## Download

> [**Zenodo**](https://doi.org/10.5281/zenodo.19582132)

## Expected layout

The frame-census command can read the downloaded `images_pos_1.zip` through
`images_pos_6.zip` archives directly from this directory. Model inference requires the
archives to be extracted with one placement directory above each archive's setting
directories. For example, extract `images_pos_1.zip` into `images/placement_1/`,
`images_pos_2.zip` into `images/placement_2/`, and so on.

```
images/
├── placement_1/
│   ├── <setting>/
│   │   ├── <replicate>/
│   │   └── ...
│   └── ...
└── placement_6/
    └── ...
```

`config.py` points `IMAGE_DIR` to `images/`. The processing scripts discover placement,
setting, and replicate directories beneath it.
