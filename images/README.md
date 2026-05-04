# Images

Raw endoscope images are not included in this repository due to their size.

## Download

> [**Zenodo**](https://doi.org/10.5281/zenodo.19582132)

## Expected layout

Images are organised by replicate recording. The `config.py` `IMAGE_DIR` setting points to this directory; all subdirectories are scanned recursively by `scripts/process_images.py`.

```
images/
├── <Position_id>/
│   ├── <Setting_id>/
│   └── ...
└── ...
```
