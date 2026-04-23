# Images

Raw endoscope images are not included in this repository due to their size.

## Download

> [**Zenodo:**](10.5281/zenodo.19582133)

## Expected layout

Images are organised by replicate recording. The `config.py` `IMAGE_DIR` setting points to this directory; all subdirectories are scanned recursively by `scripts/process_images.py`.

```
images/
├── <Position_id>/
│   ├── <Setting_id>/
│   └── ...
└── ...
```
