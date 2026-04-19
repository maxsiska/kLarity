# Images

Raw endoscope images are not included in this repository due to their size.

## Download

> **Zenodo:** [DOI placeholder — link will be added upon publication]

## Expected layout

Images are organised by replicate recording. The `config.py` `IMAGE_DIR` setting points to this directory; all subdirectories are scanned recursively by `scripts/process_images.py`.

```
images/
├── <recording_id>/
│   ├── <frame>.jpg
│   └── ...
└── ...
```
