#!/usr/bin/env python3
"""Prepare sanitized, auditable xanthan-rheology files for a Zenodo deposit.

The source workbook is treated as immutable. The publication copy retains every worksheet,
formula, cached value, style, and calculation-chain byte while removing document metadata
that exposes personal/internal paths and Excel revision identifiers. A tidy CSV is derived
from the exact cells consumed by ``notebooks/xanthan_rheology.ipynb``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

_SPREADSHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_NS = {"main": _SPREADSHEET_NS}
_BATCH_COLUMNS = (
    ("0.125", 1, "B", "C", "D", "E"),
    ("0.125", 2, "H", "I", "J", "K"),
    ("0.25", 1, "M", "N", "O", "P"),
    ("0.25", 2, "S", "T", "U", "V"),
)
_SANITIZED_MEMBERS = {
    "docProps/core.xml",
    "docProps/app.xml",
    "xl/workbook.xml",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _remove_once(text: str, pattern: str, label: str) -> str:
    cleaned, count = re.subn(pattern, "", text, flags=re.DOTALL)
    if count != 1:
        raise ValueError(f"Expected one {label} metadata element, found {count}")
    return cleaned


def _sanitize_member(name: str, content: bytes) -> bytes:
    text = content.decode("utf-8")
    if name == "docProps/core.xml":
        for tag in ("dc:creator", "cp:lastModifiedBy", "dcterms:created", "dcterms:modified"):
            text = _remove_once(text, rf"<{tag}(?:\s[^>]*)?>.*?</{tag}>", tag)
    elif name == "docProps/app.xml":
        text = _remove_once(text, r"<Company>.*?</Company>", "Company")
    elif name == "xl/workbook.xml":
        text = _remove_once(
            text,
            r"<mc:AlternateContent\b.*?</mc:AlternateContent>",
            "absolute-path AlternateContent",
        )
        text = _remove_once(text, r"<xr:revisionPtr\b[^>]*/>", "revision pointer")
    return text.encode("utf-8")


def sanitize_workbook(source: Path, destination: Path) -> None:
    """Write a deterministic metadata-sanitized copy of ``source``."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(source, "r") as src, ZipFile(destination, "w") as dst:
        for source_info in sorted(src.infolist(), key=lambda item: item.filename):
            content = src.read(source_info.filename)
            if source_info.filename in _SANITIZED_MEMBERS:
                content = _sanitize_member(source_info.filename, content)

            output_info = ZipInfo(source_info.filename, date_time=(1980, 1, 1, 0, 0, 0))
            output_info.compress_type = ZIP_DEFLATED
            output_info.external_attr = source_info.external_attr
            output_info.internal_attr = source_info.internal_attr
            output_info.create_system = source_info.create_system
            dst.writestr(output_info, content)


def _numeric_cells(workbook_path: Path) -> dict[str, float]:
    with ZipFile(workbook_path) as archive:
        root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    values: dict[str, float] = {}
    for cell in root.findall(".//main:c", _NS):
        value = cell.find("main:v", _NS)
        if value is not None and value.text is not None and cell.attrib.get("t") != "s":
            values[cell.attrib["r"]] = float(value.text)
    return values


def _format_number(value: float) -> str:
    return format(value, ".12g")


def write_tidy_csv(workbook_path: Path, destination: Path) -> int:
    """Export the four 12-point series in machine-readable long form."""
    values = _numeric_cells(workbook_path)
    fieldnames = [
        "xanthan_wt_percent",
        "batch",
        "shear_rate_s_inv",
        "stress_pa",
        "viscosity_pa_s",
        "stress_mpa_source",
    ]
    rows: list[dict[str, str | int]] = []
    for concentration, batch, raw_stress, stress, shear_rate, viscosity in _BATCH_COLUMNS:
        for row_number in range(3, 15):
            rows.append(
                {
                    "xanthan_wt_percent": concentration,
                    "batch": batch,
                    "shear_rate_s_inv": _format_number(values[f"{shear_rate}{row_number}"]),
                    "stress_pa": _format_number(values[f"{stress}{row_number}"]),
                    "viscosity_pa_s": _format_number(values[f"{viscosity}{row_number}"]),
                    "stress_mpa_source": _format_number(values[f"{raw_stress}{row_number}"]),
                }
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def validate_publication_copy(source: Path, publication_copy: Path) -> None:
    """Verify workbook integrity, sanitization, and preservation of analytical content."""
    sensitive_fragments = (
        b"dtudk-my.sharepoint.com",
        b"eoje_dtu_dk",
        b"Emilie Overgaard Willer",
        b"TECHNICAL UNIVERSITY OF DENMARK",
        b"revisionPtr",
    )
    with ZipFile(source) as src, ZipFile(publication_copy) as published:
        if published.testzip() is not None:
            raise ValueError("The publication XLSX contains a corrupt ZIP member")
        if set(src.namelist()) != set(published.namelist()):
            raise ValueError("The publication XLSX package member list changed")
        for name in src.namelist():
            source_content = src.read(name)
            published_content = published.read(name)
            if name not in _SANITIZED_MEMBERS and source_content != published_content:
                raise ValueError(f"Unexpected workbook-content change in {name}")
        package_bytes = b"\n".join(published.read(name) for name in published.namelist())
        remaining = [
            fragment.decode() for fragment in sensitive_fragments if fragment in package_bytes
        ]
        if remaining:
            raise ValueError(f"Sensitive metadata remains in publication XLSX: {remaining}")

    if _numeric_cells(source) != _numeric_cells(publication_copy):
        raise ValueError("Numeric worksheet values changed during sanitization")


def write_readme(
    destination: Path,
    *,
    source: Path,
    workbook: Path,
    csv_path: Path,
    observation_count: int,
) -> None:
    text = f"""# Xanthan rheology reference data

This deposit component contains the rheology measurements consumed by
`notebooks/xanthan_rheology.ipynb` in the kLarity repository.

## Files

- `{workbook.name}`: metadata-sanitized Excel publication copy preserving the original
  worksheet, formulas, cached values, units, and formatting.
- `{csv_path.name}`: tidy, machine-readable representation with one observation per row.

## Data structure

The dataset contains {observation_count} observations: 12 shear-rate measurements for each
combination of xanthan concentration (0.125 or 0.25 wt%) and batch (1 or 2).

| CSV column | Meaning | Unit |
|---|---|---|
| `xanthan_wt_percent` | Xanthan concentration | wt% |
| `batch` | Batch identifier | - |
| `shear_rate_s_inv` | Shear rate | s^-1 |
| `stress_pa` | Shear stress used by the analysis | Pa |
| `viscosity_pa_s` | Dynamic viscosity | Pa s |
| `stress_mpa_source` | Source-workbook stress retained for traceability | MPa |

The Pa values equal the source-workbook MPa values multiplied by 10^6. The Excel file
stores this conversion as formulas; the CSV stores the evaluated values.

## Publication sanitization

Only document metadata was removed from the publication workbook: creator/modifier names,
document timestamps, organization metadata, an absolute private SharePoint path, and Excel
revision identifiers. Worksheet data, formulas, cached results, styles, and calculation
chain were not changed.

## Checksums (SHA-256)

- Immutable local source `{source.name}`: `{_sha256(source)}`
- Publication workbook `{workbook.name}`: `{_sha256(workbook)}`
- Tidy CSV `{csv_path.name}`: `{_sha256(csv_path)}`

Prepared on {date.today().isoformat()}.

## Metadata to supply in the Zenodo record

The source workbook does not provide sufficient dataset-level provenance for the rheometer
model, measurement temperature, sample preparation, experimental protocol, dataset
creators, or reuse license. Supply these from the experimental record and manuscript when
creating the Zenodo record; they have not been inferred here.
"""
    destination.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/xanthan_rheology.xlsx"),
        help="Immutable source workbook",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/zenodo/rheology"),
        help="Zenodo staging directory",
    )
    args = parser.parse_args()

    if not args.source.is_file():
        raise FileNotFoundError(args.source)

    source_hash_before = _sha256(args.source)
    workbook = args.output_dir / "xanthan_rheology.xlsx"
    csv_path = args.output_dir / "xanthan_rheology.csv"
    readme = args.output_dir / "README.md"

    sanitize_workbook(args.source, workbook)
    observation_count = write_tidy_csv(workbook, csv_path)
    validate_publication_copy(args.source, workbook)
    if _sha256(args.source) != source_hash_before:
        raise RuntimeError("The immutable source workbook changed")
    write_readme(
        readme,
        source=args.source,
        workbook=workbook,
        csv_path=csv_path,
        observation_count=observation_count,
    )

    print(f"Source preserved: {args.source} ({source_hash_before})")
    print(f"Publication XLSX: {workbook} ({_sha256(workbook)})")
    print(f"Tidy CSV:        {csv_path} ({observation_count} rows, {_sha256(csv_path)})")
    print(f"Zenodo README:   {readme}")


if __name__ == "__main__":
    main()
