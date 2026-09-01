"""I/O helpers for derived analysis DataFrames."""

import os
import pickle
from pathlib import Path
from uuid import uuid4

import pandas as pd


def _corrupt_artifact_error(path: Path, exc: BaseException) -> RuntimeError:
    """Return an actionable error for an incomplete/corrupt analysis artifact."""
    return RuntimeError(
        f"Could not read analysis artifact {path}: {exc}. "
        "Delete the incomplete file and rebuild from the project root with "
        "`python scripts/build_dataframes.py --force`."
    )


def read_dataframe(path: Path) -> pd.DataFrame:
    """Read a supported analysis-table format with a clear corruption error."""
    try:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix in {".pkl", ".pickle"}:
            return pd.read_pickle(path)
    except (EOFError, OSError, pickle.UnpicklingError) as exc:
        raise _corrupt_artifact_error(path, exc) from exc
    raise ValueError(f"Unsupported DataFrame format: {path.suffix}")


def read_bubble_dataframe() -> pd.DataFrame:
    """Load the bubble table, preferring robust Parquet over the legacy pickle."""
    import config

    if config.BUBBLE_LEVEL_PARQUET.exists():
        return read_dataframe(config.BUBBLE_LEVEL_PARQUET)
    if config.BUBBLE_LEVEL_PKL.exists():
        return read_dataframe(config.BUBBLE_LEVEL_PKL)
    raise FileNotFoundError(
        "Bubble-level analysis table is missing. Run "
        "`python scripts/build_dataframes.py --force` from the project root."
    )


def read_frame_dataframe() -> pd.DataFrame:
    """Load the frame-level analysis table."""
    import config

    if not config.FRAME_LEVEL_PKL.exists():
        raise FileNotFoundError(
            "Frame-level analysis table is missing. Run "
            "`python scripts/build_dataframes.py --force` from the project root."
        )
    return read_dataframe(config.FRAME_LEVEL_PKL)


def _write_parquet_chunked(
    frame: pd.DataFrame, path: Path, *, row_group_rows: int = 1_000_000
) -> None:
    """Write Parquet in bounded-size row groups instead of one oversized conversion."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if frame.empty:
        frame.to_parquet(path, engine="pyarrow", compression="zstd", index=True)
        return

    writer: pq.ParquetWriter | None = None
    try:
        for start in range(0, len(frame), row_group_rows):
            chunk = frame.iloc[start : start + row_group_rows]
            table = pa.Table.from_pandas(chunk, preserve_index=True)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            writer.write_table(table, row_group_size=row_group_rows)
    finally:
        if writer is not None:
            writer.close()


def write_dataframe_atomic(
    frame: pd.DataFrame, path: Path, *, parquet_row_group_rows: int = 1_000_000
) -> None:
    """Write ``frame`` completely before atomically replacing ``path``.

    A failed serialization therefore leaves any previous valid artifact untouched and
    never exposes a truncated destination to the notebooks.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        if path.suffix == ".parquet":
            _write_parquet_chunked(
                frame,
                temporary,
                row_group_rows=parquet_row_group_rows,
            )
        elif path.suffix in {".pkl", ".pickle"}:
            frame.to_pickle(temporary, protocol=5)
        else:
            raise ValueError(f"Unsupported DataFrame format: {path.suffix}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _current_bubble_path() -> Path:
    """Return the preferred bubble artifact, or the legacy path before migration."""
    import config

    if config.BUBBLE_LEVEL_PARQUET.exists():
        return config.BUBBLE_LEVEL_PARQUET
    if config.BUBBLE_LEVEL_PKL.exists():
        return config.BUBBLE_LEVEL_PKL
    return config.BUBBLE_LEVEL_PARQUET


def check_dataframes_stale() -> None:
    """
    Print a warning if the bubble- or frame-level artifact is older
    than the newest Parquet file in the output directory.

    Call this at the top of any plotting notebook before loading analysis artifacts.
    """
    import config

    parquet_files = list(Path(config.OUTPUT_DIR).glob("*.parquet"))
    if not parquet_files:
        print("WARNING: No Parquet files found in output directory. Run process_images.py first.")
        return

    newest_parquet = max(f.stat().st_mtime for f in parquet_files)

    stale = []
    for artifact in (_current_bubble_path(), config.FRAME_LEVEL_PKL):
        if not artifact.exists():
            stale.append(f"  - {artifact.name} is missing")
        elif artifact.stat().st_mtime < newest_parquet:
            stale.append(f"  - {artifact.name} is older than the newest Parquet")

    if stale:
        print("=" * 60)
        print("WARNING: DataFrames are out of date.")
        print("Run:  python scripts/build_dataframes.py")
        print("Reason:")
        for s in stale:
            print(s)
        print("=" * 60)
