"""Safe ingestion of user-uploaded tabular files (CSV / Parquet).

Uploaded content is only ever *parsed* as data — never executed. Filenames are
sanitised and raw files are stored under a content-addressed directory, so a
client-supplied name can never influence the storage path.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import pandas as pd

ALLOWED_EXTENSIONS = {".csv": "csv", ".parquet": "parquet", ".pq": "parquet"}
PARQUET_MAGIC = b"PAR1"
CHUNK = 1024 * 1024


class IngestError(ValueError):
    """Raised for user-facing ingestion problems (bad type, parse failure, ...)."""


@dataclass
class StoredUpload:
    path: Path
    sha256: str
    size_bytes: int


def sanitize_filename(name: str | None) -> str:
    """Return a safe display/storage filename. Strips any directory components."""
    base = (name or "").replace("\\", "/").split("/")[-1].strip()
    base = re.sub(r"[^A-Za-z0-9._ -]", "_", base).strip(" .")
    base = re.sub(r"\.{2,}", ".", base)
    if not base:
        raise IngestError("The uploaded file has no usable filename.")
    stem, dot, ext = base.rpartition(".")
    if not dot:
        stem, ext = base, ""
    stem = stem[:100] or "dataset"
    return f"{stem}.{ext.lower()}" if ext else stem


def detect_format(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise IngestError(
            f"Unsupported file type '{ext or '(none)'}'. Allowed types: CSV (.csv) and Parquet (.parquet, .pq)."
        )
    return ALLOWED_EXTENSIONS[ext]


def copy_stream(src: BinaryIO, dest: Path, max_bytes: int) -> StoredUpload:
    """Copy an upload stream to ``dest`` while hashing it and enforcing the size limit."""
    digest = hashlib.sha256()
    size = 0
    with dest.open("wb") as out:
        while True:
            chunk = src.read(CHUNK)
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                out.close()
                dest.unlink(missing_ok=True)
                raise IngestError(f"File exceeds the upload limit of {max_bytes // (1024 * 1024)} MB.")
            digest.update(chunk)
            out.write(chunk)
    if size == 0:
        dest.unlink(missing_ok=True)
        raise IngestError("The uploaded file is empty.")
    return StoredUpload(dest, digest.hexdigest(), size)


def validate_content_signature(path: Path, fmt: str) -> None:
    with path.open("rb") as fh:
        head = fh.read(64 * 1024)
        if fmt == "parquet":
            fh.seek(-4, io.SEEK_END)
            tail = fh.read(4)
    if fmt == "parquet":
        if not (head.startswith(PARQUET_MAGIC) and tail == PARQUET_MAGIC):
            raise IngestError("The file has a .parquet extension but is not a valid Parquet file.")
    else:
        if b"\x00" in head:
            raise IngestError("The file does not look like a text CSV (binary content detected).")


def _sniff_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def read_csv(path: Path) -> pd.DataFrame:
    last_err: Exception | None = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            with path.open("r", encoding=encoding, newline="") as fh:
                sample = fh.read(64 * 1024)
            sep = _sniff_delimiter(sample)
            # pandas silently renames duplicate headers (a, a.1); detect them on the raw header.
            header = next(csv.reader(io.StringIO(sample), delimiter=sep), [])
            names = [h.strip() for h in header]
            dupes = sorted({n for n in names if n and names.count(n) > 1})
            if dupes:
                raise IngestError("Duplicate column names: " + ", ".join(dupes[:10]))
            return pd.read_csv(path, sep=sep, encoding=encoding, low_memory=False, engine="c")
        except UnicodeDecodeError as exc:
            last_err = exc
            continue
        except pd.errors.EmptyDataError as exc:
            raise IngestError("The CSV file contains no data.") from exc
        except pd.errors.ParserError as exc:
            raise IngestError(f"CSV parsing failed: {exc}") from exc
    raise IngestError(f"Could not decode the CSV file as UTF-8 or Latin-1: {last_err}")


def read_parquet(path: Path) -> pd.DataFrame:
    import pyarrow as pa
    import pyarrow.parquet as pq

    try:
        schema = pq.read_schema(path)
    except Exception as exc:  # pyarrow raises various error types
        raise IngestError(f"Parquet schema could not be read: {exc}") from exc
    nested = [f.name for f in schema if pa.types.is_nested(f.type)]
    if nested:
        raise IngestError(
            "Nested Parquet columns are not supported (flatten them first): " + ", ".join(nested[:10])
        )
    try:
        return pq.read_table(path).to_pandas()
    except Exception as exc:
        raise IngestError(f"Parquet file could not be read: {exc}") from exc


def parse_file(path: Path, fmt: str, max_columns: int) -> pd.DataFrame:
    df = read_csv(path) if fmt == "csv" else read_parquet(path)
    validate_frame(df, max_columns)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def validate_frame(df: pd.DataFrame, max_columns: int) -> None:
    if df.shape[1] == 0:
        raise IngestError("No columns were found in the file.")
    if df.shape[1] > max_columns:
        raise IngestError(f"Too many columns ({df.shape[1]}); the limit is {max_columns}.")
    if df.shape[0] < 2:
        raise IngestError("The dataset must contain at least two rows.")
    names = [str(c).strip() for c in df.columns]
    if any(n == "" for n in names):
        raise IngestError("The file contains a column with an empty name.")
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise IngestError("Duplicate column names: " + ", ".join(dupes[:10]))
    if df.shape[1] == 1 and any(d in names[0] for d in ",;\t|"):
        # A single wide column usually means the delimiter was not detected.
        raise IngestError("Only one column was parsed — the delimiter may be unsupported.")
