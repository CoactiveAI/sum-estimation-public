"""Hugging Face streaming helpers shared by the generation scripts.

Datasets are always opened with `streaming=True`, so nothing is materialised on
disk: rows arrive lazily and only the vectors are written out. Two item types
cover every combination in this repo:

* images - `stream_images`, which handles both dataset layouts found on the Hub:
  an `Image` feature (bytes embedded in the parquet files) and a plain URL
  column (Open Images mirrors typically store Flickr URLs, not pixels).
* text   - `stream_texts`, which joins one or more string columns per row.
"""

from __future__ import annotations

import argparse
import io
import itertools
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Iterator

#: Sent when fetching images by URL; some hosts reject the default UA.
USER_AGENT = "sum-estimation-embeddings/1.0"


def load_stream(dataset: str, config: str | None, split: str, data_files: list[str] | None = None):
    """Open a Hub dataset (or explicit data files) in streaming mode."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - dependency hint
        raise SystemExit(
            "The 'datasets' package is required. Install it with: pip install -r requirements.txt"
        ) from exc

    if data_files is not None:
        return load_dataset("json", data_files=data_files, split=split, streaming=True)
    return load_dataset(dataset, config, split=split, streaming=True)


def _fetch_image(url: str, timeout: float):
    """Download and decode one image; return None if it cannot be retrieved."""
    from PIL import Image

    try:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
        return Image.open(io.BytesIO(payload)).convert("RGB")
    except (urllib.error.URLError, OSError, ValueError):
        return None


def stream_images(
    rows: Iterable[dict],
    column: str,
    id_column: str | None = None,
    cursor=None,
    download_workers: int = 16,
    timeout: float = 10.0,
    prefetch: int = 256,
):
    """Yield `(asset_id, RGB PIL image)` pairs from a streamed dataset.

    If `column` holds URLs the images are downloaded with a thread pool
    (`download_workers` at a time, `prefetch` per wave) since the network, not
    the GPU, is the bottleneck there. Rows whose image is missing or undecodable
    are skipped, so callers should treat the stream as "usable items only" - and
    the id must therefore travel with the image rather than be inferred from
    position.

    The asset id is `id_column` when given, else the URL when the image column
    holds one, else the row's ordinal in the stream.

    A `cursor` makes the stream resumable: `cursor.skip_rows` rows are discarded
    before anything is fetched (so resuming costs a parquet scan, not a re-download),
    and every yielded row's ordinal is recorded for the next resume point.
    """
    from PIL import Image

    def get(row: dict):
        if column not in row:
            raise SystemExit(
                f"Column '{column}' not found in dataset rows. Available: "
                f"{', '.join(sorted(row))}. Set --image-column."
            )
        return row[column]

    def asset_id(row: dict, ordinal: int, value):
        if id_column:
            if id_column not in row:
                raise SystemExit(
                    f"Id column '{id_column}' not found in dataset rows. Available: "
                    f"{', '.join(sorted(row))}. Set --id-column."
                )
            return row[id_column]
        if isinstance(value, str):
            return value
        return ordinal

    start = cursor.skip_rows if cursor else 0
    if start:
        rows = itertools.islice(rows, start, None)

    pending: list[tuple[object, str, int]] = []
    skipped = 0

    def note(ordinal: int) -> None:
        if cursor:
            cursor.note(ordinal)

    def drain(batch: list[tuple[object, str, int]]):
        nonlocal skipped
        with ThreadPoolExecutor(max_workers=download_workers) as pool:
            images = pool.map(lambda item: _fetch_image(item[1], timeout), batch)
            for (identifier, _, ordinal), image in zip(batch, images):
                if image is None:
                    skipped += 1
                else:
                    note(ordinal)
                    yield identifier, image

    for ordinal, row in enumerate(rows, start=start):
        value = get(row)
        identifier = asset_id(row, ordinal, value)
        if isinstance(value, str):
            pending.append((identifier, value, ordinal))
            if len(pending) >= prefetch:
                yield from drain(pending)
                pending = []
            continue
        if isinstance(value, Image.Image):
            note(ordinal)
            yield identifier, value.convert("RGB")
            continue
        if isinstance(value, dict) and value.get("bytes"):
            note(ordinal)
            yield identifier, Image.open(io.BytesIO(value["bytes"])).convert("RGB")
            continue
        skipped += 1

    if pending:
        yield from drain(pending)
    if skipped:
        print(f"\n[Stream] Skipped {skipped} unusable image rows.")


def stream_texts(
    rows: Iterable[dict],
    columns: list[str],
    id_columns: list[str] | None = None,
    cursor=None,
    min_chars: int = 1,
) -> Iterator[tuple[object, str]]:
    """Yield `(asset_id, text)` pairs, joining `columns` per row with a space.

    The asset id joins `id_columns` with '|' (a review has no single natural key,
    so e.g. user_id|asin|timestamp identifies it); rows missing every id column
    fall back to their ordinal in the stream. Empty texts are skipped, so ids
    cannot be recovered from position afterwards.

    A `cursor` makes the stream resumable: `cursor.skip_rows` rows are discarded
    up front and each yielded row's ordinal is recorded as the next resume point.
    """
    start = cursor.skip_rows if cursor else 0
    if start:
        rows = itertools.islice(rows, start, None)
    for ordinal, row in enumerate(rows, start=start):
        if not any(c in row for c in columns):
            raise SystemExit(
                f"None of the columns {columns} exist in dataset rows. Available: "
                f"{', '.join(sorted(row))}. Set --text-columns."
            )
        text = " ".join(str(row[c]) for c in columns if row.get(c)).strip()
        if len(text) < min_chars:
            continue
        parts = [str(row[c]) for c in (id_columns or []) if row.get(c) is not None]
        if cursor:
            cursor.note(ordinal)
        yield ("|".join(parts) if parts else ordinal), text


def add_image_stream_args(parser: argparse.ArgumentParser, dataset: str, split: str, column: str):
    """Register the streaming flags shared by the image encoders."""
    parser.add_argument("--hf-dataset", default=dataset, help=f"Hub dataset id (default: {dataset}).")
    parser.add_argument("--hf-config", default=None, help="Dataset config name (default: none).")
    parser.add_argument("--split", default=split, help=f"Split to stream (default: {split}).")
    parser.add_argument(
        "--image-column", default=column,
        help=f"Column holding the image or its URL (default: {column}).",
    )
    parser.add_argument(
        "--id-column", default=None,
        help="Column to record as each row's asset id. Defaults to the image URL "
             "when the image column holds one, else the row's ordinal.",
    )
    parser.add_argument(
        "--download-workers", type=int, default=16,
        help="Parallel downloads when the image column holds URLs (default: 16).",
    )
    parser.add_argument(
        "--download-timeout", type=float, default=10.0,
        help="Per-image download timeout in seconds (default: 10).",
    )
    return parser
