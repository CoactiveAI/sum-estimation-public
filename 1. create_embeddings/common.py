"""Shared plumbing for the per-combination embedding generation scripts.

Every `generate_<dataset>_<encoder>.py` script in this directory owns exactly one
(dataset, encoder) combination: it knows how to stream that dataset's items and
how to encode them. Everything that is identical across combinations lives here:

* `StreamingEncoderSource` - batching / truncation loop around a stream of items
* `write_embeddings`       - memory-mapped `.npy` writing (raw and/or normalised)
* `build_parser` / `Combination` - the CLI shared by every script

Adding a new combination therefore means one new script that subclasses
`StreamingEncoderSource` and declares a `Combination`; nothing here changes.

Output is sharded by default (`--chunk-rows`, DEFAULT_CHUNK_ROWS rows each) so a
failed run can resume, i.e. for prefix `<dataset>_<encoder>`:

    <prefix>_00000_embeddings.npy       (raw vectors; _normalised.npy per --write)
    <prefix>_00000_ids.txt              (asset id per row, line N <-> row N)
    <prefix>_manifest.json              (shard list, row counts, resume point)

`--chunk-rows 0` writes one unnumbered matrix per variant instead
(`<prefix>_embeddings.npy`, `<prefix>_ids.txt`), which is what
`2. create qdrant cluster/qdrant_insert.py` currently reads.
"""

from __future__ import annotations

import argparse
import ast
import itertools
import json
import os
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence, TypeVar

import numpy as np
import torch

#: Items encoded per forward pass. Small enough for a single mid-range GPU at
#: CLIP ViT-L/14-336 resolution; raise it for text-only or CPU-bound runs.
DEFAULT_BATCH_SIZE = 256

#: Rows per output shard. ~0.7 GB per shard at 768-d and ~2 GB at 2048-d: small
#: enough that a crash or an interrupted run costs little, large enough that a
#: 10M-row run stays a few dozen files.
DEFAULT_CHUNK_ROWS = 250_000

#: Repo-root `embeddings/`, matching EMBEDDINGS_DIR in step 2. Absolute, so output
#: lands in the same place no matter which directory a script is run from.
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embeddings"
)

T = TypeVar("T")


@dataclass
class Cursor:
    """Position bookkeeping that lets a chunked run resume mid-dataset.

    `skip_rows` is set by the writer before iteration; a source must discard that
    many raw dataset rows before encoding anything (discarding a row must not
    fetch or decode it, or resuming would cost as much as starting over).
    `last_row` is the ordinal of the most recent row the source actually yielded,
    which is what a resumed run skips past.
    """

    skip_rows: int = 0
    last_row: int = -1

    def note(self, ordinal: int) -> None:
        """Record that the row at `ordinal` was yielded for encoding."""
        self.last_row = ordinal

    @property
    def rows_consumed(self) -> int:
        """Dataset rows consumed so far, i.e. where a resumed run should start."""
        return self.last_row + 1


@dataclass(frozen=True)
class Combination:
    """Static description of one (dataset, encoder) pair.

    Attributes
    ----------
    collection_key:
        Key into `Settings.COLLECTION_NAME` / `KNOWN_COLLECTION_CONFIGS` in
        step 2. Also the default output prefix, so the file this script writes is
        the one `qdrant_insert.py --collection <key> --embeddings-prefix <key>`
        reads back.
    dim:
        Embedding dimensionality the encoder produces. Must equal the `size` of
        the matching collection config in step 2.
    normalised:
        Whether the target collection stores L2-normalised vectors (Dot-product
        distance). Drives the default of `--write`.
    """

    collection_key: str
    dim: int
    normalised: bool


class StreamingEncoderSource(ABC):
    """Streams dataset items and encodes them into fixed-size vector batches.

    Subclasses implement `iter_items` (pull `(asset_id, item)` pairs - images,
    review strings, ...) and `encode` (turn a list of items into a
    `(len(items), dim)` float32 array). This class handles batching, the
    `num_embeddings` cut-off and dtype coercion so every combination behaves
    identically.

    Asset ids travel with the items rather than being derived from row position,
    because a source may skip rows (an unreachable image, an empty review) and
    positions would then no longer line up with the dataset.
    """

    #: Embedding dimensionality produced by `encode`.
    dim: int

    #: Resume bookkeeping, lazily created. A source that streams a real dataset
    #: honours `cursor.skip_rows` and calls `cursor.note(ordinal)` per yielded row.
    _cursor: Cursor | None = None

    @property
    def cursor(self) -> Cursor:
        if self._cursor is None:
            self._cursor = Cursor()
        return self._cursor

    @abstractmethod
    def iter_items(self) -> Iterator[tuple[object, object]]:
        """Yield `(asset_id, item)` pairs, already filtered for usability."""

    @abstractmethod
    def encode(self, items: Sequence) -> np.ndarray:
        """Encode a batch of items into a `(len(items), dim)` float32 array."""

    def iter_batches(
        self, num_embeddings: int, batch_size: int
    ) -> Iterator[tuple[list[str], np.ndarray, int]]:
        """Yield `(asset_ids, vectors, source_rows)` per batch, aligned row for row.

        `source_rows` is how far into the dataset this batch reaches, captured
        when the batch was pulled rather than read off the cursor later: the next
        batch is pulled from the source before this one finishes being written, so
        the live cursor runs ahead of what is safely on disk. A resumed run that
        trusted the live value would skip rows that were pulled but never written.
        """
        stream = itertools.islice(self.iter_items(), num_embeddings)
        for chunk in batched(stream, batch_size):
            source_rows = self.cursor.rows_consumed
            asset_ids = [clean_asset_id(asset_id) for asset_id, _ in chunk]
            vectors = np.asarray(self.encode([item for _, item in chunk]), dtype=np.float32)
            if vectors.shape != (len(chunk), self.dim):
                raise RuntimeError(
                    f"{type(self).__name__}.encode returned {vectors.shape}, "
                    f"expected {(len(chunk), self.dim)}"
                )
            yield asset_ids, vectors, source_rows


def batched(iterable: Iterable[T], size: int) -> Iterator[list[T]]:
    """Yield successive lists of at most `size` items (final one may be short)."""
    iterator = iter(iterable)
    while chunk := list(itertools.islice(iterator, size)):
        yield chunk


def clean_asset_id(asset_id: object) -> str:
    """Render an asset id as one line of text for the ids sidecar file.

    The sidecar is line-delimited and positionally joined to the vectors, so an
    embedded newline would silently shift every later row.
    """
    text = str(asset_id).replace("\r", " ").replace("\n", " ").strip()
    return text or "(unknown)"


def l2_normalise(batch: np.ndarray) -> np.ndarray:
    """L2-normalise rows to unit length (for Dot-product / cosine collections)."""
    return torch.nn.functional.normalize(torch.from_numpy(batch), p=2.0, dim=1).numpy()


def resolve_device(device: str) -> str:
    """Resolve 'auto' to the best available torch device."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _truncate_npy(path: str, rows: int) -> None:
    """Shrink an already-written `.npy` to `rows` rows, in place.

    Outputs are pre-allocated at the requested size, but a streamed dataset can
    run out early (or items can be skipped). Rewriting the header - it is padded
    to a fixed length, and a smaller row count is never longer - and truncating
    the data avoids copying a multi-GB matrix just to shorten it.
    """
    with open(path, "r+b") as f:
        prefix = f.read(10)
        if prefix[:6] != b"\x93NUMPY":
            raise ValueError(f"{path} is not a .npy file")
        if prefix[6] != 1:
            raise ValueError(f"Unsupported .npy major version {prefix[6]} in {path}")
        header_len = struct.unpack("<H", prefix[8:10])[0]
        header = ast.literal_eval(f.read(header_len).decode("latin1"))

        cols = header["shape"][1]
        header["shape"] = (rows, cols)
        new_header = (
            "{'descr': '%s', 'fortran_order': %s, 'shape': (%d, %d), }"
            % (header["descr"], header["fortran_order"], rows, cols)
        )
        if len(new_header) + 1 > header_len:
            raise ValueError("Rewritten .npy header does not fit its padded slot")
        f.seek(10)
        f.write(new_header.ljust(header_len - 1).encode("latin1") + b"\n")
        f.truncate(10 + header_len + rows * cols * np.dtype(header["descr"]).itemsize)


class _ShardWriter:
    """One output shard: the vector matrices plus the ids sidecar beside them.

    Vectors are pre-allocated at `capacity` rows and written through memory maps,
    so a shard never has to fit in RAM. `close` shrinks the matrices to the rows
    actually used, which is what makes a short final shard valid rather than
    zero-padded.
    """

    def __init__(
        self,
        output_dir: str,
        stem: str,
        dim: int,
        capacity: int,
        write_raw: bool,
        write_normalised: bool,
    ):
        self.stem = stem
        self.capacity = capacity
        self.rows = 0
        self.write_raw = write_raw
        self.write_normalised = write_normalised
        self.raw_path = os.path.join(output_dir, f"{stem}_embeddings.npy")
        self.norm_path = os.path.join(output_dir, f"{stem}_embeddings_normalised.npy")
        self.ids_path = os.path.join(output_dir, f"{stem}_ids.txt")

        def allocate(path: str) -> np.memmap:
            return np.lib.format.open_memmap(
                path, mode="w+", dtype=np.float32, shape=(capacity, dim)
            )

        self.raw = allocate(self.raw_path) if write_raw else None
        self.norm = allocate(self.norm_path) if write_normalised else None
        self.ids = open(self.ids_path, "w", encoding="utf-8")

    @property
    def space(self) -> int:
        return self.capacity - self.rows

    def append(self, asset_ids: Sequence[str], vectors: np.ndarray) -> None:
        end = self.rows + len(asset_ids)
        if end > self.capacity:
            raise RuntimeError(f"Shard {self.stem} overflow: {end} rows > {self.capacity}")
        if self.raw is not None:
            self.raw[self.rows:end] = vectors
        if self.norm is not None:
            self.norm[self.rows:end] = l2_normalise(vectors)
        self.ids.write("".join(f"{asset_id}\n" for asset_id in asset_ids))
        self.rows = end

    def close(self) -> None:
        """Flush, release the memory maps, then shrink to the rows used."""
        if self.raw is not None:
            self.raw.flush()
        if self.norm is not None:
            self.norm.flush()
        # Drop the mappings before truncating: shortening a file that is still
        # mapped is not portable.
        self.raw = None
        self.norm = None
        self.ids.close()
        if self.rows < self.capacity:
            for path, enabled in (
                (self.raw_path, self.write_raw),
                (self.norm_path, self.write_normalised),
            ):
                if enabled:
                    _truncate_npy(path, self.rows)

    def files(self) -> dict:
        """Basenames of what this shard wrote, for the manifest."""
        written = {"ids": os.path.basename(self.ids_path)}
        if self.write_raw:
            written["raw"] = os.path.basename(self.raw_path)
        if self.write_normalised:
            written["normalised"] = os.path.basename(self.norm_path)
        return written


def _save_manifest(path: str, manifest: dict) -> None:
    """Write the manifest atomically, so a crash cannot leave it half-written."""
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(temporary, path)


def _load_manifest(path: str, dim: int, chunk_rows: int, write_raw: bool, write_normalised: bool) -> dict:
    """Load a manifest for resuming, or start a fresh one.

    A manifest whose geometry differs from this run is refused rather than
    appended to, since mixing dimensionalities or output variants across shards
    would produce a set no consumer could read as one dataset.
    """
    if not os.path.exists(path):
        return {
            "dim": dim,
            "chunk_rows": chunk_rows,
            "write_raw": write_raw,
            "write_normalised": write_normalised,
            "rows": 0,
            "source_rows_consumed": 0,
            "chunks": [],
        }

    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)

    incompatible = {
        "dim": (manifest.get("dim"), dim),
        "chunk_rows": (manifest.get("chunk_rows"), chunk_rows),
        "write_raw": (manifest.get("write_raw"), write_raw),
        "write_normalised": (manifest.get("write_normalised"), write_normalised),
    }
    mismatches = [f"{k}: manifest={was!r} run={now!r}" for k, (was, now) in incompatible.items() if was != now]
    if mismatches:
        raise SystemExit(
            f"Existing manifest {path} does not match this run:\n  "
            + "\n  ".join(mismatches)
            + "\nUse matching flags to resume, a different --output-prefix, or delete "
              "the manifest and its shards to start over."
        )
    return manifest


def write_embeddings(
    source: StreamingEncoderSource,
    prefix: str,
    output_dir: str,
    num_embeddings: int,
    batch_size: int,
    write_raw: bool,
    write_normalised: bool,
    chunk_rows: int = 0,
) -> None:
    """Stream `source` to disk as `.npy` matrices plus an ids sidecar.

    With `chunk_rows == 0` the run writes one matrix per variant, which is the
    simplest layout and fine into the low millions of rows. With `chunk_rows > 0`
    it writes numbered shards of at most that many rows plus a manifest, which is
    what makes very large runs practical: a crash costs one shard instead of the
    whole file, re-running resumes after the last recorded shard, and disjoint
    shard ranges can be produced on separate machines.

    Asset ids are written in the same loop as the vectors, so row N of a matrix
    always corresponds to line N of that shard's ids file.
    """
    if not (write_raw or write_normalised):
        raise SystemExit("Nothing to write: enable at least one of raw / normalised output.")
    os.makedirs(output_dir, exist_ok=True)

    if chunk_rows > 0:
        _write_chunked(
            source, prefix, output_dir, num_embeddings, batch_size,
            write_raw, write_normalised, chunk_rows,
        )
    else:
        _write_single(
            source, prefix, output_dir, num_embeddings, batch_size, write_raw, write_normalised,
        )


def _write_single(
    source: StreamingEncoderSource,
    prefix: str,
    output_dir: str,
    num_embeddings: int,
    batch_size: int,
    write_raw: bool,
    write_normalised: bool,
) -> None:
    """Write the whole run as one matrix per variant (no manifest, no resume)."""
    shard = _ShardWriter(
        output_dir, prefix, source.dim, num_embeddings, write_raw, write_normalised
    )
    try:
        for asset_ids, vectors, _ in source.iter_batches(num_embeddings, batch_size):
            shard.append(asset_ids, vectors)
            print(f"\r[Encode] {shard.rows}/{num_embeddings} vectors", end="", flush=True)
    except KeyboardInterrupt:
        # Long real-world runs get stopped by hand; keep the prefix already
        # encoded rather than discarding hours of work.
        print("\n[Encode] Interrupted - keeping the vectors encoded so far.")
    finally:
        print()
        written = shard.rows
        if written < num_embeddings:
            print(f"[Encode] Stopped after {written} items; truncating outputs.")
        shard.close()

    if written == 0:
        raise SystemExit("The source yielded no items - nothing was written.")
    for path, enabled, kind in (
        (shard.raw_path, write_raw, "raw"),
        (shard.norm_path, write_normalised, "normalised"),
    ):
        if enabled:
            print(f"[Done] Wrote {written} x {source.dim} {kind} embeddings -> {path}")
    print(f"[Done] Wrote {written} asset ids -> {shard.ids_path}")


def _write_chunked(
    source: StreamingEncoderSource,
    prefix: str,
    output_dir: str,
    num_embeddings: int,
    batch_size: int,
    write_raw: bool,
    write_normalised: bool,
    chunk_rows: int,
) -> None:
    """Write numbered shards plus a manifest, resuming any earlier run."""
    if chunk_rows < batch_size:
        raise SystemExit(
            f"--chunk-rows ({chunk_rows}) must be at least --batch-size ({batch_size}); "
            f"shards end on batch boundaries so the resume point stays exact."
        )

    manifest_path = os.path.join(output_dir, f"{prefix}_manifest.json")
    manifest = _load_manifest(manifest_path, source.dim, chunk_rows, write_raw, write_normalised)
    manifest["target_rows"] = num_embeddings

    written = manifest["rows"]
    index = len(manifest["chunks"])
    if written >= num_embeddings:
        print(f"[Resume] {manifest_path} already holds {written} rows across "
              f"{index} shards; nothing to do.")
        return
    if index:
        source.cursor.skip_rows = manifest["source_rows_consumed"]
        print(f"[Resume] {index} shards / {written} rows already written; "
              f"skipping {source.cursor.skip_rows} dataset rows.")

    def record(shard: _ShardWriter, source_rows: int) -> None:
        """Close a shard and commit it to the manifest."""
        nonlocal written
        shard.close()
        written += shard.rows
        # Where the next run must resume from: how far the last batch written into
        # this shard reached. Sources that do not track dataset ordinals never skip
        # rows, so the cumulative row count is the position.
        consumed = source_rows or written
        manifest["chunks"].append({
            "index": len(manifest["chunks"]),
            "stem": shard.stem,
            "rows": shard.rows,
            "source_rows": consumed,
            "files": shard.files(),
        })
        manifest["rows"] = written
        manifest["source_rows_consumed"] = consumed
        _save_manifest(manifest_path, manifest)
        print(f"\r[Shard] {shard.stem}: {shard.rows} rows "
              f"({written}/{num_embeddings} total)")

    shard = None
    boundary = 0
    try:
        for asset_ids, vectors, source_rows in source.iter_batches(
            num_embeddings - written, batch_size
        ):
            # Shards end on batch boundaries: splitting a batch would leave the
            # resume point between two ordinals, which cannot be expressed.
            if shard is not None and len(asset_ids) > shard.space:
                record(shard, boundary)
                shard = None
            if shard is None:
                capacity = min(chunk_rows, num_embeddings - written)
                shard = _ShardWriter(
                    output_dir, f"{prefix}_{index:05d}", source.dim, capacity,
                    write_raw, write_normalised,
                )
                index += 1
            shard.append(asset_ids, vectors)
            # `source_rows` already counts this batch's rows, so it is the resume
            # point as soon as the batch is on disk.
            boundary = source_rows
            print(f"\r[Encode] {written + shard.rows}/{num_embeddings} vectors",
                  end="", flush=True)
    except KeyboardInterrupt:
        print("\n[Encode] Interrupted - committing the shard in progress.")
    finally:
        print()
        if shard is not None and shard.rows:
            record(shard, boundary)
        elif shard is not None:
            shard.close()

    if written == 0:
        raise SystemExit("The source yielded no items - nothing was written.")
    print(f"[Done] {written} x {source.dim} embeddings across "
          f"{len(manifest['chunks'])} shards; manifest -> {manifest_path}")


def build_parser(combination: Combination, description: str) -> argparse.ArgumentParser:
    """Build the CLI shared by every generation script."""
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--num-embeddings", type=int, default=100_000,
        help="Number of embeddings to generate (default: 100000).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Items encoded per forward pass (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help="Directory to write .npy files to (default: the repo's embeddings/ "
             "directory, which is where step 2 reads from).",
    )
    parser.add_argument(
        "--output-prefix", default=combination.collection_key,
        help=f"Output file prefix (default: {combination.collection_key}). Pass the "
             f"same value to qdrant_insert.py --embeddings-prefix.",
    )
    parser.add_argument(
        "--write", choices=("default", "raw", "normalised", "both"), default="default",
        help="Which files to write. 'default' writes the variant this collection "
             f"needs ({'normalised' if combination.normalised else 'raw'}).",
    )
    parser.add_argument(
        "--device", default="auto",
        help="Torch device: 'auto' (default), 'cpu', 'cuda' or 'mps'.",
    )
    parser.add_argument(
        "--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS,
        help=f"Rows per output shard (default: {DEFAULT_CHUNK_ROWS}). Shards come "
             f"with a manifest and let a failed run resume where it stopped. Pass 0 "
             f"to write one matrix instead. Must be >= --batch-size.",
    )
    return parser


def resolve_write_flags(combination: Combination, choice: str) -> tuple[bool, bool]:
    """Map `--write` to (write_raw, write_normalised)."""
    if choice == "default":
        return (not combination.normalised, combination.normalised)
    return (choice in ("raw", "both"), choice in ("normalised", "both"))


def run(source: StreamingEncoderSource, combination: Combination, args: argparse.Namespace) -> None:
    """Validate the source against the collection config and write its output."""
    if source.dim != combination.dim:
        raise SystemExit(
            f"Encoder produced dim {source.dim}, but collection "
            f"'{combination.collection_key}' expects {combination.dim}."
        )
    write_raw, write_normalised = resolve_write_flags(combination, args.write)
    write_embeddings(
        source=source,
        prefix=args.output_prefix,
        output_dir=args.output_dir,
        num_embeddings=args.num_embeddings,
        batch_size=args.batch_size,
        write_raw=write_raw,
        write_normalised=write_normalised,
        chunk_rows=getattr(args, "chunk_rows", 0),
    )
