"""Exact ground truth computed locally from the step-1 embedding shards.

Scoring every dataset item against a query is what the true sums and the recall
baseline need, and asking Qdrant for it costs one filtered search per 500 ids -
20,000 requests per query at 10M items. The same numbers come out of one pass of
BLAS over the memory-mapped step-1 matrices, in well under a second per query.

This is only used for ground truth, never for the algorithms themselves: their
Qdrant calls are what the experiment times, so they stay exactly as they were.

Scores match Qdrant's conventions, verified against a live collection:

    Euclid -> ||x - q||   (the distance itself, not squared; lower is better)
    Dot    -> q . x       (higher is better)

Row order is the point id: step 2 uses the row's position in the run as the id,
so row `i` of the concatenated shards is point `i`.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

DISTANCES = ("Euclid", "Dot")


class LocalVectorStore:
    """Memory-mapped access to one step-1 run, scored in batches."""

    def __init__(self, embeddings_dir: str, prefix: str, normalised: bool, distance: str):
        if distance not in DISTANCES:
            raise SystemExit(f"Unsupported distance '{distance}'; expected one of {DISTANCES}.")
        self.distance = distance
        self.prefix = prefix
        self.variant = "normalised" if normalised else "raw"
        self.shard_paths = self._resolve(embeddings_dir, prefix, normalised)

        first = np.load(self.shard_paths[0], mmap_mode="r")
        self.dim = int(first.shape[1])
        self.shard_rows = [int(np.load(p, mmap_mode="r").shape[0]) for p in self.shard_paths]
        self.total_rows = int(sum(self.shard_rows))
        # Row norms are reused by every query batch, so they are computed once.
        self._row_sq: Optional[List[np.ndarray]] = None

    # ------------------------------------------------------------------
    # Locating step-1 output (sharded, or a single matrix)
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve(embeddings_dir: str, prefix: str, normalised: bool) -> List[str]:
        variant = "normalised" if normalised else "raw"
        manifest_path = os.path.join(embeddings_dir, f"{prefix}_manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            paths = []
            for chunk in manifest.get("chunks", []):
                files = chunk["files"]
                if variant not in files:
                    raise SystemExit(
                        f"{manifest_path} holds no '{variant}' vectors; re-run step 1 "
                        f"with --write {variant}."
                    )
                paths.append(os.path.join(embeddings_dir, files[variant]))
            if not paths:
                raise SystemExit(f"{manifest_path} lists no shards.")
            return paths

        suffix = "_embeddings_normalised.npy" if normalised else "_embeddings.npy"
        single = os.path.join(embeddings_dir, f"{prefix}{suffix}")
        if not os.path.exists(single):
            raise SystemExit(
                f"No step-1 vectors for prefix '{prefix}' in {embeddings_dir} "
                f"(looked for {os.path.basename(manifest_path)} and "
                f"{os.path.basename(single)}). Local ground truth needs the "
                f"embedding files; use --ground-truth qdrant to score in Qdrant instead."
            )
        return [single]

    def iter_shards(self) -> Iterator[Tuple[int, np.ndarray]]:
        offset = 0
        for path, rows in zip(self.shard_paths, self.shard_rows):
            yield offset, np.load(path, mmap_mode="r")
            offset += rows

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _row_squares(self) -> List[np.ndarray]:
        if self._row_sq is None:
            self._row_sq = [
                np.einsum("ij,ij->i", block, block, dtype=np.float64)
                for _, block in self.iter_shards()
            ]
        return self._row_sq

    def scores(self, queries: np.ndarray) -> np.ndarray:
        """Qdrant-equivalent scores for every row: shape `(len(queries), total_rows)`.

        One pass over the matrices covers the whole batch, so the disk cost is
        shared: at 10M x 768 that is ~30 GB read once per batch rather than once
        per query.
        """
        Q = np.ascontiguousarray(np.atleast_2d(queries), dtype=np.float32)
        if Q.shape[1] != self.dim:
            raise SystemExit(f"Query dim {Q.shape[1]} != vector dim {self.dim}.")

        out = np.empty((Q.shape[0], self.total_rows), dtype=np.float32)
        row_sq = self._row_squares() if self.distance == "Euclid" else None
        q_sq = (Q.astype(np.float64) ** 2).sum(axis=1) if self.distance == "Euclid" else None

        for index, (offset, block) in enumerate(self.iter_shards()):
            rows = block.shape[0]
            dots = Q @ np.asarray(block).T
            if self.distance == "Dot":
                out[:, offset:offset + rows] = dots
                continue
            # ||x - q||^2 = ||x||^2 - 2 q.x + ||q||^2, in float64 so the
            # cancellation for near-identical vectors stays bounded.
            squared = row_sq[index][None, :] - 2.0 * dots.astype(np.float64) + q_sq[:, None]
            np.maximum(squared, 0.0, out=squared)
            out[:, offset:offset + rows] = np.sqrt(squared)
        return out


class LocalGroundTruth:
    """Per-query scores for one collection, computed a batch at a time."""

    def __init__(self, store: LocalVectorStore, batch_size: int = 8):
        self.store = store
        self.batch_size = max(1, batch_size)
        self._scores: Dict[object, np.ndarray] = {}

    @property
    def total_rows(self) -> int:
        return self.store.total_rows

    def prime(self, query_ids: Sequence, query_vectors: Sequence[np.ndarray]) -> None:
        """Score `query_ids` in one pass each batch, replacing anything cached."""
        self._scores = {}
        ids = list(query_ids)
        vectors = np.asarray(query_vectors, dtype=np.float32)
        for start in range(0, len(ids), self.batch_size):
            chunk_ids = ids[start:start + self.batch_size]
            block = self.store.scores(vectors[start:start + self.batch_size])
            for position, qid in enumerate(chunk_ids):
                self._scores[qid] = block[position]

    def scores_for(self, query_id, query_vector: np.ndarray) -> np.ndarray:
        """Raw scores for one query, scoring it on the spot if it was not primed."""
        cached = self._scores.get(query_id)
        if cached is None:
            cached = self.store.scores(np.asarray(query_vector, dtype=np.float32))[0]
            self._scores[query_id] = cached
        return cached
