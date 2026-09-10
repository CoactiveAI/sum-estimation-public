"""Draw query and dataset items uniformly from a whole collection.

Experiments used to take the first `N` records a scroll returned and pick queries
out of that prefix, which biases every run towards one corner of the index -
whatever Qdrant happens to return first, typically the earliest-inserted points.
This samples over the entire collection instead.

Two id layouts are handled:

* **integer ids** (what step 2 assigns: row position, `0 .. count-1`) - sampled
  directly, no scan at all.
* **anything else** (e.g. UUIDs in a pre-existing collection) - the id space is
  listed once by scrolling and cached under `ID_CACHE_DIR`, then sampled from.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional, Sequence

import numpy as np

from .config import settings
from .qdrant_data_classes import EmbeddingObject

#: Ids per `retrieve` request. Qdrant takes the id list in the request body, so a
#: single call asking about millions of ids is a request no server will accept -
#: every lookup here is chunked at this size.
RETRIEVE_CHUNK = 2_000

#: Vectors per `retrieve` request. Smaller, because each record carries `dim`
#: floats: 2000 x 2048 float32 would be a ~16 MB response.
VECTOR_RETRIEVE_CHUNK = 500

#: Random ids probed to decide whether the id space is contiguous integers.
CONTIGUITY_PROBES = 8


class CollectionSampler:
    """Uniform sampling of point ids, and their vectors, from one collection."""

    def __init__(
        self,
        client,
        collection_name: str,
        vector_name: str,
        rng: Optional[np.random.Generator] = None,
        id_cache_dir: Optional[str] = None,
    ):
        self.client = client
        self.collection_name = collection_name
        self.vector_name = vector_name
        self.rng = rng or np.random.default_rng()
        self.id_cache_dir = id_cache_dir or settings.ID_CACHE_DIR

        self.total = client.count(collection_name=collection_name, exact=True).count
        if self.total == 0:
            raise SystemExit(
                f"Collection '{collection_name}' is empty - index it with step 2 first."
            )
        self._integer_ids = self._detect_integer_ids()
        self._id_pool: Optional[List] = None if self._integer_ids else self._load_id_pool()

    # ------------------------------------------------------------------
    # Id space
    # ------------------------------------------------------------------

    def _detect_integer_ids(self) -> bool:
        """True when ids look like step 2's contiguous integers `0 .. count-1`.

        Both ends plus a few random interior ids are probed. If any is missing the
        space is treated as arbitrary, which falls back to listing ids explicitly -
        slower to set up, but correct for sparse or non-integer ids.
        """
        probes = {0, self.total - 1}
        if self.total > 2:
            probes.update(
                int(x) for x in self.rng.integers(0, self.total, size=min(CONTIGUITY_PROBES, self.total))
            )
        probes = sorted(probes)
        found = self.client.retrieve(
            collection_name=self.collection_name, ids=probes, with_vectors=False
        )
        return len(found) == len(probes)

    def _id_cache_path(self) -> str:
        return os.path.join(self.id_cache_dir, self.collection_name, "point_ids.txt")

    def _load_id_pool(self) -> List:
        """List every id once (cached), for collections without integer ids."""
        path = self._id_cache_path()
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                cached = f.read().splitlines()
            if len(cached) == self.total:
                print(f"[Sampler] Reusing {len(cached)} cached ids for {self.collection_name}")
                return cached
            print(f"[Sampler] Cached id list is stale "
                  f"({len(cached)} cached vs {self.total} points); rebuilding")

        print(f"[Sampler] Listing {self.total} ids for {self.collection_name} "
              f"(one-off scan, cached afterwards)")
        ids, offset = [], None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=10_000,
                with_payload=False,
                with_vectors=False,
                offset=offset,
            )
            ids.extend(str(record.id) for record in records)
            if offset is None or not records:
                break

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(ids) + "\n")
        return ids

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _draw_indices(self, m: int, upper: int) -> np.ndarray:
        """`m` distinct integers from `[0, upper)`.

        A full permutation costs 8 bytes per point (80 MB at 10M) and is only worth
        it when the draw is a large fraction of the space; for the usual case of a
        small sample from a big collection, draw with repeats and dedupe instead.
        """
        if m >= upper:
            return np.arange(upper, dtype=np.int64)
        if m > upper // 4:
            return self.rng.permutation(upper)[:m].astype(np.int64)

        chosen: set = set()
        while len(chosen) < m:
            draw = self.rng.integers(0, upper, size=int((m - len(chosen)) * 1.3) + 16)
            chosen.update(int(x) for x in draw)
        return np.fromiter(chosen, dtype=np.int64, count=len(chosen))[:m]

    def _existing_ids(self, ids: Sequence) -> List:
        """Which of `ids` are actually present, asked in chunked requests."""
        present = []
        for start in range(0, len(ids), RETRIEVE_CHUNK):
            chunk = list(ids[start:start + RETRIEVE_CHUNK])
            records = self.client.retrieve(
                collection_name=self.collection_name, ids=chunk, with_vectors=False
            )
            present.extend(int(record.id) for record in records)
        return present

    def sample_ids(self, n: int, exclude: Iterable = ()) -> List:
        """Draw `n` distinct ids uniformly from the collection."""
        excluded = set(exclude)
        wanted = min(n, self.total - len(excluded))
        if wanted <= 0:
            raise SystemExit(f"Cannot sample {n} ids from {self.total} points.")

        if not self._integer_ids:
            pool = self._id_pool if not excluded else [p for p in self._id_pool if p not in excluded]
            picked = self._draw_indices(wanted, len(pool))
            return [pool[int(i)] for i in picked]

        # Taking (nearly) the whole collection: the ids are `0 .. total-1`, checked
        # at both ends and at random interior points, so there is nothing to look
        # up - which matters when "whole collection" means 10M ids.
        if wanted >= self.total - len(excluded):
            if not excluded:
                return list(range(self.total))[:wanted]
            return [i for i in range(self.total) if i not in excluded][:wanted]

        # A strict subset: draw, then confirm the ids exist. Requests are chunked,
        # and a shortfall (a sparse id space) is topped up rather than failing.
        collected: List[int] = []
        seen = set(int(x) for x in excluded)
        for _ in range(10):
            if len(collected) >= wanted:
                break
            short = wanted - len(collected)
            draw = self._draw_indices(min(int(short * 1.2) + 8, self.total), self.total)
            candidates = [int(i) for i in draw if int(i) not in seen]
            seen.update(candidates)
            collected.extend(self._existing_ids(candidates))
        if len(collected) < wanted:
            raise SystemExit(
                f"Only found {len(collected)} of {wanted} sampled ids in "
                f"'{self.collection_name}'; the id space looks sparse - delete the "
                f"cached id list and re-run to list ids explicitly instead."
            )
        return collected[:wanted]

    def fetch_embedding_objects(self, ids: List) -> List[EmbeddingObject]:
        """Retrieve vectors for `ids` and wrap them as EmbeddingObjects."""
        records = []
        for start in range(0, len(ids), VECTOR_RETRIEVE_CHUNK):
            chunk = list(ids[start:start + VECTOR_RETRIEVE_CHUNK])
            records.extend(self.client.retrieve(
                collection_name=self.collection_name, ids=chunk, with_vectors=True
            ))

        objects = []
        for record in records:
            raw = record.vector
            vector = raw[self.vector_name] if isinstance(raw, dict) else raw
            if vector is None:
                raise SystemExit(
                    f"Point {record.id} has no '{self.vector_name}' vector; check the "
                    f"vector name for '{self.collection_name}'."
                )
            objects.append(
                EmbeddingObject(image_id=record.id, embedding=np.array(vector, dtype=float))
            )
        return objects

    def sample_queries(self, n: int) -> List[EmbeddingObject]:
        """Sample `n` points uniformly and return them with their vectors."""
        return self.fetch_embedding_objects(self.sample_ids(n))

    def sample_dataset(self, n: int, exclude: Iterable = ()) -> List[EmbeddingObject]:
        """Sample the dataset whose sum is being estimated: ids only, no vectors."""
        return [EmbeddingObject(image_id=pid) for pid in self.sample_ids(n, exclude=exclude)]
