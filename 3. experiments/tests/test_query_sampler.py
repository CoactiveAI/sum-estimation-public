"""Tests for uniform sampling, with the request sizes a 10M collection implies.

Uses an embedded Qdrant, so no server is needed. The point of most of these is
scale behaviour: a sample of millions of ids must not become one giant request,
and taking the whole collection must not look anything up at all.

    python test_query_sampler.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
from qdrant_client import QdrantClient, models

# The project lives one directory up, so put it on the path before importing it.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from helper import query_sampler
from helper.query_sampler import RETRIEVE_CHUNK, VECTOR_RETRIEVE_CHUNK, CollectionSampler

COLLECTION = "sampler_test"
VECTOR_NAME = "abs"
DIM = 4


class RecordingClient:
    """Wraps a client and records the size of every retrieve request."""

    def __init__(self, inner):
        self.inner = inner
        self.retrieve_sizes: list[int] = []       # id-only lookups
        self.vector_retrieve_sizes: list[int] = []  # lookups that pull vectors

    def retrieve(self, collection_name, ids, **kwargs):
        size = len(list(ids))
        if kwargs.get("with_vectors"):
            self.vector_retrieve_sizes.append(size)
        else:
            self.retrieve_sizes.append(size)
        return self.inner.retrieve(collection_name=collection_name, ids=ids, **kwargs)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def build(path: str, ids) -> RecordingClient:
    client = QdrantClient(path=path)
    client.create_collection(
        COLLECTION,
        vectors_config={VECTOR_NAME: models.VectorParams(
            size=DIM, distance=models.Distance.EUCLID)},
    )
    ids = list(ids)
    vectors = np.random.default_rng(0).standard_normal((len(ids), DIM)).astype(np.float32)
    client.upload_collection(
        collection_name=COLLECTION, vectors={VECTOR_NAME: vectors}, ids=ids, batch_size=256
    )
    return RecordingClient(client)


def sampler_for(client, cache_dir, seed=0):
    return CollectionSampler(
        client, COLLECTION, VECTOR_NAME,
        rng=np.random.default_rng(seed), id_cache_dir=cache_dir,
    )


def main():
    root = tempfile.mkdtemp(prefix="sampler-test-")
    checks = []

    def check(name, fn):
        try:
            checks.append((name, "PASS", fn()))
        except AssertionError as error:
            checks.append((name, "FAIL", str(error)))

    # --- contiguous integer ids, as step 2 assigns --------------------------
    total = 5_000
    client = build(os.path.join(root, "dense"), range(total))
    cache = os.path.join(root, "cache")

    def subset_is_chunked():
        s = sampler_for(client, cache)
        client.retrieve_sizes.clear()
        ids = s.sample_ids(4_500)
        assert len(ids) == len(set(ids)) == 4_500, f"got {len(ids)} ids, {len(set(ids))} distinct"
        assert all(0 <= i < total for i in ids), "id outside the collection"
        biggest = max(client.retrieve_sizes)
        assert biggest <= RETRIEVE_CHUNK, f"a retrieve asked about {biggest} ids"
        return f"4500 ids, largest request {biggest} <= {RETRIEVE_CHUNK}"

    def whole_collection_needs_no_lookup():
        s = sampler_for(client, cache)
        client.retrieve_sizes.clear()
        ids = s.sample_ids(total)
        assert sorted(ids) == list(range(total)), "whole-collection draw is not the full id range"
        assert client.retrieve_sizes == [], f"issued {len(client.retrieve_sizes)} retrieves for the whole collection"
        return f"{total} ids, 0 requests"

    def oversized_request_asked_for_more_than_exists():
        s = sampler_for(client, cache)
        ids = s.sample_ids(total * 3)
        assert len(ids) == total, f"asked for more than exists and got {len(ids)}"
        return f"clamped to {total}"

    def exclusions_are_respected():
        s = sampler_for(client, cache)
        excluded = {0, 1, 2, 7, 4_999}
        ids = s.sample_ids(total - len(excluded), exclude=excluded)
        assert not (set(ids) & excluded), "excluded id present in the sample"
        assert len(ids) == total - len(excluded), f"got {len(ids)} ids"
        return f"{len(excluded)} ids held out of a whole-collection draw"

    def vectors_are_chunked():
        s = sampler_for(client, cache)
        client.vector_retrieve_sizes.clear()
        objects = s.sample_queries(1_200)
        assert len(objects) == 1_200, f"got {len(objects)} query objects"
        assert all(o.embedding is not None and len(o.embedding) == DIM for o in objects), \
            "query came back without a vector"
        sizes = client.vector_retrieve_sizes
        assert sizes, "no vector retrieve was recorded"
        assert max(sizes) <= VECTOR_RETRIEVE_CHUNK, \
            f"a vector retrieve asked for {max(sizes)} points"
        assert sum(sizes) == 1_200, f"vector requests covered {sum(sizes)} points"
        return f"1200 vectors in {len(sizes)} requests of <= {VECTOR_RETRIEVE_CHUNK}"

    def sampling_is_uniform():
        # Ten draws of 500 from 5000: the mean id should sit near the midpoint,
        # which a prefix-biased sampler would not manage.
        means = []
        for seed in range(10):
            s = sampler_for(client, cache, seed=seed)
            means.append(np.mean(s.sample_ids(500)))
        overall = float(np.mean(means))
        assert abs(overall - total / 2) < total * 0.06, \
            f"mean sampled id {overall:.0f}, expected near {total/2}"
        return f"mean id {overall:.0f} over 10 draws (midpoint {total//2})"

    check("subset chunked", subset_is_chunked)
    check("whole collection", whole_collection_needs_no_lookup)
    check("n > collection", oversized_request_asked_for_more_than_exists)
    check("exclusions", exclusions_are_respected)
    check("vector chunking", vectors_are_chunked)
    check("uniformity", sampling_is_uniform)

    # --- sparse, non-contiguous ids: must fall back to listing ids ----------
    sparse_ids = list(range(0, 4_000, 7))
    sparse_client = build(os.path.join(root, "sparse"), sparse_ids)

    def sparse_falls_back_and_is_exact():
        s = sampler_for(sparse_client, os.path.join(root, "cache_sparse"))
        assert not s._integer_ids, "sparse id space was treated as contiguous"
        ids = s.sample_ids(200)
        valid = {str(i) for i in sparse_ids}
        assert len(ids) == 200 and all(str(i) in valid for i in ids), \
            "sampled an id that does not exist"
        return f"{len(sparse_ids)} sparse ids listed and cached"

    def sparse_cache_is_reused():
        cache_dir = os.path.join(root, "cache_sparse")
        path = os.path.join(cache_dir, COLLECTION, "point_ids.txt")
        assert os.path.exists(path), f"no cached id list at {path}"
        with open(path) as f:
            assert len(f.read().splitlines()) == len(sparse_ids), "cached id list is the wrong length"
        return os.path.relpath(path, cache_dir)

    check("sparse fallback", sparse_falls_back_and_is_exact)
    check("sparse id cache", sparse_cache_is_reused)

    print("\n" + "=" * 74)
    for name, status, detail in checks:
        print(f"{status:4}  {name:22} {detail if status == 'PASS' else ''}")
        if status == "FAIL":
            print(f"      {detail}")
    raise SystemExit(any(status == "FAIL" for _, status, _ in checks))


if __name__ == "__main__":
    main()
