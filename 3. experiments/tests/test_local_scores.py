"""Local ground truth must agree with what Qdrant would have returned.

Builds a collection and the matching step-1 files from the same vectors, then
compares scores, true sums and the exact-recall baseline between the two paths.

    python test_local_scores.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

import numpy as np
from qdrant_client import QdrantClient, models

# The project lives one directory up, so put it on the path before importing it.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from helper.local_scores import LocalGroundTruth, LocalVectorStore

POINTS = 900
CHUNK = 400
DIM = 16


def write_step1_files(directory: str, prefix: str, vectors: np.ndarray, normalised: bool) -> None:
    """Write the sharded layout step 1 produces, manifest included."""
    os.makedirs(directory, exist_ok=True)
    variant = "normalised" if normalised else "raw"
    suffix = "_embeddings_normalised.npy" if normalised else "_embeddings.npy"
    chunks = []
    for index, start in enumerate(range(0, len(vectors), CHUNK)):
        end = min(start + CHUNK, len(vectors))
        stem = f"{prefix}_{index:05d}"
        np.save(os.path.join(directory, f"{stem}{suffix}"), vectors[start:end])
        with open(os.path.join(directory, f"{stem}_ids.txt"), "w") as f:
            f.write("\n".join(str(i) for i in range(start, end)) + "\n")
        chunks.append({"index": index, "stem": stem, "rows": end - start,
                       "source_rows": end,
                       "files": {variant: f"{stem}{suffix}", "ids": f"{stem}_ids.txt"}})
    with open(os.path.join(directory, f"{prefix}_manifest.json"), "w") as f:
        json.dump({"dim": vectors.shape[1], "rows": len(vectors), "chunks": chunks}, f)


def build_collection(path: str, name: str, vectors: np.ndarray, distance: str) -> QdrantClient:
    client = QdrantClient(path=path)
    client.create_collection(
        name,
        vectors_config={"v": models.VectorParams(size=vectors.shape[1],
                                                 distance=models.Distance(distance))},
    )
    client.upload_collection(collection_name=name, vectors={"v": vectors},
                             ids=range(len(vectors)), batch_size=128)
    return client


def compare(distance: str, normalised: bool, root: str) -> str:
    rng = np.random.default_rng(3)
    vectors = rng.standard_normal((POINTS, DIM)).astype(np.float32)
    if normalised:
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    name = f"cmp_{distance.lower()}"
    workdir = os.path.join(root, name)
    embeddings = os.path.join(workdir, "embeddings")
    write_step1_files(embeddings, "run", vectors, normalised)
    client = build_collection(os.path.join(workdir, "store"), name, vectors, distance)

    store = LocalVectorStore(embeddings, "run", normalised=normalised, distance=distance)
    assert store.total_rows == POINTS, f"store has {store.total_rows} rows"
    assert len(store.shard_paths) == 3, f"expected 3 shards, got {len(store.shard_paths)}"
    truth = LocalGroundTruth(store, batch_size=2)

    query_ids = [5, 111, 640]
    queries = vectors[query_ids]
    truth.prime(query_ids, queries)

    worst = 0.0
    for qid, vector in zip(query_ids, queries):
        local = truth.scores_for(qid, vector)
        # Ask Qdrant for the same scores, one id-filtered search per 300 points.
        remote = np.full(POINTS, np.nan, dtype=np.float64)
        for start in range(0, POINTS, 300):
            ids = list(range(start, min(start + 300, POINTS)))
            hits = client.search(
                collection_name=name,
                query_vector=models.NamedVector(name="v", vector=vector.tolist()),
                query_filter=models.Filter(must=[models.HasIdCondition(has_id=ids)]),
                limit=len(ids),
            )
            for hit in hits:
                remote[hit.id] = hit.score
        assert not np.isnan(remote).any(), "Qdrant did not score every point"
        worst = max(worst, float(np.abs(local - remote).max()))

    tolerance = 2e-3
    assert worst < tolerance, f"{distance}: local scores differ from Qdrant by {worst:.2e}"

    # Ranking must agree too, which is what the recall baseline depends on.
    local = truth.scores_for(query_ids[0], queries[0])
    best_first = np.argsort(local if distance == "Euclid" else -local, kind="stable")[:20]
    hits = client.search(
        collection_name=name,
        query_vector=models.NamedVector(name="v", vector=queries[0].tolist()),
        limit=20,
    )
    assert [h.id for h in hits] == [int(i) for i in best_first], \
        f"{distance}: top-20 ordering differs\n  qdrant: {[h.id for h in hits]}\n  local:  {list(best_first)}"
    client.close()
    return f"max |local - qdrant| = {worst:.2e} over {POINTS * len(query_ids)} scores; top-20 identical"


def main():
    root = tempfile.mkdtemp(prefix="local-scores-")
    results = []
    for distance, normalised in (("Euclid", False), ("Dot", True)):
        try:
            results.append((distance, "PASS", compare(distance, normalised, root)))
        except AssertionError as error:
            results.append((distance, "FAIL", str(error)))

    print("\n" + "=" * 78)
    for distance, status, detail in results:
        print(f"{status:4}  {distance:8} {detail if status == 'PASS' else ''}")
        if status == "FAIL":
            print(f"      {detail}")
    shutil.rmtree(root, ignore_errors=True)
    raise SystemExit(any(status == "FAIL" for _, status, _ in results))


if __name__ == "__main__":
    main()
