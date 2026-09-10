"""End-to-end smoke test: index a tiny collection, then run the experiments on it.

Uses the serverless embedded Qdrant, so it needs no Docker and no cloud cluster.
It indexes synthetic vectors the same way step 2 does (levels, asset ids), runs
main.py over a couple of queries, and checks the Parquet output.

    python test_experiments.py
    python test_experiments.py --keep
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient, models

# The project lives one directory up, so put it on the path before importing it.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

COLLECTION = "amazon-reviews_distilbert"   # 768-d, Euclid, vector name 'abs'
VECTOR_NAME = "abs"
DIM = 8            # small enough to be fast; the collection's dim is not checked here
POINTS = 400
NUM_LEVELS = 10
MAX_RARITY = 40
QUERIES = 2
NUM_DATASET = 120


def write_step1_files(directory: str, prefix: str, vectors: np.ndarray) -> None:
    """Write the sharded layout step 1 produces, so local ground truth can be used."""
    os.makedirs(directory, exist_ok=True)
    chunk = 150
    chunks = []
    for index, start in enumerate(range(0, len(vectors), chunk)):
        end = min(start + chunk, len(vectors))
        stem = f"{prefix}_{index:05d}"
        np.save(os.path.join(directory, f"{stem}_embeddings.npy"), vectors[start:end])
        with open(os.path.join(directory, f"{stem}_ids.txt"), "w") as f:
            f.write("\n".join(str(i) for i in range(start, end)) + "\n")
        chunks.append({"index": index, "stem": stem, "rows": end - start, "source_rows": end,
                       "files": {"raw": f"{stem}_embeddings.npy", "ids": f"{stem}_ids.txt"}})
    with open(os.path.join(directory, f"{prefix}_manifest.json"), "w") as f:
        json.dump({"dim": int(vectors.shape[1]), "rows": len(vectors), "chunks": chunks}, f)


def build_index(storage: str, index_dir: str, embeddings_dir: str) -> None:
    """Create the collection, upload points with levels, record max levels."""
    client = QdrantClient(path=storage)
    client.create_collection(
        COLLECTION,
        vectors_config={VECTOR_NAME: models.VectorParams(
            size=DIM, distance=models.Distance.EUCLID)},
    )
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((POINTS, DIM)).astype(np.float32)
    levels = np.minimum(rng.geometric(0.5, size=(POINTS, NUM_LEVELS)), MAX_RARITY)

    client.upload_collection(
        collection_name=COLLECTION,
        vectors={VECTOR_NAME: vectors},
        payload=(
            {**{f"level_{j}": int(levels[i, j]) for j in range(NUM_LEVELS)},
             "asset_id": f"asset-{i:05d}"}
            for i in range(POINTS)
        ),
        ids=range(POINTS),
        batch_size=64,
    )
    write_step1_files(embeddings_dir, "run", vectors)
    maxima = {f"level_{j}": int(levels[:, j].max()) for j in range(NUM_LEVELS)}
    os.makedirs(os.path.join(index_dir, COLLECTION), exist_ok=True)
    with open(os.path.join(index_dir, COLLECTION, "max_levels.json"), "w") as f:
        json.dump({COLLECTION: maxima}, f, indent=2)
    client.close()
    print(f"[Setup] {POINTS} points indexed, max levels {maxima['level_0']} (level_0)")


def run_main(storage: str, index_dir: str, results: str, ground_truth: str = "local",
             embeddings_dir: str = None, extra: list = ()) -> str:
    command = [
        sys.executable, "main.py",
        "--queries", str(QUERIES),
        "--settings", "text_kde", "text_ball_counting",
        "--num-dataset", str(NUM_DATASET),
        "--results-path", results,
        "--seed", "0",
        "--k-values", "5",
        "--topk-values", "10",
        "--random-values", "50",
        "--qdrant", "embedded",
        "--embedded-path", storage,
        "--ground-truth", ground_truth,
        "--score-batch", "2",
    ] + list(extra)
    if embeddings_dir:
        command += ["--embeddings-dir", embeddings_dir,
                    "--embeddings-prefix", f"{COLLECTION}=run"]
    environment = dict(os.environ, HNSW_INDEX_DIR=index_dir, ID_CACHE_DIR=index_dir)
    completed = subprocess.run(command, cwd=PROJECT_DIR, capture_output=True, text=True, env=environment)
    if completed.returncode != 0:
        raise AssertionError(
            f"main.py exited {completed.returncode}\n"
            f"--- stdout ---\n{completed.stdout[-3000:]}\n--- stderr ---\n{completed.stderr[-3000:]}"
        )
    return completed.stdout


def check_results(results: str) -> str:
    """Every task must produce every result kind, with all methods represented."""
    details = []
    for task in ("text_kde", "text_ball_counting"):
        sums_dir = os.path.join(results, f"{task}_sum_estimates")
        assert os.path.isdir(sums_dir), f"missing {sums_dir}"
        files = sorted(os.listdir(sums_dir))
        assert len(files) == QUERIES, f"{task}: expected {QUERIES} query files, got {files}"

        frames = [pd.read_parquet(os.path.join(sums_dir, f)) for f in files]
        methods = sorted({m.split("_")[0] for frame in frames for m in frame["method"]})
        assert methods == ["combined", "our", "random", "topk"], f"{task}: methods {methods}"

        # Estimates must be finite, and the parameter columns must line up with
        # the true sums written alongside them.
        truth = pd.read_parquet(os.path.join(results, f"{task}_true_sum", files[0]))
        estimates = frames[0]
        param_columns = [c for c in estimates.columns if c not in ("method", "query_id")]
        assert param_columns == [c for c in truth.columns if c not in ("method", "query_id")]
        assert np.isfinite(estimates[param_columns].to_numpy(dtype=float)).all(), \
            f"{task}: non-finite estimate"
        assert np.isfinite(truth[param_columns].to_numpy(dtype=float)).all(), \
            f"{task}: non-finite true sum"

        # A true sum is a property of the query, not of the method that estimated
        # it, so every method must report the same value for a given parameter.
        for column in param_columns[:3]:
            assert truth[column].nunique() == 1, \
                f"{task}: true sum for {column} differs across methods"

        times = pd.read_parquet(os.path.join(results, f"{task}_time_estimates", files[0]))
        assert (times["time"] >= 0).all(), f"{task}: negative timing"

        recall = pd.read_parquet(os.path.join(results, f"{task}_recall_qdrant", files[0]))
        assert {"query_id", "k", "level", "topk"} <= set(recall.columns), recall.columns
        assert (recall["level"] >= 1).any(), f"{task}: no per-level recall rows"
        details.append(f"{task}: {len(estimates)} rows x {len(param_columns)} params")
    return "; ".join(details)


def check_uniform_sampling(storage: str, index_dir: str) -> str:
    """Sampled ids must span the collection, not just the first records."""
    os.environ["ID_CACHE_DIR"] = index_dir
    from helper.query_sampler import CollectionSampler

    client = QdrantClient(path=storage)
    sampler = CollectionSampler(client, COLLECTION, VECTOR_NAME,
                                rng=np.random.default_rng(1), id_cache_dir=index_dir)
    ids = sampler.sample_ids(120)
    assert len(ids) == len(set(ids)) == 120, "sample_ids returned duplicates or the wrong count"
    assert max(ids) > POINTS * 0.8, f"sample never reached the tail of the collection: max id {max(ids)}"
    assert min(ids) < POINTS * 0.2, f"sample never reached the head: min id {min(ids)}"
    objects = sampler.sample_queries(4)
    assert all(o.embedding is not None and len(o.embedding) == DIM for o in objects), \
        "sampled queries came back without vectors"
    client.close()
    return f"120 ids spanning {min(ids)}..{max(ids)} of {POINTS}"


def check_ground_truth_paths(local_results: str, qdrant_results: str) -> str:
    """The two ground-truth backends must produce the same true sums.

    Not bit-identical: Qdrant returns float32 scores, and the true sum
    `sum(exp((s - max) / (2 b^2)))` amplifies a score error by `1 / (2 b^2)`. At
    the smallest bandwidth in the grid (10^-0.7) that turns the ~6e-7 score
    difference measured in test_local_scores.py into ~7.5e-6 on the sum, so the
    tolerance sits above that and well below anything a real discrepancy
    (different scaling constant, wrong rows, wrong exclusion) would produce.
    """
    tolerance = 1e-4
    worst = 0.0
    worst_where = ""
    for task in ("text_kde", "text_ball_counting"):
        for name in sorted(os.listdir(os.path.join(local_results, f"{task}_true_sum"))):
            a = pd.read_parquet(os.path.join(local_results, f"{task}_true_sum", name))
            b = pd.read_parquet(os.path.join(qdrant_results, f"{task}_true_sum", name))
            columns = [c for c in a.columns if c not in ("method", "query_id")]
            assert columns == [c for c in b.columns if c not in ("method", "query_id")], \
                f"{task}: parameter columns differ between backends"
            left = a[columns].to_numpy(dtype=float)
            right = b[columns].to_numpy(dtype=float)
            denominator = np.maximum(np.abs(right), 1e-9)
            difference = float(np.max(np.abs(left - right) / denominator))
            if difference > worst:
                worst, worst_where = difference, f"{task}/{name}"
    assert worst < tolerance, (
        f"true sums differ between local and qdrant by {worst:.2e} relative "
        f"(worst at {worst_where}), above the {tolerance:.0e} float32 allowance"
    )
    return f"max relative difference {worst:.2e} (allowance {tolerance:.0e})"


def check_sharding(storage: str, index_dir: str, root: str, embeddings_dir: str) -> str:
    """Two shards must together cover exactly the queries one run covers."""
    whole = os.path.join(root, "results_whole")
    run_main(storage, index_dir, whole, "local", embeddings_dir)
    expected = set(os.listdir(os.path.join(whole, "text_kde_sum_estimates")))

    sharded = os.path.join(root, "results_sharded")
    for shard in (0, 1):
        run_main(storage, index_dir, sharded, "local", embeddings_dir,
                 extra=["--shard", str(shard), "--num-shards", "2"])
    produced = set(os.listdir(os.path.join(sharded, "text_kde_sum_estimates")))
    assert produced == expected, f"sharded run covered {produced}, whole run {expected}"
    return f"{len(expected)} query files from 2 shards"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true", help="Keep the scratch directory.")
    args = parser.parse_args()

    root = tempfile.mkdtemp(prefix="experiments-smoke-")
    storage = os.path.join(root, "qdrant_store")
    index_dir = os.path.join(root, "hnsw_index")
    embeddings_dir = os.path.join(root, "embeddings")
    results = os.path.join(root, "results")
    qdrant_results = os.path.join(root, "results_qdrant")

    checks = []
    try:
        build_index(storage, index_dir, embeddings_dir)
        checks.append(("uniform sampling", check_uniform_sampling(storage, index_dir)))
        run_main(storage, index_dir, results, "local", embeddings_dir)
        checks.append(("experiment output", check_results(results)))
        run_main(storage, index_dir, qdrant_results, "qdrant")
        checks.append(("qdrant backend", check_results(qdrant_results)))
        checks.append(("backends agree", check_ground_truth_paths(results, qdrant_results)))
        checks.append(("query sharding", check_sharding(storage, index_dir, root, embeddings_dir)))
        status = "PASS"
    except AssertionError as error:
        checks.append(("FAILED", str(error)))
        status = "FAIL"

    print("\n" + "=" * 78)
    for label, detail in checks:
        print(f"{label:20} {detail}")
    print(f"{status}")
    if args.keep:
        print(f"\nScratch kept at {root}")
    else:
        shutil.rmtree(root, ignore_errors=True)
    raise SystemExit(status == "FAIL")


if __name__ == "__main__":
    main()
