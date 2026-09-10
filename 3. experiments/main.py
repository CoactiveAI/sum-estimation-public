"""Run sum-estimation experiments and save results as Parquet files.

Each iteration draws one query uniformly from the whole collection, evaluates
every (algorithm, hyperparameter) combination against it, and writes one Parquet
file per result type per query.

Examples
--------
Full run against whatever QDRANT_HOST / local server is configured:
    python main.py

A short run over one task, useful for checking a setup end to end:
    python main.py --queries 2 --settings image_kde --num-dataset 500 \\
        --k-values 5 --topk-values 10 --random-values 50 --seed 0
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from helper import qdrant_helpers
from helper.config import settings
from local_problem_settings import LOCAL_EQUIVALENT
from helper.local_scores import LocalGroundTruth, LocalVectorStore
from my_datasets import (Dataset_Image_BallCounting, Dataset_Image_KDE,
                         Dataset_Image_Softmax, Dataset_Text_BallCounting,
                         Dataset_Text_KDE)
from helper.qdrant_connection import add_qdrant_args, connect
from helper.qdrant_connection import KNOWN_DISTANCES
from qdrant_sum_estimation_algorithm import (Combined, OurAlgorithm,
                                             RandomSample,
                                             SumEstimationAlgorithm, TopK)
from qdrant_sum_problem_settings import (Problem_Image_BallCounting,
                                         Problem_Image_KDE,
                                         Problem_Image_Softmax,
                                         Problem_Text_BallCounting,
                                         Problem_Text_KDE, SumProblemSetting)

# task name -> (problem setting class, dataset class)
SETTINGS = {
    "image_kde": (Problem_Image_KDE, Dataset_Image_KDE),
    "image_softmax": (Problem_Image_Softmax, Dataset_Image_Softmax),
    "image_ball_counting": (Problem_Image_BallCounting, Dataset_Image_BallCounting),
    "text_kde": (Problem_Text_KDE, Dataset_Text_KDE),
    "text_ball_counting": (Problem_Text_BallCounting, Dataset_Text_BallCounting),
}

# Hyperparameter grids
K_VALUES_OUR = [25, 50, 100, 200]
TOPK_VALUES = [250, 500, 1000, 2000]
RANDOM_VALUES = [500, 1000, 2000, 5000, 10000, 20000]

RESULT_KINDS = ("sum_estimates", "time_estimates", "true_sum", "recall_exact", "recall_qdrant")


@dataclass
class Combination:
    sum_problem_setting: type
    sum_estimation_algorithm: type
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.param_suffix = "_".join(str(v) for v in self.params.values())


def build_combinations(setting_classes, k_values_our, topk_values, random_values):
    combos: List[Combination] = []
    for setting_class in setting_classes:
        for k in k_values_our:
            combos.append(Combination(setting_class, OurAlgorithm, {'k': k}))
        for r in random_values:
            combos.append(Combination(setting_class, RandomSample, {'r': r}))
        for k in topk_values:
            combos.append(Combination(setting_class, TopK, {'k': k}))
        for k in topk_values:
            for r in random_values:
                combos.append(Combination(setting_class, Combined, {'k': k, 'r': r}))
    return combos


def write_results(base_path: str, setting_name: str, query_id, rows: dict) -> None:
    """Write one Parquet file per result kind for this query."""
    filename = f"{str(query_id).strip('/').replace('/', '_')}.parquet"
    for kind in RESULT_KINDS:
        if not rows[kind]:
            continue
        directory = os.path.join(base_path, f"{setting_name}_{kind}")
        os.makedirs(directory, exist_ok=True)
        pd.DataFrame(rows[kind]).to_parquet(os.path.join(directory, filename))


def resolve_prefixes(pairs, collections) -> dict:
    """Map collection -> step-1 prefix, defaulting to the collection name."""
    prefixes = {name: name for name in collections}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--embeddings-prefix wants COLLECTION=PREFIX, got '{pair}'")
        collection, prefix = pair.split("=", 1)
        if collection not in prefixes:
            raise SystemExit(
                f"--embeddings-prefix names unknown collection '{collection}'. "
                f"Known: {', '.join(sorted(prefixes))}"
            )
        prefixes[collection] = prefix
    return prefixes


def build_ground_truth(collections, embeddings_dir, prefixes, batch_size, required):
    """Open a local score source per collection, or return {} if unavailable."""
    sources = {}
    for collection in collections:
        distance = KNOWN_DISTANCES[collection]
        try:
            store = LocalVectorStore(
                embeddings_dir=embeddings_dir,
                prefix=prefixes[collection],
                normalised=distance["normalised"],
                distance=distance["distance"],
            )
        except SystemExit as unavailable:
            if required:
                raise
            print(f"[GroundTruth] {collection}: falling back to Qdrant ({unavailable})")
            return {}
        print(f"[GroundTruth] {collection}: {store.total_rows} x {store.dim} local vectors "
              f"({distance['distance']}, {len(store.shard_paths)} file(s))")
        sources[collection] = LocalGroundTruth(store, batch_size=batch_size)
    return sources


def validate_grids(base_datasets, random_values, names) -> None:
    """Refuse sample sizes larger than the dataset before any work starts.

    `RandomSample` and `Combined` draw `r` distinct items from the dataset without
    replacement, so `r` cannot exceed it. Left unchecked this surfaces as a numpy
    error inside the algorithm on the first combination.

    `--k-values` and `--topk-values` need no such check: those are Qdrant search
    limits, and a limit above the collection size simply returns fewer points.
    """
    for name in names:
        setting_class, _ = SETTINGS[name]
        dataset = base_datasets[setting_class.__name__]
        # The query is held out of its own dataset, so one fewer is available.
        usable = len(dataset.dataset_embedding_objects) - 1
        too_large = [r for r in random_values if r > usable]
        if not too_large:
            continue
        fits = [r for r in random_values if r <= usable]
        raise SystemExit(
            f"{name}: --random-values {too_large} exceed the {usable} dataset items "
            f"available for '{dataset.collection_name}' (one is held out as the query).\n"
            f"Random sampling draws without replacement, so r must be <= {usable}.\n"
            f"Either index more embeddings, or pass a grid that fits, e.g.\n"
            f"    --random-values {' '.join(str(r) for r in (fits or [max(1, usable // 4), max(2, usable // 2), usable]))}"
        )


def query_indices(total: int, shard: int, num_shards: int) -> list:
    """Which query positions this worker runs. Queries are independent."""
    if num_shards < 1 or not (0 <= shard < num_shards):
        raise SystemExit(f"--shard {shard} is not in range for --num-shards {num_shards}.")
    return list(range(shard, total, num_shards))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--queries", type=int, default=settings.NUM_QUERIES,
                        help=f"Queries to run (default: {settings.NUM_QUERIES}). Each is a "
                             f"uniform draw from the whole collection.")
    parser.add_argument("--settings", nargs="+", choices=sorted(SETTINGS), default=sorted(SETTINGS),
                        help="Which tasks to run (default: all).")
    parser.add_argument("--num-dataset", default=str(settings.NUM_DATASET_EMBEDDINGS),
                        help=f"Dataset items to sample, or 'all' for the whole collection "
                             f"(default: {settings.NUM_DATASET_EMBEDDINGS}).")
    parser.add_argument("--ground-truth", choices=("auto", "local", "qdrant"), default="auto",
                        help="Where true sums and the recall baseline come from. 'local' "
                             "scores the step-1 embedding files directly, which is orders of "
                             "magnitude cheaper at scale; 'qdrant' issues one filtered search "
                             "per 500 items. 'auto' (default) uses local when the files are "
                             "there. Algorithm timings are unaffected either way.")
    parser.add_argument("--embeddings-dir", default=settings.EMBEDDINGS_DIR,
                        help=f"Where step 1 wrote its vectors, for --ground-truth local "
                             f"(default: {settings.EMBEDDINGS_DIR}).")
    parser.add_argument("--embeddings-prefix", nargs="*", default=[], metavar="COLLECTION=PREFIX",
                        help="Step-1 prefix per collection, when it differs from the "
                             "collection name (e.g. amazon-reviews_distilbert=synthetic_x).")
    parser.add_argument("--score-batch", type=int, default=8,
                        help="Queries scored per pass over the embedding files (default: 8). "
                             "Higher shares the disk read across more queries, at "
                             "batch x collection_size x 4 bytes of memory.")
    parser.add_argument("--shard", type=int, default=0,
                        help="This worker's index, to split queries across processes (default: 0).")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="How many workers are splitting the queries (default: 1).")
    parser.add_argument("--results-path", default=settings.RESULTS_PATH,
                        help=f"Where to write Parquet output (default: {settings.RESULTS_PATH}).")
    parser.add_argument("--seed", type=int,
                        default=int(settings.RANDOM_SEED) if settings.RANDOM_SEED else None,
                        help="Seed for query/dataset selection; set it to repeat a run.")
    parser.add_argument("--oversampling", type=float, default=settings.OVERSAMPLING,
                        help=f"Qdrant rescoring oversampling (default: {settings.OVERSAMPLING}).")
    parser.add_argument("--k-values", type=int, nargs="+", default=K_VALUES_OUR,
                        help="k grid for OurAlgorithm.")
    parser.add_argument("--topk-values", type=int, nargs="+", default=TOPK_VALUES,
                        help="k grid for TopK and Combined.")
    parser.add_argument("--random-values", type=int, nargs="+", default=RANDOM_VALUES,
                        help="sample-size grid for RandomSample and Combined.")
    add_qdrant_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    client = connect(args.qdrant, args.embedded_path)
    qdrant_helpers.init_client(client)

    print("Sampling datasets and queries …")
    setting_classes = [SETTINGS[name][0] for name in args.settings]
    num_dataset = None if str(args.num_dataset).lower() == "all" else int(args.num_dataset)
    base_datasets = {}
    for name in args.settings:
        setting_class, dataset_class = SETTINGS[name]
        base_datasets[setting_class.__name__] = dataset_class(
            client=client, rng=rng,
            # None means "the whole collection", resolved by the sampler.
            num_dataset=num_dataset or 10 ** 12,
            num_queries=args.queries,
        )
        dataset = base_datasets[setting_class.__name__]
        print(f"  {name}: {len(dataset.dataset_embedding_objects)} dataset items, "
              f"{len(dataset.query_pool)} queries from '{dataset.collection_name}'")

    # Ground truth: local scoring where possible, else Qdrant as before.
    collections = sorted({d.collection_name for d in base_datasets.values()})
    ground_truth = {}
    if args.ground_truth != "qdrant":
        ground_truth = build_ground_truth(
            collections=collections,
            embeddings_dir=args.embeddings_dir,
            prefixes=resolve_prefixes(args.embeddings_prefix, collections),
            batch_size=args.score_batch,
            required=args.ground_truth == "local",
        )
    if ground_truth:
        for collection, source in ground_truth.items():
            dataset_sizes = {
                len(d.dataset_embedding_objects) for d in base_datasets.values()
                if d.collection_name == collection
            }
            for size in dataset_sizes:
                if size > source.total_rows:
                    raise SystemExit(
                        f"{collection}: sampled {size} dataset items but the local "
                        f"vectors hold {source.total_rows} rows - the embedding files "
                        f"are not the ones this collection was built from."
                    )
    else:
        print("[GroundTruth] Scoring true sums in Qdrant")

    validate_grids(base_datasets, args.random_values, args.settings)
    all_combos = build_combinations(
        setting_classes, args.k_values, args.topk_values, args.random_values
    )
    random.shuffle(all_combos)
    os.makedirs(args.results_path, exist_ok=True)

    positions = query_indices(args.queries, args.shard, args.num_shards)
    if args.num_shards > 1:
        print(f"[Shard] worker {args.shard}/{args.num_shards} runs {len(positions)} "
              f"of {args.queries} queries")

    # Score the embedding files once per batch of queries rather than once per
    # query: the pass over 10M vectors is a disk read, so sharing it matters.
    for batch_start in range(0, len(positions), args.score_batch):
        batch = positions[batch_start:batch_start + args.score_batch]
        for collection, source in ground_truth.items():
            queries = [
                dataset.query_pool[p % len(dataset.query_pool)]
                for dataset in base_datasets.values()
                if dataset.collection_name == collection
                for p in batch
            ]
            unique = {obj.image_id: obj for obj in queries}
            source.prime(list(unique), [obj.embedding for obj in unique.values()])

        for q in batch:
            run_query(q, args, setting_classes, base_datasets, client, ground_truth, all_combos)


def run_query(q, args, setting_classes, base_datasets, client, ground_truth, all_combos) -> None:
    """Run every combination against query position `q` and write its results."""
    current_level = q % 10

    # 1. One query per task, and a per-query setting object holding the caches.
    query_setting_objs: dict[str, SumProblemSetting] = {}
    for setting_class in setting_classes:
        base_dataset = base_datasets[setting_class.__name__]
        query_obj = base_dataset.query_pool[q % len(base_dataset.query_pool)]

        # The query must not be part of the dataset it is summed over.
        objects, dataset_ids = base_dataset.without(query_obj.image_id)

        query_dataset = base_dataset.copy()
        query_dataset.query_embedding_objects = [query_obj]
        query_dataset.dataset_embedding_objects = objects
        query_dataset.dataset_ids = dataset_ids

        source = ground_truth.get(base_dataset.collection_name)
        concrete = LOCAL_EQUIVALENT[setting_class] if source else setting_class
        setting_obj = concrete(query_dataset, client, args.oversampling)
        if source:
            setting_obj.attach_ground_truth(source)
        setting_obj.SetNewLevel(current_level)

        # Cached once per (query, task): used by GetTrueEstimate / f_vals.
        setting_obj._get_cached_all_scores()
        setting_obj._get_cached_max_sims()
        query_setting_objs[setting_class.__name__] = setting_obj

    # 2. Every algorithm combination, reusing those caches.
    rows = {
        setting_class: {kind: [] for kind in RESULT_KINDS}
        for setting_class in setting_classes
    }

    # The true sum is a property of (task, parameter) - the estimator has no
    # say in it - so it is computed once here instead of inside every one of
    # the ~38 combinations below.
    true_sums = {}
    for setting_class in setting_classes:
        setting_obj = query_setting_objs[setting_class.__name__]
        baseline = SumEstimationAlgorithm(setting_obj)
        true_sums[setting_class] = [
            baseline.GetTrueEstimate(b)[0]
            for b in setting_obj.setting_dataset.setting_params
        ]
    for combination in tqdm(all_combos, desc=f"q={q}"):
        setting_class = combination.sum_problem_setting
        setting_obj = query_setting_objs[setting_class.__name__]

        QUERY_ID = setting_obj.query_ids[0]
        params = combination.params
        setting_params = setting_obj.setting_dataset.setting_params

        algo_obj: SumEstimationAlgorithm = combination.sum_estimation_algorithm(
            sum_problem_setting=setting_obj, params=params
        )
        method = algo_obj.name

        sum_estimates = [algo_obj.GetEstimateForSettingParam(b)[0] for b in setting_params]
        time_estimate = algo_obj.GetTimeEstimate()
        true_sum = true_sums[setting_class]

        recall_exact = None
        recall_qdrant = None
        if method == 'our':
            recall_qdrant = algo_obj.GetQdrantRecall()[0]
        elif method != 'random':
            recall_exact = algo_obj.GetExactRecall()[0]
            recall_qdrant = algo_obj.GetQdrantRecall()[0]

        entry_name = f"{method}_{combination.param_suffix}"
        setting_rows = rows[setting_class]
        setting_rows["sum_estimates"].append(
            {"method": entry_name, "query_id": QUERY_ID} |
            {str(param): est for param, est in zip(setting_params, sum_estimates)}
        )
        setting_rows["time_estimates"].append({
            "method": entry_name, "query_id": QUERY_ID, "time": time_estimate,
        })
        setting_rows["true_sum"].append(
            {"method": entry_name, "query_id": QUERY_ID} |
            {str(param): ts for param, ts in zip(setting_params, true_sum)}
        )

        if method == 'our':
            setting_rows["recall_qdrant"].extend([
                {"query_id": QUERY_ID, "k": params['k'], "level": l, "topk": topk}
                for l, topk in recall_qdrant
            ])
        elif method != 'random':
            setting_rows["recall_exact"].append({
                "query_id": QUERY_ID, "k": params['k'],
                "level": recall_exact[0], "topk": recall_exact[1],
            })
            setting_rows["recall_qdrant"].append({
                "query_id": QUERY_ID, "k": params['k'],
                "level": recall_qdrant[0], "topk": recall_qdrant[1],
            })

    # 3. One write per (task, query), after every combination has been run.
    for setting_class in setting_classes:
        setting_obj = query_setting_objs[setting_class.__name__]
        write_results(
            args.results_path, setting_obj.name, setting_obj.query_ids[0], rows[setting_class]
        )
    print(f"[q={q}] wrote results for {len(setting_classes)} task(s) -> {args.results_path}")


if __name__ == "__main__":
    main()
