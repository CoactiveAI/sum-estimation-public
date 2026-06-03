"""Run sum-estimation experiments and save results as Parquet files.

Usage
-----
    python main.py

Configuration via environment variables (see .env.example):
    QDRANT_HOST, QDRANT_API_KEY, RESULTS_PATH, NUM_DATASET_EMBEDDINGS,
    NUM_QUERY_CANDIDATES
"""

import os
import random
from dataclasses import dataclass, field

import pandas as pd
from tqdm import tqdm

from config import settings
from my_datasets import (Dataset, Dataset_Image_BallCounting,
                         Dataset_Image_KDE, Dataset_Image_Softmax,
                         Dataset_Text_BallCounting, Dataset_Text_KDE)
from qdrant_helpers import qdrant
from qdrant_sum_estimation_algorithm import (Combined, OurAlgorithm,
                                             RandomSample,
                                             SumEstimationAlgorithm, TopK)
from qdrant_sum_problem_settings import (Problem_Image_BallCounting, Problem_Text_BallCounting,
                                         Problem_Image_KDE, Problem_Text_KDE, Problem_Image_Softmax,
                                         SumProblemSetting)

# ---------------------------------------------------------------------------
# Hyperparameter grids
# ---------------------------------------------------------------------------
k_values_our = [25, 50, 100, 200]
topk_values = [250, 500, 1000, 2000]
random_values = [500, 1000, 2000, 5000, 10000, 20000]

sum_problem_settings = [
    KDE_Image,
    Softmax_Image,
    BallCounting_Image,
    KDE_Text,
    BallCounting_Text,
]

# ---------------------------------------------------------------------------
# Load datasets (IDs + query pool from Qdrant – no local embedding files)
# ---------------------------------------------------------------------------
print("Loading datasets from Qdrant …")
setting_dataset_mapping = {
    Problem_Image_KDE.__name__: Dataset_Image_KDE(),
    Problem_Image_Softmax.__name__: Dataset_Image_Softmax(),
    Problem_Image_BallCounting.__name__: Dataset_Image_BallCounting(),
    Problem_Text_KDE.__name__: Dataset_Text_KDE(),
    Problem_Text_BallCounting.__name__: Dataset_Text_BallCounting(),
}
print("Datasets loaded.")

# ---------------------------------------------------------------------------
# Build all (setting, algorithm, params) combinations
# ---------------------------------------------------------------------------
@dataclass
class Combination:
    sum_problem_setting: type  # SumProblemSetting subclass
    sum_estimation_algorithm: type  # SumEstimationAlgorithm subclass
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.param_suffix = "_".join(str(v) for v in self.params.values())


all_combos: list[Combination] = []

for setting_class in sum_problem_settings:
    for k in k_values_our:
        all_combos.append(Combination(setting_class, OurAlgorithm, {'k': k}))
    for r in random_values:
        all_combos.append(Combination(setting_class, RandomSample, {'r': r}))
    for k in topk_values:
        all_combos.append(Combination(setting_class, TopK, {'k': k}))
    for k in topk_values:
        for r in random_values:
            all_combos.append(Combination(setting_class, Combined, {'k': k, 'r': r}))

random.shuffle(all_combos)

# ---------------------------------------------------------------------------
# Result accumulators  (cleared after each write to bound memory usage)
# ---------------------------------------------------------------------------
results_sum_estimates  = {sc: [] for sc in sum_problem_settings}
results_time_estimates = {sc: [] for sc in sum_problem_settings}
results_true_sum       = {sc: [] for sc in sum_problem_settings}
results_recall_exact   = {sc: [] for sc in sum_problem_settings}
results_recall_qdrant  = {sc: [] for sc in sum_problem_settings}

os.makedirs(settings.RESULTS_PATH, exist_ok=True)

# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------
oversampling = 2.5

for q in range(settings.NUM_QUERY_CANDIDATES):
    current_level = q % 10

    # ------------------------------------------------------------------
    # 1. Pick a random query vector for each setting class and build a
    #    per-query dataset object.  All-scores and max-sims are cached on
    #    the setting object so they are computed only once per (q, class).
    # ------------------------------------------------------------------
    query_setting_objs: dict[str, SumProblemSetting] = {}

    for setting_class in sum_problem_settings:
        base_dataset = setting_dataset_mapping[setting_class.__name__]
        query_obj = random.choice(base_dataset.query_pool)

        # Build dataset excluding the chosen query item
        dataset = [obj for obj in base_dataset.dataset_embedding_objects
                   if obj.image_id != query_obj.image_id]

        query_dataset = base_dataset.copy()
        query_dataset.query_embedding_objects = [query_obj]
        query_dataset.dataset_embedding_objects = dataset

        setting_obj = setting_class(query_dataset, qdrant, oversampling)
        setting_obj.SetNewLevel(current_level)

        # Pre-compute and cache all scores for this query (used by
        # GetTrueEstimate and GetExactRecall across all algorithms)
        print(f"  [q={q}] caching all-scores for {setting_class.__name__} …")
        setting_obj._get_cached_all_scores()
        setting_obj._get_cached_max_sims()

        query_setting_objs[setting_class.__name__] = setting_obj

    # ------------------------------------------------------------------
    # 2. Run every algorithm combination, reusing the cached setting obj
    # ------------------------------------------------------------------
    for combination in tqdm(all_combos, desc=f"q={q}"):
        setting_class = combination.sum_problem_setting
        setting_obj = query_setting_objs[setting_class.__name__]

        QUERY_ID = setting_obj.query_ids[0]
        params = combination.params
        param_suffix = combination.param_suffix
        setting_params = base_dataset.setting_params  # same for all queries

        # Instantiate algorithm (runs Qdrant queries internally)
        algo_obj: SumEstimationAlgorithm = combination.sum_estimation_algorithm(
            sum_problem_setting=setting_obj,
            params=params
        )
        method = algo_obj.name

        # Estimates
        sum_estimates = [algo_obj.GetEstimateForSettingParam(b)[0] for b in setting_params]
        time_estimate = algo_obj.GetTimeEstimate()
        true_sum = [algo_obj.GetTrueEstimate(b)[0] for b in setting_params]

        # Recall (not applicable for random sampling)
        recall_exact = None
        recall_qdrant = None
        if method == 'our':
            recall_qdrant = algo_obj.GetQdrantRecall()[0]
        elif method != 'random':
            recall_exact = algo_obj.GetExactRecall()[0]
            recall_qdrant = algo_obj.GetQdrantRecall()[0]

        entry_name = f"{method}_{param_suffix}"
        QUERY_ID_filename = str(QUERY_ID).strip("/").replace("/", "_")

        # Accumulate results
        results_sum_estimates[setting_class].append(
            {"method": entry_name, "query_id": QUERY_ID} |
            {str(param): est for param, est in zip(setting_params, sum_estimates)}
        )
        results_time_estimates[setting_class].append({
            "method": entry_name,
            "query_id": QUERY_ID,
            "time": time_estimate,
        })
        results_true_sum[setting_class].append(
            {"method": entry_name, "query_id": QUERY_ID} |
            {str(param): ts for param, ts in zip(setting_params, true_sum)}
        )

        if method == 'our':
            results_recall_qdrant[setting_class].extend([
                {"query_id": QUERY_ID, "k": params['k'], "level": l, "topk": topk}
                for l, topk in recall_qdrant
            ])
        elif method != 'random':
            results_recall_exact[setting_class].append({
                "query_id": QUERY_ID,
                "k": params['k'],
                "level": recall_exact[0],
                "topk": recall_exact[1],
            })
            results_recall_qdrant[setting_class].append({
                "query_id": QUERY_ID,
                "k": params['k'],
                "level": recall_qdrant[0],
                "topk": recall_qdrant[1],
            })

        # Write and clear accumulators each iteration to keep memory bounded
        name = setting_obj.name
        base_path = settings.RESULTS_PATH

        pd.DataFrame(results_sum_estimates[setting_class]).to_parquet(
            f"{base_path}/{name}_sum_estimates/{QUERY_ID_filename}.parquet")
        pd.DataFrame(results_time_estimates[setting_class]).to_parquet(
            f"{base_path}/{name}_time_estimates/{QUERY_ID_filename}.parquet")
        pd.DataFrame(results_true_sum[setting_class]).to_parquet(
            f"{base_path}/{name}_true_sum/{QUERY_ID_filename}.parquet")
        pd.DataFrame(results_recall_exact[setting_class]).to_parquet(
            f"{base_path}/{name}_recall_exact/{QUERY_ID_filename}.parquet")
        pd.DataFrame(results_recall_qdrant[setting_class]).to_parquet(
            f"{base_path}/{name}_recall_qdrant/{QUERY_ID_filename}.parquet")

        results_sum_estimates[setting_class]  = []
        results_time_estimates[setting_class] = []
        results_true_sum[setting_class]       = []
        results_recall_exact[setting_class]   = []
        results_recall_qdrant[setting_class]  = []
