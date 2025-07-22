!pip install -r requirements.txt

import random
from dataclasses import dataclass

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
from qdrant_sum_problem_settings import (BallCounting_Image, BallCounting_Text,
                                         KDE_Image, KDE_Text, Softmax_Image,
                                         SumProblemSetting)

# Hyperparameter domains
k_values_our = [25, 50, 100, 200]
topk_values = [250, 500, 1000, 2000]
random_values = [500, 1000, 2000, 5000, 10000, 20000]

sum_problem_settings = [
    KDE_Image,
    Softmax_Image,
    BallCounting_Image,
    KDE_Text,
    BallCounting_Text
]

setting_dataset_mapping = {
    KDE_Image.__name__: Dataset_Image_KDE(spark, sc), 
    Softmax_Image.__name__: Dataset_Image_Softmax(spark, sc),
    BallCounting_Image.__name__: Dataset_Image_BallCounting(spark, sc),
    KDE_Text.__name__: Dataset_Text_KDE(spark, sc),
    BallCounting_Text.__name__: Dataset_Text_BallCounting(spark, sc)
}


@dataclass
class Combination:
    sum_problem_setting: SumProblemSetting
    sum_estimation_algorithm: SumEstimationAlgorithm

    setting_dataset: Dataset = None  # Initially optional
    params: dict = None

    def __post_init__(self):
        self.setting_dataset = setting_dataset_mapping[self.sum_problem_setting.__name__]
        self.param_suffix = "_".join(str(v) for v in self.params.values())


# Pre-generate all combinations
all_combos = []

for sum_problem_setting in sum_problem_settings:
    # Our
    for k in k_values_our:
        all_combos.append(Combination(sum_problem_setting=sum_problem_setting, sum_estimation_algorithm=OurAlgorithm, params={'k':k}))

    # Random
    for r in random_values:
        all_combos.append(Combination(sum_problem_setting=sum_problem_setting, sum_estimation_algorithm=RandomSample, params={'r':r}))

    # TopK
    for k in topk_values:
        all_combos.append(Combination(sum_problem_setting=sum_problem_setting, sum_estimation_algorithm=TopK, params={'k':k}))

    # combined
    for k in topk_values:
        for r in random_values:
            all_combos.append(Combination(sum_problem_setting=sum_problem_setting, sum_estimation_algorithm=Combined, params={'k':k, 'r':r}))

random.shuffle(all_combos)

################################################

# Results dictionary
results_sum_estimates = {setting: [] for setting in sum_problem_settings}
results_time_estimates = {setting: [] for setting in sum_problem_settings}
results_true_sum = {setting: [] for setting in sum_problem_settings}
results_recall_exact = {setting: [] for setting in sum_problem_settings}
results_recall_qdrant = {setting: [] for setting in sum_problem_settings}

################################################

for q in range(1):
    current_level = q%10
    oversampling = 2.5

    # Execute the loop
    for combination in tqdm(all_combos):

        # get info from Combination object
        sum_problem_setting = combination.sum_problem_setting
        sum_estimation_algorithm = combination.sum_estimation_algorithm
        setting_dataset = combination.setting_dataset
        params = combination.params
        param_suffix = combination.param_suffix
        
        # get info from Dataset child object
        query_embedding_objects = setting_dataset.query_embedding_objects
        dataset_embedding_objects = setting_dataset.dataset_embedding_objects
        setting_params = setting_dataset.setting_params

        # select query and filter dataset
        query_embedding_obj = random.choice(query_embedding_objects)
        query_embedding = query_embedding_obj.embedding
        QUERY_ID = query_embedding_obj.image_id        
        DATASET = [obj for obj in dataset_embedding_objects if obj.image_id != QUERY_ID] # exclude query object from dataset

        # create new Dataset object with updated values
        query_setting_dataset = setting_dataset.copy()
        query_setting_dataset.query_embedding_objects = [query_embedding_obj]
        query_setting_dataset.dataset_embedding_objects = DATASET
        
        # instantiate SumProblemSetting child object
        sum_problem_setting_obj = sum_problem_setting(query_setting_dataset, qdrant, oversampling)
        sum_problem_setting_obj.SetNewLevel(current_level)

        # instantiate SumEstimationAlgorithm child object
        sum_estimation_algorithm_obj = sum_estimation_algorithm(sum_problem_setting=sum_problem_setting_obj, params=params)
        method = sum_estimation_algorithm_obj.name

        # Calculate estimates ad true values
        sum_estimates = [sum_estimation_algorithm_obj.GetEstimateForSettingParam(b)[0] for b in setting_params]
        time_estimate = sum_estimation_algorithm_obj.GetTimeEstimate()
        true_sum = [sum_estimation_algorithm_obj.GetTrueEstimate(b)[0] for b in setting_params]
        if method == 'our':
            # TODO: Unblock this
            # recall_exact = sum_estimation_algorithm_obj.GetExactRecall()[0]
            recall_qdrant = sum_estimation_algorithm_obj.GetQdrantRecall()[0]            
        elif method != 'random':
            recall_exact = sum_estimation_algorithm_obj.GetExactRecall()[0]
            recall_qdrant = sum_estimation_algorithm_obj.GetQdrantRecall()[0]

        entry_name = f"{method}_{param_suffix}"
       

        # Construct the results dictionaries
        results_sum_estimates[sum_problem_setting].append(
            {"method": entry_name, "query_id": QUERY_ID} | 
            {str(param): estimate for param, estimate in zip(setting_params, sum_estimates)}
        )

        results_time_estimates[sum_problem_setting].append({
            "method": entry_name,
            "query_id": QUERY_ID,
            "time": time_estimate,
        })

        results_true_sum[sum_problem_setting].append(
            {"method": entry_name, "query_id": QUERY_ID} | 
            {str(param): estimate for param, estimate in zip(setting_params, true_sum)}
        )

        if method == 'our': # Our results will have nested dictionaries
            # TODO: Unblock this
            # results_recall_exact[sum_problem_setting].extend([
            #         {
            #             "query_id": QUERY_ID, 
            #             "k": params['k'],
            #             "level": l,
            #             "topk": topk
            #         } for l, topk in recall_exact
            # ])
            results_recall_qdrant[sum_problem_setting].extend([
                    {
                        "query_id": QUERY_ID, 
                        "k": params['k'],
                        "level": l,
                        "topk": topk
                    } for l, topk in recall_qdrant
            ])            
        elif method != 'random':
            results_recall_exact[sum_problem_setting].append({
                    "query_id": QUERY_ID, 
                    "k": params['k'],
                    "level": recall_exact[0],
                    "topk": recall_exact[1]                
            })
            results_recall_qdrant[sum_problem_setting].append({
                    "query_id": QUERY_ID, 
                    "k": params['k'],
                    "level": recall_qdrant[0],
                    "topk": recall_qdrant[1]                
            })

        QUERY_ID_filename = QUERY_ID.strip("/").replace("/", "_")
        pd.DataFrame(results_sum_estimates[sum_problem_setting]).to_parquet(f"{settings.RESULTS_PATH}/{sum_problem_setting_obj.name}_sum_estimates/{QUERY_ID_filename}.parquet")
        pd.DataFrame(results_time_estimates[sum_problem_setting]).to_parquet(f"{settings.RESULTS_PATH}/{sum_problem_setting_obj.name}_time_estimates/{QUERY_ID_filename}.parquet")
        pd.DataFrame(results_true_sum[sum_problem_setting]).to_parquet(f"{settings.RESULTS_PATH}/{sum_problem_setting_obj.name}_true_sum/{QUERY_ID_filename}.parquet")
        pd.DataFrame(results_recall_exact[sum_problem_setting]).to_parquet(f"{settings.RESULTS_PATH}/{sum_problem_setting_obj.name}_recall_exact/{QUERY_ID_filename}.parquet")
        pd.DataFrame(results_recall_qdrant[sum_problem_setting]).to_parquet(f"{settings.RESULTS_PATH}/{sum_problem_setting_obj.name}_recall_qdrant/{QUERY_ID_filename}.parquet")

        results_sum_estimates[sum_problem_setting] = []
        results_time_estimates[sum_problem_setting] = []
        results_true_sum[sum_problem_setting] = []
        results_recall_exact[sum_problem_setting] = []
        results_recall_qdrant[sum_problem_setting] = []