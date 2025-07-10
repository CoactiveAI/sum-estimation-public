from time import time
from qdrant_helpers import (
    qdrant,
    batch_qdrant_search,
    make_search_request,
    max_levels_dict,
    GetTopK,
    GetTopKWithLevel,
    GetRandomSample,
    GetAllScores,
)

from qdrant_data_classes import EmbeddingObject, EmbeddingObjectWithSim
from qdrant_sum_problem_settings import SumProblemSetting, KDE_Image, Softmax_Image, BallCounting_Image, KDE_Text, BallCounting_Text
from qdrant_sum_estimation_algorithm import RandomSample, TopK, Combined, OurAlgorithm
from datasets import get_query_dataset_kde_image, get_query_dataset_softmax_image, get_query_dataset_kde_text

def run_time_topk_for_query_k(qid, qe, collection_name, k: int, oversampling=2.5) -> float:
    start = time()
    _ = GetTopK(
        k=k,
        qid=qid,
        qe=qe,
        collection_name=collection_name,
        oversampling=oversampling
    )
    return time() - start


def run_time_random_for_query_r(qid, qe, dataset_embedding_objects, collection_name, r: int, oversampling=2.5) -> float:
    start = time()
    _ = GetRandomSample(
        m=r,
        qid=qid,
        qe=qe,
        DATASET_EMBEDDING_OBJECTS=dataset_embedding_objects,
        collection_name=collection_name,
        oversampling=oversampling
    )
    return time() - start


def run_time_our_for_query_k(qid, qe, collection_name, k: int, current_level: int, oversampling=2.5) -> float:
    start = time()
    _ = GetTopKWithLevel(
        k=k,
        max_level=max_levels_dict[collection_name][f"level_{current_level}"],
        current_level=current_level,
        qid=qid,
        qe=qe,
        collection_name=collection_name,
        oversampling=oversampling
    )
    return time() - start


def run_time_uai_for_query_k_r(qid, qe, dataset_embedding_objects, collection_name, k: int, r: int, current_level: int, oversampling=2.5) -> float:
    start = time()
    _ = GetTopK(
        k=k,
        qid=qid,
        qe=qe,
        collection_name=collection_name,
        oversampling=oversampling
    )
    _ = GetRandomSample(
        m=r,
        qid=qid,
        qe=qe,
        DATASET_EMBEDDING_OBJECTS=dataset_embedding_objects,
        collection_name=collection_name,
        oversampling=oversampling
    )
    return time() - start


def run_topk_algorithm(query_embedding_obj, DATASET_EMBEDDING_OBJECTS, setting_params, method, level, k, oversampling=2.5):
    query_id = query_embedding_obj.image_id

    # Exclude the query itself from the dataset
    DATASET = [obj for obj in DATASET_EMBEDDING_OBJECTS if obj.image_id != query_id]

    # Instantiate the method-specific object (e.g., KDE, Softmax, etc.)
    topk_obj = method(DATASET, [query_embedding_obj], qdrant, oversampling)
    topk_obj.SetNewLevel(level)

    # Run TopK for the specified k
    topk_algorithm = TopK(topk_obj, k)
    sum_estimates = [topk_algorithm.GetEstimateForSettingParam(b)[0] for b in setting_params]
    time_estimate = topk_algorithm.GetTimeEstimate()

    # Return result as {query_id: {k: sum_estimates}}
    return {query_id: {k: sum_estimates, 'time': time_estimate}}


def run_random_algorithm(query_embedding_obj, DATASET_EMBEDDING_OBJECTS, setting_params, method, level, m, oversampling=2.5):
    query_id = query_embedding_obj.image_id

    # Exclude the query itself from the dataset
    DATASET = [obj for obj in DATASET_EMBEDDING_OBJECTS if obj.image_id != query_id]

    # Instantiate method-specific object (e.g., KDE, Softmax)
    random_obj = method(DATASET, [query_embedding_obj], qdrant, oversampling)
    random_obj.SetNewLevel(level)

    # Run RandomSample with m points
    random_algorithm = RandomSample(random_obj, m)
    sum_estimates = [random_algorithm.GetEstimateForSettingParam(b)[0] for b in setting_params]
    time_estimate = random_algorithm.GetTimeEstimate()

    return {query_id: {m: sum_estimates, 'time': time_estimate}}


def run_our_algorithm(query_embedding_obj, DATASET_EMBEDDING_OBJECTS, setting_params, method, level, k, oversampling=2.5):
    query_id = query_embedding_obj.image_id

    # Exclude the query itself from the dataset
    DATASET = [obj for obj in DATASET_EMBEDDING_OBJECTS if obj.image_id != query_id]

    # Instantiate method-specific object
    our_obj = method(DATASET, [query_embedding_obj], qdrant, oversampling)
    our_obj.SetNewLevel(level)

    # Run OurAlgorithm
    our_algorithm = OurAlgorithm(our_obj, k)
    sum_estimates = [our_algorithm.GetEstimateForSettingParam(b)[0][0] for b in setting_params]
    time_estimate = our_algorithm.GetTimeEstimate()

    return {query_id: {k: sum_estimates,'time': time_estimate}}


def run_combined_algorithm(query_embedding_obj, DATASET_EMBEDDING_OBJECTS, setting_params, method, level, k, m, oversampling=2.5):
    query_id = query_embedding_obj.image_id

    # Exclude the query itself from the dataset
    DATASET = [obj for obj in DATASET_EMBEDDING_OBJECTS if obj.image_id != query_id]

    # Instantiate method-specific object
    uai_obj = method(DATASET, [query_embedding_obj], qdrant, oversampling)
    uai_obj.SetNewLevel(level)

    # Run combined algorithm with k and m
    uai_algorithm = Combined(uai_obj, k=k, m=m)
    sum_estimates = [uai_algorithm.GetEstimateForSettingParam(b)[0] for b in setting_params]
    time_estimate = uai_algorithm.GetTimeEstimate()

    return {query_id: {k: {m: sum_estimates,'time': time_estimate}}}


import random
import numpy as np
from tqdm import tqdm

def build_setting_class_instance_lookup(setting_dataset_pairs, spark, sc):
    def get_query_dataset_objects_for(setting_class, spark, sc):
        name = setting_class.__name__

        if name in {"KDE_Image", "BallCounting_Image"}:
            return get_query_dataset_kde_image(spark, sc)
        elif name == "Softmax_Image":
            return get_query_dataset_softmax_image(spark, sc)
        elif name in {"KDE_Text", "BallCounting_Text"}:
            return get_query_dataset_kde_text(spark, sc)
        else:
            raise ValueError(f"No loader implemented for class: {name}")

    setting_class_instance_lookup = {}

    for setting_class, _ in setting_dataset_pairs:
        setting_name = setting_class.__name__

        if setting_name in setting_class_instance_lookup: continue

        # Load query and dataset objects
        QUERY_EMBEDDING_OBJECTS, DATASET_EMBEDDING_OBJECTS = get_query_dataset_objects_for(setting_class, spark, sc)

        # # Exclude this query from dataset
        # filtered_dataset = [obj for obj in DATASET_EMBEDDING_OBJECTS if obj.image_id != query_id]

        # Fixed parameters
        # current_level = 0
        oversampling = 2.5

        # Select setting-specific parameter range
        if "KDE" in setting_name:
            setting_params = [10 ** p for p in np.arange(-0.25, 1.75, 0.05)]
        elif "Softmax" in setting_name:
            setting_params = [10 ** p for p in np.arange(-3.0, 1.0, 0.1)]
        elif "BallCounting" in setting_name:
            setting_params = sorted(set(
                [10 ** p for p in np.arange(-3.0, 2.0, 0.1)] +
                [10 ** p for p in np.arange(0.5, 1.8, 0.05)]
            ))
        else:
            raise ValueError(f"Unknown setting class: {setting_name}")

        # Define the config dictionary
        instance_config = {
            "query_embedding_objects": QUERY_EMBEDDING_OBJECTS,
            "dataset_embedding_objects": DATASET_EMBEDDING_OBJECTS,
            # "dataset_embedding_objects": filtered_dataset,
            "oversampling": oversampling,
            "setting_params": setting_params,
        }

        setting_class_instance_lookup[setting_name] = instance_config

    return setting_class_instance_lookup
