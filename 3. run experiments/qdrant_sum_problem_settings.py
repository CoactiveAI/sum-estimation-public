from qdrant_data_classes import EmbeddingObject, EmbeddingObjectWithSim
from my_datasets import Dataset
from tqdm import tqdm
from typing import List, Callable, Optional
from time import sleep

import math
import numpy as np
import random
import torch
import uuid

from qdrant_client import models, QdrantClient
from qdrant_client.http.models import (
    HasIdCondition,
    NamedVector,
    QuantizationSearchParams,
    ScoredPoint,
    SearchParams,
    SearchRequest,
)

from qdrant_helpers import (
    batch_qdrant_search,
    collections_dict,
    make_search_request,
    max_levels_dict,
    get_top_k_for_query,
    get_top_k_with_level,
    get_random_sample_for_query,
    get_all_scores_for_query
)
from qdrant_client import QdrantClient, models
from qdrant_client.models import SearchRequest, NamedVector, SearchParams, QuantizationSearchParams


class SumProblemSetting:
    def __init__(
        self,
        setting_dataset: Dataset,
        fn_for_nn_sims_calc: Callable,
        qdrant: QdrantClient,
        oversampling: float = 2.5,
    ):
        self.setting_dataset = setting_dataset
        dataset_embedding_objects = setting_dataset.dataset_embedding_objects
        query_embedding_objects = setting_dataset.query_embedding_objects
        collection_name = setting_dataset.collection_name

        self.N_d = len(dataset_embedding_objects)
        self.N_q = len(query_embedding_objects)
        self.dim = len(query_embedding_objects[0].embedding) if query_embedding_objects[0].embedding is not None else None
        self.fn_for_nn_sims_calc = fn_for_nn_sims_calc

        self.query_ids = [obj.image_id for obj in query_embedding_objects]
        self.query_embeddings = [obj.embedding for obj in query_embedding_objects]
        self.dataset_embedding_objects = dataset_embedding_objects
        self.dataset_ids = [obj.image_id for obj in dataset_embedding_objects]

        self.collection_name = collection_name
        self.vector_name = collections_dict[collection_name]['vector_name']
        self.is_list_of_ids_uuids = collections_dict[collection_name]['is_list_of_ids_uuids']

        self.qdrant = qdrant
        self.oversampling = oversampling
        self.current_level = 0
        self.max_level = None
        

    def GetTopK(self, k: int) -> List[List[EmbeddingObjectWithSim]]:
        return [
            get_top_k_for_query(
                qid=qi,
                qe=qe,
                k=k,
                collection_name=self.collection_name,
                vector_name=self.vector_name,
                is_uuid=self.is_list_of_ids_uuids,
                oversampling=self.oversampling,
                fn_for_nn_sims_calc=self.fn_for_nn_sims_calc
            )
            for qi, qe in zip(self.query_ids, self.query_embeddings)
        ]

    def GetTopKWithLevel(self, k: int) -> List[List[EmbeddingObjectWithSim]]:
        return [
            get_top_k_with_level(
                qid=qi,
                qe=qe,
                k=k,
                current_level=self.current_level,
                max_level=self.max_level,
                collection_name=self.collection_name,
                vector_name=self.vector_name,
                is_uuid=self.is_list_of_ids_uuids,
                oversampling=self.oversampling,
                fn_for_nn_sims_calc=self.fn_for_nn_sims_calc
            )
            for qi, qe in zip(self.query_ids, self.query_embeddings)
        ]

    def GetRandomSample(self, m: int) -> List[List[EmbeddingObjectWithSim]]:
        return [
            get_random_sample_for_query(
                qe=qe,
                m=m,
                dataset_embedding_objects=self.dataset_embedding_objects,
                collection_name=self.collection_name,
                vector_name=self.vector_name,
                oversampling=self.oversampling,
                fn_for_nn_sims_calc=self.fn_for_nn_sims_calc
            )
            for qe in self.query_embeddings
        ]

    def GetMaxSims(self) -> List[float]:
        queries = [
            make_search_request(
                vector=qe,
                vector_name=self.vector_name,
                k=1,
                oversampling=self.oversampling,
                must=None,
                must_not=[models.HasIdCondition(has_id=self.query_ids)] if self.is_list_of_ids_uuids else []
            )
            for qe in self.query_embeddings
        ]
        results = batch_qdrant_search(self.collection_name, queries, debug_tag="GetMaxSims")
        return [self.fn_for_nn_sims_calc(batch[0].score) for batch in results]

    def GetAllScores(self) -> List[float]:
        return [
            get_all_scores_for_query(
                qid=qid,
                qe=qe,
                ids=self.dataset_ids,
                collection_name=self.collection_name,
                vector_name=self.vector_name,
                oversampling=self.oversampling,
                fn_for_nn_sims_calc=self.fn_for_nn_sims_calc
            )
            for qe, qid in zip(self.query_embeddings, self.query_ids)
        ]

    def GetNNSims(self, selected_embedding_objects_with_sims: List[List[EmbeddingObjectWithSim]]) -> List[List[float]]:
        return [[obj.nn_sim_to_q for obj in objs] for objs in selected_embedding_objects_with_sims]

    def SetNewLevel(self, l: int) -> None:
        self.current_level = l
        self.max_level = max_levels_dict[self.collection_name][f"level_{l}"]

    def GetN_d(self) -> int:
        return self.N_d

    def GetN_q(self) -> int:
        return self.N_q
    

########################################################

class KDE_Image(SumProblemSetting):
    def __init__(
        self, 
        setting_dataset: Dataset,
        qdrant: QdrantClient,
        oversampling: float = 2.5,       
    ):
        self.name = 'image_kde'
        super().__init__(
            setting_dataset = setting_dataset,
            fn_for_nn_sims_calc = self.fn_for_nn_sims_calc,
            qdrant = qdrant,
            oversampling = oversampling,
        )

    def fn_for_nn_sims_calc(self, score:float) -> Callable:
        return -score**2

    def f_vals(self, bandwidth: np.float64, selected_embedding_objects: List[List[EmbeddingObject]] = None) -> List[List[np.float64]]:
        f_vals_per_query = []

        nn_sims = self.GetNNSims(selected_embedding_objects)
        max_sims = self.GetMaxSims()

        for q_idx in range(self.N_q):
            nn_sims_to_q_scaled = np.array(nn_sims[q_idx]) - max_sims[q_idx]
            f_vals = np.exp(nn_sims_to_q_scaled / (2 * (bandwidth**2)), dtype=np.float64)
            f_vals_per_query.append(f_vals.tolist())
            
        return f_vals_per_query

class Softmax_Image(SumProblemSetting):
    def __init__(
        self, 
        setting_dataset: Dataset,
        qdrant: QdrantClient,
        oversampling: float = 2.5,
    ):
        self.name = 'image_softmax'
        super().__init__(
            setting_dataset = setting_dataset, 
            fn_for_nn_sims_calc = self.fn_for_nn_sims_calc,
            qdrant = qdrant,
            oversampling = oversampling,            
        )

    def fn_for_nn_sims_calc(self, score:float) -> Callable:
        return score

    def f_vals(self, temperature: np.float64, selected_embedding_objects: List[List[EmbeddingObject]] = None) -> List[List[np.float64]]:
        f_vals_per_query = []

        nn_sims = self.GetNNSims(selected_embedding_objects)
        max_sims = self.GetMaxSims()

        for q_idx in range(self.N_q):
            nn_sims_to_q_scaled = np.array(nn_sims[q_idx]) - max_sims[q_idx]
            f_vals = np.exp(nn_sims_to_q_scaled / temperature, dtype=np.float64)
            f_vals_per_query.append(f_vals.tolist())
            
        return f_vals_per_query

class BallCounting_Image(SumProblemSetting):
    def __init__(
        self, 
        setting_dataset: Dataset,
        qdrant: QdrantClient,
        oversampling: float = 2.5,        
    ):
        self.name = 'image_ball_counting'
        super().__init__(
            setting_dataset = setting_dataset, 
            fn_for_nn_sims_calc = self.fn_for_nn_sims_calc,
            qdrant = qdrant,
            oversampling = oversampling,
        )

    def fn_for_nn_sims_calc(self, score:float) -> Callable:
        return -score

    def f_vals(self, r: float = None, selected_embedding_objects: List[List[EmbeddingObject]] = None) -> List[List[np.float64]]:
        f_vals_per_query = []

        nn_sims = self.GetNNSims(selected_embedding_objects)

        for q_idx in range(self.N_q):
            nn_sims_to_q = np.abs(np.array(nn_sims[q_idx]))
            f_vals = (nn_sims_to_q <= r).astype(int)
            f_vals_per_query.append(f_vals.tolist())
            
        return f_vals_per_query    
    

class KDE_Text(SumProblemSetting):
    def __init__(
        self, 
        setting_dataset: Dataset,
        qdrant: QdrantClient,
        oversampling: float = 2.5,       
    ):
        self.name = 'text_kde'
        super().__init__(
            setting_dataset = setting_dataset, 
            fn_for_nn_sims_calc = self.fn_for_nn_sims_calc,
            qdrant = qdrant,
            oversampling = oversampling,
        )

    def fn_for_nn_sims_calc(self, score:float) -> Callable:
        return -score**2

    def f_vals(self, bandwidth: np.float64, selected_embedding_objects: List[List[EmbeddingObject]] = None) -> List[List[np.float64]]:
        f_vals_per_query = []

        nn_sims = self.GetNNSims(selected_embedding_objects)
        max_sims = self.GetMaxSims()

        for q_idx in range(self.N_q):
            nn_sims_to_q_scaled = np.array(nn_sims[q_idx]) - max_sims[q_idx]
            f_vals = np.exp(nn_sims_to_q_scaled / (2 * (bandwidth**2)), dtype=np.float64)
            f_vals_per_query.append(f_vals.tolist())
            
        return f_vals_per_query

class BallCounting_Text(SumProblemSetting):
    def __init__(
        self, 
        setting_dataset: Dataset,
        qdrant: QdrantClient,
        oversampling: float = 2.5,        
    ):
        self.name = 'text_ball_counting'
        super().__init__(
            setting_dataset = setting_dataset, 
            fn_for_nn_sims_calc = self.fn_for_nn_sims_calc,
            qdrant = qdrant,
            oversampling = oversampling,
        )

    def fn_for_nn_sims_calc(self, score:float) -> Callable:
        return -score

    def f_vals(self, r: float = None, selected_embedding_objects: List[List[EmbeddingObject]] = None) -> List[List[np.float64]]:
        f_vals_per_query = []

        nn_sims = self.GetNNSims(selected_embedding_objects)

        for q_idx in range(self.N_q):
            nn_sims_to_q = np.abs(np.array(nn_sims[q_idx]))
            f_vals = (nn_sims_to_q <= r).astype(int)
            f_vals_per_query.append(f_vals.tolist())
            
        return f_vals_per_query