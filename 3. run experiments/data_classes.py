from dataclasses import dataclass

import numpy as np

from my_datasets import Dataset
from qdrant_sum_problem_settings import SumProblemSetting
from qdrant_sum_estimation_algorithm import SumEstimationAlgorithm


@dataclass
class EmbeddingObject:
    image_id: str
    embedding: np.ndarray = None
    bias: np.float64 = None

@dataclass
class EmbeddingObjectWithSim:
    embedding_object: EmbeddingObject
    nn_sim_to_q: np.float64

    # for set comparison
    def __eq__(self, other):
        return self.embedding_object.image_id == other.embedding_object.image_id

    # for set comparison
    def __hash__(self):
        return hash(self.embedding_object.image_id)  


@dataclass
class Combination:
    sum_problem_setting: SumProblemSetting
    sum_estimation_algorithm: SumEstimationAlgorithm

    setting_dataset: Dataset = None  # Initially optional
    params: dict = None

    def __post_init__(self):
        self.setting_dataset = setting_dataset_mapping[self.sum_problem_setting.__name__]
        self.param_suffix = "_".join(str(v) for v in self.params.values())
         
