
import numpy as np
from dataclasses import dataclass
from qdrant_data_classes import EmbeddingObject
from qdrant_sum_problem_settings import SumProblemSetting

MAX_RARITY_IN_VECTOR_DB = 40

class SumEstimationAlgorithm:

    def __init__(self, sum_problem_setting: SumProblemSetting):
        self.sum_problem_setting = sum_problem_setting
        self.N_d = self.sum_problem_setting.GetN_d()
        self.N_q = self.sum_problem_setting.GetN_q()        

    def GetEstimateForSettingParam(self):
        raise NotImplementedError("Subclasses should implement this method")


class RandomSample(SumEstimationAlgorithm):

    def __init__(self, sum_problem_setting: SumProblemSetting, m: int):
        super().__init__(sum_problem_setting)
        self.m = m
        self.random_sample_embedding_objects = self.sum_problem_setting.GetRandomSample(self.m)

    def GetEstimateForSettingParam(self, setting_param: np.float64 = None) -> np.ndarray:
        random_sample_f_vals = self.sum_problem_setting.f_vals(setting_param, self.random_sample_embedding_objects)
        random_sample_sums = (self.N_d/self.m) * np.array([np.sum(row) for row in random_sample_f_vals])
        return random_sample_sums
    

class TopK(SumEstimationAlgorithm):

    def __init__(self, sum_problem_setting: SumProblemSetting, k: int):
        super().__init__(sum_problem_setting)
        self.k = k
        self.top_k_embedding_objects = self.sum_problem_setting.GetTopK(self.k)

    def GetEstimateForSettingParam(self, setting_param: np.float64 = None) -> np.ndarray:
        topk_f_vals = self.sum_problem_setting.f_vals(setting_param, self.top_k_embedding_objects)
        topk_sums = np.sum(topk_f_vals,axis=1)
        return topk_sums


class Combined(SumEstimationAlgorithm):

    def __init__(self, sum_problem_setting: SumProblemSetting, k: int, m: int):
        super().__init__(sum_problem_setting)
        self.k = k
        self.m = m

        self.top_k_embedding_objects = self.sum_problem_setting.GetTopK(self.k)
        random_sample_embedding_objects = self.sum_problem_setting.GetRandomSample(self.m)                
        self.T_embedding_objects = [
            set(random_sample_embedding_objects[q_idx]) - set(self.top_k_embedding_objects[q_idx])
            for q_idx in range(self.N_q)
        ]

    def GetEstimateForSettingParam(self, setting_param: np.float64 = None) -> np.ndarray:
        topk_f_vals = self.sum_problem_setting.f_vals(setting_param, self.top_k_embedding_objects)
        topk_sums = np.sum(topk_f_vals,axis=1)

        T_f_vals = self.sum_problem_setting.f_vals(setting_param, self.T_embedding_objects)
        T_sums = [((self.N_d-self.k)/len(T_f_vals_q)) * np.sum(T_f_vals_q) for T_f_vals_q in T_f_vals]

        return topk_sums + T_sums


@dataclass
class EmbeddingObjectWithCabooseLevel:
    embedding_object: EmbeddingObject
    caboose_level: int
    random_tiebreaker: np.float64


class OurAlgorithm(SumEstimationAlgorithm):

    def __init__(self, sum_problem_setting: SumProblemSetting, k: int):
        super().__init__(sum_problem_setting)
        self.k = k

        embedding_objects_all_levels = self.sum_problem_setting.GetTopKWithLevel(self.k)

        self.qidx_to_weighted = {}
        self.qidx_to_backhalf = {}
        for q_idx in range(self.N_q):
            U = []
            nonfull = []
            for level in range(self.sum_problem_setting.max_level):
                for embedding_object in embedding_objects_all_levels[q_idx][level]:
                    U.append( (embedding_object,level) )

                if len(embedding_objects_all_levels[q_idx][level]) < self.k:
                    nonfull += embedding_objects_all_levels[q_idx][level]

            self.qidx_to_backhalf[q_idx] = sorted(nonfull, key=lambda x: (x.nn_sim_to_q, x.id), reverse=True)[len(nonfull)//2:]

            weighted = []
            level_to_count = {}
            p=1
            weight = 1/p
            for emb_obj, level in sorted(U, key=lambda x: (x[0].nn_sim_to_q, x[0].id), reverse=True):
                weighted.append( (emb_obj, weight))
                
                if level not in level_to_count:
                    level_to_count[level] = 0
                level_to_count[level] += 1

                if level_to_count[level] == k:
                    p -= 2**(-level)
                    weight = 1/p

            self.qidx_to_weighted[q_idx] = weighted


    def GetEstimateForSettingParam(self, setting_param: np.float64 = None): # -> Tuple[np.ndarray, List[int]]:
        results = []
        
        for qidx in range(self.N_q):
            c = np.mean(self.sum_problem_setting.f_vals(setting_param, self.qidx_to_backhalf[qidx]))

            f_vals = self.sum_problem_setting.f_vals(setting_param, [x[0] for x in self.qidx_to_weighted[qidx]])
            weights = [x[1] for x in self.qidx_to_weighted[qidx]]

            uncorrected_estimate = sum([f_val*weight for f_val, weight in zip(f_vals,weights)])
            correction = c * (self.N_d - np.sum(weights))

            results.append(uncorrected_estimate + correction)

        return results