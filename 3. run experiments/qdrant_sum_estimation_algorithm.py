
from dataclasses import dataclass
from time import time

import numpy as np

from qdrant_data_classes import EmbeddingObject
from qdrant_sum_problem_settings import SumProblemSetting

MAX_RARITY_IN_VECTOR_DB = 40

class SumEstimationAlgorithm:

    def __init__(self, sum_problem_setting: SumProblemSetting):
        self.sum_problem_setting = sum_problem_setting
        self.N_d = self.sum_problem_setting.GetN_d()
        self.N_q = self.sum_problem_setting.GetN_q()

    def GetTimeEstimate(self) -> float:
        return self.end - self.start

    def GetEstimateForSettingParam(self):
        raise NotImplementedError("Subclasses should implement this method")

    def GetTrueEstimate(self, setting_param: np.float64) -> np.ndarray:
        # TODO: Replace with AllScores() implementation
        df = self.sum_problem_setting.setting_dataset.true_estimates
        
        all_sums = []
        for query_id in self.sum_problem_setting.query_ids:
            row = df[df['query_id'] == query_id]
            all_sums.append(row[str(setting_param)].iloc[0])
        return all_sums
    
    def GetExactRecall(self) -> list[tuple]:
        # TODO: Replace with AllScores() implementation
        df = self.sum_problem_setting.setting_dataset.exact_recall

        all_recalls = []
        for query_id in self.sum_problem_setting.query_ids:
            row = df[df['query_id'] == query_id]
            if self.__class__.__name__ == "OurAlgorithm":
                all_recalls.append(
                    [(l, row['Top200'].iloc[0][:self.k]) for l in range(1,self.sum_problem_setting.max_level+1)]
                )
            elif hasattr(self, 'k'): 
                all_recalls.append(
                    (-1, row['Top5000'].iloc[0][:self.k])
                )
        return all_recalls      

    def GetQdrantRecall(self) -> np.ndarray:
        if self.__class__.__name__ == "OurAlgorithm":
            qdrant_recalls = [
                [
                    (
                        l + 1,
                        [obj.embedding_object.image_id for obj in objs_in_level]
                    )
                    for l, objs_in_level in enumerate(top_k_objects)
                ]
                for top_k_objects in self.top_k_embedding_objects
            ]
        elif hasattr(self, 'k'):
            qdrant_recalls = [
                (
                    -1,
                    [obj.embedding_object.image_id for obj in top_k_objects]
                )
                for l, top_k_objects in enumerate(self.top_k_embedding_objects)
            ]
        return qdrant_recalls


class RandomSample(SumEstimationAlgorithm):

    def __init__(self, sum_problem_setting: SumProblemSetting, params: dict[str, int]):
        super().__init__(sum_problem_setting)
        self.name = 'random'
        self.m = params['r']
        self.start = time()
        self.random_sample_embedding_objects = self.sum_problem_setting.GetRandomSample(self.m)
        self.end = time()

    def GetEstimateForSettingParam(self, setting_param: np.float64 = None) -> np.ndarray:
        random_sample_f_vals = self.sum_problem_setting.f_vals(setting_param, self.random_sample_embedding_objects)
        random_sample_sums = (self.N_d/self.m) * np.array([np.sum(row) for row in random_sample_f_vals])
        return random_sample_sums
    
    def GetExactRecall(self) -> list[list[tuple]]:
        pass

    def GetQdrantRecall(self) -> list[list[tuple]]:
        pass        
    

class TopK(SumEstimationAlgorithm):

    def __init__(self, sum_problem_setting: SumProblemSetting, params: dict[str, int]):
        super().__init__(sum_problem_setting)
        self.name = 'topk'
        self.k = params['k']
        self.start = time()
        self.top_k_embedding_objects = self.sum_problem_setting.GetTopK(self.k)
        self.end = time()

    def GetEstimateForSettingParam(self, setting_param: np.float64 = None) -> np.ndarray:
        topk_f_vals = self.sum_problem_setting.f_vals(setting_param, self.top_k_embedding_objects)
        topk_sums = np.sum(topk_f_vals,axis=1)
        return topk_sums    


class Combined(SumEstimationAlgorithm):

    def __init__(self, sum_problem_setting: SumProblemSetting, params: dict[str, int]):
        super().__init__(sum_problem_setting)
        self.name = 'combined'
        self.k = params['k']
        self.m = params['r']

        self.start = time()
        self.top_k_embedding_objects = self.sum_problem_setting.GetTopK(self.k)
        random_sample_embedding_objects = self.sum_problem_setting.GetRandomSample(self.m)
        self.end = time()
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

    def __init__(self, sum_problem_setting: SumProblemSetting, params: dict[str, int]):
        super().__init__(sum_problem_setting)
        self.name = 'our'
        self.k = params['k']

        self.start = time()
        embedding_objects_all_levels = self.sum_problem_setting.GetTopKWithLevel(self.k)
        self.top_k_embedding_objects = embedding_objects_all_levels
        self.end = time()

        # self.qidx_to_weighted = {}
        # self.qidx_to_backhalf = {}
        self.qidx_to_weighted = []
        self.qidx_to_backhalf = []
        for q_idx in range(self.N_q):
            U = []
            nonfull = []
            for level in range(self.sum_problem_setting.max_level):
                for embedding_object in embedding_objects_all_levels[q_idx][level]:
                    U.append( (embedding_object,level+1) ) # levels should run from 1 to max_level, not 0 to max_level-1

                if len(embedding_objects_all_levels[q_idx][level]) < self.k:
                    nonfull += embedding_objects_all_levels[q_idx][level]            

            # self.qidx_to_backhalf[q_idx] = sorted(nonfull, key=lambda x: (x.nn_sim_to_q, x.embedding_object.image_id), reverse=True)[len(nonfull)//2:]
            self.qidx_to_backhalf.append(
                sorted(nonfull, key=lambda x: (x.nn_sim_to_q, x.embedding_object.image_id), reverse=True)[len(nonfull)//2:]
            )

            weighted = []
            level_to_count = {}
            p=1
            weight = 1/p
            for emb_obj, level in sorted(U, key=lambda x: (x[0].nn_sim_to_q, x[0].embedding_object.image_id), reverse=True): # x[0] is the embedding_object
                weighted.append( (emb_obj, weight))
                
                if level not in level_to_count:
                    level_to_count[level] = 0
                level_to_count[level] += 1

                if level_to_count[level] == self.k:
                    p -= 2**(-level)
                    weight = 1/p

            # self.qidx_to_weighted[q_idx] = weighted
            self.qidx_to_weighted.append(weighted)


    def GetEstimateForSettingParam(self, setting_param: np.float64 = None): # -> Tuple[np.ndarray, List[int]]:
        results = []

        all_c = self.sum_problem_setting.f_vals(setting_param, self.qidx_to_backhalf)
        all_f_vals = self.sum_problem_setting.f_vals(
            setting_param, 
            [[x[0] for x in self.qidx_to_weighted[qidx]] for qidx in range(self.N_q)]
        )

        for qidx in range(self.N_q):
            c = np.mean(all_c[qidx])
            f_vals = all_f_vals[qidx]
            weights = [x[1] for x in self.qidx_to_weighted[qidx]]

            uncorrected_estimate = sum([f_val*weight for f_val, weight in zip(f_vals,weights)])
            correction = c * (self.N_d - np.sum(weights))
            results.append(uncorrected_estimate + correction)

        return results    