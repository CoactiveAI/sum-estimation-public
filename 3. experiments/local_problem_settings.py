"""Problem settings whose ground truth is computed locally instead of in Qdrant.

Each class here pairs the untouched `Problem_*` from
`qdrant_sum_problem_settings.py` with a mixin that overrides only the three
ground-truth accessors:

    GetAllScores          scores of every dataset item (true sums)
    GetMaxSims            the scaling constant f_vals subtracts
    _get_cached_true_topk the exact-recall baseline

The scoring functions, the weighting in `OurAlgorithm`, and everything the
experiment times are inherited unchanged - the algorithms still issue their own
Qdrant queries.

`GetNNSims` is also overridden, so all-scores can stay a numpy array rather than
becoming one `EmbeddingObjectWithSim` per item: at 10M that difference is 3.6 GB
and 27 s per query, per task. `f_vals` calls `np.array(...)` on whatever it gets
back, so an array passes straight through.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from helper.qdrant_data_classes import EmbeddingObject, EmbeddingObjectWithSim
from qdrant_sum_problem_settings import (Problem_Image_BallCounting,
                                         Problem_Image_KDE,
                                         Problem_Image_Softmax,
                                         Problem_Text_BallCounting,
                                         Problem_Text_KDE)


class LocalGroundTruthMixin:
    """Serves ground truth from `local_scores.LocalGroundTruth`."""

    ground_truth = None

    def attach_ground_truth(self, ground_truth) -> "LocalGroundTruthMixin":
        self.ground_truth = ground_truth
        self._cached_local_topk: Optional[List[List[EmbeddingObjectWithSim]]] = None
        self._cached_dataset_rows: Optional[np.ndarray] = None
        return self

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_ground_truth(self):
        if self.ground_truth is None:
            raise SystemExit(
                "No local ground truth attached; call attach_ground_truth() after "
                "constructing the setting (main.py does this)."
            )
        return self.ground_truth

    def _dataset_rows(self) -> np.ndarray:
        """Dataset ids as row indices into the step-1 matrices."""
        if self._cached_dataset_rows is None:
            self._cached_dataset_rows = np.fromiter(
                (int(i) for i in self.dataset_ids), dtype=np.int64, count=len(self.dataset_ids)
            )
        return self._cached_dataset_rows

    def _transformed_scores(self, q_idx: int) -> np.ndarray:
        """Every row's score for query `q_idx`, put through fn_for_nn_sims_calc."""
        ground_truth = self._require_ground_truth()
        raw = ground_truth.scores_for(self.query_ids[q_idx], self.query_embeddings[q_idx])
        # Every fn_for_nn_sims_calc in the suite is elementwise (-s, -s**2, s), so
        # it vectorises over the whole array.
        return np.asarray(self.fn_for_nn_sims_calc(raw.astype(np.float64)))

    def _rows_to_keep(self, q_idx: int) -> Optional[np.ndarray]:
        """Rows to ignore when taking a maximum or a top-k: the query itself.

        Mirrors the Qdrant path, which excludes the query id with a HasIdCondition
        only for collections flagged `is_list_of_ids_uuids`.
        """
        if not self.is_list_of_ids_uuids:
            return None
        try:
            return np.int64(int(self.query_ids[q_idx]))
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def GetNNSims(self, selected_embedding_objects) -> List:
        """Pass arrays through; unwrap object lists as the base class does."""
        return [
            objs if isinstance(objs, np.ndarray) else [obj.nn_sim_to_q for obj in objs]
            for objs in selected_embedding_objects
        ]

    def GetAllScores(self) -> List[np.ndarray]:
        rows = self._dataset_rows()
        return [self._transformed_scores(q_idx)[rows] for q_idx in range(self.N_q)]

    def GetMaxSims(self) -> List[float]:
        maxima = []
        for q_idx in range(self.N_q):
            sims = self._transformed_scores(q_idx)
            excluded = self._rows_to_keep(q_idx)
            if excluded is not None and 0 <= int(excluded) < sims.shape[0]:
                sims = sims.copy()
                sims[int(excluded)] = -np.inf
            maxima.append(float(sims.max()))
        return maxima

    def _get_cached_true_topk(self, k: int = 5000) -> List[List[EmbeddingObjectWithSim]]:
        """Exact top-k over the collection, as the recall baseline.

        The Qdrant path used an approximate top-k search as its "exact" baseline,
        which measured Qdrant against itself. This is the real thing, and cheaper:
        an argpartition over scores already in memory.
        """
        if getattr(self, "_cached_local_topk", None) is not None:
            return self._cached_local_topk

        per_query = []
        for q_idx in range(self.N_q):
            sims = self._transformed_scores(q_idx)
            excluded = self._rows_to_keep(q_idx)
            if excluded is not None and 0 <= int(excluded) < sims.shape[0]:
                sims = sims.copy()
                sims[int(excluded)] = -np.inf

            take = min(k, sims.shape[0])
            candidates = np.argpartition(-sims, take - 1)[:take]
            ordered = candidates[np.argsort(-sims[candidates], kind="stable")]
            per_query.append([
                EmbeddingObjectWithSim(EmbeddingObject(int(row)), float(sims[row]))
                for row in ordered
            ])
        self._cached_local_topk = per_query
        return per_query


class Local_Problem_Image_KDE(LocalGroundTruthMixin, Problem_Image_KDE):
    pass


class Local_Problem_Image_Softmax(LocalGroundTruthMixin, Problem_Image_Softmax):
    pass


class Local_Problem_Image_BallCounting(LocalGroundTruthMixin, Problem_Image_BallCounting):
    pass


class Local_Problem_Text_KDE(LocalGroundTruthMixin, Problem_Text_KDE):
    pass


class Local_Problem_Text_BallCounting(LocalGroundTruthMixin, Problem_Text_BallCounting):
    pass


#: Qdrant-backed class -> locally-scored equivalent.
LOCAL_EQUIVALENT = {
    Problem_Image_KDE: Local_Problem_Image_KDE,
    Problem_Image_Softmax: Local_Problem_Image_Softmax,
    Problem_Image_BallCounting: Local_Problem_Image_BallCounting,
    Problem_Text_KDE: Local_Problem_Text_KDE,
    Problem_Text_BallCounting: Local_Problem_Text_BallCounting,
}
