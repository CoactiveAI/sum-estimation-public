from typing import Self

import numpy as np

from config import settings
from qdrant_data_classes import EmbeddingObject
from qdrant_helpers import collections_dict, qdrant, scroll_collection


class Dataset:
    """Base class for experiment datasets.

    Loads dataset item IDs and a pool of query vectors directly from Qdrant,
    so no local embedding files are required.  Subclasses set:

        collection_name  – which Qdrant collection to use
        setting_params   – list of hyperparameter values for the scoring function
    """

    def __init__(self):
        self.setting_params = self._get_setting_params()
        self.dataset_embedding_objects = self._get_dataset_embedding_objects()
        self.query_pool = self._get_query_pool()

    # ------------------------------------------------------------------
    # Override points
    # ------------------------------------------------------------------

    def _get_setting_params(self) -> list[float]:
        return self.setting_params

    def _get_dataset_embedding_objects(self) -> list[EmbeddingObject]:
        """Load dataset item IDs (no embeddings needed) from Qdrant."""
        records = scroll_collection(
            self.collection_name,
            n=settings.NUM_DATASET_EMBEDDINGS,
            with_vectors=None,
        )
        return [EmbeddingObject(image_id=r.id) for r in records]

    def _get_query_pool(self) -> list[EmbeddingObject]:
        """Load a pool of candidate query vectors (with embeddings) from Qdrant."""
        vector_name = collections_dict[self.collection_name]['vector_name']
        records = scroll_collection(
            self.collection_name,
            n=settings.NUM_QUERY_CANDIDATES,
            with_vectors=[vector_name],
        )
        pool = []
        for r in records:
            raw = r.vector
            # scroll returns either a plain list or a dict of named vectors
            vec = raw[vector_name] if isinstance(raw, dict) else raw
            pool.append(EmbeddingObject(image_id=r.id, embedding=np.array(vec, dtype=float)))
        return pool

    # ------------------------------------------------------------------
    # Shallow copy used to build per-query dataset objects in main.py
    # ------------------------------------------------------------------

    def copy(self) -> Self:
        new_obj = self.__class__.__new__(self.__class__)
        for attr in ["collection_name", "setting_params"]:
            if hasattr(self, attr):
                setattr(new_obj, attr, getattr(self, attr))
        new_obj.query_pool = None
        new_obj.dataset_embedding_objects = None
        return new_obj


# ---------------------------------------------------------------------------
# Concrete dataset classes
# ---------------------------------------------------------------------------

class Dataset_Image_KDE(Dataset):
    def __init__(self):
        self.collection_name = settings.COLLECTION_NAME["open-images_resnet-50"]
        self.setting_params = [10 ** p for p in np.arange(-0.25, 1.75, 0.05)]
        super().__init__()


class Dataset_Image_Softmax(Dataset):
    def __init__(self):
        self.collection_name = settings.COLLECTION_NAME["open-images_clip_vit_l14_336"]
        self.setting_params = [10 ** p for p in np.arange(-3.0, 1.0, 0.1)]
        super().__init__()


class Dataset_Image_BallCounting(Dataset):
    def __init__(self):
        self.collection_name = settings.COLLECTION_NAME["open-images_resnet-50"]
        self.setting_params = sorted(set(
            [10 ** p for p in np.arange(-3.0, 2.0, 0.1)] +
            [10 ** p for p in np.arange(0.5, 1.8, 0.05)]
        ))
        super().__init__()


class Dataset_Text_KDE(Dataset):
    def __init__(self):
        self.collection_name = settings.COLLECTION_NAME["amazon-reviews_distilbert"]
        self.setting_params = [10 ** p for p in np.arange(-0.70, 1.50, 0.05)]
        super().__init__()


class Dataset_Text_BallCounting(Dataset):
    def __init__(self):
        self.collection_name = settings.COLLECTION_NAME["amazon-reviews_distilbert"]
        self.setting_params = sorted(set(
            [10 ** p for p in np.arange(-5.0, 1.0, 0.1)] +
            [10 ** p for p in np.arange(0, 1.8, 0.05)]
        ))
        super().__init__()
