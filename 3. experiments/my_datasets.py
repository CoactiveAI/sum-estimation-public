"""Experiment datasets: one class per (task, collection) pairing.

Same structure as before - each subclass fixes a collection and the list of
scoring-function hyperparameters - with the selection changed: queries and
dataset items are now drawn uniformly from the whole collection through
`query_sampler.CollectionSampler`, instead of taking whatever prefix a scroll
returned.
"""

from typing import List, Optional

import numpy as np

from helper.config import settings
from helper.qdrant_data_classes import EmbeddingObject
from helper.qdrant_helpers import collections_dict
from helper.query_sampler import CollectionSampler


class Dataset:
    """Base class for experiment datasets.

    Subclasses set:
        collection_name  - which Qdrant collection to use
        setting_params   - hyperparameter values for the scoring function

    `query_pool` holds the queries to run, each a uniform draw from the whole
    collection. `dataset_embedding_objects` is the sample whose sum is estimated;
    ids only, since scores come from Qdrant.
    """

    def __init__(
        self,
        client,
        rng: Optional[np.random.Generator] = None,
        num_dataset: Optional[int] = None,
        num_queries: Optional[int] = None,
    ):
        self.setting_params = self._get_setting_params()
        self.vector_name = collections_dict[self.collection_name]["vector_name"]
        self.sampler = CollectionSampler(
            client=client,
            collection_name=self.collection_name,
            vector_name=self.vector_name,
            rng=rng,
        )
        num_dataset = num_dataset or settings.NUM_DATASET_EMBEDDINGS
        num_queries = num_queries or settings.NUM_QUERIES

        self.dataset_embedding_objects = self._get_dataset_embedding_objects(num_dataset)
        # Ids and their positions, built once: every query reuses this list minus
        # its own point, so rebuilding it per (query, task) is pure overhead.
        self.dataset_ids = [obj.image_id for obj in self.dataset_embedding_objects]
        self._id_positions = {image_id: index for index, image_id in enumerate(self.dataset_ids)}
        self.query_pool = self._get_query_pool(num_queries)

    # ------------------------------------------------------------------
    # Override points
    # ------------------------------------------------------------------

    def _get_setting_params(self) -> List[float]:
        return self.setting_params

    def _get_dataset_embedding_objects(self, n: int) -> List[EmbeddingObject]:
        """Sample the dataset items whose sum is being estimated (ids only)."""
        return self.sampler.sample_dataset(n)

    def _get_query_pool(self, n: int) -> List[EmbeddingObject]:
        """Sample the query points, with their vectors."""
        return self.sampler.sample_queries(n)

    # ------------------------------------------------------------------
    # Shallow copy used to build per-query dataset objects in main.py
    # ------------------------------------------------------------------

    def without(self, image_id):
        """The dataset with one point removed: `(objects, ids)`, order preserved.

        The query point must not appear in the dataset it is summed over. Slicing
        around its position keeps exactly the membership and order the previous
        per-query list comprehension produced, without walking 10M objects in
        Python to find it.
        """
        position = self._id_positions.get(image_id)
        if position is None:
            return self.dataset_embedding_objects, self.dataset_ids
        objects = (self.dataset_embedding_objects[:position]
                   + self.dataset_embedding_objects[position + 1:])
        ids = self.dataset_ids[:position] + self.dataset_ids[position + 1:]
        return objects, ids

    def copy(self) -> "Dataset":
        new_obj = self.__class__.__new__(self.__class__)
        for attr in ["collection_name", "setting_params", "vector_name", "sampler"]:
            if hasattr(self, attr):
                setattr(new_obj, attr, getattr(self, attr))
        new_obj.query_pool = None
        new_obj.dataset_embedding_objects = None
        return new_obj


# ---------------------------------------------------------------------------
# Concrete dataset classes
# ---------------------------------------------------------------------------

class Dataset_Image_KDE(Dataset):
    def __init__(self, *args, **kwargs):
        self.collection_name = settings.COLLECTION_NAME["open-images_resnet-50"]
        self.setting_params = [10 ** p for p in np.arange(-0.25, 1.75, 0.05)]
        super().__init__(*args, **kwargs)


class Dataset_Image_Softmax(Dataset):
    def __init__(self, *args, **kwargs):
        self.collection_name = settings.COLLECTION_NAME["open-images_clip_vit_l14_336"]
        self.setting_params = [10 ** p for p in np.arange(-3.0, 1.0, 0.1)]
        super().__init__(*args, **kwargs)


class Dataset_Image_BallCounting(Dataset):
    def __init__(self, *args, **kwargs):
        self.collection_name = settings.COLLECTION_NAME["open-images_resnet-50"]
        self.setting_params = sorted(set(
            [10 ** p for p in np.arange(-3.0, 2.0, 0.1)] +
            [10 ** p for p in np.arange(0.5, 1.8, 0.05)]
        ))
        super().__init__(*args, **kwargs)


class Dataset_Text_KDE(Dataset):
    def __init__(self, *args, **kwargs):
        self.collection_name = settings.COLLECTION_NAME["amazon-reviews_distilbert"]
        self.setting_params = [10 ** p for p in np.arange(-0.70, 1.50, 0.05)]
        super().__init__(*args, **kwargs)


class Dataset_Text_BallCounting(Dataset):
    def __init__(self, *args, **kwargs):
        self.collection_name = settings.COLLECTION_NAME["amazon-reviews_distilbert"]
        self.setting_params = sorted(set(
            [10 ** p for p in np.arange(-5.0, 1.0, 0.1)] +
            [10 ** p for p in np.arange(0, 1.8, 0.05)]
        ))
        super().__init__(*args, **kwargs)
