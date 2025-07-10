from typing import Optional, Callable, List, Dict
from dataclasses import dataclass, field
import numpy as np

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
