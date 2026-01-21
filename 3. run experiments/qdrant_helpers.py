from time import sleep
from typing import Callable, List

import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import (NamedVector, QuantizationSearchParams,
                                       SearchParams, SearchRequest)

from config import settings
from data_classes import EmbeddingObject, EmbeddingObjectWithSim

qdrant_client_params = {
    "url": settings.QDRANT_HOST,
    "api_key": settings.QDRANT_API_KEY,    
    "port": settings.QDRANT_PORT,
    "timeout": settings.QDRANT_TIMEOUT
}
qdrant = QdrantClient(**qdrant_client_params)


max_levels_dict = {
    settings.COLLECTION_NAME["open-images_resnet-50"]: {
        f'level_{i}': v for i, v in enumerate([24, 24, 25, 22, 24, 22, 24, 24, 28, 25])
    },
    settings.COLLECTION_NAME["open-images_clip_vit_l14_336"]: {
        f'level_{i}': v for i, v in enumerate([21, 26, 25, 24, 24, 25, 23, 25, 24, 22])
    },
    settings.COLLECTION_NAME["amazon-reviews_distilbert"]: {
        f'level_{i}': v for i, v in enumerate([28, 23, 23, 23, 23, 27, 25, 24, 26, 22])
    }
}

collections_dict = {
    settings.COLLECTION_NAME["open-images_resnet-50"]:{'vector_name': 'abs1', 'is_list_of_ids_uuids': True},
    settings.COLLECTION_NAME["open-images_clip_vit_l14_336"]:{'vector_name': 'unit', 'is_list_of_ids_uuids': False},
    settings.COLLECTION_NAME["amazon-reviews_distilbert"]:{'vector_name': 'abs', 'is_list_of_ids_uuids': True},    
}


def make_search_request(
    vector: List[float],
    vector_name: str,
    k: int,
    offset: int = 0,
    oversampling: float = 2.5,
    must=None,
    must_not=None
) -> SearchRequest:
    return SearchRequest(
        vector=NamedVector(name=vector_name, vector=list(vector)),
        filter=models.Filter(
            must=must or [],
            must_not=must_not or []
        ),
        params=SearchParams(
            quantization=QuantizationSearchParams(rescore=True, oversampling=oversampling)
        ),
        limit=k,
        offset=offset,
        with_vector=False,
        with_payload=False
    )


def batch_qdrant_search(
    collection_name: str,
    queries: List[SearchRequest],
    batch_chunk: int = 40,
    retries: int = 5,
    retry_sleep_sec: int = 2,
    debug_tag: str = ""
):
    results = []
    # for chunk in tqdm(range(0, len(queries), batch_chunk)):
    for chunk in range(0, len(queries), batch_chunk):
        for attempt in range(retries):
            try:
                batch = qdrant.search_batch(
                    collection_name=collection_name,
                    requests=queries[chunk:chunk+batch_chunk],
                )
                results.extend(batch)
                break
            except Exception as e:
                if attempt == retries - 1:
                    print(f"[ERROR {debug_tag}] {e}")
                sleep(retry_sleep_sec)
    return results


def get_top_k_for_query(
    qid: str,
    qe: List[float],
    k: int,
    collection_name: str,
    vector_name: str,
    is_uuid: bool,
    oversampling: float,
    fn_for_nn_sims_calc: Callable
) -> List[EmbeddingObjectWithSim]:
    query = make_search_request(
        vector=qe,
        vector_name=vector_name,
        k=k,
        oversampling=oversampling,
        must_not=[models.HasIdCondition(has_id=[qid])] if is_uuid else []
    )
    results = batch_qdrant_search(collection_name, [query], debug_tag=f"TopK(qid={qid})")
    return [EmbeddingObjectWithSim(EmbeddingObject(r.id), fn_for_nn_sims_calc(r.score)) for r in results[0]]


def get_top_k_with_level(
    qid: str,
    qe: List[float],
    k: int,
    current_level: int,
    max_level: int,
    collection_name: str,
    vector_name: str,
    is_uuid: bool,
    oversampling: float,
    fn_for_nn_sims_calc: Callable,
) -> List[List[EmbeddingObjectWithSim]]:
    queries = [
        make_search_request(
            vector=qe,
            vector_name=vector_name,
            k=k,
            oversampling=oversampling,
            must=[models.FieldCondition(key=f'level_{current_level}', match=models.MatchValue(value=l))],
            must_not=[models.HasIdCondition(has_id=[qid])] if is_uuid else []
        )
        for l in range(1, max_level + 1)
    ]
    results = batch_qdrant_search(collection_name, queries, debug_tag=f"TopKWithLevel(qid={qid})")
    return [[EmbeddingObjectWithSim(EmbeddingObject(r.id), fn_for_nn_sims_calc(r.score)) for r in batch] for batch in results]


def get_random_sample_for_query(
    qe: List[float],
    m: int,
    dataset_embedding_objects: List[EmbeddingObject],
    collection_name: str,
    vector_name: str,
    oversampling: float,
    fn_for_nn_sims_calc: Callable
) -> List[EmbeddingObjectWithSim]:
    sample_ids = [dataset_embedding_objects[i].image_id for i in np.random.choice(len(dataset_embedding_objects), m, replace=False)]
    queries = [
        make_search_request(
            vector=qe,
            vector_name=vector_name,
            k=500,
            oversampling=oversampling,
            must=[models.HasIdCondition(has_id=sample_ids[start:start+500])]
        )
        for start in range(0, m, 500)
    ]
    results = batch_qdrant_search(collection_name, queries, debug_tag="RandomSample")
    return [EmbeddingObjectWithSim(EmbeddingObject(r.id), fn_for_nn_sims_calc(r.score)) for batch in results for r in batch]


def get_all_scores_for_query(
    qid: str,
    qe: List[float],
    ids: List[str],
    collection_name: str,
    vector_name: str,
    oversampling: float,
    fn_for_nn_sims_calc: Callable
) -> List[float]:
    queries = [
        make_search_request(
            vector=qe,
            vector_name=vector_name,
            k=500,
            oversampling=oversampling,
            must=[models.HasIdCondition(has_id=ids[start:start+500])]
        )
        for start in range(0, len(ids), 500)
    ]
    results = batch_qdrant_search(collection_name, queries, debug_tag=f"AllScores(qid={qid})")
    return [EmbeddingObjectWithSim(EmbeddingObject(r.id), fn_for_nn_sims_calc(r.score)) for batch in results for r in batch]
