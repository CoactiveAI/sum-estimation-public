"""Qdrant query helpers used by the problem settings and the algorithms.

Same functions and signatures as before, with two changes:

* the client is resolved at runtime by `init_client()` instead of being built at
  import, so the experiments can run against cloud, a local server, or embedded;
* `max_levels_dict` is loaded from what step 2 recorded rather than hard-coded, so
  re-indexing a collection cannot leave stale level bounds behind.
"""

from time import sleep
from typing import Callable, List, Optional

import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import (NamedVector, QuantizationSearchParams,
                                       SearchParams, SearchRequest)

from .config import settings
from .max_levels import load_max_levels
from .qdrant_data_classes import EmbeddingObject, EmbeddingObjectWithSim

# Collection -> {level_i: max value present}, read from step 2's output.
max_levels_dict = load_max_levels()

# Collection -> vector name and whether ids can be filtered with HasIdCondition.
collections_dict = {
    settings.COLLECTION_NAME[key]: dict(config)
    for key, config in settings.COLLECTIONS.items()
}

# Set by init_client(); every helper reads it through _client().
qdrant: Optional[QdrantClient] = None


def init_client(client: QdrantClient) -> QdrantClient:
    """Install the client the helpers should use."""
    global qdrant
    qdrant = client
    return qdrant


def _client() -> QdrantClient:
    if qdrant is None:
        raise SystemExit(
            "No Qdrant client configured. Call qdrant_helpers.init_client(...) "
            "before running an experiment (main.py does this at startup)."
        )
    return qdrant


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
    for chunk in range(0, len(queries), batch_chunk):
        for attempt in range(retries):
            try:
                batch = _client().search_batch(
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


def scroll_collection(
    collection_name: str,
    n: int,
    with_vectors: Optional[List[str]] = None,
) -> List:
    """Scroll a collection to retrieve up to n records, in storage order.

    Kept for inspection and debugging. Experiments sample with
    `query_sampler.CollectionSampler` instead, because a scroll prefix is not a
    uniform sample of the collection.
    """
    items = []
    offset = None
    while len(items) < n:
        batch_size = min(1000, n - len(items))
        results, offset = _client().scroll(
            collection_name=collection_name,
            limit=batch_size,
            with_vectors=with_vectors if with_vectors else False,
            offset=offset,
        )
        if not results:
            break
        items.extend(results)
        if offset is None:
            break
    return items


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
) -> List[EmbeddingObjectWithSim]:
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
