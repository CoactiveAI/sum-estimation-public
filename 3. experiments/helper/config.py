"""Settings for the experiments, read from the environment (see `.env.example`).

Connection settings mirror step 2, so the experiments can run against the same
three targets: a cloud cluster, a local Docker Qdrant, or the serverless embedded
client used by the tests.
"""

import os

from dotenv import load_dotenv

load_dotenv()

#: helper/ -> this step's directory -> the repo root.
_STEP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_STEP_DIR)


class Settings:

    # --- Qdrant connection (same variables step 2 uses) --------------------
    QDRANT_HOST = os.getenv("QDRANT_HOST", "")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "443"))
    QDRANT_TIMEOUT = int(os.getenv("QDRANT_TIMEOUT", "10000"))

    LOCAL_QDRANT_PORT = int(os.getenv("LOCAL_QDRANT_PORT", "6333"))
    QDRANT_EMBEDDED_PATH = os.getenv("QDRANT_EMBEDDED_PATH", ":memory:")

    # Where step 1 wrote its vectors, for local ground truth. Shared with steps 1
    # and 2, so one setting points the whole pipeline at the same run.
    EMBEDDINGS_DIR = os.getenv("EMBEDDINGS_DIR") or os.path.join(_REPO_ROOT, "embeddings")

    # Where step 2 recorded each collection's per-level maxima.
    HNSW_INDEX_DIR = os.getenv("HNSW_INDEX_DIR", os.path.join(_REPO_ROOT, "hnsw_index"))

    # --- Collections ------------------------------------------------------
    # Collection names are the keys themselves (set in step 2). `vector_name` must
    # match what step 2 created; `is_list_of_ids_uuids` says whether a point id can
    # be excluded with a HasIdCondition filter, which is how a query point is kept
    # out of its own results.
    COLLECTIONS = {
        "open-images_resnet-50": {"vector_name": "abs1", "is_list_of_ids_uuids": True},
        "open-images_clip_vit_l14_336": {"vector_name": "unit", "is_list_of_ids_uuids": False},
        "amazon-reviews_distilbert": {"vector_name": "abs", "is_list_of_ids_uuids": True},
    }
    COLLECTION_NAME = {key: key for key in COLLECTIONS}

    # --- Experiment scale -------------------------------------------------
    # Parquet output, at the repo root so steps 3 and 4 agree on one location
    # regardless of the directory a script is run from.
    RESULTS_PATH = os.getenv("RESULTS_PATH") or os.path.join(_REPO_ROOT, "experiments_results")

    # Dataset items sampled per run: the sum being estimated is over these.
    NUM_DATASET_EMBEDDINGS = int(os.getenv("NUM_DATASET_EMBEDDINGS", "100000"))

    # How many queries to run. Each is drawn uniformly from the whole collection.
    NUM_QUERIES = int(os.getenv("NUM_QUERIES", "100"))

    # Seed for query/dataset selection. Set it to repeat an exact run.
    RANDOM_SEED = os.getenv("RANDOM_SEED")

    # Rescoring oversampling factor passed to Qdrant searches.
    OVERSAMPLING = float(os.getenv("OVERSAMPLING", "2.5"))

    # Cache of point ids per collection, used when ids are not plain integers.
    ID_CACHE_DIR = os.getenv("ID_CACHE_DIR", os.path.join(_REPO_ROOT, "hnsw_index"))


settings = Settings()
