import os

from dotenv import load_dotenv

load_dotenv()


class Settings:

    # Cloud Qdrant URL. If set, scripts connect to it directly.
    # If left empty, the scripts spin up a local Qdrant (Docker) instead.
    QDRANT_HOST = os.getenv("QDRANT_HOST", "")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "443"))
    QDRANT_TIMEOUT = int(os.getenv("QDRANT_TIMEOUT", "10000"))

    # Local Qdrant (used only when QDRANT_HOST is empty).
    LOCAL_QDRANT_PORT = int(os.getenv("LOCAL_QDRANT_PORT", "6333"))
    LOCAL_QDRANT_IMAGE = os.getenv("LOCAL_QDRANT_IMAGE", "qdrant/qdrant:latest")
    LOCAL_QDRANT_CONTAINER = os.getenv("LOCAL_QDRANT_CONTAINER", "sum-estimation-qdrant")
    LOCAL_QDRANT_VOLUME = os.getenv("LOCAL_QDRANT_VOLUME", "./qdrant_storage")

    # Qdrant collection names. The name is the key itself, so a collection is
    # identifiable from its (dataset, encoder) pair alone; the mapping stays so a
    # deployment can point a key at a differently named collection if it must.
    COLLECTION_KEYS = (
        "open-images_resnet-50",
        "open-images_clip_vit_l14_336",
        "amazon-reviews_distilbert",
    )
    COLLECTION_NAME = {key: key for key in COLLECTION_KEYS}

    # Number of independent rarity-level assignments stored on every point
    # (payload fields level_0 .. level_{NUM_LEVELS-1}). Each is an independent
    # draw, so experiments can be re-run against a different level index without
    # re-inserting — i.e. "multiple runs with re-assigned levels".
    NUM_LEVELS = int(os.getenv("NUM_LEVELS", "10"))

    # Geometric rarity draws are capped at this value (matches the original pipeline).
    MAX_RARITY_IN_VECTOR_DB = int(os.getenv("MAX_RARITY_IN_VECTOR_DB", "40"))

    # Serverless embedded mode (`--qdrant embedded`): a directory to persist to,
    # or ":memory:" for a throwaway index.
    QDRANT_EMBEDDED_PATH = os.getenv("QDRANT_EMBEDDED_PATH", ":memory:")

    # Where step 1 (create_embeddings) wrote its .npy files.
    EMBEDDINGS_DIR = os.getenv("EMBEDDINGS_DIR") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embeddings"
    )

    # HNSW graph parameters used when creating a collection.
    HNSW_M = int(os.getenv("HNSW_M", "32"))
    HNSW_EF_CONSTRUCT = int(os.getenv("HNSW_EF_CONSTRUCT", "32"))
    HNSW_FULL_SCAN_THRESHOLD = int(os.getenv("HNSW_FULL_SCAN_THRESHOLD", "10000"))

    # Points a segment must hold before Qdrant starts building its HNSW index.
    # Bulk loads set this to 0 first and restore it afterwards, so the graph is
    # built once at the end instead of being rebuilt as segments grow.
    INDEXING_THRESHOLD = int(os.getenv("INDEXING_THRESHOLD", "10000"))

    # Where index metadata is recorded, one directory per collection:
    #     <HNSW_INDEX_DIR>/<collection>/max_levels.json
    # Absolute by default, so the location does not depend on the directory a
    # script is run from.
    HNSW_INDEX_DIR = os.getenv(
        "HNSW_INDEX_DIR",
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hnsw_index"
        ),
    )


settings = Settings()
