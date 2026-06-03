import os

from dotenv import load_dotenv

load_dotenv()


class Settings:

    QDRANT_HOST = os.environ["QDRANT_HOST"]
    QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "443"))
    QDRANT_TIMEOUT = int(os.getenv("QDRANT_TIMEOUT", "10000"))

    COLLECTION_NAME = {
        "open-images_resnet-50": "SEPub_OI_train8M_resnet_50",
        "open-images_clip_vit_l14_336": "SEPub_OI_train8M_clip_vit_normalised",
        "amazon-reviews_distilbert": "SEPub_AR_10M_bert"
    }

    RESULTS_PATH = os.getenv("RESULTS_PATH", "./results")

    # How many dataset IDs to load from Qdrant for experiments.
    # Higher values make true-sum estimates more accurate but slow down computation.
    NUM_DATASET_ITEMS = int(os.getenv("NUM_DATASET_ITEMS", "100000"))

    # Size of the pool from which random query vectors are drawn each run.
    NUM_QUERY_CANDIDATES = int(os.getenv("NUM_QUERY_CANDIDATES", "1000"))


settings = Settings()
