"""Settings for the plots, read from the environment (see `.env.example`)."""

import os

from dotenv import load_dotenv

load_dotenv()

#: This step's directory (helper/ lives inside it), and the repo root above it.
_STEP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_STEP_DIR)


class Settings:

    # Where step 3 wrote its per-query Parquet shards, and where combine_shards.py
    # writes the combined `<data>_<task>_<kind>.parquet` files the plots read.
    # Same default as step 3: the repo-root experiments_results/.
    EXPERIMENT_RESULTS_PATH = os.getenv("RESULTS_PATH") or os.path.join(
        _REPO_ROOT, "experiments_results"
    )

    # Where figures are written: the repo-root plots/, beside experiments_results/.
    PLOTS_DIR = os.getenv("PLOTS_DIR") or os.path.join(_REPO_ROOT, "plots")

    TASK_DATA_COMBINATIONS = [
        {'task': 'kde',           'data': 'image'},
        {'task': 'softmax',       'data': 'image'},
        {'task': 'ball_counting', 'data': 'image'},
        {'task': 'kde',           'data': 'text'},
        {'task': 'ball_counting', 'data': 'text'},
    ]

    RESULT_KINDS = (
        'sum_estimates',
        'time_estimates',
        'true_sum',
        'recall_exact',
        'recall_qdrant',
    )

    # Levels shown on the recall plot, and the k values it draws a series for.
    RECALL_MAX_LEVEL = int(os.getenv("RECALL_MAX_LEVEL", "40"))

    # Older result sets had OurAlgorithm's Qdrant levels off by one against the
    # exact baseline for the image KDE/softmax tasks, and the original script
    # shifted them to compensate. Current step-3 output is consistent, so this is
    # off by default; set it to re-plot those legacy runs.
    LEGACY_IMAGE_LEVEL_SHIFT = os.getenv("LEGACY_IMAGE_LEVEL_SHIFT", "").lower() in ("1", "true", "yes")


settings = Settings()
