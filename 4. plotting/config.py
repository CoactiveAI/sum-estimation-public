import os

from dotenv import load_dotenv

load_dotenv()


class Settings:

    # Path to results produced by 3. run experiments/main.py
    # Use a local path (./results) or an S3 path (s3://bucket/prefix)
    EXPERIMENT_RESULTS_PATH = os.getenv("RESULTS_PATH", "./results")

    TASK_DATA_COMBINATIONS = [
        {'task': 'kde',           'data': 'image'},
        {'task': 'softmax',       'data': 'image'},
        {'task': 'ball_counting', 'data': 'image'},
        {'task': 'kde',           'data': 'text'},
        {'task': 'ball_counting', 'data': 'text'},
    ]


settings = Settings()
