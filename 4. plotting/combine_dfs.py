"""Combine per-query Parquet shards into one file per result type.

Works with local paths or S3 paths (configured via RESULTS_PATH in .env).
"""

import os

import pandas as pd
from tqdm import tqdm

from config import settings

EXPERIMENT_RESULTS_PATH = settings.EXPERIMENT_RESULTS_PATH

sum_problem_setting_names = [
    'image_kde',
    'image_softmax',
    'image_ball_counting',
    'text_kde',
    'text_ball_counting',
]

data_types = [
    'sum_estimates',
    'time_estimates',
    'true_sum',
    'recall_exact',
    'recall_qdrant',
]


def list_parquet_files(dir_path: str) -> list[str]:
    """List .parquet files under dir_path (local or S3)."""
    if dir_path.startswith("s3://"):
        import s3fs
        fs = s3fs.S3FileSystem(anon=False)
        try:
            entries = fs.ls(dir_path)
            return [f"s3://{e}" if not e.startswith("s3://") else e for e in entries
                    if e.endswith(".parquet")]
        except Exception as e:
            print(f"Could not list {dir_path}: {e}")
            return []
    else:
        if not os.path.isdir(dir_path):
            return []
        return [
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if f.endswith(".parquet")
        ]


results_dir_paths = [
    f"{name}_{d}"
    for name in sum_problem_setting_names
    for d in data_types
]

for dir_name in tqdm(results_dir_paths):
    full_dir_path = os.path.join(EXPERIMENT_RESULTS_PATH, dir_name)
    paths = list_parquet_files(full_dir_path)

    dfs = []
    for path in paths:
        try:
            dfs.append(pd.read_parquet(path))
        except Exception as e:
            print(f"Failed to read {path}: {e}")

    if dfs:
        df_all = pd.concat(dfs, ignore_index=True)
        out_path = f"{EXPERIMENT_RESULTS_PATH}/{dir_name}.parquet"
        df_all.to_parquet(out_path)
        print(f"Wrote {len(df_all)} rows → {out_path}")
    else:
        print(f"No readable parquet files found in {dir_name}, skipping.")
