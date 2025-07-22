from config import settings

import pandas as pd
from tqdm import tqdm
import s3fs

EXPERIMENT_RESULTS_PATH = settings.EXPERIMENT_RESULTS_PATH

sum_problem_setting_names = [
    'image_kde',
    'image_softmax',
    'image_ball_counting',
    'text_kde',
    'text_ball_counting'
]

data = [
    'sum_estimates',
    'time_estimates',
    'true_sum',
    'recall_exact',
    'recall_qdrant'
]

results_dir_paths = []

for name in sum_problem_setting_names:
    for d in data:
        path = f"{name}_{d}"
        results_dir_paths.append(path)


fs = s3fs.S3FileSystem(anon=False)

for dir in tqdm(results_dir_paths):
    full_dir_path = f"{EXPERIMENT_RESULTS_PATH}/{dir}/"
    paths = [obj.path for obj in dbutils.fs.ls(full_dir_path)]
    dfs = []
    for path in paths:
        try:
            df = pd.read_parquet(path)
            dfs.append(df)
        except Exception as e:
            print(f"Failed to read {path}: {e}")
    
    if dfs:  # Proceed only if at least one DataFrame was loaded
        df_all = pd.concat(dfs, ignore_index=True)
        df_all.reset_index(drop=True, inplace=True)
        df_all.to_parquet(f"{EXPERIMENT_RESULTS_PATH}/{dir}.parquet")
    else:
        print(f"No readable parquet files found in {dir}, skipping.")