"""Combine step 3's per-query Parquet shards into one file per result type.

Step 3 writes `<data>_<task>_<kind>/<query_id>.parquet`; the plots read
`<data>_<task>_<kind>.parquet`. Works with a local path or an S3 path (set
RESULTS_PATH in .env).

    python combine_shards.py
    python combine_shards.py --results-path s3://bucket/prefix
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helper.config import settings

SETTING_NAMES = [
    'image_kde',
    'image_softmax',
    'image_ball_counting',
    'text_kde',
    'text_ball_counting',
]


def list_parquet_files(dir_path: str) -> List[str]:
    """List .parquet files under `dir_path` (local or S3)."""
    if dir_path.startswith("s3://"):
        import s3fs
        filesystem = s3fs.S3FileSystem(anon=False)
        try:
            entries = filesystem.ls(dir_path)
        except Exception as error:
            print(f"Could not list {dir_path}: {error}")
            return []
        return [
            entry if entry.startswith("s3://") else f"s3://{entry}"
            for entry in entries if entry.endswith(".parquet")
        ]
    if not os.path.isdir(dir_path):
        return []
    return [
        os.path.join(dir_path, name)
        for name in sorted(os.listdir(dir_path))
        if name.endswith(".parquet")
    ]


def combine(results_path: str, names: List[str], kinds) -> int:
    written = 0
    directories = [f"{name}_{kind}" for name in names for kind in kinds]
    for dir_name in tqdm(directories, desc="combining"):
        paths = list_parquet_files(os.path.join(results_path, dir_name))
        frames = []
        for path in paths:
            try:
                frames.append(pd.read_parquet(path))
            except Exception as error:
                print(f"Failed to read {path}: {error}")
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True)
        out_path = f"{results_path}/{dir_name}.parquet"
        combined.to_parquet(out_path)
        print(f"Wrote {len(combined):>6} rows -> {out_path}")
        written += 1
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-path", default=settings.EXPERIMENT_RESULTS_PATH,
                        help=f"Step-3 results directory (default: {settings.EXPERIMENT_RESULTS_PATH}).")
    parser.add_argument("--settings", nargs="+", default=SETTING_NAMES, choices=SETTING_NAMES,
                        help="Which tasks to combine (default: all).")
    args = parser.parse_args()

    written = combine(args.results_path, args.settings, settings.RESULT_KINDS)
    if not written:
        raise SystemExit(
            f"Nothing to combine under {args.results_path}. Run step 3 first, or "
            f"point --results-path at its output."
        )
    print(f"[Done] {written} combined file(s) in {args.results_path}")


if __name__ == "__main__":
    main()
