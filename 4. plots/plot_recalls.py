"""Recall of OurAlgorithm's per-level retrieval against the exact top-k.

Writes `<data>_<task>_recalls.png`: mean recall per rarity level, one series per
k, with a 95% interval across queries.

    python plot_recalls.py
    python plot_recalls.py --results-path ../results

How the ground truth is matched
-------------------------------
`recall_exact` rows carry `level = -1` and a list truncated to the method's own
k, with one row per method that reported it. `recall_qdrant` rows carry the real
levels (1..max_level) for OurAlgorithm. The exact top-k does not depend on the
level - `GetExactRecall` returns the same list for every level - so the baseline
for a query is the longest exact list recorded for it, and recall at k compares
against its first k ids.

The original script instead kept whichever exact row it read last and asserted
`k <= that row's k`, which fails as soon as two methods with different k report
for the same query.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helper import results_io
from helper.config import settings


def exact_topk_per_query(path: str) -> dict:
    """query_id -> the longest exact top-k list recorded for it."""
    frame = pd.read_parquet(path)
    best = {}
    for row in frame.itertuples():
        current = best.get(row.query_id)
        if current is None or len(row.topk) > len(current):
            best[row.query_id] = list(row.topk)
    return best


def recalls_by_k_and_level(path: str, exact: dict, level_shift: bool) -> dict:
    """(k, level) -> list of per-query recalls."""
    frame = pd.read_parquet(path)
    per_key = defaultdict(list)
    skipped_queries, short_baselines = set(), 0

    for row in frame.itertuples():
        level = int(row.level)
        if level_shift and level >= 0:
            level += 1
        baseline = exact.get(row.query_id)
        if baseline is None:
            skipped_queries.add(row.query_id)
            continue
        truth = baseline[:row.k]
        if len(truth) < row.k:
            # The baseline is shorter than this k, so recall is not defined here.
            short_baselines += 1
            recall = np.nan
        else:
            recall = len(set(row.topk).intersection(truth)) / len(truth)
        per_key[(row.k, level)].append(recall)

    if skipped_queries:
        print(f"  {len(skipped_queries)} query(ies) had no exact baseline and were skipped")
    if short_baselines:
        print(f"  {short_baselines} row(s) asked for more ids than the baseline holds "
              f"(recall undefined; raise EXACT_RECALL_K in step 3 to cover them)")
    return per_key


def recall_figure(data, task, per_key, plots_dir, plot_format, max_level):
    k_values = sorted({k for k, _ in per_key})
    means, stds, counts = {}, {}, {}
    for k in k_values:
        means[k] = np.full((max_level,), np.nan)
        stds[k] = np.full((max_level,), np.nan)
        counts[k] = np.full((max_level,), np.nan)

    aggregate_only = []
    for (k, level), recalls in per_key.items():
        if level < 0:
            # level -1 is TopK/Combined: a single number, not a per-level series.
            usable = [r for r in recalls if not np.isnan(r)]
            if usable:
                aggregate_only.append((k, float(np.mean(usable)),
                                       2 * float(np.std(usable)) / np.sqrt(len(usable))))
            continue
        if level >= max_level or np.isnan(recalls).all():
            continue
        means[k][level] = np.nanmean(recalls)
        stds[k][level] = np.nanstd(recalls)
        counts[k][level] = np.sum(~np.isnan(recalls))

    for k, mean, interval in sorted(aggregate_only):
        print(f"  k={k}: recall {mean:.4f} +/- {interval:.4f} (level-independent methods)")

    plotted = [k for k in k_values if not np.isnan(means[k]).all()]
    if not plotted:
        print(f"  {data}_{task}: no per-level recall rows, skipping figure")
        return None

    plt.figure()
    levels = np.arange(max_level)
    for k in plotted:
        plt.plot(levels, means[k], "o-", label=f"k={k}")
        error = 2 * stds[k] / np.sqrt(counts[k])
        # matplotlib 3.5 crashes in fill_between when nothing in the inputs is masked
        # (the internal mask comes back 0-dimensional). Passing `where` explicitly is
        # equivalent to the default and works on every version.
        plt.fill_between(levels, means[k] - error, means[k] + error, alpha=0.3,
                         where=np.ones(len(levels), dtype=bool))
    plt.legend()
    plt.xlabel("Level")
    plt.ylabel("Recall")
    # Ticks over the levels that carry data, so the axis is readable at any scale.
    highest = int(np.nanmax([np.nanmax(np.where(~np.isnan(means[k]))[0]) for k in plotted]))
    ticks = np.arange(1, highest + 2)
    plt.xticks(ticks=ticks, labels=[str(t) for t in ticks])
    plt.tight_layout()
    path = os.path.join(plots_dir, f"{data}_{task}_recalls.{plot_format}")
    plt.savefig(path, format=plot_format, bbox_inches="tight")
    plt.close()
    return path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-path", default=settings.EXPERIMENT_RESULTS_PATH)
    parser.add_argument("--plots-dir", default=settings.PLOTS_DIR)
    parser.add_argument("--format", default="png", help="Figure format (default: png).")
    parser.add_argument("--max-level", type=int, default=settings.RECALL_MAX_LEVEL,
                        help=f"Levels to allocate (default: {settings.RECALL_MAX_LEVEL}).")
    return parser.parse_args()


def main():
    args = parse_args()
    present, missing = results_io.available_combinations(
        args.results_path, settings.TASK_DATA_COMBINATIONS, ("recall_exact", "recall_qdrant"))
    if not present:
        raise SystemExit(f"No combined recall files under {args.results_path}.")
    for combination in missing:
        print(f"  skipping {combination['data']}_{combination['task']}: no recall results")

    results_io.ensure_dir(args.plots_dir)
    written = []
    for combination in present:
        task, data = combination["task"], combination["data"]
        shift = settings.LEGACY_IMAGE_LEVEL_SHIFT and data == "image" and task in ("kde", "softmax")
        exact = exact_topk_per_query(
            results_io.combined_path(args.results_path, data, task, "recall_exact"))
        per_key = recalls_by_k_and_level(
            results_io.combined_path(args.results_path, data, task, "recall_qdrant"),
            exact, shift)
        path = recall_figure(data, task, per_key, args.plots_dir, args.format, args.max_level)
        if path:
            written.append(path)
    print(f"[Done] {len(written)} figure(s) in {args.plots_dir}")


if __name__ == "__main__":
    main()
