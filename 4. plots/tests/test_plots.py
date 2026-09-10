"""Tests for the plotting step: statistics, recall matching, and the four figures.

Writes a small results tree with pandas (step 3 is not needed), combines it, and
runs every plot script over it.

    python tests/test_plots.py
    python tests/test_plots.py --keep
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from helper import results_io
from plot_synthetic import Estimate, SyntheticTask, estimate_many

TASKS = (("text", "kde"), ("text", "ball_counting"))
QUERIES = 8
PARAMS = [0.5, 1.0, 2.0, 4.0]
METHODS = {"our_25": 25, "our_50": 50, "topk_100": 100, "combined_100_50": 100, "random_50": None}


def write_results(root: str) -> str:
    """A results tree shaped exactly like step 3's output."""
    results = os.path.join(root, "results")
    rng = np.random.default_rng(0)
    for data, task in TASKS:
        name = f"{data}_{task}"
        truth = {q: rng.uniform(5, 50, size=len(PARAMS)) for q in range(QUERIES)}

        for kind in ("sum_estimates", "true_sum", "time_estimates",
                     "recall_exact", "recall_qdrant"):
            os.makedirs(os.path.join(results, f"{name}_{kind}"), exist_ok=True)

        for q in range(QUERIES):
            estimates, true_rows, times, exact, approximate = [], [], [], [], []
            for method, k in METHODS.items():
                noise = 1 + rng.normal(0, 0.05, size=len(PARAMS))
                estimates.append({"method": method, "query_id": q}
                                 | {str(p): v for p, v in zip(PARAMS, truth[q] * noise)})
                true_rows.append({"method": method, "query_id": q}
                                 | {str(p): v for p, v in zip(PARAMS, truth[q])})
                times.append({"method": method, "query_id": q, "time": float(rng.uniform(0.1, 2.0))})

                if method.startswith("our"):
                    # OurAlgorithm reports per-level retrieval, no exact rows.
                    for level in range(1, 6):
                        approximate.append({"query_id": q, "k": k, "level": level,
                                            "topk": list(range(level, level + k))})
                elif k is not None:
                    exact.append({"query_id": q, "k": k, "level": -1,
                                  "topk": list(range(k))})
                    approximate.append({"query_id": q, "k": k, "level": -1,
                                        "topk": list(range(k))})

            for kind, rows in (("sum_estimates", estimates), ("true_sum", true_rows),
                               ("time_estimates", times), ("recall_exact", exact),
                               ("recall_qdrant", approximate)):
                pd.DataFrame(rows).to_parquet(
                    os.path.join(results, f"{name}_{kind}", f"{q}.parquet"))
    return results


def run(script: str, *args) -> str:
    done = subprocess.run([sys.executable, script, *args], cwd=PROJECT_DIR,
                          capture_output=True, text=True)
    if done.returncode != 0:
        raise AssertionError(f"{script} exited {done.returncode}\n{done.stdout[-1500:]}\n{done.stderr[-1500:]}")
    return done.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    root = tempfile.mkdtemp(prefix="plots-test-")
    plots = os.path.join(root, "plots")
    checks = []

    def check(name, fn):
        try:
            checks.append((name, "PASS", fn()))
        except AssertionError as error:
            checks.append((name, "FAIL", str(error)))

    # --- statistics ---------------------------------------------------------
    def ci_matches_original():
        assert results_io.median_ci_indices(100) == (40, 60), results_io.median_ci_indices(100)
        assert results_io.quantile_ci_indices(1000, 0.95) == (949, 936, 963), \
            results_io.quantile_ci_indices(1000, 0.95)
        for n in (1, 3, 7, 250):
            lower, upper = results_io.median_ci_indices(n)
            assert 0 <= lower <= upper <= n - 1, f"n={n} gave ({lower}, {upper})"
        return "n=100 -> (40, 60); n=1000 q=.95 -> (949, 936, 963), as hard-coded originally"

    def relative_error_drops_zero_truth():
        estimates = {1: np.array([2.0, 3.0])}
        truth = {1: np.array([1.0, 0.0])}
        matrix = results_io.relative_error_matrix(estimates, truth)
        assert matrix[0, 0] == 1.0, matrix
        assert np.isnan(matrix[0, 1]), "zero true sum should give NaN, not an infinity"
        return "zero true sum -> NaN"

    def vectorised_estimator_matches_original():
        x_values = [int(10 ** p) for p in np.arange(0.0, 7.1, 0.5)]
        task = SyntheticTask(10 ** 6, 1)
        U, non_full = task.GetU(200), task.GetNonFullLevels(200)
        slow = np.array([Estimate(U, 200, 10 ** 6, x, non_full) for x in x_values])
        fast = estimate_many(U, 200, 10 ** 6, x_values, non_full)
        worst = float(np.max(np.abs(slow - fast) / np.maximum(np.abs(slow), 1e-12)))
        assert worst < 1e-12, f"vectorised estimator differs by {worst:.2e} relative"
        return f"max relative difference {worst:.1e} over {len(x_values)} x values"

    check("confidence intervals", ci_matches_original)
    check("relative error", relative_error_drops_zero_truth)
    check("estimator equivalence", vectorised_estimator_matches_original)

    # --- end to end ---------------------------------------------------------
    results = write_results(root)

    def combine_writes_one_file_per_kind():
        run("combine_shards.py", "--results-path", results, "--settings", *[f"{d}_{t}" for d, t in TASKS])
        for data, task in TASKS:
            for kind in ("sum_estimates", "true_sum", "time_estimates",
                         "recall_exact", "recall_qdrant"):
                path = os.path.join(results, f"{data}_{task}_{kind}.parquet")
                assert os.path.exists(path), f"missing {path}"
            frame = pd.read_parquet(os.path.join(results, f"{data}_{task}_sum_estimates.parquet"))
            assert len(frame) == QUERIES * len(METHODS), f"{len(frame)} rows"
        return f"{len(TASKS) * 5} combined files"

    def results_figures():
        run("plot_results.py", "--results-path", results, "--plots-dir", plots, "--format", "png")
        made = sorted(os.listdir(plots))
        for data, task in TASKS:
            for kind in ("quality", "tradeoff"):
                name = f"{data}_{task}_{kind}.png"
                assert name in made, f"{name} not written (got {made})"
                assert os.path.getsize(os.path.join(plots, name)) > 5000, f"{name} looks empty"
        return f"{2 * len(TASKS)} figures"

    def recall_matches_known_overlap():
        output = run("plot_recalls.py", "--results-path", results, "--plots-dir", plots,
                     "--format", "png", "--max-level", "10")
        for data, task in TASKS:
            name = f"{data}_{task}_recalls.png"
            assert name in os.listdir(plots), f"{name} not written"
        # our_25 at level L retrieves ids L..L+24 against an exact baseline of 0..24,
        # so recall is (25 - L) / 25. Recompute it the way the script does.
        import plot_recalls
        exact = plot_recalls.exact_topk_per_query(
            os.path.join(results, "text_kde_recall_exact.parquet"))
        per_key = plot_recalls.recalls_by_k_and_level(
            os.path.join(results, "text_kde_recall_qdrant.parquet"), exact, False)
        for level in range(1, 6):
            expected = (25 - level) / 25
            got = float(np.mean(per_key[(25, level)]))
            assert abs(got - expected) < 1e-12, f"k=25 level={level}: {got} != {expected}"
        assert float(np.mean(per_key[(100, -1)])) == 1.0, "identical lists should give recall 1"
        return "recall at k=25 matches (25-level)/25 for every level"

    def synthetic_figure():
        run("plot_synthetic.py", "--seeds", "3", "--n", "100000", "--plots-dir", plots,
            "--format", "png")
        path = os.path.join(plots, "synthetic.png")
        assert os.path.exists(path) and os.path.getsize(path) > 5000, "synthetic.png looks empty"
        return "synthetic.png written"

    def missing_tasks_are_skipped():
        output = run("plot_results.py", "--results-path", results, "--plots-dir", plots,
                     "--format", "png")
        for data, task in (("image", "kde"), ("image", "softmax"), ("image", "ball_counting")):
            assert f"skipping {data}_{task}" in output, f"did not skip {data}_{task}:\n{output}"
        return "3 absent tasks skipped instead of raising"

    check("combine shards", combine_writes_one_file_per_kind)
    check("results figures", results_figures)
    check("recall figures", recall_matches_known_overlap)
    check("synthetic figure", synthetic_figure)
    check("missing tasks", missing_tasks_are_skipped)

    print("\n" + "=" * 80)
    for name, status, detail in checks:
        print(f"{status:4}  {name:24} {detail if status == 'PASS' else ''}")
        if status == "FAIL":
            print(f"      {detail}")
    if args.keep:
        print(f"\nScratch kept at {root}")
    else:
        shutil.rmtree(root, ignore_errors=True)
    raise SystemExit(any(status == "FAIL" for _, status, _ in checks))


if __name__ == "__main__":
    main()
