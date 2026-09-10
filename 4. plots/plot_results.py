"""Quality and time/quality trade-off figures, one pair per (data, task).

Writes `<data>_<task>_quality.pdf` and `<data>_<task>_tradeoff.pdf`:

* quality  - median relative error against the scoring-function parameter
* tradeoff - median relative error at the worst parameter against median runtime,
             with confidence intervals on both axes

    python plot_results.py
    python plot_results.py --results-path ../results --format png
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helper import results_io, style
from helper.config import settings


def quality_figure(task, data, param_values, error_matrices, plots_dir, plot_format):
    """Median relative error against the task parameter, one line per method."""
    plt.figure()
    seen = []
    # Sorted so 'combined' is drawn first, as in the original.
    for method in sorted(error_matrices):
        algorithm = style.algorithm_of(method)
        if algorithm in seen:
            legend_name = None
        else:
            legend_name = style.ALG_TO_LEGEND_NAME[algorithm]
            seen.append(algorithm)

        matrix = error_matrices[method]
        with warnings.catch_warnings():
            # A parameter where every query had a zero true sum is all-NaN by
            # construction; it is blanked on the next line anyway.
            warnings.simplefilter("ignore", RuntimeWarning)
            y = np.nanmedian(matrix, axis=0)
        # A parameter where most queries had no defined error says nothing.
        y[np.mean(np.isnan(matrix), axis=0) > 0.75] = np.nan
        plt.plot(param_values, y, color=style.ALG_TO_COLOR[algorithm],
                 label=legend_name, alpha=0.8, linewidth=1.5)

    key = f"{data}_{task}"
    plt.xlabel(style.TASK_TO_PARAM_NAME[task])
    plt.ylabel('Median Relative Error')
    plt.xscale('log')
    plt.xlim(style.PARAM_LOWER[key], style.PARAM_UPPER[key])
    plt.ylim(-0.01, style.YLIM_UPPER[key])
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(loc="best")
    plt.tight_layout()
    path = os.path.join(plots_dir, f"{key}_quality.{plot_format}")
    plt.savefig(path, format=plot_format, bbox_inches="tight")
    plt.close()
    return path


def tradeoff_figure(task, data, error_matrices, times, plots_dir, plot_format):
    """Error at the worst parameter against runtime, with intervals on both axes."""
    plt.figure()
    per_algorithm = {}

    for method in sorted(error_matrices):
        algorithm = style.algorithm_of(method)
        per_algorithm.setdefault(algorithm, [])

        method_times = times[method]
        lower_index, upper_index = results_io.median_ci_indices(len(method_times))
        sorted_times = np.sort(method_times)
        x_centre = float(np.median(method_times))
        x_lower = x_centre - sorted_times[lower_index]
        x_upper = sorted_times[upper_index] - x_centre

        # The parameter where this method does worst, then its spread there.
        matrix = np.nan_to_num(error_matrices[method])
        worst = int(np.argmax(np.median(matrix, axis=0)))
        errors = matrix[:, worst]
        lower_index, upper_index = results_io.median_ci_indices(len(errors))
        sorted_errors = np.sort(errors)
        y_centre = float(np.median(errors))
        y_lower = y_centre - sorted_errors[lower_index]
        y_upper = sorted_errors[upper_index] - y_centre

        per_algorithm[algorithm].append(
            (x_centre, x_lower, x_upper, y_centre, y_lower, y_upper, style.params_of(method))
        )

    for algorithm, points in sorted(per_algorithm.items()):
        plt.errorbar(
            np.array([p[0] for p in points]),
            np.array([p[3] for p in points]),
            xerr=np.array([[p[1] for p in points], [p[2] for p in points]]),
            yerr=np.array([[p[4] for p in points], [p[5] for p in points]]),
            fmt='o',
            label=style.ALG_TO_LEGEND_NAME[algorithm],
            capsize=4,
            color=style.ALG_TO_COLOR[algorithm],
        )
        for point in points:
            plt.annotate(point[6], (point[0], point[3]), fontsize=10,
                         color=style.ALG_TO_COLOR[algorithm])

    key = f"{data}_{task}"
    plt.xlabel("Median Inference Time")
    plt.ylabel(f"Median Relative Error at Worst {style.TASK_TO_PARAM_NAME[task]}")
    plt.xlim(0, style.TIME_UPPER[key])
    plt.ylim(-0.01, style.YLIM_UPPER[key])
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    path = os.path.join(plots_dir, f"{key}_tradeoff.{plot_format}")
    plt.savefig(path, format=plot_format, bbox_inches="tight")
    plt.close()
    return path


def plot_combination(combination, results_path, plots_dir, plot_format):
    task, data = combination["task"], combination["data"]

    param_values, true_sums = results_io.load_true_sums(
        results_io.combined_path(results_path, data, task, "true_sum"))
    estimate_params, per_method = results_io.load_param_matrix(
        results_io.combined_path(results_path, data, task, "sum_estimates"))
    if not np.array_equal(param_values, estimate_params):
        raise SystemExit(
            f"{data}_{task}: parameter columns differ between true_sum and "
            f"sum_estimates; the two files are not from the same run."
        )
    times = results_io.load_times(
        results_io.combined_path(results_path, data, task, "time_estimates"))
    missing = set(per_method) - set(times)
    if missing:
        raise SystemExit(f"{data}_{task}: no timings for {sorted(missing)}")

    error_matrices = {
        method: results_io.relative_error_matrix(estimates, true_sums)
        for method, estimates in per_method.items()
    }
    written = [
        quality_figure(task, data, param_values, error_matrices, plots_dir, plot_format),
        tradeoff_figure(task, data, error_matrices, times, plots_dir, plot_format),
    ]
    queries = len(next(iter(per_method.values())))
    print(f"  {data}_{task}: {len(per_method)} methods, {queries} queries, "
          f"{len(param_values)} parameters")
    return written


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-path", default=settings.EXPERIMENT_RESULTS_PATH,
                        help="Directory holding the combined Parquet files.")
    parser.add_argument("--plots-dir", default=settings.PLOTS_DIR,
                        help="Where to write the figures.")
    parser.add_argument("--format", default="pdf", help="Figure format (default: pdf).")
    return parser.parse_args()


def main():
    args = parse_args()
    needed = ("true_sum", "sum_estimates", "time_estimates")
    present, missing = results_io.available_combinations(
        args.results_path, settings.TASK_DATA_COMBINATIONS, needed)
    if not present:
        raise SystemExit(
            f"No combined results under {args.results_path}. Run step 3, then "
            f"combine_shards.py."
        )
    for combination in missing:
        print(f"  skipping {combination['data']}_{combination['task']}: no results")

    results_io.ensure_dir(args.plots_dir)
    written = []
    for combination in present:
        written += plot_combination(combination, args.results_path, args.plots_dir, args.format)
    print(f"[Done] {len(written)} figure(s) in {args.plots_dir}")


if __name__ == "__main__":
    main()
