"""Synthetic validation of the estimator's error bound.

Writes `synthetic.pdf`: the 95th-percentile relative error of the algorithm on a
simulated rarity-level structure, against the number of non-zero f values, with
the analytical bound drawn alongside.

    python plot_synthetic.py                 # 1000 seeds, as in the paper
    python plot_synthetic.py --seeds 50      # quick check

The estimator is the same arithmetic as `Estimate` in the original script, with
the per-x_value loop replaced by prefix sums - the weight a position carries does
not depend on x, so it is computed once per seed instead of once per x value.
`tests/test_plots.py` checks the two agree.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helper import results_io
from helper.config import settings


class SyntheticTask:
    """Items scattered along a line, each assigned a geometric rarity level.

    A level stops accepting items once it holds `max_k`, which is what makes the
    gaps between successive items widen as the simulation proceeds.
    """

    def __init__(self, n, seed, max_k=1000):
        self.n = n
        self.max_k = max_k
        self.items_per_level = {}

        rand_gen = np.random.default_rng(seed=seed)
        min_remaining_level = 1
        cut_levels = []
        p = 1
        loc = 0
        while loc < n:
            while True:
                level = rand_gen.geometric(0.5) + (min_remaining_level - 1)
                if level not in cut_levels:
                    break

            if level not in self.items_per_level:
                self.items_per_level[level] = []
            self.items_per_level[level].append(loc)

            if len(self.items_per_level[level]) >= max_k:
                cut_levels.append(level)
                p -= 2 ** (-level)
                while min_remaining_level in cut_levels:
                    min_remaining_level += 1

            loc += rand_gen.geometric(p)

    def GetU(self, k):
        assert k <= self.max_k
        items = []
        for level in self.items_per_level.keys():
            for item in self.items_per_level[level][:k]:
                items.append((item, level))
        return items

    def GetNonFullLevels(self, k):
        assert k <= self.max_k
        items = []
        for level in self.items_per_level.keys():
            if len(self.items_per_level[level]) < k:
                items += self.items_per_level[level]
        return items


def Estimate(U, k, n, num_nonzero, non_full):
    """The estimator, exactly as originally written (kept for the equivalence test)."""
    median_non_full = np.median(non_full)
    back_half = [num for num in non_full if num >= median_non_full]
    c = np.mean([int(num < num_nonzero) for num in back_half])

    E = 0
    p = 1
    level_to_count = {}

    for index, level in sorted(U):
        f_value = int(index < num_nonzero) - c
        E += f_value / p

        if level not in level_to_count:
            level_to_count[level] = 0
        level_to_count[level] += 1

        if level_to_count[level] == k:
            p -= 2 ** (-level)

    return E + c * n


def position_weights(U, k):
    """`(indices, weights)` in the estimator's traversal order.

    The weight `1/p` a position carries depends only on how many items of each
    level precede it, not on `num_nonzero`, so it is the same for every x value.
    """
    ordered = sorted(U)
    weights = np.empty(len(ordered), dtype=np.float64)
    indices = np.empty(len(ordered), dtype=np.int64)
    p = 1.0
    level_to_count = {}
    for position, (index, level) in enumerate(ordered):
        indices[position] = index
        weights[position] = 1.0 / p
        level_to_count[level] = level_to_count.get(level, 0) + 1
        if level_to_count[level] == k:
            p -= 2 ** (-level)
    return indices, weights


def estimate_many(U, k, n, x_values, non_full):
    """`Estimate` for every x value at once, via prefix sums."""
    indices, weights = position_weights(U, k)
    cumulative = np.cumsum(weights)
    total_weight = cumulative[-1] if len(cumulative) else 0.0

    median_non_full = np.median(non_full)
    back_half = np.sort([num for num in non_full if num >= median_non_full])

    x = np.asarray(x_values)
    below = np.searchsorted(indices, x, side="left")
    weight_below = np.where(below > 0, cumulative[np.clip(below - 1, 0, None)], 0.0)
    c = np.searchsorted(back_half, x, side="left") / len(back_half)
    return weight_below - c * total_weight + c * n


def simulate(n, k, x_values, seeds, quiet=False):
    """Relative error per (seed, x value)."""
    x = np.asarray(x_values, dtype=np.float64)
    rows = []
    for seed in range(seeds):
        if not quiet and seed % 100 == 0:
            print(f"  seed {seed}/{seeds}")
        task = SyntheticTask(n, seed)
        estimates = estimate_many(task.GetU(k), k, n, x_values, task.GetNonFullLevels(k))
        rows.append(np.abs(estimates - x) / x)
    return np.array(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=1000, help="Simulation runs (default: 1000).")
    parser.add_argument("--n", type=int, default=10 ** 7, help="Items per run (default: 10^7).")
    parser.add_argument("--k", type=int, default=200, help="k per level (default: 200).")
    parser.add_argument("--quantile", type=float, default=0.95,
                        help="Error quantile to plot (default: 0.95).")
    parser.add_argument("--analysis-bound", type=float, default=0.1631,
                        help="The analytical bound to draw (default: 0.1631).")
    parser.add_argument("--plots-dir", default=settings.PLOTS_DIR)
    parser.add_argument("--format", default="pdf")
    return parser.parse_args()


def main():
    args = parse_args()
    x_values = [int(10 ** power) for power in np.arange(0.0, 7.1, 0.1)]
    print(f"Simulating {args.seeds} seeds at n={args.n}, k={args.k} …")
    errors = simulate(args.n, args.k, x_values, args.seeds)

    ordered = np.sort(errors, axis=0)
    centre, lower, upper = results_io.quantile_ci_indices(args.seeds, args.quantile)
    centres, lowers, uppers = ordered[centre, :], ordered[lower, :], ordered[upper, :]

    plt.figure()
    plt.plot(x_values, centres, label="Our algorithm")
# matplotlib 3.5 crashes in fill_between when nothing in the inputs is masked
# (the internal mask comes back 0-dimensional). Passing `where` explicitly is
# equivalent to the default and works on every version.
    plt.fill_between(x_values, lowers, uppers, alpha=0.3,
                     where=np.ones(len(x_values), dtype=bool))
    plt.plot(x_values, len(x_values) * [args.analysis_bound], label="Our analysis", c="r")
    plt.xlabel("Number of non-zero f values")
    plt.ylabel(f"{int(args.quantile * 100)}th-percentile Relative Error")
    plt.xscale('log')
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(loc="best")
    plt.tight_layout()

    results_io.ensure_dir(args.plots_dir)
    path = os.path.join(args.plots_dir, f"synthetic.{args.format}")
    plt.savefig(path, format=args.format, bbox_inches="tight")
    plt.close()
    print(f"[Done] {path}")


if __name__ == "__main__":
    main()
