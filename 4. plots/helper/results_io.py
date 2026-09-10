"""Reading step-3 results, and the statistics the figures need.

The plots read the combined files `combine_shards.py` writes:
`<data>_<task>_<kind>.parquet`, one per (task, result kind).

Columns are selected by name rather than by position. The originals indexed
`df.columns[2:]` and `row[3:]`, which silently mis-reads if the column order ever
changes.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

#: Columns that identify a row rather than carrying a parameter value.
INDEX_COLUMNS = ("method", "query_id")


def combined_path(results_path: str, data: str, task: str, kind: str) -> str:
    return os.path.join(results_path, f"{data}_{task}_{kind}.parquet")


def available_combinations(results_path: str, combinations, kinds) -> Tuple[list, list]:
    """Split the configured combinations into those with data and those without."""
    present, missing = [], []
    for combination in combinations:
        needed = [
            combined_path(results_path, combination["data"], combination["task"], kind)
            for kind in kinds
        ]
        (present if all(os.path.exists(p) for p in needed) else missing).append(combination)
    return present, missing


def param_columns(frame: pd.DataFrame) -> List[str]:
    """The scoring-function parameter columns, in file order."""
    return [column for column in frame.columns if column not in INDEX_COLUMNS]


def load_param_matrix(path: str) -> Tuple[np.ndarray, Dict[str, Dict[object, np.ndarray]]]:
    """Read a wide (method, query_id, <param>...) file.

    Returns the parameter values and `{method: {query_id: values}}`.
    """
    frame = pd.read_parquet(path)
    columns = param_columns(frame)
    values = np.array([float(column) for column in columns])

    per_method: Dict[str, Dict[object, np.ndarray]] = {}
    for method, rows in frame.groupby("method", sort=True):
        per_method[method] = {
            row.query_id: np.asarray(row_values, dtype=float)
            for row, row_values in zip(rows.itertuples(), rows[columns].to_numpy(dtype=float))
        }
    return values, per_method


def load_true_sums(path: str) -> Tuple[np.ndarray, Dict[object, np.ndarray]]:
    """Read the true sums as `{query_id: values}`.

    Every method reports the same true sum for a query, so duplicates collapse.
    """
    frame = pd.read_parquet(path)
    columns = param_columns(frame)
    values = np.array([float(column) for column in columns])
    per_query = {
        row.query_id: np.asarray(row_values, dtype=float)
        for row, row_values in zip(frame.itertuples(), frame[columns].to_numpy(dtype=float))
    }
    return values, per_query


def load_times(path: str) -> Dict[str, List[float]]:
    frame = pd.read_parquet(path)
    return {method: rows["time"].tolist() for method, rows in frame.groupby("method", sort=True)}


def relative_error_matrix(
    estimates: Dict[object, np.ndarray],
    true_sums: Dict[object, np.ndarray],
) -> np.ndarray:
    """`|estimate - truth| / truth` per (query, parameter); NaN where truth <= 0.

    A zero true sum has no relative error to speak of - for ball counting the
    radius can be small enough that nothing falls inside it - so those entries are
    dropped rather than turned into infinities.
    """
    rows = []
    for query_id, estimate in estimates.items():
        truth = true_sums[query_id]
        denominator = np.where(truth > 0, truth, -1)
        rows.append(np.where(truth > 0, np.abs(estimate - truth) / denominator, np.nan))
    return np.array(rows)


def quantile_ci_indices(n: int, quantile: float, alpha: float = 0.05) -> Tuple[int, int, int]:
    """Sorted-array indices for a quantile and its confidence interval.

    The count of samples below the true quantile is Binomial(n, q), so the
    interval comes straight from that distribution's tails. Exactly reproduces the
    indices the original scripts hard-coded, and generalises to any n:

        n=100,  q=0.50 -> lower 40,  upper 60   (original plot_results.py)
        n=1000, q=0.95 -> lower 936, upper 963  (original plot_synthetic.py)

    Returns `(centre, lower, upper)` as 0-based indices into the sorted values.
    """
    if n < 1:
        raise ValueError("need at least one sample")
    centre = int(np.clip(np.ceil(quantile * n) - 1, 0, n - 1))
    try:
        from scipy.stats import binom
        lower = int(binom.ppf(alpha / 2, n, quantile))
        upper = int(binom.isf(alpha / 2, n, quantile))
    except ImportError:  # pragma: no cover - normal approximation fallback
        spread = 1.96 * np.sqrt(n * quantile * (1 - quantile))
        lower = int(np.floor(quantile * n - spread))
        upper = int(np.ceil(quantile * n + spread)) - 1
    lower = int(np.clip(lower, 0, n - 1))
    upper = int(np.clip(upper, 0, n - 1))
    return centre, lower, upper


def median_ci_indices(n: int) -> Tuple[int, int]:
    """`(lower, upper)` sorted-array indices bracketing the median."""
    _, lower, upper = quantile_ci_indices(n, 0.5)
    return lower, upper


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
