# Step 4 — Plots

Turns step 3's Parquet output into the paper's figures.

```
combine_shards.py   per-query shards -> one file per (task, result kind)
plot_results.py     quality and time/quality trade-off figures
plot_recalls.py     recall per rarity level
plot_synthetic.py   synthetic validation of the error bound
helper/
  config.py       settings, read from .env
  style.py        colours, legend names, axis limits
  results_io.py   reading results, relative error, confidence intervals
tests/
  test_plots.py   statistics, recall matching, and every figure end to end
```

## Running

```bash
cd "4. plots"
python combine_shards.py            # once per experiment run
python plot_results.py              # <data>_<task>_quality.pdf, _tradeoff.pdf
python plot_recalls.py              # <data>_<task>_recalls.png
python plot_synthetic.py            # synthetic.pdf (1000 seeds, ~30s)
```

Results are read from the repo-root `experiments_results/` (step 3's default) and
figures go to the repo-root `plots/`. Every script takes `--results-path`, `--plots-dir` and
`--format`; `plot_synthetic.py` also takes `--seeds`, `--n`, `--k` and
`--quantile`. Tasks with no results are skipped with a message rather than
raising, so a run covering one task plots that task.

## The figures

| File | Shows |
|---|---|
| `<data>_<task>_quality.pdf` | median relative error against the task parameter, one line per method |
| `<data>_<task>_tradeoff.pdf` | error at the worst parameter against median runtime, with intervals on both axes |
| `<data>_<task>_recalls.png` | mean recall per rarity level, one series per k |
| `synthetic.pdf` | 95th-percentile error of the estimator against the analytical bound |

## Confidence intervals

The count of samples below a quantile is Binomial(n, q), so the interval indices
come from that distribution's tails (`results_io.quantile_ci_indices`). This
reproduces the indices the original scripts hard-coded — `(40, 60)` for the
median at n=100, `(949, 936, 963)` for the 95th percentile at n=1000 — and works
for any number of queries or seeds. The originals asserted exactly 100 queries.

## Recall ground truth

`recall_exact` holds rows at `level = -1`, one per method that reported, each
truncated to that method's own k. `recall_qdrant` holds the real levels for
OurAlgorithm. Since `GetExactRecall` returns the same list for every level, the
baseline for a query is the longest exact list recorded for it, and recall at k
compares against its first k ids.

The original kept whichever exact row it read last and asserted `k <= that row's
k`, which fails as soon as two methods with different k report for the same
query — which is the normal case. If a k exceeds the longest baseline, those rows
are reported and left out rather than silently counted; raise `EXACT_RECALL_K` in
step 3 to cover them.

`LEGACY_IMAGE_LEVEL_SHIFT=1` restores the original's `level += 1` fix-up for
image KDE/softmax, which compensated for an off-by-one in older result sets.
Current step-3 output is consistent, so it is off by default.

## Testing

```bash
python tests/test_plots.py
```

Writes a small results tree, combines it, and runs all four scripts. Beyond
checking the figures appear, it pins the confidence-interval indices to the
originals, checks a zero true sum yields NaN rather than an infinity, verifies
recall against a case with a known overlap, and asserts the vectorised synthetic
estimator agrees with the original loop.

## A note on `plot_synthetic.py`

The estimator's arithmetic is unchanged, but the per-x-value loop is replaced by
prefix sums: the weight a position carries depends only on the levels preceding
it, not on the x value, so it is computed once per seed rather than once per x
value. That is ~48× faster (85 ms to 1.8 ms per seed) and agrees with the
original to 5e-15 relative. The original function is kept in the module and the
test compares against it.
