# Step 3 — Experiments

Estimates the sum of a scoring function over a dataset with four samplers, and
records how close each gets and how long it took. Reads the collections step 2
indexed; writes Parquet for step 4 to plot.

```
main.py                             runner
my_datasets.py                      one class per (task, collection): grids and sampling
qdrant_sum_estimation_algorithm.py  OurAlgorithm, TopK, RandomSample, Combined — unchanged
qdrant_sum_problem_settings.py      scoring functions per task — unchanged
local_problem_settings.py           the same settings with locally-computed ground truth
helper/
  config.py            settings, read from .env
  qdrant_connection.py cloud / local Docker / embedded connections
  qdrant_helpers.py    Qdrant query helpers
  qdrant_data_classes.py
  max_levels.py        loads the level bounds step 2 recorded
  query_sampler.py     uniform selection of queries and dataset items
  local_scores.py      exact ground truth from the step-1 files
tests/
  test_experiments.py    end to end, both ground-truth backends
  test_query_sampler.py  chunking, uniformity, sparse ids
  test_local_scores.py   local scores vs Qdrant, per distance
```

What defines the experiment sits at the top level; `helper/` holds the plumbing.

The two files marked unchanged differ from the originals only in their import
lines, which now point at `helper/`; `diff` against
`3. run experiments/` shows nothing else. The summing logic, the weighting in
`OurAlgorithm`, and the `f_vals` definitions are untouched.

## Running

```bash
cd "3. experiments"
python main.py                        # all five tasks, NUM_QUERIES queries
python main.py --settings image_kde --queries 10
```

Flags: `--queries`, `--settings`, `--num-dataset` (or `all`), `--results-path`,
`--seed`, `--oversampling`, `--k-values`, `--topk-values`, `--random-values`,
`--qdrant {auto,cloud,docker,embedded}` (same targets as step 2), plus the
scaling flags below.

Output, one file per query per task:

```
experiments_results/
  image_kde_sum_estimates/<query_id>.parquet     one row per (method, params)
  image_kde_true_sum/<query_id>.parquet          the exact sum, for comparison
  image_kde_time_estimates/<query_id>.parquet
  image_kde_recall_exact/<query_id>.parquet
  image_kde_recall_qdrant/<query_id>.parquet
```

## Scaling to large collections

Nothing the experiment *measures* gets slower as the collection grows: every
timed call is an ANN or id-filtered query whose cost barely depends on `N_d`.
What grows is the ground truth around it — scoring every dataset item, which in
Qdrant costs one filtered search per 500 items (20,000 requests per query at
10M). Three things address that:

**Ground truth is computed locally by default.** `--ground-truth
{auto,local,qdrant}`: `local` scores the step-1 matrices directly with BLAS over
memory maps, `auto` (the default) uses it when those files are present and falls
back to Qdrant otherwise. Measured on 4,000 items: 8 search requests and 172 ms
versus 0 requests and 8 ms — and the gap widens linearly with collection size.
The algorithms still issue their own Qdrant queries, so **timings are unaffected**.

Point ids are row positions (step 2 assigns them that way), so row `i` of the
concatenated shards is point `i`. Point `--embeddings-dir` at step 1's output and
use `--embeddings-prefix collection=prefix` when the prefix differs from the
collection name. Local scores reproduce Qdrant's conventions exactly — `||x-q||`
for Euclid, `q·x` for Dot — agreeing to ~6e-7 with identical top-k ordering
(`test_local_scores.py`).

**All-scores stays a numpy array.** The locally-scored settings keep scores as
`float32` rather than one `EmbeddingObjectWithSim` per item: 0.04 GB instead of
3.6 GB at 10M, per query per task. This is done by subclassing in
`local_problem_settings.py` — `qdrant_sum_problem_settings.py` is untouched.

**True sums are computed once per (query, task).** They depend only on the task
and the parameter, not the estimator, so they are no longer recomputed inside
each of the ~38 combinations.

Two more knobs for scale:

- `--score-batch N` (default 8) scores N queries per pass over the embedding
  files, so the disk read is shared. Costs `N × collection_size × 4` bytes.
- `--shard i --num-shards n` splits queries across processes. Queries are
  independent and output is one file per query, so `n` workers give a near-linear
  speedup:

```bash
for i in 0 1 2 3; do python main.py --shard $i --num-shards 4 & done; wait
```

## Query selection

Queries and dataset items are drawn **uniformly from the entire collection**.
Previously both came from the prefix a scroll returned — the earliest-inserted
points — so every run sampled the same corner of the index.

`query_sampler.CollectionSampler` handles two id layouts:

- **Integer ids** (what step 2 assigns — row position, `0 .. count-1`): sampled
  directly from `count()`, with no scan at all. Both ends and a few random
  interior ids are probed to confirm the space really is contiguous. Taking the
  whole collection issues no lookups; a strict subset is confirmed to exist in
  chunked requests, and gaps are re-drawn.
- **Anything else** (e.g. UUIDs in a pre-existing collection): the id space is
  listed once by scrolling and cached at `<ID_CACHE_DIR>/<collection>/point_ids.txt`,
  then sampled from. The scan happens once; later runs reuse the cache, and it is
  rebuilt automatically if the collection's point count changes.

`--seed` makes a selection reproducible. The query point is always excluded from
the dataset it is summed over.

Requests are chunked (`RETRIEVE_CHUNK`, `VECTOR_RETRIEVE_CHUNK`), so a
10M-id sample is thousands of small lookups rather than one impossible request,
and drawing a large fraction of a collection avoids materialising a full
permutation. `test_query_sampler.py` covers these at scale.

## Level bounds

`OurAlgorithm` issues one query per level value, so it needs to know how many
values a level field takes. Those bounds come from
`hnsw_index/<collection>/max_levels.json`, written by step 2 — they used to be
hard-coded here, which silently went stale whenever a collection was re-indexed
with fresh level draws. A collection with no recorded bounds fails with a message
naming the file it wanted.

Each run uses level field `q % 10`, so a 100-query run exercises all ten
independent level assignments.

## Testing

```bash
python tests/test_experiments.py     # index, run both backends, check the output
python tests/test_query_sampler.py   # sampling: chunking, uniformity, sparse ids
python tests/test_local_scores.py    # local scores vs Qdrant, per distance
```

Each test and `main.py` work from any working directory.

Indexes 400 synthetic points into an embedded Qdrant, runs two queries over two
tasks with **both** ground-truth backends, and checks that they agree on the true
sums (to within the float32 allowance the score precision implies), that two
query shards cover exactly what one run does, and the Parquet output: every task produces every result kind, all
four methods appear, estimates and true sums are finite, the true sum for a
parameter agrees across methods (it is a property of the query, not the
estimator), per-level recall rows exist, and sampled ids span the collection
rather than clustering at the start.

## Grid sizes and dataset size

`--random-values` are sample sizes drawn from the dataset **without replacement**,
so each must be at most `--num-dataset` minus one (the query is held out of its
own dataset). The defaults start at 500, so a small collection needs a smaller
grid; the run refuses up front and suggests one rather than failing inside the
first algorithm:

```bash
python main.py --num-dataset all --random-values 25 50 99 --topk-values 10 25 --k-values 5 10
```

`--k-values` and `--topk-values` need no such limit: those are Qdrant search
limits, and a limit above the collection size just returns fewer points. Note
though that a `k` at or above the dataset size makes `TopK` sum nearly everything,
so its "error" stops being informative.

## A caveat worth knowing

`--num-dataset` samples the items whose sum is being estimated, but the samplers
query the *whole* collection. When the dataset is a strict subset, `TopK` and
`OurAlgorithm` draw on points outside it, so their estimates are not directly
comparable to the true sum over the subset — the effect is large when the subset
is small. This is inherent to the original design, not new here. For a
like-for-like comparison, set `--num-dataset` to the collection size.
