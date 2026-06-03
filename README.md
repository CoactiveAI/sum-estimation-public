# SumEstimation

**SumEstimation** is a framework for efficiently estimating the sum of scoring functions over large-scale embedding datasets using a variety of sampling strategies. It targets scenarios where computing the exact sum over millions of vectors is too expensive and a high-quality approximation is sufficient.

## Supported Methods

| Method | Description |
|---|---|
| **OurAlgorithm** | Adaptive sampler that uses rarity-level structure in a Qdrant index |
| **TopK** | Sum over the nearest neighbours only |
| **Random** | Uniform random sample, scaled to the full dataset size |
| **Combined** | Hybrid: TopK + Random for the complement |

Each method is evaluated on five (task, data) combinations:
- KDE / Softmax / Ball-counting on image embeddings (Open Images, ResNet-50, CLIP ViT-L14-336)
- KDE / Ball-counting on text embeddings (Amazon Reviews, DistilBERT)

---

## Prerequisites

- Python 3.10+
- A running **Qdrant** cluster pre-loaded with the embedding collections (see `2. create qdrant cluster/`)
- The Qdrant cluster does **not** need to be on the same machine; embeddings are never downloaded locally

---

## Installation

```bash
git clone https://github.com/your-org/sum-estimation-public.git
cd sum-estimation-public
pip install -r requirements.txt
```

---

## Configuration

Copy the example environment file and fill in your values:

```bash
cp .env.example .env
```

Edit `.env`:

```dotenv
# Required – Qdrant connection
QDRANT_HOST=http://your-qdrant-host:6333
QDRANT_API_KEY=your-api-key

# Optional – defaults shown
QDRANT_PORT=443
QDRANT_TIMEOUT=10000

# Where to write result Parquet files (local path or s3://bucket/prefix)
RESULTS_PATH=./results

# Dataset size trade-off
# Higher = more accurate true-sum estimates, slower per-query computation.
# ~100 000 is a good starting point for reproduction; increase for full-scale runs.
NUM_DATASET_EMBEDDINGS=100000

# How many candidate query vectors to load into the random-query pool
NUM_QUERY_CANDIDATES=1000
```

> **Security note**: `.env` is listed in `.gitignore` and must never be committed.

---

## Running Experiments

```bash
cd "3. run experiments"
python main.py
```

What happens:
1. Dataset item IDs and a pool of query candidate vectors are loaded from Qdrant (no local files needed).
2. For each of 100 experiment iterations a **random** query vector is drawn from the pool.
3. All-scores (similarities of every dataset item to the query) are computed **once** per query and cached.
4. Every algorithm × hyperparameter combination is evaluated and results are written to `RESULTS_PATH` as Parquet shards.

### Result layout

```
results/
  image_kde_sum_estimates/<query_id>.parquet
  image_kde_time_estimates/<query_id>.parquet
  image_kde_true_sum/<query_id>.parquet
  image_kde_recall_exact/<query_id>.parquet
  image_kde_recall_qdrant/<query_id>.parquet
  image_softmax_*/...
  image_ball_counting_*/...
  text_kde_*/...
  text_ball_counting_*/...
```

---

## Generating Plots

### Step 1 – Combine shards

```bash
cd "4. plotting"
python combine_dfs.py
```

This merges per-query Parquet shards into one file per result type (works with local paths and S3).

### Step 2 – Plot

```bash
python plot_results.py      # error and time trade-off figures
python plot_recalls.py      # recall analysis
python plot_synthetic.py    # synthetic validation
```

PDF outputs are written to `4. plotting/plots/`.

---

## Repository Structure

```
.
├── .env.example                      ← copy to .env and fill in credentials
├── requirements.txt
├── 1. create_embeddings/             ← (optional) scripts for generating embeddings
├── 2. create qdrant cluster/
│   ├── qdrant_insert.py              ← create collections and insert vectors
│   └── store_levels.py               ← compute and store rarity-level payloads
├── 3. run experiments/
│   ├── config.py                     ← reads settings from .env
│   ├── main.py                       ← experiment entry point
│   ├── my_datasets.py                ← dataset classes (load from Qdrant)
│   ├── qdrant_helpers.py             ← Qdrant query helpers + scroll utility
│   ├── qdrant_data_classes.py        ← EmbeddingObject, EmbeddingObjectWithSim
│   ├── qdrant_sum_problem_settings.py← scoring functions per task
│   └── qdrant_sum_estimation_algorithm.py ← OurAlgorithm, TopK, Random, Combined
└── 4. plotting/
    ├── config.py
    ├── combine_dfs.py
    ├── plot_results.py
    ├── plot_recalls.py
    └── plot_synthetic.py
```

---

## Reproducing Results

The experiments assume a Qdrant cluster with the following collections already populated:

| Collection | Dataset | Encoder |
|---|---|---|
| `open-images_resnet-50` | Open Images (8M train) | ResNet-50 |
| `open-images_clip_vit_l14_336` | Open Images (8M train) | CLIP ViT-L14-336 |
| `amazon-reviews_distilbert` | Amazon Reviews (10M) | DistilBERT |

Each point must carry `level_0` … `level_9` payload fields (computed by `2. create qdrant cluster/store_levels.py`).

For smaller-scale reproduction, set `NUM_DATASET_EMBEDDINGS` to a few thousand; the algorithms and relative rankings remain the same.

---

## Contact

Steve Mussmann – mussmann@gatech.edu  
Mehul Smriti Raje – mehul@coactive.ai, mehul.raje@gmail.com
