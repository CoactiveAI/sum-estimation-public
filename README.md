# SumEstimation

**SumEstimation** estimates the sum of a scoring function over a large embedding
dataset by sampling, for cases where the exact sum over millions of vectors is
too expensive and a good approximation will do.

| Method | Description |
|---|---|
| **OurAlgorithm** | Adaptive sampler that exploits rarity-level structure in an HNSW index |
| **TopK** | Sum over the nearest neighbours only |
| **Random** | Uniform random sample, scaled to the dataset size |
| **Combined** | TopK plus a random sample of the complement |

Each is evaluated on five (task, data) combinations — KDE, softmax and
ball-counting on image embeddings; KDE and ball-counting on text embeddings —
over three collections:

| Collection | Dataset | Encoder | Dim | Distance |
|---|---|---|---|---|
| `open-images_resnet-50` | Open Images | ResNet-50 | 2048 | Euclid |
| `open-images_clip_vit_l14_336` | Open Images | CLIP ViT-L/14-336 | 768 | Dot (unit vectors) |
| `amazon-reviews_distilbert` | Amazon Reviews 2023 | DistilBERT | 768 | Euclid |

## The pipeline

Four steps, each with its own README covering its flags, outputs and tests:

| Step | Does | Reads | Writes |
|---|---|---|---|
| [1. create_embeddings](1.%20create_embeddings/) | encodes datasets into sharded `.npy` matrices | Hugging Face Hub | `embeddings/` |
| [2. create_hnsw_index](2.%20create_hnsw_index/) | builds a Qdrant HNSW collection with rarity-level payloads | `embeddings/` | Qdrant, `hnsw_index/` |
| [3. experiments](3.%20experiments/) | runs every sampler against random queries | Qdrant, `embeddings/`, `hnsw_index/` | `experiments_results/` |
| [4. plots](4.%20plots/) | turns results into the paper's figures | `experiments_results/` | `plots/` |

## Prerequisites

- Python 3.9+
- `pip install -r requirements.txt`
- Somewhere to put the index. Steps 2 and 3 take `--qdrant`:
  **`embedded`** (no server, no Docker — exact search, so it validates the
  pipeline but not HNSW), **`docker`** (a local server, started for you) or
  **`cloud`** (the cluster in `QDRANT_HOST`). See
  [2. create_hnsw_index](2.%20create_hnsw_index/#where-the-index-runs).

## Configuration

```bash
cp .env.example .env
```

`.env.example` documents every setting. The ones that span steps:

| Setting | Used by | Meaning |
|---|---|---|
| `QDRANT_HOST`, `QDRANT_API_KEY` | 2, 3 | cloud cluster; leave empty for local |
| `EMBEDDINGS_DIR` | 1, 2, 3 | where vectors live — one directory per run |
| `HNSW_INDEX_DIR` | 2, 3 | per-collection `max_levels.json` |
| `RESULTS_PATH` | 3, 4 | Parquet results |
| `PLOTS_DIR` | 4 | figures |

Point `EMBEDDINGS_DIR` at one directory per experiment (e.g.
`../embeddings/50k_run`) so all three steps that touch vectors agree on which
run they mean. Step 3 refuses to start if the vectors and the indexed collection
disagree.

> `.env` is gitignored and must never be committed.

## End to end

```bash
export EMBEDDINGS_DIR=../embeddings/50k_run

cd "1. create_embeddings"
python generate_amazon_reviews_distilbert.py --num-embeddings 50000
python generate_open_images_resnet50.py --num-embeddings 50000
python generate_open_images_clip_vit_l14_336.py --num-embeddings 50000

cd "../2. create_hnsw_index"
for c in amazon-reviews_distilbert open-images_resnet-50 open-images_clip_vit_l14_336; do
    python qdrant_insert.py --collection "$c" --embeddings-prefix "$c" --qdrant docker
done

cd "../3. experiments"
python main.py --num-dataset all

cd "../4. plots"
python combine_shards.py && python plot_results.py && python plot_recalls.py
python plot_synthetic.py
```

Two things to size before a real run: the image encoders fetch Flickr URLs one
image at a time (~2 images/sec, so 50k takes hours — a Hub mirror with embedded
images avoids this), and `--random-values` must not exceed the dataset size.

## Testing

Every step has a smoke test. Steps 2-4 need no Qdrant server and no downloads —
they use the embedded client and generate their own data, so they run in seconds:

```bash
python "2. create_hnsw_index/test_index.py"        # indexing, levels, ids
python "3. experiments/tests/test_experiments.py"  # experiments end to end
python "4. plots/tests/test_plots.py"              # statistics and every figure
```

Step 1's test drives the real encoders, so it downloads model weights and streams
each dataset (a few minutes; `synthetic` alone needs no network):

```bash
python "1. create_embeddings/test_generation.py"            # all four generators
python "1. create_embeddings/test_generation.py" synthetic   # just the offline one
```

`3. experiments/tests/` also holds tests for query sampling and for local scoring
against Qdrant.

## Layout

```
1. create_embeddings/   2. create_hnsw_index/   3. experiments/   4. plots/

embeddings/             generated vectors             (gitignored)
hnsw_index/             per-collection index metadata (gitignored)
experiments_results/    Parquet results               (gitignored)
plots/                  figures                       (committed)
```

## Citation

If you use this repository, please cite the paper or contact us here.

## Contact

Steve Mussmann – mussmann@gatech.edu  
Mehul Smriti Raje – mehul@coactive.ai, mehul.raje@gmail.com
