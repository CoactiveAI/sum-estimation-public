# Step 1 — Embedding generation

One script per (dataset, encoder) combination. Each streams its dataset from the
Hugging Face Hub, encodes it, and writes sharded `.npy` files to the repo-root
`embeddings/` directory.

| Script | Dataset | Encoder | Dim | Writes |
|---|---|---|---|---|
| `generate_open_images_resnet50.py` | Open Images | ResNet-50 (post-avgpool) | 2048 | raw |
| `generate_open_images_clip_vit_l14_336.py` | Open Images | CLIP ViT-L/14-336 | 768 | normalised |
| `generate_amazon_reviews_distilbert.py` | Amazon Reviews 2023 | DistilBERT | 768 | raw |
| `generate_synthetic.py` | — random Gaussian, for local testing | — | 768 | raw |

## Generating

```bash
pip install -r ../requirements.txt
cd "1. create_embeddings"
python generate_amazon_reviews_distilbert.py --num-embeddings 100000 --device cuda
```

Common flags, shared by every script: `--num-embeddings`, `--batch-size`,
`--chunk-rows`, `--output-dir`, `--output-prefix`, `--device` (`auto` picks CUDA,
then MPS, then CPU) and `--write {default,raw,normalised,both}`. `default` writes
the variant in the table above — normalised (unit) vectors for CLIP, raw for the
rest. `--help` lists each script's dataset-specific flags.

Nothing is downloaded in full: datasets stream, and vectors are written through a
memory map, so a run is bounded by disk, not RAM.

To check an install works, `python test_generation.py` runs every script at tiny
sizes (12 rows in shards of 8) into `embeddings/test_run/` and verifies shards,
ids and resume. Pass a substring — `python test_generation.py synthetic` — to run
just one.

`generate_synthetic.py` needs no dataset, GPU or network — use it to exercise the
pipeline locally. It is seeded, so a given `--num-embeddings`/`--seed` always
gives the same vectors. Add `--collection <key>` to match one of the real
combinations' dimensionality and normalisation.

Open Images has no official Hub mirror. Both image scripts default to
`bitmind/open-images-v7`, whose `url` column holds Flickr URLs, so images are
fetched over HTTP as they stream and dead links are skipped. A mirror with
embedded images is much faster: `--hf-dataset your-org/x --image-column image`.

## Output files

Output is sharded: **250,000 rows per shard** by default (`--chunk-rows`), about
0.7 GB at 768-d and 2 GB at 2048-d. For prefix `<prefix>` (defaults to the
combination's key, set with `--output-prefix`):

```
<prefix>_00000_embeddings.npy    vectors, float32 (n, dim)   — _embeddings_normalised.npy per --write
<prefix>_00000_ids.txt           asset id per row, line N ↔ row N
<prefix>_00001_...               next shard
<prefix>_manifest.json           shard list, row counts, resume point
```

Read a full run by iterating the manifest:

```python
import json, numpy as np
manifest = json.load(open("embeddings/<prefix>_manifest.json"))
for shard in manifest["chunks"]:
    vectors = np.load(f"embeddings/{shard['files']['raw']}", mmap_mode="r")
    ids = open(f"embeddings/{shard['files']['ids']}").read().splitlines()
```

`--chunk-rows 0` writes one unnumbered `<prefix>_embeddings.npy` plus
`<prefix>_ids.txt` instead, with no manifest and no resume.

**Ids matter** because row position is not identity: unreachable images and empty
reviews are skipped, so row 500 is not dataset record 500. Ids are the image URL
(`--id-column` to choose another column) or
`category|user_id|asin|timestamp` for reviews (`--id-columns`):

```
All_Beauty|AGKHLEW2SOWHNMFQIJGBECAF7INQ|B00YQ6X8EO|1588687728923
```

## Rerunning after a failure

Re-run the same command. Shards are committed to the manifest only once complete,
so a run skips what is finished and continues from the next shard:

```
[Resume] 12 shards / 3000000 rows already written; skipping 3014892 dataset rows.
```

- Rows are skipped before any image is fetched, so resuming is cheap.
- A crash costs at most the shard in progress, which the next run rewrites. `Ctrl-C`
  commits the shard in progress before exiting.
- Raising `--num-embeddings` extends an existing run; re-running a finished one
  does nothing.
- `--chunk-rows` must be ≥ `--batch-size`, and shards end on batch boundaries, so
  they hold *at most* `--chunk-rows` rows.

Reviews resume by row offset within one streamed file, so `--chunk-rows` with
several `--categories` is refused. Shard those by category instead — one run each,
with its own `--output-prefix`:

```bash
for category in Books Electronics Movies_and_TV; do
    python generate_amazon_reviews_distilbert.py --categories "$category" \
        --num-embeddings 1000000 --output-prefix "amazon-reviews_distilbert_$category"
done
```

## Adding a combination

Copy the closest `generate_*.py` and change four things:

1. `Combination(key, dim, normalised)` — `dim` is what the encoder outputs;
   `normalised` says whether unit vectors are what `--write default` produces.
2. `iter_items()` — yield `(asset_id, item)` pairs. Honour `self.cursor.skip_rows`
   and call `self.cursor.note(row_ordinal)` per row so the combination stays
   resumable; `hf_streams.stream_images` / `stream_texts` already do both.
3. `encode(items)` — return a `(len(items), dim)` float32 array.
4. `run(source, COMBINATION, args)` in `__main__`.

Batching, sharding, the manifest, resume, normalisation and the shared CLI all
come from `common.py`; Hub streaming comes from `hf_streams.py`. Neither needs
changes to add a combination.
