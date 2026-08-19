"""Amazon Reviews x DistilBERT -> 768-d embeddings.

Collection: `amazon-reviews_distilbert`.

Streams review text from the Hugging Face Hub (`McAuley-Lab/Amazon-Reviews-2023`)
and encodes it with `distilbert-base-uncased`, mean-pooling the final hidden
states over non-padding tokens. The target collection uses Euclidean distance
over raw vectors, so `--write` defaults to raw only.

Categories are streamed in the order given and concatenated, so reaching tens of
millions of reviews is a matter of listing enough of them. The default set is
the largest categories first; `--categories` takes any comma-separated list of
category names from the dataset's `raw/review_categories/` files.

Examples
--------
100k review vectors:
    python generate_amazon_reviews_distilbert.py --num-embeddings 100000 --device cuda

Specific categories, CLS pooling:
    python generate_amazon_reviews_distilbert.py \\
        --categories Books,Movies_and_TV --pooling cls

Then load them into Qdrant (step 2):
    python qdrant_insert.py --collection amazon-reviews_distilbert \\
        --embeddings-prefix amazon-reviews_distilbert
"""

from typing import Sequence

import numpy as np
import torch

from common import Combination, StreamingEncoderSource, build_parser, resolve_device, run
from hf_streams import load_stream, stream_texts

COMBINATION = Combination(collection_key="amazon-reviews_distilbert", dim=768, normalised=False)

DEFAULT_MODEL = "distilbert-base-uncased"
DEFAULT_HF_DATASET = "McAuley-Lab/Amazon-Reviews-2023"
DEFAULT_CATEGORIES = "Books,Home_and_Kitchen,Clothing_Shoes_and_Jewelry,Electronics,Movies_and_TV"
DEFAULT_TEXT_COLUMNS = "title,text"
#: A review has no single natural key; these three together identify one.
DEFAULT_ID_COLUMNS = "user_id,asin,timestamp"


class AmazonReviewsDistilBertSource(StreamingEncoderSource):
    """Mean-pooled (or CLS) DistilBERT encodings of Amazon review text."""

    dim = 768

    def __init__(self, args):
        from transformers import AutoModel, AutoTokenizer

        self.args = args
        self.device = resolve_device(args.device)

        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.model = AutoModel.from_pretrained(args.model).eval().to(self.device)
        hidden = self.model.config.hidden_size
        if hidden != self.dim:
            raise SystemExit(
                f"Model '{args.model}' has hidden size {hidden}, but this "
                f"combination expects {self.dim}."
            )
        print(f"[Model] {args.model} ({args.pooling} pooling) on {self.device}")

    def iter_items(self):
        columns = [c.strip() for c in self.args.text_columns.split(",") if c.strip()]
        id_columns = [c.strip() for c in self.args.id_columns.split(",") if c.strip()]
        # Read the per-category JSONL files directly, which avoids the dataset's
        # loading script (and therefore trust_remote_code) entirely.
        for category in [c.strip() for c in self.args.categories.split(",") if c.strip()]:
            path = f"hf://datasets/{self.args.hf_dataset}/raw/review_categories/{category}.jsonl"
            print(f"\n[Stream] Reading category '{category}'")
            rows = load_stream(self.args.hf_dataset, None, "train", data_files=[path])
            # Prefix the category so ids stay unique across concatenated files.
            for identifier, text in stream_texts(
                rows, columns, id_columns=id_columns, cursor=self.cursor,
                min_chars=self.args.min_chars,
            ):
                yield f"{category}|{identifier}", text

    @torch.no_grad()
    def encode(self, items: Sequence) -> np.ndarray:
        inputs = self.tokenizer(
            list(items),
            padding=True,
            truncation=True,
            max_length=self.args.max_length,
            return_tensors="pt",
        ).to(self.device)
        hidden_states = self.model(**inputs).last_hidden_state

        if self.args.pooling == "cls":
            pooled = hidden_states[:, 0]
        else:
            mask = inputs["attention_mask"].unsqueeze(-1).to(hidden_states.dtype)
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return pooled.float().cpu().numpy()


def parse_args():
    parser = build_parser(COMBINATION, __doc__)
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Text encoder checkpoint (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--hf-dataset", default=DEFAULT_HF_DATASET,
        help=f"Hub dataset id (default: {DEFAULT_HF_DATASET}).",
    )
    parser.add_argument(
        "--categories", default=DEFAULT_CATEGORIES,
        help="Comma-separated review categories, streamed in order "
             f"(default: {DEFAULT_CATEGORIES}).",
    )
    parser.add_argument(
        "--text-columns", default=DEFAULT_TEXT_COLUMNS,
        help=f"Comma-separated fields joined per review (default: {DEFAULT_TEXT_COLUMNS}).",
    )
    parser.add_argument(
        "--id-columns", default=DEFAULT_ID_COLUMNS,
        help="Comma-separated fields joined with '|' to identify each review; the "
             f"category is prepended (default: {DEFAULT_ID_COLUMNS}).",
    )
    parser.add_argument(
        "--pooling", choices=("mean", "cls"), default="mean",
        help="How to pool token states into one vector (default: mean).",
    )
    parser.add_argument(
        "--max-length", type=int, default=256,
        help="Maximum tokens per review (default: 256).",
    )
    parser.add_argument(
        "--min-chars", type=int, default=1,
        help="Skip reviews shorter than this many characters (default: 1).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    if args.chunk_rows > 0 and len(categories) > 1:
        raise SystemExit(
            "--chunk-rows resumes from a row ordinal within a single streamed file, "
            f"but {len(categories)} categories were given. Shard per category "
            "instead: one run each with --categories <one> and its own "
            "--output-prefix."
        )
    run(AmazonReviewsDistilBertSource(args), COMBINATION, args)
