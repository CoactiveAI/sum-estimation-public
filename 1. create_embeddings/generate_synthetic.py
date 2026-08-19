"""Synthetic embeddings for local testing - no dataset, no encoder, no network.

Draws deterministic random Gaussian vectors, so the whole pipeline (insert ->
rarity levels -> experiments -> plots) can be exercised end to end on a laptop
without downloading Open Images or Amazon Reviews or loading an encoder. The
numbers are meaningless as embeddings; only the shapes and the plumbing are real.

By default it produces free-form vectors (`--dim`, prefix `synthetic_synthetic`).
Pass `--collection` to stand in for a specific real collection instead: the
dimensionality, the raw/normalised choice and the output prefix are then taken
from that collection's config, so step 2 accepts the vectors as-is.

Because runs are seeded, a given (`--seed`, `--dim`, `--num-embeddings`) always
produces the same matrix. Vectors are always drawn on CPU to keep that true;
`--device` is accepted for symmetry with the real encoders but unused here.

Examples
--------
Quick local run (10k x 768, raw):
    python generate_synthetic.py

Stand in for the CLIP collection (768-d unit vectors, Dot product):
    python generate_synthetic.py --collection open-images_clip_vit_l14_336

Stand in for the ResNet-50 collection (2048-d raw vectors):
    python generate_synthetic.py --collection open-images_resnet-50

Then load into Qdrant (step 2), which starts a local Docker Qdrant when
QDRANT_HOST is empty:
    python qdrant_insert.py --collection open-images_resnet-50 \\
        --embeddings-prefix synthetic_open-images_resnet-50
"""

import argparse
import itertools
from typing import Iterator, Sequence

import numpy as np
import torch

from common import Combination, StreamingEncoderSource, build_parser, run

#: (dim, normalised) per collection a synthetic run can stand in for, mirroring
#: `KNOWN_COLLECTION_CONFIGS` in `2. create qdrant cluster/qdrant_connection.py`.
MIMICKABLE = {
    "open-images_resnet-50": (2048, False),
    "open-images_clip_vit_l14_336": (768, True),
    "amazon-reviews_distilbert": (768, False),
}

#: Free-form default: 768-d raw vectors under the `synthetic_synthetic` prefix.
DEFAULT_DIM = 768
DEFAULT_NUM_EMBEDDINGS = 10_000


class SyntheticSource(StreamingEncoderSource):
    """Deterministic random Gaussian vectors.

    `iter_items` yields row indices and `encode` turns each batch of indices into
    vectors, so the batching and writing path is identical to the real encoders.
    The generator is consumed sequentially, which makes the output independent of
    `--batch-size` for a given seed.
    """

    def __init__(self, dim: int, seed: int):
        self.dim = dim
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def iter_items(self) -> Iterator[tuple[str, int]]:
        # Unbounded: `--num-embeddings` is what stops the stream. There is no real
        # asset behind a synthetic vector, so the id just records the row.
        start = self.cursor.skip_rows
        if start:
            # Burn the draws a previous run consumed, in slices, so a resumed run
            # continues the same sequence without allocating them all at once.
            remaining = start
            while remaining:
                step = min(remaining, 8192)
                torch.randn(step, self.dim, generator=self.generator)
                remaining -= step
        for row in itertools.count(start):
            self.cursor.note(row)
            yield f"synthetic-{row}", row

    def encode(self, items: Sequence) -> np.ndarray:
        batch = torch.randn(len(items), self.dim, generator=self.generator)
        return batch.to(torch.float32).numpy()


def parse_args() -> tuple[Combination, argparse.Namespace]:
    # `--collection` decides the dimensionality, the raw/normalised default and
    # the output prefix, so it has to be read before the main parser is built.
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--collection", choices=tuple(MIMICKABLE), default=None)
    preliminary, _ = bootstrap.parse_known_args()

    if preliminary.collection:
        dim, normalised = MIMICKABLE[preliminary.collection]
        # `collection_key` doubles as the default output prefix, so the
        # `synthetic_` marker keeps stand-in vectors distinguishable from real ones.
        combination = Combination(f"synthetic_{preliminary.collection}", dim, normalised)
    else:
        combination = Combination("synthetic_synthetic", DEFAULT_DIM, False)

    parser = build_parser(combination, __doc__)
    parser.set_defaults(num_embeddings=DEFAULT_NUM_EMBEDDINGS)
    parser.add_argument(
        "--collection", choices=tuple(MIMICKABLE), default=None,
        help="Produce vectors shaped for this real collection instead of the "
             "free-form default (sets dimensionality, normalisation and prefix).",
    )
    parser.add_argument(
        "--dim", type=int, default=None,
        help=f"Embedding dimensionality (default: {DEFAULT_DIM}). Ignored when "
             f"--collection is given, which fixes the dimensionality.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed for deterministic generation (default: 0).",
    )
    args = parser.parse_args()

    if args.collection and args.dim is not None and args.dim != combination.dim:
        raise SystemExit(
            f"--dim {args.dim} conflicts with --collection {args.collection}, "
            f"which requires {combination.dim}-d vectors. Drop --dim."
        )
    return combination, args


if __name__ == "__main__":
    combination, args = parse_args()
    dim = combination.dim if args.collection else (args.dim or DEFAULT_DIM)
    source = SyntheticSource(dim=dim, seed=args.seed)
    # Free-form runs may use any dimensionality, so the collection stand-in that
    # `run` validates against is rebuilt from what was actually requested.
    run(source, Combination(combination.collection_key, dim, combination.normalised), args)
