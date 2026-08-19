"""Open Images x CLIP ViT-L/14-336 -> 768-d embeddings.

Collection: `open-images_clip_vit_l14_336`.

Streams Open Images from the Hugging Face Hub and encodes each image with the
CLIP image tower (`openai/clip-vit-large-patch14-336`), taking the 768-d
projected image features. The target collection is a Dot-product collection over
unit vectors, so `--write` defaults to the L2-normalised file only; pass
`--write both` if you also want the raw features.

Open Images has no official Hub mirror; the default (`bitmind/open-images-v7`)
is a community mirror whose `url` column holds Flickr image URLs, which this
script downloads on the fly. Point `--hf-dataset` / `--image-column` at any
other mirror - including one with embedded image bytes - to avoid the downloads.

Examples
--------
100k unit vectors on a GPU:
    python generate_open_images_clip_vit_l14_336.py --num-embeddings 100000 --device cuda

Raw features as well:
    python generate_open_images_clip_vit_l14_336.py --write both

Then load them into Qdrant (step 2):
    python qdrant_insert.py --collection open-images_clip_vit_l14_336 \\
        --embeddings-prefix open-images_clip_vit_l14_336
"""

from typing import Sequence

import numpy as np
import torch

from common import Combination, StreamingEncoderSource, build_parser, resolve_device, run
from hf_streams import add_image_stream_args, load_stream, stream_images

COMBINATION = Combination(collection_key="open-images_clip_vit_l14_336", dim=768, normalised=True)

DEFAULT_MODEL = "openai/clip-vit-large-patch14-336"
DEFAULT_HF_DATASET = "bitmind/open-images-v7"
DEFAULT_SPLIT = "train"
DEFAULT_IMAGE_COLUMN = "url"


class OpenImagesClipSource(StreamingEncoderSource):
    """CLIP ViT-L/14-336 projected image features for Open Images."""

    dim = 768

    def __init__(self, args):
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

        self.args = args
        self.device = resolve_device(args.device)

        self.processor = CLIPImageProcessor.from_pretrained(args.model)
        # Vision tower + projection head only; the text tower is never used.
        self.model = (
            CLIPVisionModelWithProjection.from_pretrained(args.model).eval().to(self.device)
        )
        projection_dim = self.model.config.projection_dim
        if projection_dim != self.dim:
            raise SystemExit(
                f"Model '{args.model}' projects to {projection_dim}-d, but this "
                f"combination expects {self.dim}-d."
            )
        print(f"[Model] {args.model} on {self.device}")

    def iter_items(self):
        rows = load_stream(self.args.hf_dataset, self.args.hf_config, self.args.split)
        return stream_images(
            rows,
            column=self.args.image_column,
            id_column=self.args.id_column,
            cursor=self.cursor,
            download_workers=self.args.download_workers,
            timeout=self.args.download_timeout,
        )

    @torch.no_grad()
    def encode(self, items: Sequence) -> np.ndarray:
        inputs = self.processor(images=list(items), return_tensors="pt").to(self.device)
        return self.model(**inputs).image_embeds.float().cpu().numpy()


def parse_args():
    parser = build_parser(COMBINATION, __doc__)
    add_image_stream_args(parser, DEFAULT_HF_DATASET, DEFAULT_SPLIT, DEFAULT_IMAGE_COLUMN)
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"CLIP checkpoint to encode with (default: {DEFAULT_MODEL}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(OpenImagesClipSource(args), COMBINATION, args)
