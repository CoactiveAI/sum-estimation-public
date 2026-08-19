"""Open Images x ResNet-50 -> 2048-d embeddings (collection `open-images_resnet-50`).

Streams Open Images from the Hugging Face Hub and encodes each image with the
torchvision ResNet-50 classifier truncated at the global average pool, giving
the 2048-d penultimate feature vector. The target collection uses Euclidean
distance over raw (un-normalised) vectors, so `--write` defaults to raw only.

Open Images has no official Hub mirror; the default (`bitmind/open-images-v7`)
is a community mirror whose `url` column holds Flickr image URLs, which this
script downloads on the fly. Point `--hf-dataset` / `--image-column` at any
other mirror - including one with embedded image bytes - to avoid the downloads.

Examples
--------
100k vectors on a GPU:
    python generate_open_images_resnet50.py --num-embeddings 100000 --device cuda

A different mirror that stores decoded images:
    python generate_open_images_resnet50.py \\
        --hf-dataset your-org/open-images --image-column image

Then load them into Qdrant (step 2):
    python qdrant_insert.py --collection open-images_resnet-50 \\
        --embeddings-prefix open-images_resnet-50
"""

from typing import Sequence

import numpy as np
import torch

from common import Combination, StreamingEncoderSource, build_parser, resolve_device, run
from hf_streams import add_image_stream_args, load_stream, stream_images

COMBINATION = Combination(collection_key="open-images_resnet-50", dim=2048, normalised=False)

DEFAULT_HF_DATASET = "bitmind/open-images-v7"
DEFAULT_SPLIT = "train"
DEFAULT_IMAGE_COLUMN = "url"


class OpenImagesResNet50Source(StreamingEncoderSource):
    """ResNet-50 penultimate-layer (post-avgpool) features for Open Images."""

    dim = 2048

    def __init__(self, args):
        from torchvision.models import ResNet50_Weights, resnet50

        self.args = args
        self.device = resolve_device(args.device)

        weights = ResNet50_Weights[args.weights]
        model = resnet50(weights=weights)
        # Drop the 1000-way classifier: the avgpool output is the embedding.
        model.fc = torch.nn.Identity()
        self.model = model.eval().to(self.device)
        self.preprocess = weights.transforms()
        print(f"[Model] ResNet-50 ({args.weights}) on {self.device}")

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
        batch = torch.stack([self.preprocess(image) for image in items]).to(self.device)
        return self.model(batch).float().cpu().numpy()


def parse_args():
    parser = build_parser(COMBINATION, __doc__)
    add_image_stream_args(parser, DEFAULT_HF_DATASET, DEFAULT_SPLIT, DEFAULT_IMAGE_COLUMN)
    parser.add_argument(
        "--weights", default="IMAGENET1K_V2",
        help="torchvision ResNet50_Weights member (default: IMAGENET1K_V2).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(OpenImagesResNet50Source(args), COMBINATION, args)
