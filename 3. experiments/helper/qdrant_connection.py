"""Resolve a Qdrant client for the experiments.

Same three targets as step 2 - a cloud cluster, a local Docker server, or the
serverless embedded client - so an experiment can be smoke-tested with no server
running. Embedded mode answers by exact search, which is fine for checking the
estimators but tells you nothing about approximate-search recall.
"""

from __future__ import annotations

import urllib.request

from qdrant_client import QdrantClient

from .config import settings

MODES = ("auto", "cloud", "docker", "embedded")

#: Distance and vector variant per collection, matching what step 2 created.
#: Local scoring has to reproduce the same score Qdrant would return.
KNOWN_DISTANCES = {
    "open-images_resnet-50": {"distance": "Euclid", "normalised": False},
    "open-images_clip_vit_l14_336": {"distance": "Dot", "normalised": True},
    "amazon-reviews_distilbert": {"distance": "Euclid", "normalised": False},
}


def _server_ready(url: str) -> bool:
    for endpoint in ("/readyz", "/healthz"):
        try:
            with urllib.request.urlopen(f"{url}{endpoint}", timeout=2) as response:
                if response.status == 200:
                    return True
        except Exception:
            continue
    return False


def connect(mode: str = "auto", embedded_path: str | None = None) -> QdrantClient:
    """Return a client for the requested mode; 'auto' prefers cloud, then Docker."""
    if mode not in MODES:
        raise SystemExit(f"Unknown --qdrant mode '{mode}'. Choose from: {', '.join(MODES)}.")
    if mode == "auto":
        mode = "cloud" if settings.QDRANT_HOST else "docker"

    if mode == "cloud":
        if not settings.QDRANT_HOST:
            raise SystemExit("--qdrant cloud needs QDRANT_HOST set in .env.")
        print(f"[Qdrant] Cloud cluster at {settings.QDRANT_HOST}")
        return QdrantClient(
            url=settings.QDRANT_HOST,
            api_key=settings.QDRANT_API_KEY or None,
            port=settings.QDRANT_PORT,
            timeout=settings.QDRANT_TIMEOUT,
        )

    if mode == "docker":
        url = f"http://localhost:{settings.LOCAL_QDRANT_PORT}"
        if not _server_ready(url):
            raise SystemExit(
                f"No Qdrant answering at {url}. Start one by indexing with step 2 "
                f"(`qdrant_insert.py --qdrant docker`), or use --qdrant embedded."
            )
        print(f"[Qdrant] Local server at {url}")
        return QdrantClient(url=url, timeout=settings.QDRANT_TIMEOUT)

    location = embedded_path or settings.QDRANT_EMBEDDED_PATH
    if location in ("", ":memory:"):
        print("[Qdrant] Embedded in-memory (exact search)")
        return QdrantClient(location=":memory:")
    print(f"[Qdrant] Embedded at {location} (exact search)")
    return QdrantClient(path=location)


def add_qdrant_args(parser) -> None:
    parser.add_argument(
        "--qdrant", choices=MODES, default="auto",
        help="Where to query: 'auto' (cloud when QDRANT_HOST is set, else a local "
             "Docker server), or 'embedded' for a serverless client.",
    )
    parser.add_argument(
        "--embedded-path", default=None,
        help="Directory for --qdrant embedded (default: QDRANT_EMBEDDED_PATH).",
    )
