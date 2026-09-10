"""Load the per-level maxima step 2 recorded for each collection.

Step 2 writes `<HNSW_INDEX_DIR>/<collection>/max_levels.json` while indexing. The
levels are queried by exact value, so the maximum is the iteration bound: a level
field takes values `1 .. max`, and `OurAlgorithm` issues one query per value.

These were previously hard-coded in `qdrant_helpers.py`, which silently went stale
whenever a collection was re-indexed with fresh level draws.
"""

from __future__ import annotations

import json
import os

from .config import settings


class MaxLevels(dict):
    """Collection -> {level_i: max}, with an explanation when one is missing."""

    def __missing__(self, collection_name: str):
        looked_at = os.path.join(settings.HNSW_INDEX_DIR, collection_name, "max_levels.json")
        known = ", ".join(sorted(self)) or "(none loaded)"
        raise SystemExit(
            f"No max levels for collection '{collection_name}'.\n"
            f"Expected {looked_at}, written by step 2 when the collection is indexed.\n"
            f"Loaded so far: {known}"
        )


def load_max_levels(index_dir: str | None = None) -> MaxLevels:
    """Read every `<collection>/max_levels.json` under the index directory."""
    index_dir = index_dir or settings.HNSW_INDEX_DIR
    loaded = MaxLevels()
    if not os.path.isdir(index_dir):
        return loaded

    for entry in sorted(os.listdir(index_dir)):
        path = os.path.join(index_dir, entry, "max_levels.json")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        # Each file is keyed by its collection name, so several merge cleanly.
        for collection_name, maxima in payload.items():
            loaded[collection_name] = {field: int(value) for field, value in maxima.items()}
    return loaded
