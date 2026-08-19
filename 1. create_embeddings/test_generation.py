"""Smoke-test every generation script end to end, small and fast.

Each script is run twice with tiny sizes: once to check it produces correctly
shaped shards, then again with a higher target to check the re-run resumes rather
than starting over. Output goes to `embeddings/test_run/<script>/`, kept for
inspection and never mixed with real generation output.

    python test_generation.py                # all scripts
    python test_generation.py synthetic      # only scripts matching a substring
    python test_generation.py --clean        # delete embeddings/test_run/ afterwards

The image scripts download model weights on first use (CLIP is ~1.7 GB) and fetch
images over HTTP, so they take minutes; `synthetic` needs no network at all.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

from common import DEFAULT_OUTPUT_DIR

HERE = os.path.dirname(os.path.abspath(__file__))

#: Test output lives beside real embeddings but in its own directory, so a smoke
#: run can never be mistaken for - or overwrite - a real one.
TEST_RUN_DIR = os.path.join(DEFAULT_OUTPUT_DIR, "test_run")

# Deliberately tiny: 12 rows in shards of 8 means two shards, so shard geometry
# and the resume path are exercised in seconds rather than hours.
ROWS = 12
RESUMED_ROWS = 20
BATCH_SIZE = 4
CHUNK_ROWS = 8

#: script -> (expected dim, expected variant, extra args)
CASES = {
    "generate_synthetic.py": (768, "raw", []),
    # All_Beauty is one of the smallest review categories, so it streams quickly.
    "generate_amazon_reviews_distilbert.py": (768, "raw", ["--categories", "All_Beauty"]),
    "generate_open_images_resnet50.py": (2048, "raw", []),
    "generate_open_images_clip_vit_l14_336.py": (768, "normalised", []),
}


def run_script(script, output_dir, num_embeddings, extra):
    """Run one generation script; return its combined output."""
    command = [
        sys.executable, script,
        "--num-embeddings", str(num_embeddings),
        "--batch-size", str(BATCH_SIZE),
        "--chunk-rows", str(CHUNK_ROWS),
        "--output-dir", output_dir,
        "--output-prefix", "smoke",
    ] + extra
    completed = subprocess.run(
        command, cwd=HERE, capture_output=True, text=True,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"{script} exited {completed.returncode}\n"
            f"--- stdout ---\n{completed.stdout[-2000:]}\n"
            f"--- stderr ---\n{completed.stderr[-2000:]}"
        )
    return completed.stdout + completed.stderr


def load_run(output_dir, variant):
    """Read a sharded run back: (manifest, vectors, ids)."""
    with open(os.path.join(output_dir, "smoke_manifest.json")) as f:
        manifest = json.load(f)
    vectors, ids = [], []
    for shard in manifest["chunks"]:
        vectors.append(np.load(os.path.join(output_dir, shard["files"][variant])))
        with open(os.path.join(output_dir, shard["files"]["ids"])) as f:
            ids += f.read().splitlines()
    return manifest, np.concatenate(vectors), ids


def check(script, dim, variant, extra, output_dir):
    """Generate, verify, then resume and verify again."""
    run_script(script, output_dir, ROWS, extra)
    manifest, vectors, ids = load_run(output_dir, variant)

    rows = [shard["rows"] for shard in manifest["chunks"]]
    assert len(rows) > 1, f"expected more than one shard, got {rows}"
    assert max(rows) <= CHUNK_ROWS, f"shard exceeds --chunk-rows: {rows}"
    assert sum(rows) == manifest["rows"] == len(ids), f"{rows} vs {manifest['rows']} vs {len(ids)}"
    assert vectors.shape == (ROWS, dim), f"expected {(ROWS, dim)}, got {vectors.shape}"
    assert vectors.dtype == np.float32, vectors.dtype
    assert np.isfinite(vectors).all(), "non-finite values in output"
    assert (np.abs(vectors).sum(axis=1) > 0).all(), "all-zero row (unwritten shard tail?)"
    assert len(set(ids)) == len(ids), "duplicate asset ids"
    if variant == "normalised":
        norms = np.linalg.norm(vectors, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5), f"not unit vectors: {norms[:3]}"

    # Re-running with a higher target must resume, keeping what is already on disk.
    output = run_script(script, output_dir, RESUMED_ROWS, extra)
    assert "[Resume]" in output, "second run did not resume"
    resumed_manifest, resumed_vectors, resumed_ids = load_run(output_dir, variant)
    assert resumed_manifest["rows"] == RESUMED_ROWS, resumed_manifest["rows"]
    assert np.array_equal(resumed_vectors[:ROWS], vectors), "resume rewrote earlier rows"
    assert resumed_ids[:ROWS] == ids, "resume shifted earlier ids"
    assert len(set(resumed_ids)) == RESUMED_ROWS, "duplicate ids across the resume seam"

    return f"{len(rows)}+{len(resumed_manifest['chunks']) - len(rows)} shards, {RESUMED_ROWS} rows"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pattern", nargs="?", default="",
                        help="Only run scripts whose name contains this substring.")
    parser.add_argument("--clean", action="store_true",
                        help="Delete embeddings/test_run/ when finished.")
    args = parser.parse_args()

    scripts = [s for s in CASES if args.pattern in s]
    if not scripts:
        raise SystemExit(f"No script matches {args.pattern!r}. Known: {', '.join(CASES)}")

    print(f"Output: {TEST_RUN_DIR}\n")
    results = []
    for script in scripts:
        dim, variant, extra = CASES[script]
        output_dir = os.path.join(TEST_RUN_DIR, script.replace(".py", ""))
        # Start clean: a manifest left by an earlier smoke run would otherwise be
        # resumed, and the first-run assertions expect a fresh set of shards.
        shutil.rmtree(output_dir, ignore_errors=True)
        print(f"[Test] {script} ...", flush=True)
        started = time.time()
        try:
            detail = check(script, dim, variant, extra, output_dir)
            results.append((script, "PASS", f"{detail}, {time.time() - started:.0f}s"))
        except AssertionError as error:
            results.append((script, "FAIL", str(error)))
        print()

    print("=" * 72)
    for script, status, detail in results:
        print(f"{status:4}  {script:44} {detail if status == 'PASS' else ''}")
        if status == "FAIL":
            print(detail)
    if args.clean:
        shutil.rmtree(TEST_RUN_DIR, ignore_errors=True)
        print(f"\nRemoved {TEST_RUN_DIR}")
    else:
        print(f"\nOutput left in {TEST_RUN_DIR}")
    raise SystemExit(any(status == "FAIL" for _, status, _ in results))


if __name__ == "__main__":
    main()
