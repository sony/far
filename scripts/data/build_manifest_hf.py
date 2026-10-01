"""
Build a FAR manifest from a HuggingFace WebDataset repository.

Lists all .tar shards under a given repo directory using the HuggingFace Hub
API and writes one manifest entry per shard.  No data is downloaded; the
WebDatasetBackend handles streaming and optional local caching at training time.

Manifest entry fields:
    shard_id     — shard filename stem (e.g. "shard-000000")
    source_type  — "webdataset"
    path         — HuggingFace download URL for the shard tar
    num_frames   — 0  (unknown until the shard is streamed)
    fps          — 0.0
    width/height — null
    dataset_name, split

Usage:
    python scripts/data/build_manifest_hf.py \\
        --repo_id      <hf-user>/<webdataset-repo> \\
        --path         val \\
        --split        val \\
        --output       results/manifests/loopnav_wds/manifest.json \\
        --dataset_name loopnav_wds

Set HF_TOKEN in the environment for private repositories.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class WDSShardSample:
    shard_id: str
    source_type: str      # "webdataset"
    path: str             # HuggingFace download URL
    num_frames: int       # 0 = unknown (resolved at load time)
    fps: float
    width: int | None
    height: int | None
    dataset_name: str
    split: str


def list_shards(repo_id: str, path: str, revision: str = "main") -> list[str]:
    from huggingface_hub import list_repo_files
    all_files = list_repo_files(repo_id, repo_type="dataset", revision=revision)
    if path:
        prefix = path.rstrip("/") + "/"
        return sorted(f for f in all_files if f.startswith(prefix) and f.endswith(".tar"))
    return sorted(f for f in all_files if f.endswith(".tar"))


def shard_url(repo_id: str, filename: str, revision: str = "main") -> str:
    from huggingface_hub import hf_hub_url
    return hf_hub_url(repo_id=repo_id, filename=filename, repo_type="dataset", revision=revision)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a FAR manifest from a HuggingFace WebDataset repo."
    )
    parser.add_argument("--repo_id", required=True,
                        help="HuggingFace dataset repo (e.g. <user>/<dataset-repo>)")
    parser.add_argument("--paths", nargs="*", default=[""],
                        help="Directories within the repo containing .tar shards (default: repo root). Multiple values allowed.")
    parser.add_argument("--split", required=True,
                        help="Split label to assign to all found shards (e.g. train, val)")
    parser.add_argument("--output", required=True,
                        help="Output manifest JSON path")
    parser.add_argument("--dataset_name", default="loopnav_wds",
                        help="Dataset name tag in manifest entries")
    parser.add_argument("--revision", default="main",
                        help="HuggingFace repo revision (default: main)")
    args = parser.parse_args()

    seen: set[str] = set()
    shards: list[str] = []
    for path in args.paths:
        for s in list_shards(args.repo_id, path, args.revision):
            if s not in seen:
                seen.add(s)
                shards.append(s)

    if not shards:
        paths_str = ", ".join(f"'{p or '(repo root)'}'" for p in args.paths)
        print(f"No .tar shards found under {paths_str} — exiting")
        return

    paths_str = ", ".join(f"'{p or '(repo root)'}'" for p in args.paths)
    print(f"Found {len(shards)} shard(s) under {paths_str}")

    samples: list[WDSShardSample] = []
    for filename in shards:
        url = shard_url(args.repo_id, filename, args.revision)
        shard_id = filename
        samples.append(WDSShardSample(
            shard_id=shard_id,
            source_type="webdataset",
            path=url,
            num_frames=0,
            fps=0.0,
            width=None,
            height=None,
            dataset_name=args.dataset_name,
            split=args.split,
        ))
        print(f"  {filename}  →  {url}")

    manifest = {
        "version": "v1",
        "num_samples": len(samples),
        "samples": [asdict(s) for s in samples],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote {len(samples)} shard entries → {output_path}")


if __name__ == "__main__":
    main()
