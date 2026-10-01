"""Build the SoundSpaces latent manifest + per-split index files.

Scans the latent dataset produced by `build_latent_dataset_soundspaces.py`
(one dir per episode: latents.npy + npz/json sidecars) and writes:
    --manifest-out   {"samples": [{video_id, source_type, path, dataset_name,
                                   split}, ...]}
    --index-dir/soundspaces_latent_{train,test}.json   {"sample_ids": [...]}

SPLITS: two splits only, mapped directly from each episode's meta.json
`split` field — train -> train, val/test -> test. Scene-disjointness is
inherited from the dataset's native carve (and asserted here).

    python scripts/data/build_manifest_soundspaces.py \\
        --latent-root datasets/soundspaces/latent \\
        --manifest-out results/manifests/soundspaces/soundspaces_latent.json \\
        --index-dir    results/indices/soundspaces
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--latent-root", default="datasets/soundspaces/latent")
    parser.add_argument("--manifest-out",
                        default="results/manifests/soundspaces/soundspaces_latent.json")
    parser.add_argument("--index-dir", default="results/indices/soundspaces")
    parser.add_argument("--dataset-name", default="soundspaces")
    args = parser.parse_args()

    root = Path(args.latent_root)
    metas, skipped = {}, 0
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if not (d / "latents.npy").is_file() or not (d / "meta.json").is_file():
            skipped += 1
            continue
        metas[d] = json.loads((d / "meta.json").read_text())
    assert metas, f"no complete episodes under {root}"

    samples = [{
        "video_id": d.name,
        "source_type": "soundspaces_latent",
        "path": str(d.resolve()),
        "dataset_name": args.dataset_name,
        "split": "train" if m.get("split", "train") == "train" else "test",
    } for d, m in metas.items()]
    scenes = defaultdict(set)
    for d, m in metas.items():
        scenes["train" if m.get("split", "train") == "train" else "test"].add(m["scene"])
    assert not (scenes["train"] & scenes["test"]), \
        f"scene overlap across splits: {scenes['train'] & scenes['test']}"
    print(f"scenes: train {len(scenes['train'])}, test {len(scenes['test'])} "
          f"(test episode fraction {sum(s['split'] == 'test' for s in samples) / len(samples):.4f})")

    out = Path(args.manifest_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"samples": samples}, indent=1))

    idx_dir = Path(args.index_dir)
    idx_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter(s["split"] for s in samples)
    for split in counts:
        ids = [s["video_id"] for s in samples if s["split"] == split]
        (idx_dir / f"soundspaces_latent_{split}.json").write_text(
            json.dumps({"sample_ids": ids}, indent=1))
    print(f"manifest: {out} ({len(samples)} episodes, {skipped} incomplete skipped)")
    print("indices:", dict(counts))


if __name__ == "__main__":
    main()
