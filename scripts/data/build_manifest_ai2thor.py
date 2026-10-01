"""Build the AI2-THOR latent manifest + per-split index files.

Scans the latent dataset produced by `build_latent_dataset_ai2thor.py`
(one dir per episode: latents.npy + npz/json sidecars) and writes:
    --manifest-out   {"samples": [{video_id, source_type, path, dataset_name,
                                   split, kind}, ...]}
    --index-dir/ai2thor_latent_{train,test}.json   {"sample_ids": [...]}

SPLITS: two splits only, mapped directly from each episode's meta.json
`split` field — train -> train, val/test -> test. Scene-disjointness is
inherited from the dataset's native carve (and asserted here: 95 train
scenes vs 24 test scenes, zero overlap at the time of writing).

KINDS: `<traj>_act` episodes fire their interaction chains; `<traj>_ctrl` are
the seed-matched STATIC siblings (identical scene, trajectory and choreography,
no state change) that exist purely as paired eval controls. The generator
already emits every ctrl episode into the test split, which this script
asserts — a ctrl episode in train would leak a near-duplicate of a test
trajectory into training. `--kind` additionally restricts what lands in the
indices: `act` is the right choice for retriever pre-training and world-model
training (a ctrl sibling is a near-duplicate trajectory, which inflates
contrastive recall and wastes gradient), while `both` keeps the controls
available to the eval batteries.

    python scripts/data/build_manifest_ai2thor.py \\
        --latent-root datasets/ai2thor/latent \\
        --manifest-out results/manifests/ai2thor/ai2thor_latent.json \\
        --index-dir    results/indices/ai2thor
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--latent-root", default="datasets/ai2thor/latent")
    parser.add_argument("--manifest-out",
                        default="results/manifests/ai2thor/ai2thor_latent.json")
    parser.add_argument("--index-dir", default="results/indices/ai2thor")
    parser.add_argument("--dataset-name", default="ai2thor")
    parser.add_argument("--index-prefix", default="ai2thor_latent",
                        help="Index file stem: <index-dir>/<prefix>_{train,test}.json "
                             "(e.g. ai2thor_v2_latent for the v6 corpus)")
    parser.add_argument("--kind", choices=("act", "both"), default="act",
                        help="Which episodes enter the INDEX files (the manifest "
                             "always lists everything found): act = interaction "
                             "episodes only (training/pre-training default); "
                             "both = also the static _ctrl eval controls")
    parser.add_argument("--split-field", default="split",
                        help="meta.json field carrying the split (default `split`). The "
                             "door-corridor corpus also stores `split_seed` (the original "
                             "seed%%5 carve: every scene in both splits, all ctrl twins in "
                             "test) next to the scene-disjoint `split`.")
    parser.add_argument("--allow-scene-overlap", action="store_true",
                        help="Downgrade the scene-disjointness check to a warning — for a "
                             "SEED-based carve where every scene appears in both splits "
                             "(in-scene protocol: unseen episodes of seen scenes).")
    parser.add_argument("--allow-ctrl-in-train", action="store_true",
                        help="Permit _ctrl episodes in the TRAIN split. The THOR "
                             "generator emits every ctrl into test (asserted by "
                             "default); a scene-disjoint corpus (door-corridor "
                             "SPLITS.json) instead keeps train-scene ctrl twins "
                             "in train — there they are the no-agent negatives "
                             "that teach conditional rendering, not test leaks "
                             "(test scenes are held out wholesale).")
    args = parser.parse_args()

    root = Path(args.latent_root)
    metas = {}
    skipped = 0
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if not (d / "latents.npy").is_file() or not (d / "meta.json").is_file():
            skipped += 1                       # still encoding, or a stray dir
            continue
        metas[d] = json.loads((d / "meta.json").read_text())
    assert metas, f"no complete episodes under {root}"

    samples = [{
        "video_id": d.name,
        "source_type": "ai2thor_latent",
        "path": str(d),
        "dataset_name": args.dataset_name,
        "split": "train" if m.get(args.split_field, "train") == "train" else "test",
        "kind": "ctrl" if d.name.endswith("_ctrl") else "act",
    } for d, m in metas.items()]

    # Scene-disjointness: the generalization claim rests on held-out SCENES,
    # so a shared FloorPlan across splits must fail loudly, not silently.
    scenes = defaultdict(set)
    for s, m in zip(samples, metas.values()):
        scenes[s["split"]].add(m["scene"])
    overlap = scenes["train"] & scenes["test"]
    if overlap and args.allow_scene_overlap:
        print(f"  [info] {len(overlap)} scene(s) in BOTH splits (seed carve, field `{args.split_field}`): "
              f"in-scene protocol — unseen episodes of seen scenes")
    else:
        assert not overlap, f"scene overlap across splits: {sorted(overlap)}"

    # Static controls are eval-only by construction; one in train would put a
    # near-duplicate of a test trajectory into the training set.
    leaked = [s["video_id"] for s in samples
              if s["kind"] == "ctrl" and s["split"] == "train"]
    if leaked and args.allow_ctrl_in_train:
        print(f"  [info] {len(leaked)} _ctrl episodes in train (scene-disjoint "
              f"split: no-agent negatives, twins of TRAIN-scene trajectories)")
    else:
        assert not leaked, f"{len(leaked)} _ctrl episodes in the train split, e.g. {leaked[:3]}"

    manifest_path = Path(args.manifest_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"samples": samples}, indent=1))

    idx_dir = Path(args.index_dir)
    idx_dir.mkdir(parents=True, exist_ok=True)
    indexed = [s for s in samples if args.kind == "both" or s["kind"] == "act"]
    counts = Counter(s["split"] for s in indexed)
    # BOTH index files are always written, even when a split is still empty:
    # the dataset config references them by path, so a missing file turns a
    # partially-encoded corpus into an opaque instantiate error instead of an
    # obvious "0 samples". Rebuild once encoding finishes.
    for split in ("train", "test"):
        ids = [s["video_id"] for s in indexed if s["split"] == split]
        (idx_dir / f"{args.index_prefix}_{split}.json").write_text(
            json.dumps({"sample_ids": sorted(ids)}, indent=1))
        if not ids:
            print(f"  [warn] {split} index is EMPTY — encoding still in progress? "
                  f"re-run this script when build_latent_dataset_ai2thor.py completes")

    kinds = Counter((s["kind"], s["split"]) for s in samples)
    print(f"manifest: {len(samples)} episodes -> {manifest_path}"
          + (f"  ({skipped} incomplete dirs skipped)" if skipped else ""))
    print(f"  scenes: train {len(scenes['train'])} / test {len(scenes['test'])} ({'overlap ' + str(len(overlap)) if overlap else 'disjoint'})")
    print(f"  kind x split: {dict(kinds)}")
    print(f"  indices (--kind {args.kind}): "
          + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
          + f" -> {idx_dir}")


if __name__ == "__main__":
    main()
