"""Build the EVENT INDEX for an AI2-THOR latent corpus: the subset of an index's
clips on which `ContextGoalSampler._event_draw` can actually anchor a pair.

Motivation (measured on ai2thor_v2 train, event_phase=eval): 79% of act clips
have NO eval-leg interaction span at all — the eval leg only re-opens container
stations — so the sampler's per-clip Bernoulli(event_bias_p) roll silently
falls back to a uniform draw on them and the realized event-pair rate is
p x 21%, not p. The fix is to bias CLIP selection: `VideoDataset` (event_frac /
event_index_path) substitutes, with probability p, a clip drawn from THIS index
and forces the event draw on it — making the marginal event-pair rate exactly p.

Eligibility is `far.data.samplers.eligible_event_spans` — the SAME predicate
`_event_draw` applies at train time, so this artifact cannot drift from the
sampler's behaviour; if the event params change, rebuild with matching flags
(they are recorded in the output for the consumer to verify).

    python scripts/data/build_event_index_ai2thor.py \
        --manifest results/manifests/ai2thor_v2/ai2thor_v2_latent.json \
        --index results/indices/ai2thor_v2/ai2thor_v2_latent_train.json \
        --output results/indices/ai2thor_v2/ai2thor_v2_latent_train_events.json
"""

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from far.data.datasets import _event_spans
from far.data.samplers import eligible_event_spans

INTERACT_CODES = [1, 2, 3, 4]                       # open / close / pickup / put
CODE_NAMES = {0: "open", 1: "close", 2: "pickup", 3: "put"}   # span codes are COLUMN indices


def scan_clip(sample: dict, params: dict):
    """(video_id, eligible spans) — spans rebuilt exactly as the backend does:
    interaction spans from actions.npz, or (--spans reveal) the door-corridor
    REVEAL ONSETS from meta.json's legs via THORDynBackend.reveal_spans."""
    d = Path(sample["path"])
    meta = json.loads((d / "meta.json").read_text())
    if params["spans"] == "reveal":
        from far.data.backends import THORDynBackend
        spans = THORDynBackend.reveal_spans(meta, params["reveal_start_offset"])
    else:
        interact = np.load(d / "actions.npz")["interact"][: int(meta["n_frames"])]
        onehot = np.stack([(interact == c) for c in INTERACT_CODES], axis=1)
        spans = _event_spans(np.cumsum(onehot, axis=0).astype(np.float32))
    eligible = eligible_event_spans(spans, int(meta["n_frames"]), int(meta["eval_start"]),
                                    params["event_codes"], params["event_phase"],
                                    params["event_pre"], params["event_post"],
                                    params["event_goal_anchor"])
    return sample["video_id"], eligible


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--index", type=Path, required=True,
                    help="Source index whose sample_ids are scanned (e.g. the train index)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--event-phase", default="eval", choices=["any", "eval"])
    ap.add_argument("--event-codes", type=int, nargs="*", default=None,
                    help="Interaction span columns that qualify (0 open / 1 close / 2 pickup / 3 put — the sampler's event_codes convention); default all")
    ap.add_argument("--event-pre", type=int, nargs=2, default=[-3, 5])
    ap.add_argument("--event-post", type=int, nargs=2, default=[0, 8])
    ap.add_argument("--event-goal-anchor", default="start", choices=["start", "end"])
    ap.add_argument("--spans", default="interact", choices=["interact", "reveal"],
                    help="interact = interaction spans from actions.npz (TOUR corpora); reveal = the door-corridor "
                         "reveal onsets (code 0 look_A, 1 look_B) served by THORDynBackend")
    ap.add_argument("--reveal-start-offset", type=int, default=12,
                    help="--spans reveal: span start = eval_start + this (frames); must match the "
                         "backend's reveal_start_offset (agent-independent current window)")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    params = {"spans": args.spans, "event_phase": args.event_phase,
              "event_codes": None if not args.event_codes else set(args.event_codes),
              "event_pre": tuple(args.event_pre), "event_post": tuple(args.event_post),
              "event_goal_anchor": args.event_goal_anchor,
              "reveal_start_offset": int(args.reveal_start_offset)}

    manifest = json.loads(args.manifest.read_text())
    by_id = {s["video_id"]: s for s in manifest["samples"]}
    ids = json.loads(args.index.read_text())["sample_ids"]
    missing = [i for i in ids if i not in by_id]
    assert not missing, f"{len(missing)} index ids not in manifest, e.g. {missing[:3]}"

    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(lambda i: scan_clip(by_id[i], params), ids))

    kept = [(vid, el) for vid, el in results if el]
    names = {0: "reveal_A", 1: "reveal_B"} if args.spans == "reveal" else CODE_NAMES
    code_counts = Counter(names[c] for _, el in kept for c, _, _ in el)
    out = {
        "sample_ids": [vid for vid, _ in kept],
        "event_params": {**{k: v for k, v in params.items() if k not in ("spans", "reveal_start_offset")},
                         "event_codes": sorted(params["event_codes"]) if params["event_codes"] else None},
        "spans": args.spans,
        "reveal_start_offset": int(args.reveal_start_offset) if args.spans == "reveal" else None,
        "built_from": {"manifest": str(args.manifest), "index": str(args.index), "n_source_ids": len(ids)},
        "eligible_spans_per_clip": {vid: len(el) for vid, el in kept},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1))
    print(f"{len(kept)} / {len(ids)} clips have >=1 eligible span ({len(kept) / len(ids):.1%}) -> {args.output}")
    print(f"eligible spans: {sum(len(el) for _, el in kept)} total | by code: {dict(code_counts)}")


if __name__ == "__main__":
    main()
