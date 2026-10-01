"""Pre-compute per-frame LOSS-MASK sidecars for an AI2-THOR latent corpus.
Three families via --family:

  change    "what changed vs MEMORY": every eval frame is diffed against an
            exact-view counterpart from earlier in the episode (closures.npz
            pairs at 2 cm / 1 deg / 1 deg). --reference picks the counterpart:
            `smallest` (default) = the closest earlier world state that
            differs at all (peak token error > --min-peak) — the item inside
            an open container, the missing item on an approach — never the
            door angle (mid-swing frames pair with same-angle exploration
            swing frames) or a lookback duplicate; `earliest` = first
            sighting ("changed since"); `tightest` = the tightest exact-view
            pair (an earlier rule, kept so those sidecars reproduce bit-for-bit).
            --label-context-frames extends labels to exploration-phase frames
            (lookback frames). Empty for
            frames with no exact counterpart (~37%% of eval frames).
  interact  "what is changing RIGHT NOW": every frame inside an interaction
            span — and the settled frames after it while the camera holds
            still — is diffed against the PRE-SWING anchor frame. The agent
            is provably frozen through every animation (0.000 cm / 0.000 deg
            over 944 measured spans), so these are exact-view pairs by
            construction: the moving door, the item vanishing at a pickup.
            Covers BOTH phases (exploration included — the only supervision
            that exists for exploration-leg reveals).
  union     change OR interact, computed in one pass (this is what ai2thor_v3
            ships as `latents.change_interact.npy`). Swing rows are door-sized
            for cabinets/microwaves/fridges — pair with a raised
            `model.mask_max_frac` (~0.15-0.25) or they gate away at the loss.

All families threshold the per-token MSE on the (32, 32) LATENT grid (the
signal the diffusion loss is computed on — NOT a downsampled pixel diff;
measured mean recall 0.80 of decoded pixel-diff regions), with the same
recipe: mask = err > --tau x peak, empty when peak <= --min-peak (the "same
state" floor: lookback twins are only JPEG-noise identical while genuine
changes peak far above 0.1).

OUTPUT: `<episode>/latents<--suffix>` — (T, 128) uint8, np.packbits over each
frame's (32, 32) mask; loaded by `THORBackend(mask_suffix=...)`. Default
suffix per family: .change.npy / .interact.npy / .change_interact.npy.

    python scripts/data/build_masks_ai2thor.py --family union \
        --latent-root datasets/ai2thor_v3/latent --workers 48
    python scripts/data/build_masks_ai2thor.py --family change \
        --latent-root datasets/ai2thor_v2/latent --reference tightest ...

Resume-safe (existing sidecars skipped), atomic writes, torchrun-style
RANK/WORLD_SIZE sharding.
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


def _pitch_from_quat(quat: np.ndarray) -> np.ndarray:
    """Per-frame pitch (rad) from THOR's wxyz camera quaternion.

    Same convention as `far.data.backends._thor_pitch_from_quat`; inlined so
    this script has no torch/far import (it is pure numpy and runs on CPU).
    """
    q = np.asarray(quat, dtype=np.float64)
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arcsin(np.clip(2.0 * (qy * qz - qw * qx), -1.0, 1.0))


def _item_mask_frames(ep_dir: Path, n_frames: int) -> dict:
    """{frame: (32,32) bool} union of `item`-role masks, for --union-item-masks."""
    from PIL import Image
    idx_path = ep_dir / "masks_role_index.json"
    if not idx_path.is_file():
        return {}
    idx = json.loads(idx_path.read_text())
    out = {}
    for fn in idx.get("files", {}).get("item", []):
        f = int(fn.split("_")[0])
        if f >= n_frames:
            continue
        m = np.asarray(Image.open(ep_dir / "masks" / fn)) > 127
        m = m.reshape(32, 8, 32, 8).any(axis=(1, 3))
        out[f] = out[f] | m if f in out else m
    return out


def _interaction_spans(interact: np.ndarray):
    """Contiguous interaction spans (code, start, end) over the raw codes."""
    on = np.flatnonzero(interact > 0)
    if on.size == 0:
        return []
    out = []
    for run in np.split(on, np.flatnonzero(np.diff(on) > 1) + 1):
        out.append((int(interact[run[0]]), int(run[0]), int(run[-1])))
    return out


def change_masks(ep_dir: Path, lat, raw_root, tau, pos_tol, yaw_tol, pitch_tol,
                 union_items, label_context, reference, min_peak):
    """(T, 32, 32) bool change-family masks + stats — ported VERBATIM from
    build_change_masks_ai2thor.py (bit-parity guaranteed by the parity test)."""
    meta = json.loads((ep_dir / "meta.json").read_text())
    T = int(lat.shape[0])
    es = int(meta.get("eval_start", T))

    pz = np.load(ep_dir / "poses.npz")
    pos, yaw = np.asarray(pz["pos"])[:T], np.asarray(pz["heading"])[:T]
    pit = _pitch_from_quat(pz["quat"])[:T]

    cz = np.load(ep_dir / "closures.npz")
    i, j = np.asarray(cz["i"]), np.asarray(cz["j"])
    ok = (i < T) & (j < T)
    i, j = i[ok], j[ok]
    d_pos = np.linalg.norm(pos[i] - pos[j], axis=1)
    d_yaw = np.abs(np.arctan2(np.sin(yaw[i] - yaw[j]), np.cos(yaw[i] - yaw[j])))
    d_pit = np.abs(pit[i] - pit[j])
    # EXACT view match. The LATER frame of a pair is the one being labelled and
    # the earlier one is a candidate "before" reference. Default: one side in
    # each phase (label eval frames against the exploration leg only). With
    # --label-context-frames exploration-vs-exploration pairs count as well,
    # so post-change exploration frames (v7 lookbacks) are labelled too.
    exact = (d_pos < pos_tol) & (d_yaw < yaw_tol) & (d_pit < pitch_tol)
    if label_context:
        # Never eval-vs-eval: an eval frame's reference is always taken from
        # the exploration leg, so the eval labels are identical to the default
        # variant's (the variants nest) and mean "changed relative to memory".
        exact &= ~((i >= es) & (j >= es))
    else:
        exact &= ((i < es) & (j >= es)) | ((j < es) & (i >= es))
    sel = np.flatnonzero(exact)

    # All exact counterparts per labelled frame (earlier frame, pose residual).
    cands = {}
    for k in sel:
        a, b = int(i[k]), int(j[k])
        ev, ctx = max(a, b), min(a, b)
        cands.setdefault(ev, []).append((ctx, float(d_pos[k] + d_yaw[k] + d_pit[k])))

    items = _item_mask_frames(Path(raw_root) / ep_dir.name, T) if union_items else {}

    def _err(x, y):
        return ((lat[x].astype(np.float32) - lat[y].astype(np.float32)) ** 2).mean(0)

    masks = np.zeros((T, 32, 32), dtype=bool)
    n_marked = n_multi = n_lookback = 0
    for ev, cs in cands.items():
        if reference == "tightest":
            # v2/v6 behaviour, reproduced exactly: pose tie-break, no floor.
            ctx = min(cs, key=lambda c: c[1])[0]
            err = _err(ev, ctx)
            peak = float(err.max())
            if peak <= 1e-8:
                continue
        else:
            errs = {ctx: _err(ev, ctx) for ctx, _ in cs}
            peaks = {ctx: float(e.max()) for ctx, e in errs.items()}
            if reference == "earliest":
                ctx = min(errs)
                if peaks[ctx] <= 1e-8:
                    continue
            else:                                   # smallest
                nz = [c for c in errs if peaks[c] > min_peak]
                if not nz:
                    continue
                ctx = min(nz, key=lambda c: float(errs[c].mean()))
            err, peak = errs[ctx], peaks[ctx]
            latest = max(errs)
            if latest != ctx:
                # A later counterpart exists; when it already shows the
                # current state it must match within the floor (deterministic
                # renderer up to JPEG noise).
                n_multi += 1
                n_lookback += int(peaks[latest] <= min_peak)
        masks[ev] = err > tau * peak
        n_marked += 1
    for f, m in items.items():
        if f >= es or label_context:
            masks[f] |= m
    return masks, n_marked, max(T - es, 0), n_multi, n_lookback


def interact_masks(ep_dir: Path, lat, tau, min_peak, pos_tol, yaw_tol, pitch_tol,
                   settle_max):
    """(T, 32, 32) bool interaction-family masks + stats — ported from
    build_interaction_masks_ai2thor.py."""
    T = int(lat.shape[0])
    itr = np.asarray(np.load(ep_dir / "actions.npz")["interact"][:T])
    pz = np.load(ep_dir / "poses.npz")
    pos, yaw = np.asarray(pz["pos"])[:T], np.asarray(pz["heading"])[:T]
    pit = _pitch_from_quat(pz["quat"])[:T]

    def same_pose(a, b):
        dy = np.arctan2(np.sin(yaw[b] - yaw[a]), np.cos(yaw[b] - yaw[a]))
        return (np.linalg.norm(pos[b] - pos[a]) < pos_tol
                and abs(dy) < yaw_tol and abs(pit[b] - pit[a]) < pitch_tol)

    masks = np.zeros((T, 32, 32), dtype=bool)
    n_frames = n_spans = 0
    for code, s0, s1 in _interaction_spans(itr):
        a = s0 - 1
        if a < 0:
            continue
        n_spans += 1
        anchor = lat[a].astype(np.float32)
        f = s0
        while f < T and (f <= s1 or (f - s1 <= settle_max and itr[f] == 0)):
            if not same_pose(a, f):
                break
            err = ((lat[f].astype(np.float32) - anchor) ** 2).mean(0)
            peak = float(err.max())
            if peak > min_peak:
                masks[f] |= err > tau * peak
                n_frames += 1
            f += 1
    return masks, n_spans, n_frames


def build_episode(args_tuple) -> tuple:
    (ep_dir, family, suffix, raw_root, tau, min_peak, pos_tol, yaw_tol, pitch_tol,
     union_items, label_context, reference, settle_max) = args_tuple
    ep_dir = Path(ep_dir)
    out_path = ep_dir / f"latents{suffix}"
    if out_path.is_file():
        return ep_dir.name, {}, True

    lat = np.load(ep_dir / "latents.npy", mmap_mode="c")
    T = int(lat.shape[0])
    masks = np.zeros((T, 32, 32), dtype=bool)
    stats = {}
    if family in ("change", "union"):
        m, nm, nev, nmu, nlb = change_masks(ep_dir, lat, raw_root, tau, pos_tol,
                                            yaw_tol, pitch_tol, union_items,
                                            label_context, reference, min_peak)
        masks |= m
        stats.update(chg_marked=nm, chg_eval=nev, chg_multi=nmu, chg_lookback=nlb)
    if family in ("interact", "union"):
        m, ns, nf = interact_masks(ep_dir, lat, tau, min_peak, pos_tol, yaw_tol,
                                   pitch_tol, settle_max)
        masks |= m
        stats.update(int_spans=ns, int_frames=nf)
    stats["big"] = int((masks.reshape(T, -1).mean(1) >= 0.10).sum())

    packed = np.packbits(masks.reshape(T, -1), axis=1)
    tmp = out_path.with_suffix(out_path.suffix + f".tmp{os.getpid()}")
    with open(tmp, "wb") as fh:
        np.save(fh, packed)
    os.replace(tmp, out_path)
    return ep_dir.name, stats, False


DEFAULT_SUFFIX = {"change": ".change.npy", "interact": ".interact.npy",
                  "union": ".change_interact.npy"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=("change", "interact", "union"), default="change")
    ap.add_argument("--latent-root", default="datasets/ai2thor_v2/latent")
    ap.add_argument("--raw-root", default=None,
                    help="Only read with --union-item-masks (the shipped masks/ PNGs)")
    ap.add_argument("--suffix", default=None,
                    help="Sidecar name (latents<suffix>); default per family: "
                         ".change.npy / .interact.npy / .change_interact.npy. Keep it "
                         "distinct from any '.keys' suffix.")
    ap.add_argument("--tau", type=float, default=0.05,
                    help="Token is 'changed' when its error exceeds tau x the frame's peak")
    ap.add_argument("--min-peak", type=float, default=0.05,
                    help="Peak-error floor below which a pair counts as 'same state'")
    ap.add_argument("--pos-tol", type=float, default=0.02, help="metres")
    ap.add_argument("--yaw-tol", type=float, default=np.radians(1.0), help="radians")
    ap.add_argument("--pitch-tol", type=float, default=np.radians(1.0), help="radians")
    ap.add_argument("--reference", choices=["smallest", "earliest", "tightest"], default="smallest",
                    help="change family: which exact counterpart is the 'before' reference "
                         "(module docstring). `tightest` reproduces the v2/v6 sidecars bit-for-bit.")
    ap.add_argument("--label-context-frames", action="store_true",
                    help="change family: also label exploration-phase frames (own --suffix!)")
    ap.add_argument("--union-item-masks", action="store_true",
                    help="change family: also OR in the shipped `item`-role mask PNGs (needs --raw-root)")
    ap.add_argument("--settle-max", type=int, default=8,
                    help="interact family: max settled frames after a span to keep labelling "
                         "(the pose lock is the real stop; the dwell pan breaks it after 1-2)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    suffix = args.suffix or DEFAULT_SUFFIX[args.family]

    need = {"change": ("latents.npy", "closures.npz", "meta.json", "poses.npz"),
            "interact": ("latents.npy", "actions.npz", "poses.npz"),
            "union": ("latents.npy", "closures.npz", "meta.json", "actions.npz", "poses.npz")}[args.family]
    root = Path(args.latent_root)
    eps = sorted(p for p in root.iterdir()
                 if p.is_dir() and all((p / f).is_file() for f in need))
    if args.limit:
        eps = eps[:args.limit]
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    mine = eps[rank::world]
    print(f"[rank {rank}/{world}] {len(mine)} of {len(eps)} episodes | family {args.family} -> "
          f"{suffix} | tau {args.tau} min-peak {args.min_peak} | tol {args.pos_tol} m / "
          f"{np.degrees(args.yaw_tol):.1f} deg / {np.degrees(args.pitch_tol):.1f} deg"
          + (f" | reference {args.reference}" if args.family != "interact" else "")
          + (" | labelling context frames too" if args.label_context_frames else ""), flush=True)

    jobs = [(str(p), args.family, suffix, args.raw_root, args.tau, args.min_peak,
             args.pos_tol, args.yaw_tol, args.pitch_tol, args.union_item_masks,
             args.label_context_frames, args.reference, args.settle_max) for p in mine]
    done = skipped = 0
    agg = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(build_episode, j) for j in jobs]
        for n, fut in enumerate(as_completed(futs), 1):
            try:
                name, stats, was_skipped = fut.result()
            except Exception as e:                       # never abort the sweep
                print(f"  [rank {rank}] ERROR {e}", flush=True)
                continue
            if was_skipped:
                skipped += 1
            else:
                done += 1
                for k, v in stats.items():
                    agg[k] = agg.get(k, 0) + v
            if n % 500 == 0 or n == len(futs):
                print(f"  [rank {rank}] {n}/{len(futs)} | written {done} skipped {skipped}", flush=True)
    print(f"[rank {rank}] done. {done} written, {skipped} already existed.", flush=True)
    if "chg_marked" in agg:
        print(f"[rank {rank}] change: coverage {agg['chg_marked'] / max(agg.get('chg_eval', 1), 1):.1%} "
              f"of eval frames"
              + (" (labelled/eval; includes context-phase labels)" if args.label_context_frames else ""),
              flush=True)
        if args.reference != "tightest":
            print(f"[rank {rank}] change: labelled frames with a LATER exact counterpart: "
                  f"{agg.get('chg_multi', 0):,} — {agg.get('chg_lookback', 0):,} "
                  f"({agg.get('chg_lookback', 0) / max(agg.get('chg_multi', 1), 1):.1%}) match it "
                  f"within min-peak (post-change lookback view present).", flush=True)
    if "int_spans" in agg:
        print(f"[rank {rank}] interact: {agg['int_spans']:,} spans -> {agg['int_frames']:,} "
              f"labelled frames ({agg['int_frames'] / max(agg['int_spans'], 1):.1f}/span)", flush=True)
    print(f"[rank {rank}] rows >= 10% of tokens: {agg.get('big', 0):,} "
          f"(gated at the loss unless mask_max_frac is raised).", flush=True)


if __name__ == "__main__":
    main()
