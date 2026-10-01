"""Build agent-silhouette loss-mask sidecars for the dynamic-agent corpus.

The door-corridor episodes ship a per-frame agent silhouette at
`masks/%06d.png` (256², {0,255}, written ONLY where the agent is on screen —
`agent2.npz['px'] > 0`). This script downsamples each to the 32×32 LATENT grid
and packs it into the same format `THORBackend(mask_suffix=...)` consumes, so
the moving agent drives the mask-weighted diffusion loss exactly like the THOR
change/union masks do — the generator is pushed to render the agent's region
rather than let it wash out at plain MSE.

Downsample: a latent cell is ON if ANY pixel in its 8×8 patch is silhouette
(`reshape(32,8,32,8).any((1,3))` — verbatim from `build_masks_ai2thor.py`),
which preserves the tiny onset agent (a few pixels at the doorway edge).
Frames with no mask file (px==0, and every ctrl frame) → an all-zero row.

OUTPUT: `<latent-dir>/<episode>/latents<--suffix>` — (T, 128) uint8,
`np.packbits` over each frame's flattened (32,32) mask; frame-aligned to
`latents.npy`. Masks are read from --source-root (they are NOT copied into the
latent tree by the encoder). Resume-safe: episodes whose sidecar exists are
skipped. CPU-only; shard with --shard/--num-shards for parallel processes.

    python scripts/data/build_agent_masks_dyn.py \\
        --source-root /path/to/door_corridor_raw_episodes \\
        --latent-dir  datasets/ai2thor_dyn/latent
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


def episode_mask(src_ep: Path, n_frames: int) -> np.ndarray:
    """(T, 32, 32) bool — the agent silhouette on the latent grid per frame."""
    masks = np.zeros((n_frames, 32, 32), dtype=bool)
    mdir = src_ep / "masks"
    if not mdir.is_dir():
        return masks
    for png in mdir.glob("*.png"):
        f = int(png.stem)
        if f >= n_frames:
            continue
        m = np.asarray(Image.open(png)) > 127
        if m.ndim == 3:                       # collapse any accidental channel dim
            m = m.any(-1)
        masks[f] = m.reshape(32, 8, 32, 8).any(axis=(1, 3))
    return masks


def atomic_save(arr: np.ndarray, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp, "wb") as fh:
        np.save(fh, arr)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-root", required=True, help="raw door-corridor episode tree (masks/%06d.png per episode)")
    ap.add_argument("--latent-dir", default="datasets/ai2thor_dyn/latent")
    ap.add_argument("--suffix", default=".agent.npy")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    src = Path(args.source_root)
    lat = Path(args.latent_dir)
    eps = sorted(d for d in lat.iterdir() if d.is_dir() and (d / "meta.json").exists())
    mine = eps[args.shard::args.num_shards]
    print(f"[shard {args.shard}/{args.num_shards}] {len(mine)}/{len(eps)} episodes "
          f"-> latents{args.suffix}", flush=True)

    done = frac_sum = frac_n = 0
    for ep in mine:
        out = ep / f"latents{args.suffix}"
        if out.exists():
            continue
        n = int(json.loads((ep / "meta.json").read_text())["n_frames"])
        m = episode_mask(src / ep.name, n)
        packed = np.packbits(m.reshape(n, -1), axis=1)     # (T, 128) uint8
        atomic_save(packed, out)
        nz = m.reshape(n, -1).any(1)
        if nz.any():
            frac_sum += m.reshape(n, -1)[nz].mean(1).sum(); frac_n += int(nz.sum())
        done += 1
        if done % 200 == 0:
            print(f"[shard {args.shard}] {done}/{len(mine)}", flush=True)
    mean_frac = (frac_sum / frac_n) if frac_n else 0.0
    print(f"[shard {args.shard}] DONE: {done} written | "
          f"agent-present rows: {frac_n:,} | mean on-cell fraction there {mean_frac:.3f}", flush=True)


if __name__ == "__main__":
    main()
