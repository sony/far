"""Per-episode OBSERVABILITY metadata for the SoundSpaces latent dataset.

For every episode, measures how well the OUTBOUND leg's views cover the
RETURN leg's views — the clip-level version of the per-step "memory support"
covariate. Pure pose geometry (poses.npz + meta.json), no rendering/model.

Per return-leg frame j:
    1. Sample probe points inside j's viewing frustum (uniform directions in
       the FOV pyramid, depths in [min_depth, max_depth]) — "what j sees".
    2. support(j) = max over outbound frames i of the fraction of j's probe
       points inside i's frustum (same FOV/depth gates) — "the best single
       outbound view of what j sees".

A probe point counts only when the outbound camera views it from a similar
DIRECTION (within DIR_THRESH_DEG of the return camera's ray) — surfaces seen
from the opposite side (heading-flip) are correctly scored unsupported.
Frustum overlap remains WALL-BLIND (through-wall visibility counts), so
support is an UPPER BOUND on true observability: clips scored unobservable
really are; some hard clips (occlusion tunneling) survive. The short
max_depth keeps overlap mostly same-room. Positions include z, so stair traversals (vertical
offsets) correctly break support.

Conventions mirror `SoundSpacesBackend`: poses standardized to the z-up frame
via `_standardize_loopnav_pose` (forward = (cos yaw, sin yaw), pitch 0), the
return leg starts at the first meta.json leg with phase == "eval", and the
camera is the Habitat 90x90 deg sensor (half-FOV 45/45).

Outputs (written atomically to --output-dir):
    soundspaces_observability.json — params header + per-episode stats:
        {video_id, route, tier, split, n_frames, ret_start, coverage,
         mean_support, max_unsupported_run, z_span}
        coverage = fraction of return frames with support > tau.
    soundspaces_observability.npz  — per-episode per-return-frame support
        curves, uint8-quantized (support * 255), keyed by video_id.

Usage:
    python scripts/data/build_observability_soundspaces.py \
        --latent-dir datasets/soundspaces/latent \
        --output-dir results/observability [--device cuda:0] [--limit N]
"""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

FOV_HALF_H = math.radians(45.0)
FOV_HALF_V = math.radians(45.0)
MIN_DEPTH, MAX_DEPTH = 0.3, 4.0
N_PROBE = 256
TAU = 0.2
# A probe point only counts as supported if the outbound camera views it from
# a SIMILAR direction as the return camera (angle between the two viewing rays
# below this threshold) — the same-surface proxy. Without it, seeing the same
# VOLUME from the opposite side (the 173-deg heading-flip case — precisely the
# no-appearance-support pathology) scores as full coverage.
DIR_THRESH_DEG = 75.0


def standardize(pos: np.ndarray, heading: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Habitat [x, height, z] -> z-up [x', y', z'] = [-z, -x, height]; yaw = heading
    (forward becomes (cos yaw, sin yaw) — same remap as _standardize_loopnav_pose)."""
    std = np.stack([-pos[:, 2], -pos[:, 0], pos[:, 1]], axis=1)
    return std, heading


def probe_points(pos: torch.Tensor, yaw: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """(T, 3), (T,) -> (T, N_PROBE, 3) points sampled in each frame's frustum."""
    T = pos.shape[0]
    dev = pos.device
    ah = (torch.rand(T, N_PROBE, generator=gen, device=dev) * 2 - 1) * FOV_HALF_H
    av = (torch.rand(T, N_PROBE, generator=gen, device=dev) * 2 - 1) * FOV_HALF_V
    d  = MIN_DEPTH + torch.rand(T, N_PROBE, generator=gen, device=dev) * (MAX_DEPTH - MIN_DEPTH)
    yawp = yaw[:, None] + ah                                   # absolute horizontal ray angle
    horiz = d * torch.cos(av)
    return torch.stack([
        pos[:, None, 0] + horiz * torch.cos(yawp),
        pos[:, None, 1] + horiz * torch.sin(yawp),
        pos[:, None, 2] + d * torch.sin(av),
    ], dim=-1)


def in_frustum(points: torch.Tensor, pos: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """points (M, 3) vs cameras (K, 3)/(K,) -> (K, M) bool visibility."""
    delta = points[None] - pos[:, None]                        # (K, M, 3)
    fwd = torch.stack([torch.cos(yaw), torch.sin(yaw)], -1)    # (K, 2)
    x = (delta[..., :2] * fwd[:, None]).sum(-1)                # forward distance
    y = delta[..., 1] * fwd[:, None, 0] - delta[..., 0] * fwd[:, None, 1]   # leftward
    z = delta[..., 2]
    dist = delta.norm(dim=-1)
    return ((x > 0)
            & (y.abs() <= x * math.tan(FOV_HALF_H))
            & (z.abs() <= x * math.tan(FOV_HALF_V))
            & (dist >= MIN_DEPTH) & (dist <= MAX_DEPTH))


@torch.no_grad()
def episode_support(pos: torch.Tensor, yaw: torch.Tensor, ret_start: int,
                    gen: torch.Generator) -> torch.Tensor:
    """(T, 3), (T,) -> (T - ret_start,) per-return-frame support in [0, 1]."""
    out_pos, out_yaw = pos[:ret_start], yaw[:ret_start]
    ret_pos, ret_yaw = pos[ret_start:], yaw[ret_start:]
    pts = probe_points(ret_pos, ret_yaw, gen)                  # (R, P, 3)
    R = pts.shape[0]
    support = torch.zeros(R, device=pos.device)
    cos_thresh = math.cos(math.radians(DIR_THRESH_DEG))
    # Chunk over return frames to bound the (K_out, R*P) visibility matrix.
    chunk = max(1, int(4e7 // max(out_pos.shape[0] * N_PROBE, 1)))
    for s in range(0, R, chunk):
        p = pts[s:s + chunk].reshape(-1, 3)                    # (r*P, 3)
        vis = in_frustum(p, out_pos, out_yaw)                  # (K_out, r*P)
        # Same-surface gate: the outbound ray to the point must agree in
        # direction with the return camera's ray to it.
        u_ret = p - ret_pos[s:s + chunk].repeat_interleave(N_PROBE, dim=0)    # (r*P, 3)
        u_ret = torch.nn.functional.normalize(u_ret, dim=-1)
        u_out = torch.nn.functional.normalize(p[None] - out_pos[:, None], dim=-1)  # (K_out, r*P, 3)
        vis = vis & ((u_out * u_ret[None]).sum(-1) > cos_thresh)
        frac = vis.view(out_pos.shape[0], -1, N_PROBE).float().mean(-1)   # (K_out, r)
        support[s:s + chunk] = frac.max(0).values
    return support


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent-dir", default="datasets/soundspaces/latent")
    ap.add_argument("--output-dir", default="results/observability")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    root = Path(args.latent_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes = sorted(p for p in root.iterdir() if (p / "poses.npz").is_file())
    if args.limit:
        episodes = episodes[:args.limit]
    print(f"episodes: {len(episodes)} | device: {args.device}")

    gen = torch.Generator(device=args.device)
    gen.manual_seed(0)
    stats, curves = [], {}
    for n, ep in enumerate(episodes):
        poses = np.load(ep / "poses.npz")
        meta = json.loads((ep / "meta.json").read_text())
        pos_std, yaw = standardize(np.asarray(poses["pos"], dtype=np.float32),
                                   np.asarray(poses["heading"], dtype=np.float32))
        T = pos_std.shape[0]
        ret_start = -1
        for leg in meta.get("legs", []):
            if leg.get("phase") == "eval":
                ret_start = min(int(leg["frames"][0]), T - 1)
                break
        if ret_start <= 0:
            continue
        pos_t = torch.as_tensor(pos_std, device=args.device)
        yaw_t = torch.as_tensor(yaw, device=args.device)
        support = episode_support(pos_t, yaw_t, ret_start, gen).cpu().numpy()

        supported = support > TAU
        runs, cur = [], 0
        for s in supported:
            cur = 0 if s else cur + 1
            runs.append(cur)
        parts = ep.name.split("_")
        stats.append({
            "video_id": ep.name,
            "route": parts[1] if len(parts) > 1 else "?",
            "tier": parts[2] if len(parts) > 2 else "?",
            "split": meta.get("split", "?"),
            "n_frames": int(T),
            "ret_start": int(ret_start),
            "coverage": round(float(supported.mean()), 4),
            "mean_support": round(float(support.mean()), 4),
            "max_unsupported_run": int(max(runs) if runs else 0),
            "z_span": round(float(pos_std[:, 2].max() - pos_std[:, 2].min()), 3),
        })
        curves[ep.name] = np.clip(support * 255, 0, 255).astype(np.uint8)
        if (n + 1) % 1000 == 0:
            print(f"  {n + 1}/{len(episodes)}")

    payload = {
        "params": {"fov_half_h_deg": 45.0, "fov_half_v_deg": 45.0,
                   "min_depth": MIN_DEPTH, "max_depth": MAX_DEPTH,
                   "n_probe": N_PROBE, "tau": TAU, "seed": 0,
                   "dir_thresh_deg": DIR_THRESH_DEG,
                   "note": "support = max over outbound frames of the fraction of the "
                           "return frame's frustum probes inside the outbound frustum; "
                           "wall-blind (upper bound on true observability)"},
        "episodes": stats,
    }
    tmp = out_dir / "soundspaces_observability.json.tmp"
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, out_dir / "soundspaces_observability.json")
    tmp_npz = out_dir / f"soundspaces_observability.npz.tmp{os.getpid()}"
    with open(tmp_npz, "wb") as f:
        np.savez_compressed(f, **curves)
    os.replace(tmp_npz, out_dir / "soundspaces_observability.npz")
    print(f"wrote {len(stats)} episodes -> {out_dir}/soundspaces_observability.{{json,npz}}")


if __name__ == "__main__":
    main()
