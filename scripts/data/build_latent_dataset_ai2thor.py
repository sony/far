"""Pre-compute SDXL-VAE latents for the AI2-THOR loopnav/TOUR episodes.

For each episode directory under --dataset-root (layout:
  <traj_id>_{act,ctrl}/{frames/*.jpg, masks/*.png, poses.npz, actions.npz,
                        closures.npz, legs.npz, objvis.npz, meta.json,
                        preview.mp4})
this script:
  1. Encodes every frame (256x256 RGB JPEG) with the SDXL autoencoder
     (`madebyollin/sdxl-vae-fp16-fix` — the stock SDXL VAE overflows in fp16)
     to a single latents.npy of shape (T, 4, 32, 32), float16, saved at
     `tokenize()` scale (multiplied by the VAE's scaling_factor, 0.13025 —
     the same convention as the soundspaces/loopnav latents: downstream
     consumers divide by the scale to decode).
  2. Copies the episode's sidecars (poses.npz, actions.npz, closures.npz,
     legs.npz, objvis.npz, meta.json) next to the latents, so
     datasets/ai2thor/latent/<traj_id>/ is self-contained for TRAINING.
     frames/ and preview.mp4 are NOT copied — the latents replace them.
     masks/ is copied only with --copy-masks (see below).

Episode selection: any directory holding a meta.json. The generator's
bookkeeping directories (`_logs`) and any leftover `.tmp_*` staging dirs are
skipped by that rule. `--kind {both,act,ctrl}` restricts to the interaction
episodes (`_act`) or their seed-matched static siblings (`_ctrl`).

masks/: per-frame per-object PNGs, needed only by the EVAL metrics (change
region scoring), not by training. Copying them means ~280 tiny files per
episode (~3.8M files corpus-wide), which can dominate wall-clock, so it is
OFF by default — eval can read them from --dataset-root, which stays put.
Pass --copy-masks when the latent tree must be self-contained (e.g. before
uploading it to another machine).

Pitch probe: the corpus stores `pos`, `quat`, `heading` but NO explicit
pitch, while the downstream generators (Plücker rays) and worldmem's
vertical FOV need it. Every --validate-every episodes this script derives
pitch from `quat` and prints its range together with the yaw/heading
agreement of the quaternion layout it used — a read-only sanity check that
the conversion the data backend will perform is sound. Nothing is rewritten:
sidecars are byte-identical copies, and the pose-convention logic stays in
the backend (as it does for loopnav's standard_frame remap).

Determinism: latents are the posterior MEAN (`latent_dist.mode()`), so
re-encoding an episode under the SAME settings is bit-identical — safe to
resume and to spot-check. Across settings the differences are small but real
(JPEG decoder ~3.4% of latent std, eager vs compiled ~0.34%), which is what
`latent_params.json` in --output-dir records and guards.

Resume-safe: episodes whose latents.npy already exists are skipped (missing
sidecar copies are still filled in). All writes are atomic (tmp + rename).

Memory: only the current batch of frames and the current episode's (T, 4,
32, 32) fp16 array (~5 MB) are ever held — nothing accumulates across
episodes.

Single GPU:
    python scripts/data/build_latent_dataset_ai2thor.py \\
        --dataset-root /path/to/ai2thor_raw_episodes \\
        --output-dir   datasets/ai2thor/latent

Multi-GPU (one independent process per GPU; episodes[rank::world_size]):
    torchrun --nproc_per_node=8 scripts/data/build_latent_dataset_ai2thor.py [...]
No process-group initialization is needed — ranks operate independently.
"""

import argparse
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    from torchvision.io import ImageReadMode, decode_jpeg
    HAVE_NVJPEG = True
except Exception:                               # torchvision absent/too old
    HAVE_NVJPEG = False

SIDECARS = ("poses.npz", "actions.npz", "closures.npz", "legs.npz",
            "objvis.npz", "meta.json")


def load_vae(name: str, device: torch.device):
    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(name, torch_dtype=torch.float16)
    vae = vae.to(device).eval()
    scale = float(vae.config.scaling_factor)
    return vae, scale


def _load_frame(path):
    return np.asarray(Image.open(path).convert("RGB"))


def _read_bytes(path):
    with open(path, "rb") as f:                 # bytearray: writable, no torch warning
        return torch.frombuffer(bytearray(f.read()), dtype=torch.uint8)


def _batch_to_device(chunk, device, decoder, pool):
    """Decode a batch of JPEG paths -> uint8 (B, 3, H, W) on `device`."""
    if decoder == "nvjpeg":
        raw = (list(pool.map(_read_bytes, chunk)) if pool is not None
               else [_read_bytes(p) for p in chunk])
        try:
            return torch.stack(decode_jpeg(raw, device=device, mode=ImageReadMode.RGB))
        except Exception as e:                  # corrupt/odd frame: fall back per batch
            print(f"[warn] nvjpeg failed ({e}); PIL for this batch", flush=True)
    imgs = np.stack(list(pool.map(_load_frame, chunk)) if pool is not None
                    else [_load_frame(p) for p in chunk])
    return torch.from_numpy(imgs).permute(0, 3, 1, 2).to(device)


@torch.no_grad()
def encode_episode(frame_paths, vae, scale, batch_size, device,
                   pool=None, decoder="nvjpeg", pad_batches=False):
    """Encode JPEG frames -> (T, 4, H/8, W/8) float16 at tokenize() scale.

    Frame decode, not the VAE, is the bottleneck: measured on this corpus the
    VAE runs at ~12.6k f/s (bs=64) while PIL decode + host->device copy
    sustains only ~0.3k f/s, so raising --batch-size does nothing. `nvjpeg`
    instead ships the compressed bytes to the GPU (~20k f/s to read) and
    decodes them there (~13k f/s), which is ~10x end-to-end. It decodes
    JPEGs slightly differently from libjpeg (mean 0.5/255 per pixel, 3.4% of
    the latent std) — far below the VAE's own reconstruction error (27 dB),
    but a corpus must not MIX decoders, which `latent_params.json` enforces.
    """
    out = []
    for start in range(0, len(frame_paths), batch_size):
        chunk = frame_paths[start:start + batch_size]
        x = _batch_to_device(chunk, device, decoder, pool)
        x = x.to(torch.float16).div_(127.5).sub_(1.0)              # [-1, 1]
        n = len(chunk)
        if pad_batches and n < batch_size:
            # Keep exactly ONE input shape: a compiled encoder would otherwise
            # recompile per trailing-batch size (episodes end anywhere), blow
            # past dynamo's cache limit and silently fall back to eager.
            x = torch.cat([x, x[-1:].expand(batch_size - n, *x.shape[1:])])
        z = vae.encode(x).latent_dist.mode().mul_(scale)           # posterior MEAN
        out.append(z[:n].to(torch.float16).cpu())
    return torch.cat(out).numpy()                                  # (T, 4, h, w) fp16


@torch.no_grad()
def roundtrip_psnr(lat, frame_path, vae, scale, device):
    """Decode latents[0] and PSNR it against the source frame (sanity only)."""
    z = torch.from_numpy(lat[:1]).to(device, torch.float16) / scale
    rec = vae.decode(z).sample.float().clamp(-1, 1).add(1).div(2)
    ref = torch.from_numpy(                     # copy: PIL buffers are read-only
        np.array(Image.open(frame_path).convert("RGB"))).permute(2, 0, 1)
    ref = ref.float().div(255).to(device)
    mse = (rec[0] - ref).pow(2).mean()
    return float(-10 * torch.log10(mse))


def probe_pitch(poses_path: Path) -> str:
    """Derive pitch from `quat` and report it (READ-ONLY diagnostic).

    The corpus does not record the quaternion component order, so both common
    layouts are tried and the one whose yaw best reproduces the stored
    `heading` wins; pitch is then read from that same layout. Unity/THOR is
    y-up with forward = +z, so for a rotation matrix R: forward = R[:, 2] and
    yaw = atan2(fx, fz). Pitch is reported in the REPO convention — the one
    `global_to_local_pose` assumes, d = [cos p cos y, cos p sin y, sin p] in
    the z-up standard frame, i.e. POSITIVE = looking up. The standard-frame
    remap sends Unity's up-axis (y) to the repo's z, so pitch = asin(fy), and
    the scan/inspection look-downs must therefore print NEGATIVE.
    """
    z = np.load(poses_path)
    if "pitch" in z.files:
        p = z["pitch"]
        return f"pitch stored: [{np.degrees(p.min()):.1f}, {np.degrees(p.max()):.1f}] deg"
    if "quat" not in z.files:
        return "no quat/pitch in poses.npz — pose pathway needs another source"
    q, head = z["quat"].astype(np.float64), z["heading"].astype(np.float64)
    best = None
    for name, (x, y, w_, zc) in (("xyzw", (0, 1, 3, 2)), ("wxyz", (1, 2, 0, 3))):
        qx, qy, qw, qz = q[:, x], q[:, y], q[:, w_], q[:, zc]
        n = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
        qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
        fx = 2 * (qx * qz + qw * qy)                  # R[:, 2] of the quaternion
        fy = 2 * (qy * qz - qw * qx)
        fz = 1 - 2 * (qx**2 + qy**2)
        yaw = np.arctan2(fx, fz)
        agree = float(np.mean(np.cos(yaw - head)))    # +1 = layout reproduces heading
        pitch = np.arcsin(np.clip(fy, -1, 1))    # repo convention: + = up
        if best is None or agree > best[1]:
            best = (name, agree, pitch)
    name, agree, pitch = best
    return (f"pitch from quat[{name}]: [{np.degrees(pitch.min()):.1f}, "
            f"{np.degrees(pitch.max()):.1f}] deg (yaw/heading agreement {agree:+.3f})")


def encoding_contract(out_root: Path, vae_name: str, scale: float, decoder: str,
                      compiled: bool):
    """Pin the encoding recipe for the corpus (mirrors soundspaces'
    audio_feat_params.json). Written once; every later rank and every resume
    checks against it, so a run cannot silently mix incompatible settings.

    STRICT fields (vae / scaling_factor / decoder / dtype / posterior) abort on
    mismatch: the decoder choice moves latents by ~3.4% of their std, and the
    VAE or scale changing makes the corpus incoherent outright. `compiled` is
    recorded for provenance but only WARNS — eager vs compiled differ by 0.34%
    of std, an order of magnitude below the decoder gap and far below the VAE's
    own reconstruction error, so blocking a resume over it would cost more than
    it buys."""
    path = out_root / "latent_params.json"
    strict = {"vae": vae_name, "scaling_factor": scale, "decoder": decoder,
              "latent_dtype": "float16", "posterior": "mode"}
    want = dict(strict, compiled=compiled)
    if path.exists():
        have = json.loads(path.read_text())
        have_strict = {k: have.get(k) for k in strict}
        if have_strict != strict:
            raise SystemExit(
                f"encoding mismatch in {out_root}:\n  existing {have_strict}\n  "
                f"requested {strict}\nRe-encode from scratch or match the existing "
                f"settings — mixing them corrupts the corpus silently.")
        if have.get("compiled") != compiled:
            print(f"[warn] resuming a corpus encoded with compiled="
                  f"{have.get('compiled')} using compiled={compiled} "
                  f"(numerically ~0.34% of latent std apart; harmless)")
        return
    out_root.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".json.tmp{os.getpid()}")
    tmp.write_text(json.dumps(want, indent=1))
    os.replace(tmp, path)          # atomic: concurrent ranks race harmlessly


def atomic_save_npy(arr, path: Path):
    for stale in path.parent.glob(path.name + ".tmp*"):
        stale.unlink()              # orphans from a killed/preempted run
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp, "wb") as f:      # file handle: np.save must NOT append ".npy"
        np.save(f, arr)
    os.replace(tmp, path)


def copy_masks(ep_dir: Path, out_dir: Path) -> None:
    """Mirror masks/ (eval-only assets); per-file skip keeps it resume-safe."""
    src = ep_dir / "masks"
    if not src.is_dir():
        return
    dst = out_dir / "masks"
    dst.mkdir(exist_ok=True)
    have = set(os.listdir(dst))
    for name in os.listdir(src):
        if name in have:
            continue
        tmp = dst / (name + f".tmp{os.getpid()}")
        shutil.copy2(src / name, tmp)
        os.replace(tmp, dst / name)


def process_episode(ep_dir: Path, out_dir: Path, vae, scale, batch_size,
                    device, validate: bool, with_masks: bool,
                    pool=None, decoder="nvjpeg",
                    pad_batches=False, sidecars=SIDECARS) -> tuple[int, bool]:
    """Encode one episode + copy sidecars. Returns (n_frames, encoded?)."""
    meta = json.loads((ep_dir / "meta.json").read_text())
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sidecar copies first (cheap; also backfills episodes whose latents were
    # written by an interrupted earlier run).
    for name in sidecars:
        src, dst = ep_dir / name, out_dir / name
        if src.exists() and not dst.exists():
            tmp = dst.with_suffix(dst.suffix + f".tmp{os.getpid()}")
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)
    if with_masks:
        copy_masks(ep_dir, out_dir)

    npy_path = out_dir / "latents.npy"
    if npy_path.exists():
        return meta["n_frames"], False

    frame_paths = sorted((ep_dir / "frames").glob("*.jpg"))
    assert len(frame_paths) == meta["n_frames"], \
        f"{ep_dir.name}: {len(frame_paths)} frames on disk vs meta n_frames={meta['n_frames']}"
    res = meta.get("resolution")               # THOR meta records it; corridor meta does not
    with Image.open(frame_paths[0]) as im:
        assert im.size[0] == im.size[1], f"{ep_dir.name}: non-square frame {im.size}"
        if res is not None:
            assert im.size == (res, res), \
                f"{ep_dir.name}: frame size {im.size} vs meta resolution {res}"

    lat = encode_episode(frame_paths, vae, scale, batch_size, device, pool,
                         decoder, pad_batches)
    if validate:
        psnr = roundtrip_psnr(lat, frame_paths[0], vae, scale, device)
        print(f"[validate] {ep_dir.name}: roundtrip PSNR {psnr:.1f} dB "
              f"| latent std {float(np.std(lat.astype(np.float32))):.3f} "
              f"| {probe_pitch(ep_dir / 'poses.npz')}", flush=True)
    atomic_save_npy(lat, npy_path)
    return meta["n_frames"], True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root",
                        required=True, help="raw episode tree (one directory per episode)")
    parser.add_argument("--output-dir", default="datasets/ai2thor/latent")
    parser.add_argument("--vae", default="madebyollin/sdxl-vae-fp16-fix",
                        help="diffusers AutoencoderKL repo id (fp16-safe SDXL VAE)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Frames per VAE forward pass (the VAE is NOT the "
                             "bottleneck — raising this does nothing; see --workers)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Threads for reading frames per rank (0 = serial). "
                             "Measured no gain on warm cache (PIL decode is "
                             "GIL-bound at this resolution); useful only when "
                             "reads are cold/IO-bound")
    parser.add_argument("--no-compile", action="store_true",
                        help="Skip torch.compile of the VAE encoder. Compiling "
                             "costs ~100 s once per rank but doubles encode "
                             "throughput (measured 458 -> 943 f/s), so it pays "
                             "for itself after ~2k frames; use this for smokes")
    parser.add_argument("--decoder", choices=("nvjpeg", "pil"), default="nvjpeg",
                        help="JPEG decoder: nvjpeg = batched GPU decode (~10x "
                             "end-to-end, the default); pil = libjpeg on CPU. "
                             "Must stay CONSISTENT across a corpus — enforced "
                             "by latent_params.json in --output-dir")
    parser.add_argument("--kind", choices=("both", "act", "ctrl"), default="both",
                        help="Encode interaction episodes (_act), their static "
                             "siblings (_ctrl), or both (default)")
    parser.add_argument("--sidecars", default=",".join(SIDECARS),
                        help="Comma-separated sidecar filenames copied next to the "
                             "latents (missing ones skipped). Default = the THOR "
                             "set; e.g. the dynamic-agent corpus wants "
                             "'poses.npz,agent2.npz,meta.json'.")
    parser.add_argument("--copy-masks", action="store_true",
                        help="Also mirror masks/ (eval-only; ~280 small files "
                             "per episode — slow, off by default)")
    parser.add_argument("--validate-every", type=int, default=200,
                        help="Decode-roundtrip PSNR + pitch probe every N encoded "
                             "episodes on this rank (0 = never)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N episodes on this rank (smoke runs)")
    args = parser.parse_args()

    rank       = int(os.environ.get("RANK",       0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    root = Path(args.dataset_root)
    # An episode is any directory carrying a meta.json — this skips the
    # generator's `_logs` bookkeeping dir and any `.tmp_*` staging leftovers.
    episodes = sorted(d for d in root.iterdir()
                      if d.is_dir() and (d / "meta.json").exists())
    if args.kind != "both":
        episodes = [d for d in episodes if d.name.endswith("_" + args.kind)]
    mine = episodes[rank::world_size]
    if args.limit is not None:
        mine = mine[: args.limit]

    decoder = args.decoder
    if decoder == "nvjpeg" and not (HAVE_NVJPEG and torch.cuda.is_available()):
        print("[warn] nvjpeg unavailable (torchvision/CUDA) — falling back to pil")
        decoder = "pil"
    vae, scale = load_vae(args.vae, device)
    # The VAE encoder — not the JPEG decode or the transfers — is the floor
    # here (profiled: 88% of wall-clock), so compiling it is the one lever
    # that matters. Weights are unchanged, so latents stay comparable.
    compiled = False
    if not args.no_compile and torch.cuda.is_available():
        try:
            torch.cuda.set_device(local_rank)   # so compile/autotune target THIS gpu
            vae.encoder = torch.compile(vae.encoder)
            compiled = True
        except Exception as e:
            print(f"[warn] torch.compile unavailable ({e}); running eager")
    encoding_contract(Path(args.output_dir), args.vae, scale, decoder, compiled)
    print(f"[rank {rank}/{world_size}] {len(mine)}/{len(episodes)} episodes "
          f"(kind={args.kind}) on {device} | vae={args.vae} scale={scale} "
          f"decoder={decoder} compiled={compiled}", flush=True)

    out_root = Path(args.output_dir)
    done_frames = done_eps = encoded_eps = 0
    pool = ThreadPoolExecutor(args.workers) if args.workers > 0 else None
    t0 = time.time()
    for i, ep_dir in enumerate(mine):
        # validate fires on every N-th ENCODED episode (skipped episodes never
        # reach the validation branch inside process_episode).
        validate = (args.validate_every > 0 and encoded_eps % args.validate_every == 0)
        n_frames, encoded = process_episode(
            ep_dir, out_root / ep_dir.name, vae, scale, args.batch_size,
            device, validate, args.copy_masks, pool, decoder, compiled,
            sidecars=tuple(s for s in args.sidecars.split(",") if s))
        done_eps += 1
        encoded_eps += int(encoded)
        if encoded:
            done_frames += n_frames
            rate = done_frames / max(time.time() - t0, 1e-9)
            print(f"[rank {rank}] {done_eps}/{len(mine)} {ep_dir.name} "
                  f"({n_frames} frames, {rate:.0f} f/s)", flush=True)
        elif done_eps % 500 == 0:
            print(f"[rank {rank}] {done_eps}/{len(mine)} (skipping existing)", flush=True)
    print(f"[rank {rank}] DONE: {encoded_eps} encoded, "
          f"{done_eps - encoded_eps} skipped, {done_frames:,} frames "
          f"in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
