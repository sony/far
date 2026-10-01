"""Pre-compute SDXL-VAE latents for the SoundSpaces loopnav episodes.

For each episode directory under --dataset-root (layout:
  <traj_id>/{frames/*.jpg, audio.wav, poses.npz, actions.npz, closures.npz,
             legs.npz, meta.json, preview.mp4})
this script:
  1. Encodes every frame (256x256 RGB JPEG) with the SDXL autoencoder
     (`madebyollin/sdxl-vae-fp16-fix` — the stock SDXL VAE overflows in fp16)
     to a single latents.npy of shape (T, 4, 32, 32), float16, saved at
     `tokenize()` scale (multiplied by the VAE's scaling_factor, 0.13025 —
     the same convention as the loopnav ViTVAE latents: downstream consumers
     divide by the scale to decode).
  2. Copies the episode's sidecars (audio.wav, poses.npz, actions.npz,
     closures.npz, legs.npz, meta.json) next to the latents, so
     datasets/soundspaces/latent/<traj_id>/ is self-contained. frames/ and
     preview.mp4 are NOT copied — the latents replace them.

Determinism: latents are the posterior MEAN (`latent_dist.mode()`), so
re-encoding an episode is bit-identical — safe to resume and to spot-check.

Resume-safe: episodes whose latents.npy already exists are skipped (missing
sidecar copies are still filled in). All writes are atomic (tmp + rename).

Memory: only the current batch of frames and the current episode's (T, 4,
32, 32) fp16 array (~5 MB) are ever held — nothing accumulates across
episodes.

Single GPU:
    python scripts/data/build_latent_dataset_soundspaces.py \\
        --dataset-root /path/to/soundspaces_raw_episodes \\
        --output-dir   datasets/soundspaces/latent

Multi-GPU (one independent process per GPU; episodes[rank::world_size]):
    torchrun --nproc_per_node=8 scripts/data/build_latent_dataset_soundspaces.py [...]
No process-group initialization is needed — ranks operate independently.
"""

import argparse
import json
import os
import shutil
import time
import wave
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SIDECARS = ("audio.wav", "poses.npz", "actions.npz", "closures.npz",
            "legs.npz", "meta.json")


def load_vae(name: str, device: torch.device):
    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(name, torch_dtype=torch.float16)
    vae = vae.to(device).eval()
    scale = float(vae.config.scaling_factor)
    return vae, scale


@torch.no_grad()
def encode_episode(frame_paths, vae, scale, batch_size, device):
    """Encode JPEG frames -> (T, 4, H/8, W/8) float16 at tokenize() scale."""
    out = []
    for start in range(0, len(frame_paths), batch_size):
        imgs = np.stack([np.asarray(Image.open(p).convert("RGB"))
                         for p in frame_paths[start:start + batch_size]])
        x = torch.from_numpy(imgs).permute(0, 3, 1, 2).to(device)
        x = x.to(torch.float16).div_(127.5).sub_(1.0)              # [-1, 1]
        z = vae.encode(x).latent_dist.mode().mul_(scale)           # posterior MEAN
        out.append(z.to(torch.float16).cpu())
    return torch.cat(out).numpy()                                  # (T, 4, h, w) fp16


@torch.no_grad()
def roundtrip_psnr(lat, frame_path, vae, scale, device):
    """Decode latents[0] and PSNR it against the source frame (sanity only)."""
    z = torch.from_numpy(lat[:1]).to(device, torch.float16) / scale
    rec = vae.decode(z).sample.float().clamp(-1, 1).add(1).div(2)
    ref = torch.from_numpy(
        np.asarray(Image.open(frame_path).convert("RGB"))).permute(2, 0, 1)
    ref = ref.float().div(255).to(device)
    mse = (rec[0] - ref).pow(2).mean()
    return float(-10 * torch.log10(mse))


def atomic_save_npy(arr, path: Path):
    for stale in path.parent.glob(path.name + ".tmp*"):
        stale.unlink()              # orphans from a killed/preempted run
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp, "wb") as f:      # file handle: np.save must NOT append ".npy"
        np.save(f, arr)
    os.replace(tmp, path)


def process_episode(ep_dir: Path, out_dir: Path, vae, scale, batch_size,
                    device, validate: bool) -> tuple[int, bool]:
    """Encode one episode + copy sidecars. Returns (n_frames, encoded?)."""
    meta = json.loads((ep_dir / "meta.json").read_text())
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sidecar copies first (cheap; also backfills episodes whose latents were
    # written by an interrupted earlier run).
    for name in SIDECARS:
        src, dst = ep_dir / name, out_dir / name
        if src.exists() and not dst.exists():
            tmp = dst.with_suffix(dst.suffix + f".tmp{os.getpid()}")
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)

    npy_path = out_dir / "latents.npy"
    if npy_path.exists():
        return meta["n_frames"], False

    frame_paths = sorted((ep_dir / "frames").glob("*.jpg"))
    assert len(frame_paths) == meta["n_frames"], \
        f"{ep_dir.name}: {len(frame_paths)} frames on disk vs meta n_frames={meta['n_frames']}"
    with Image.open(frame_paths[0]) as im:
        assert im.size == (meta["resolution"], meta["resolution"]), \
            f"{ep_dir.name}: frame size {im.size} vs meta resolution {meta['resolution']}"
    with wave.open(str(ep_dir / "audio.wav")) as w:
        audio_s = w.getnframes() / w.getframerate()
    expect_s = meta["n_frames"] / meta["fps"]
    if abs(audio_s - expect_s) > 1.0 / meta["fps"]:
        print(f"[warn] {ep_dir.name}: audio {audio_s:.2f}s vs frames {expect_s:.2f}s")

    lat = encode_episode(frame_paths, vae, scale, batch_size, device)
    if validate:
        psnr = roundtrip_psnr(lat, frame_paths[0], vae, scale, device)
        print(f"[validate] {ep_dir.name}: roundtrip PSNR {psnr:.1f} dB "
              f"| latent std {float(np.std(lat.astype(np.float32))):.3f}")
    atomic_save_npy(lat, npy_path)
    return meta["n_frames"], True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root",
                        required=True, help="raw episode tree (one directory per episode)")
    parser.add_argument("--output-dir", default="datasets/soundspaces/latent")
    parser.add_argument("--vae", default="madebyollin/sdxl-vae-fp16-fix",
                        help="diffusers AutoencoderKL repo id (fp16-safe SDXL VAE)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Frames per VAE forward pass")
    parser.add_argument("--validate-every", type=int, default=200,
                        help="Decode-roundtrip PSNR print every N encoded episodes "
                             "on this rank (0 = never)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N episodes on this rank (smoke runs)")
    args = parser.parse_args()

    rank       = int(os.environ.get("RANK",       0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    root = Path(args.dataset_root)
    episodes = sorted(d for d in root.iterdir() if d.is_dir())
    mine = episodes[rank::world_size]
    if args.limit is not None:
        mine = mine[: args.limit]

    vae, scale = load_vae(args.vae, device)
    print(f"[rank {rank}/{world_size}] {len(mine)} episodes on {device} "
          f"| vae={args.vae} scale={scale}")

    out_root = Path(args.output_dir)
    done_frames = done_eps = encoded_eps = 0
    t0 = time.time()
    for i, ep_dir in enumerate(mine):
        # validate fires on every N-th ENCODED episode (skipped episodes never
        # reach the validation branch inside process_episode).
        validate = (args.validate_every > 0 and encoded_eps % args.validate_every == 0)
        n_frames, encoded = process_episode(
            ep_dir, out_root / ep_dir.name, vae, scale, args.batch_size,
            device, validate)
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
