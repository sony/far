"""
Pre-compute ViTVAE latents for all LoopNav WebDataset shards.

For each shard assigned to this rank:
  1. Download the .tar from HuggingFace
  2. Extract all (avi, json) clip pairs
  3. Encode every frame to a latent vector using the ViTVAE tokenizer
  4. Save latents as <clip>.npy and pose data as <clip>.json to the output directory
  5. Delete the .tar file

Latents are computed from the posterior mean (mode), not a stochastic sample,
so the output is deterministic.

Resume-safe: clips whose .npy already exists are skipped.

Single-process usage:
    python scripts/data/build_latent_dataset_loopnav.py \\
        --manifest   results/manifests/loopnav/loopnav.json \\
        --output-dir results/latents/loopnav \\
        --ckpt       models/oasis_500m_vit_vae.pth \\
        [--batch-size 64]

Multi-GPU usage (torchrun); set CUDA_VISIBLE_DEVICES to control which GPUs are used:
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \\
        scripts/data/build_latent_dataset_loopnav.py \\
        --manifest   results/manifests/loopnav/loopnav.json \\
        --output-dir results/latents/loopnav \\
        --ckpt       models/oasis_500m_vit_vae.pth \\
        [--batch-size 64]

Each rank processes a disjoint subset of shards (shards[rank::world_size]).
No process-group initialization is needed — ranks operate independently.

Set HF_TOKEN in the environment for private repositories.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm

from far.data.backends import _decode_all_av_frames
from far.data.transforms import make_resize
from far.tokenizers.vit_vae import Decoder, Encoder, ViTVAE

# ── Architecture constants (matches configs/tokenizer/vit_vae.yaml @ 360×640) ──
_IMAGE_HEIGHT  = 360
_IMAGE_WIDTH   = 640
_PATCH_SIZE    = 20
_DIM           = 1024
_ENCODER_DEPTH = 6
_DECODER_DEPTH = 12
_HEADS         = 16
_LATENT_DIM    = 16


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(ckpt_path: str, device: torch.device) -> ViTVAE:
    encoder = Encoder(
        input_height=_IMAGE_HEIGHT, input_width=_IMAGE_WIDTH,
        patch_size=_PATCH_SIZE, dim=_DIM, depth=_ENCODER_DEPTH,
        heads=_HEADS, latent_dim=_LATENT_DIM,
    )
    decoder = Decoder(
        input_height=_IMAGE_HEIGHT, input_width=_IMAGE_WIDTH,
        patch_size=_PATCH_SIZE, dim=_DIM, depth=_DECODER_DEPTH,
        heads=_HEADS, latent_dim=_LATENT_DIM,
    )
    model = ViTVAE(encoder=encoder, decoder=decoder)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(sd)
    return model.eval().to(device)


# ── Encoding ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_frames(
    frames: torch.Tensor,       # (T, C, H, W) uint8
    transform: nn.Module,
    model: ViTVAE,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Return (T, latent_dim, seq_h, seq_w) float32 numpy array."""
    T = frames.shape[0]
    chunks = []
    for start in range(0, T, batch_size):
        chunk = transform(frames[start : start + batch_size]).to(device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
            latent = model.tokenize(chunk).float()
        chunks.append(latent.cpu())
    return torch.cat(chunks, dim=0).numpy()


# ── Shard helpers ─────────────────────────────────────────────────────────────

def _download(url: str, dest: Path, desc: str = "") -> None:
    token = os.environ.get("HF_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0)) or None
        with tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024,
                  desc=desc, leave=False) as bar:
            while True:
                block = resp.read(1 << 20)  # 1 MiB chunks
                if not block:
                    break
                f.write(block)
                bar.update(len(block))


def _iter_clips(tar_path: Path):
    """Yield (clip_key, avi_bytes, json_bytes) for each complete clip in the tar.

    clip_key preserves the full relative path within the tar minus the extension,
    e.g. "subdir/clip_001" for a member named "subdir/clip_001.avi".
    """
    data: dict[str, dict[str, bytes]] = {}
    with tarfile.open(tar_path, "r") as tf:
        for member in tf.getmembers():
            p = Path(member.name)
            ext = p.suffix.lstrip(".")
            if ext not in ("avi", "json"):
                continue
            fobj = tf.extractfile(member)
            if fobj is None:
                continue
            key = str(p.with_suffix(""))  # full relative path, no extension
            data.setdefault(key, {})[ext] = fobj.read()

    for key, exts in data.items():
        if "avi" in exts and "json" in exts:
            yield key, exts["avi"], exts["json"]


def process_shard(
    shard: dict,
    output_dir: Path,
    model: ViTVAE,
    transform: nn.Module,
    batch_size: int,
    device: torch.device,
) -> tuple[int, int]:
    """Download, encode, and save one shard. Returns (n_saved, n_skipped)."""
    shard_stem = Path(shard["shard_id"]).with_suffix("")  # e.g. "ABA/desert_village_1"
    shard_out  = output_dir / shard_stem
    shard_out.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
        tar_path = Path(tmp.name)

    n_saved = n_skipped = 0
    try:
        _download(shard["path"], tar_path, desc=f"↓ {shard_stem.name}")

        for clip_name, avi_bytes, json_bytes in tqdm(_iter_clips(tar_path), desc="  encoding", unit="clip", leave=False):
            npy_path  = shard_out / f"{clip_name}.npy"
            json_path = shard_out / f"{clip_name}.json"
            npy_path.parent.mkdir(parents=True, exist_ok=True)

            if npy_path.exists():
                n_skipped += 1
                continue

            frames = _decode_all_av_frames(
                io.BytesIO(avi_bytes), strict=False, name=clip_name
            )
            if frames.shape[0] == 0:
                continue

            latents = encode_frames(frames, transform, model, batch_size, device)
            np.save(npy_path, latents)
            json_path.write_bytes(json_bytes)
            n_saved += 1
    finally:
        tar_path.unlink(missing_ok=True)

    return n_saved, n_skipped


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest",   default="results/manifests/loopnav/loopnav.json",
                        help="Path to the loopnav manifest JSON")
    parser.add_argument("--output-dir", default="datasets/loopnav/latent",
                        help="Root directory for saved latents")
    parser.add_argument("--ckpt",       default="models/oasis_500m_vit_vae.pth",
                        help="Path to the ViTVAE checkpoint")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Frames per tokenizer forward pass")
    parser.add_argument("--split",      default=None,
                        help="Only process shards with this split label (default: all)")
    args = parser.parse_args()

    # Distributed: read torchrun env vars; no process group needed
    rank       = int(os.environ.get("RANK",       0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(args.manifest).read_text())
    shards   = manifest["samples"]
    if args.split:
        shards = [s for s in shards if s.get("split") == args.split]

    # Guard against duplicate shard_ids: two entries with the same id would map
    # to the same output directory and could be processed by different ranks
    # concurrently, producing undefined output.
    seen: dict[str, str] = {}
    for s in shards:
        sid = s["shard_id"]
        if sid in seen:
            raise ValueError(
                f"Manifest has duplicate shard_id '{sid}'. "
                f"Same-named tars from different directories must have distinct shard_ids."
            )
        seen[sid] = sid

    # Partition shards across ranks
    my_shards = shards[rank::world_size]

    if rank == 0:
        print(f"Manifest: {len(shards)} shards total  |  world_size: {world_size}  |  output: {output_dir}")
    print(f"[rank {rank}] device: {device}  |  shards assigned: {len(my_shards)}")

    print(f"[rank {rank}] Loading tokenizer...")
    model     = build_model(args.ckpt, device)
    transform = make_resize(_IMAGE_HEIGHT, _IMAGE_WIDTH)

    total_saved = total_skipped = 0
    for shard in tqdm(my_shards, desc=f"Shards [rank {rank}]", unit="shard",
                      position=rank, leave=True):
        shard_id = shard["shard_id"]
        try:
            n_saved, n_skipped = process_shard(
                shard, output_dir, model, transform, args.batch_size, device
            )
            total_saved   += n_saved
            total_skipped += n_skipped
            tqdm.write(f"  [rank {rank}] {shard_id}: {n_saved} saved, {n_skipped} skipped")
        except Exception as e:
            tqdm.write(f"  [rank {rank}] {shard_id}: ERROR — {e}")

    print(f"[rank {rank}] Done. {total_saved} clips saved, {total_skipped} already existed.")


if __name__ == "__main__":
    main()
