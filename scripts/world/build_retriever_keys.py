"""Pre-compute frozen-QueryViT content keys for every pre-computed latent frame.

For each `<clip>.npy` latent file under `--latent-dir`, encode every frame's
CONTENT key — `query_vit(latent_frame, no_action)` — with the frozen retriever
loaded from `--ckpt`, and save the result next to the latent as
`<clip><key-suffix>` (default `<clip>.keys.npy`), shape `(T, key_dim)` aligned
frame-for-frame with the latent.

These keys are a deterministic function of the latents + the frozen QueryViT
weights, exactly like the latents are a function of the frames + the VAE — so
they're pre-computed once and loaded by `LatentLoopNavBackend` at train time,
turning the retriever's per-frame key encodes into cached lookups (cheap
retrieval / per-frame scoring / contrastive negatives). The QueryViT's query
adapter (if any) lives only on the action path, so it does NOT affect keys; the
keys depend solely on the frozen encoder. Re-run this whenever the retriever
checkpoint changes.

Resume-safe: clips whose key file already exists are skipped.

AUDIO mode (`--source audio`, SoundSpaces): per-frame inputs are not stored on
disk — they are causal binaural feature windows derived from each episode's
`audio_feat.npy`. The windows come from `SoundSpacesBackend(audio_mode=
"frames")`, the SAME code path training uses, so the slicing / normalization /
padding contract (audio_feat_params.json) has a single source of truth and the
cached keys provably live in the training key space. One key file per episode
dir: `latents<key-suffix>` (use `--key-suffix .keys_audio.npy` — what
`SoundSpacesBackend.key_suffix` expects), shape `(T, key_dim)` aligned with
`latents.npy`. Writes are atomic (tmp + rename): a preempted write can never
leave a truncated key file that resume-skip would then treat as complete.

    torchrun --nproc_per_node=8 scripts/world/build_retriever_keys.py \
        --source audio --latent-dir datasets/soundspaces/latent \
        --ckpt results/<audio pretrain>/checkpoints/0400000.pth.tar \
        --key-suffix .keys_audio.npy \
        --override model.memory.query_vit.img_height=64 [...]

Single-process:
    python scripts/world/build_retriever_keys.py \\
        --latent-dir datasets/loopnav/latent \\
        --ckpt results/.../checkpoints/0150000.pth.tar [--batch-size 256]

Multi-GPU (torchrun); each rank processes a disjoint subset of latent files:
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \\
        scripts/world/build_retriever_keys.py \\
        --latent-dir datasets/loopnav/latent \\
        --ckpt results/.../checkpoints/0150000.pth.tar
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

from far.constants import CONFIG_ROOT


# ── Model ─────────────────────────────────────────────────────────────────────

def build_query_vit(ckpt_path: str, device: torch.device, config_name: str,
                    overrides: list[str]):
    """Instantiate QueryViT from the config (arch always matches training) and
    load its weights from the retriever checkpoint's `memory` state_dict.

    `strict=False`: the checkpoint may carry an action-path adapter (`query_proj`)
    that is irrelevant to keys, and the action MLP shapes don't matter either
    (unused on the null-action key path) — only the encoder needs to load."""
    with initialize_config_dir(version_base=None, config_dir=CONFIG_ROOT):
        cfg = compose(config_name=config_name, overrides=['model/memory=pnp_cl'] + overrides)
    qv_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model.memory.query_vit, resolve=True))
    query_vit = instantiate(qv_cfg)

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    mem_sd = ckpt['memory'] if isinstance(ckpt, dict) and 'memory' in ckpt else ckpt
    qv_sd = {
        k.replace('_orig_mod.', '').replace('module.', '')[len('query_vit.'):]: v
        for k, v in mem_sd.items()
        if k.replace('_orig_mod.', '').replace('module.', '').startswith('query_vit.')
    }
    if not qv_sd:
        raise ValueError(f"No 'query_vit.*' weights found in {ckpt_path}")
    incompat = query_vit.load_state_dict(qv_sd, strict=False)
    miss = [k for k in incompat.missing_keys if 'query_proj' not in k]
    if miss or incompat.unexpected_keys:
        print(f"[load] missing(non-adapter)={miss} unexpected={incompat.unexpected_keys}")
    print(f"[load] loaded {len(qv_sd)} query_vit tensors from {ckpt_path}")
    return query_vit.eval().to(device)


# ── Encoding ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_keys(latents: np.ndarray, query_vit, batch_size: int,
                device: torch.device) -> np.ndarray:
    """latents: (T, C, H, W) float32 → (T, key_dim) float32 content keys.

    fp32 throughout to match the memory's fp32 QueryViT islands at train time."""
    T = latents.shape[0]
    out = []
    for s in range(0, T, batch_size):
        # as_tensor: no-op for torch inputs (audio mode passes the backend's
        # strided window VIEW — slicing materializes only this batch).
        batch = torch.as_tensor(latents[s:s + batch_size]).float()
        if device.type == 'cuda':
            batch = batch.pin_memory().to(device, non_blocking=True)              # (b, C, H, W)
        else:
            batch = batch.to(device)
        keys  = query_vit(batch)                                                 # null action → KEY (b, key_dim)
        out.append(keys.float().cpu())
    return torch.cat(out, dim=0).numpy()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--latent-dir', required=True,
                        help='Root directory of pre-computed latent .npy files (searched recursively)')
    parser.add_argument('--ckpt', required=True,
                        help='Retriever checkpoint (pre-training or full-training .pth.tar with a "memory" entry)')
    parser.add_argument('--batch-size', type=int, default=256, help='Frames per QueryViT forward')
    parser.add_argument('--key-suffix', default='.keys.npy',
                        help="Suffix for the saved key file: <clip>.npy -> <clip><key-suffix>. "
                             "Keep '.keys' in it (e.g. .keys.npy, .keys_0300000.npy) so multiple "
                             "key sets coexist and are excluded from the latent scan. Select at "
                             "train time via the backend's `key_suffix`.")
    parser.add_argument('--source', choices=['latents', 'audio'], default='latents',
                        help="'latents': encode stored <clip>.npy latent frames (loopnav). "
                             "'audio': encode causal binaural windows derived per frame via "
                             "SoundSpacesBackend(audio_mode='frames') — one key file per episode dir.")
    parser.add_argument('--latent-name', default=None,
                        help="Restrict discovery to exactly this basename per dir "
                             "(e.g. 'latents.npy') — required when packed mask "
                             "sidecars (latents.agent.npy, latents.change*.npy) "
                             "already sit next to the latents. Default: all *.npy.")
    parser.add_argument('--config-name', default='world_train', help='Hydra entry config for the QueryViT arch')
    parser.add_argument('--override', action='append', default=['action_dim=5'],
                        help='Extra Hydra overrides affecting the query_vit block (repeatable)')
    parser.add_argument('--prefetch', type=int, default=4,
                        help='Clips loaded ahead of the GPU by a background thread pool (0 = synchronous). '
                             'Bounded: at most this many clips are ever resident, so host memory stays flat.')
    parser.add_argument('--shard', type=int, default=0,
                        help='With --num-shards: this launch handles items[shard::num_shards] BEFORE the '
                             'per-rank split, so several torchrun groups (e.g. two per GPU) can share one job.')
    parser.add_argument('--num-shards', type=int, default=1)
    args = parser.parse_args()

    # Distributed: read torchrun env vars; ranks operate independently (no PG).
    rank       = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    latent_dir = Path(args.latent_dir)
    if args.source == 'audio':
        from far.data.backends import SoundSpacesBackend
        backend = SoundSpacesBackend(audio_mode='frames')
        # One episode dir per key file; windows derive from audio_feat.npy.
        items = sorted(p for p in latent_dir.iterdir() if (p / 'audio_feat.npy').is_file())
    elif args.latent_name:
        # Exactly this basename per episode dir — use when other `.npy` sidecars
        # (e.g. packed loss masks `latents.agent.npy` / `latents.change.npy`)
        # coexist with the latents and would otherwise be swept in as "latents".
        items = sorted(latent_dir.rglob(args.latent_name))
    else:
        # All latent .npy files, excluding ANY key-set output (so building one key
        # set never re-encodes another). Key files are identified by `.keys` in the
        # name — keep `--key-suffix` to that convention (`.keys.npy`, `.keys_A.npy`).
        # NOTE: this sweeps in any OTHER `.npy` sidecar (mask families) — pass
        # `--latent-name latents.npy` when such sidecars already exist on disk.
        items = sorted(p for p in latent_dir.rglob('*.npy') if '.keys' not in p.name)
    items = items[args.shard::args.num_shards]
    my_items = items[rank::world_size]

    if rank == 0:
        print(f"{args.source}: {len(items)} items  |  world_size: {world_size}  |  key-suffix: {args.key_suffix}")
    print(f"[rank {rank}] device: {device}  |  items assigned: {len(my_items)}")

    query_vit = build_query_vit(args.ckpt, device, args.config_name, args.override)

    def atomic_save(arr, path: Path):
        tmp = path.with_suffix(path.suffix + f'.tmp{os.getpid()}')
        with open(tmp, 'wb') as f:      # file handle: np.save must NOT append '.npy'
            np.save(f, arr)
        os.replace(tmp, path)

    def key_path_of(item: Path) -> Path:
        return item / f'latents{args.key_suffix}' if args.source == 'audio' \
            else item.with_name(item.stem + args.key_suffix)

    def load_frames(item: Path):
        if args.source == 'audio':
            # (T, 4, n_mels, W) float32 strided window view — the training
            # windowing contract, from the backend itself.
            return backend.load_clip({'path': str(item)})[0]
        return np.load(item)                                                     # (T, C, H, W)

    todo = [it for it in my_items if not key_path_of(it).exists()]
    n_skipped = len(my_items) - len(todo)
    n_saved = 0
    # Bounded prefetch: the pool reads clip i+prefetch from disk while the GPU
    # encodes clip i. `todo` is consumed in order and at most `prefetch`
    # futures are outstanding, so resident host memory is O(prefetch clips).
    pool = ThreadPoolExecutor(max_workers=max(1, args.prefetch)) if args.prefetch > 0 else None
    pending = []
    it = iter(todo)
    def _submit():
        nxt = next(it, None)
        if nxt is not None:
            pending.append((nxt, pool.submit(load_frames, nxt) if pool else None))
    for _ in range(max(1, args.prefetch)):
        _submit()
    pbar = tqdm(total=len(todo), desc=f"Keys [rank {rank}]", unit='clip', position=rank, leave=True)
    while pending:
        item, fut = pending.pop(0)
        _submit()
        try:
            frames = fut.result() if fut is not None else load_frames(item)
            keys = encode_keys(frames, query_vit, args.batch_size, device)       # (T, key_dim)
            atomic_save(keys, key_path_of(item))
            n_saved += 1
        except Exception as e:
            tqdm.write(f"  [rank {rank}] {item}: ERROR — {e}")
        pbar.update(1)
    pbar.close()
    if pool is not None:
        pool.shutdown()

    print(f"[rank {rank}] Done. {n_saved} key files saved, {n_skipped} already existed.")


if __name__ == '__main__':
    main()
