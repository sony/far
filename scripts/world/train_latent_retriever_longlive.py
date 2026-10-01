"""LongLive-RAG retrieval-embedding pre-training (faithful port).

Trains a LongLive-RAG-style (arXiv:2606.02553) retrieval embedding on OUR
pre-computed latent clips — the action-AGNOSTIC, content-similarity baseline
retriever. The LOSSES are the faithful port from the reference repo
(github.com/qixinhu11/LongLive-RAG, `ae/train.py`); the ENCODER deliberately
departs from their conv AE: it is OUR `QueryViT` in content mode (no action /
stride → the query-token output IS the retrieval embedding, dim `key_dim`).
That gives architecture/capacity parity with the contrastive retriever, so
the baseline comparison isolates the OBJECTIVE (recon + temporal delta +
smoothness, action-agnostic) as the only difference — and the checkpoint's
`"memory"` entry is exactly `{query_vit.*}`, drop-in compatible with
`PnPRetrieval.load_weights` and `build_retriever_keys`. Losses per the paper:

    L_rec      = ||D(E(x_t)) - x_t||^2                                  (eq. 3)
    L_Delta    = lambda_D * max(0, cos(v_t, v_ref) - m)                 (eq. 6)
    L_SeqDelta = mean over offsets tau in [1, w] of mean_t L_Delta      (eq. 7)
    L_Smooth   = lambda_s * mean_t ||v_t - 2 v_{t-1} + v_{t-2}||_2      (eq. 8)
    L_Total    = L_rec + L_SeqDelta + L_Smooth                          (eq. 9)

Reference defaults (ae/configs/ae_delta.yaml): margin 0.85, window 3, both
loss weights 1.0, AdamW lr 3e-4 / wd 1e-4, global batch 128, seq_len 8
CONTIGUOUS frames (their latent_dim 1024 is replaced by the QueryViT's
key_dim). The data side must supply contiguous clips: run with the dataset's
`random_clip` sampler at frame_stride=1 (the temporal-delta window presumes
adjacent-frame redundancy).

Only the DECODER is defined here (their upsample+conv design, input adapted
to key_dim); it exists to carry the reconstruction loss and is discarded
after pre-training — retrieval uses the QueryViT + L2-normalised embeddings,
cosine similarity, query = the most recent frame's embedding (paper Alg. 1).

Entry config: `configs/world_pretrain.yaml` (paths/optimizer/schedule shared
with the contrastive pre-train; the `contrastive:` block is unused here).
LongLive knobs arrive as `+longlive.*` CLI overrides (see the launcher).
"""
import os
from datetime import datetime
from time import time

import hydra
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from torch.distributed import all_reduce, barrier, destroy_process_group, ReduceOp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import far.distributed as dist
from far.constants import CONFIG_ROOT, RESULT_ROOT
from far.distributed import create_logger
from far.experiment import save_run_info, init_wandb


def make_exp_name(cfg):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    datasets = '+'.join(cfg.datasets)
    return f"{timestamp}__pretrain_retriever__{cfg.run_name}__bs{cfg.batch_size}__{datasets}"


# ─────────────────────────────────────────────────────────────────────────────
# Decoder — ported from LongLive-RAG `ae/model.py` (upsample+conv, GroupNorm/
# SiLU/ResBlocks, zero-init residual tails). The encoder is OUR QueryViT.
# ─────────────────────────────────────────────────────────────────────────────

def _num_groups(channels: int, preferred: int = 32) -> int:
    for g in (preferred, 16, 8, 4, 1):
        if channels % g == 0:
            return g
    return 1


class ResBlock(nn.Module):
    """Pre-norm residual block: GN → SiLU → Conv → GN → SiLU → Conv + skip."""

    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(_num_groups(dim), dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(_num_groups(dim), dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1),
        )
        nn.init.zeros_(self.block[-1].weight)
        nn.init.zeros_(self.block[-1].bias)

    def forward(self, x):
        return x + self.block(x)


class Decoder(nn.Module):
    """Embedding → (C, H, W) via upsample + conv (no checkerboard); crop to size."""

    def __init__(self, latent_dim, hidden_dims, out_channels, feat_shape, target_h, target_w):
        super().__init__()
        self._feat_shape = feat_shape
        flat = 1
        for s in feat_shape:
            flat *= s
        self.fc = nn.Sequential(nn.Linear(latent_dim, flat), nn.SiLU(inplace=True))
        rev, layers = list(reversed(hidden_dims)), []
        for i in range(len(rev) - 1):
            layers += [nn.Upsample(scale_factor=2, mode='nearest'),
                       nn.Conv2d(rev[i], rev[i + 1], 3, padding=1),
                       nn.GroupNorm(_num_groups(rev[i + 1]), rev[i + 1]),
                       nn.SiLU(inplace=True),
                       ResBlock(rev[i + 1])]
        self.upsample_stack = nn.Sequential(*layers)
        self.final = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                   nn.Conv2d(rev[-1], out_channels, 3, padding=1))
        self._target_h, self._target_w = target_h, target_w

    def forward(self, z):
        h = self.fc(z).view(-1, *self._feat_shape)
        h = self.final(self.upsample_stack(h))
        return h[:, :, :self._target_h, :self._target_w]


def _open_adaln_gates(query_vit, std: float = 0.02) -> int:
    """QueryViT's adaLN-zero blocks are IDENTITY at init: every block's
    modulation weight AND bias are zero, so the q_token never reads the frame
    and the embedding is a CONSTANT (verified: cross-input cosine = 1.000000
    at init). InfoNCE has nonzero gradient at that collapse point and opens
    the gates during contrastive pre-training; the LongLive losses do NOT —
    the cosine hinge's gradient is exactly zero for identical embeddings and
    recon settles for the dataset-mean latent (the observed delta=0.1500 /
    recon=0.77 plateau). Kick the modulation with small noise so the encoder
    is input-dependent from step 0. Applied ONLY here — pnp_cl keeps its
    zero-init for the contrastive / diffusion fine-tuning paths."""
    n = 0
    with torch.no_grad():
        for name, p in query_vit.named_parameters():
            if 'adaLN_modulation' in name:
                p.normal_(0, std)
                n += 1
    return n


class LongLivePretrainModule(nn.Module):
    """QueryViT encoder (content mode) + in-script conv decoder.

    The embedding is the QueryViT's query-token output with NO action/stride
    (a content KEY, dim key_dim). The decoder is the reference repo's
    upsample+conv design with its bottleneck derived from the target spatial
    size (ceil(H / 2^n), ceil(W / 2^n) for n hidden dims) instead of a conv
    encoder's trace. state_dict splits cleanly: `query_vit.*` (the retriever —
    saved as the checkpoint's "memory" entry) and `decoder.*` (auxiliary).
    """

    def __init__(self, query_vit, in_channels, spatial_h, spatial_w, hidden_dims):
        super().__init__()
        self.query_vit = query_vit
        key_dim = query_vit.key_head.out_features
        n = len(hidden_dims)
        feat = (hidden_dims[-1], -(-spatial_h // 2 ** n), -(-spatial_w // 2 ** n))
        self.decoder = Decoder(key_dim, hidden_dims, in_channels, feat, spatial_h, spatial_w)
        self._init_decoder()

    def _init_decoder(self):
        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Global init above overwrote the ResBlock tails — restore identity-at-init.
        for m in self.decoder.modules():
            if isinstance(m, ResBlock):
                nn.init.zeros_(m.block[-1].weight)
                nn.init.zeros_(m.block[-1].bias)

    def forward(self, x):
        embed = self.query_vit(x)                 # no action/stride → content key
        return self.decoder(embed), embed

    @torch.no_grad()
    def encode(self, x, normalize: bool = True):
        embed = self.query_vit(x)
        return F.normalize(embed, dim=-1) if normalize else embed


def compute_losses(model, chunk, delta_margin, delta_weight, delta_window, smooth_weight):
    """Composite LongLive loss on one batch of contiguous chunks (B, S, C, H, W).

    Ported from the reference `ae/train.py::compute_losses`: recon MSE + the
    window temporal-delta hinge averaged over offsets 1..w + the second-order
    trajectory smoothness (L2 norm of embedding acceleration).
    """
    B, S = chunk.shape[:2]
    x_flat = chunk.flatten(0, 1)
    recon, embed = model(x_flat)
    recon_loss = F.mse_loss(recon.float(), x_flat.float())

    embed_seq = embed.view(B, S, -1)
    # Temporal losses on the L2-NORMALIZED trajectory. Deviation from the
    # reference (raw v there): our ViT embedding norm (~18) makes a raw-scale
    # smoothness term dominate the objective, and its trivial minimum is a
    # constant trajectory — observed to re-collapse the encoder within ~50
    # steps. Retrieval operates on L2-normalized embeddings (paper Alg. 1),
    # so the normalized trajectory is the retrieval-relevant one; this also
    # makes the loss balance architecture/scale-invariant. The delta term is
    # cosine-based and unaffected.
    emb_n = F.normalize(embed_seq, dim=-1)
    delta_loss = torch.zeros((), device=chunk.device)
    max_window = min(S - 1, delta_window)
    for k in range(1, max_window + 1):
        sim = (emb_n[:, k:, :] * emb_n[:, :-k, :]).sum(-1)
        delta_loss = delta_loss + delta_weight * F.relu(sim - delta_margin).mean()
    if max_window > 0:
        delta_loss = delta_loss / max_window

    smooth_loss = torch.zeros((), device=chunk.device)
    if S >= 3 and smooth_weight > 0:
        accel = emb_n[:, 2:, :] - 2 * emb_n[:, 1:-1, :] + emb_n[:, :-2, :]
        smooth_loss = torch.sqrt(accel.float().pow(2).sum(dim=-1) + 1e-8).mean() * smooth_weight

    return recon_loss + delta_loss + smooth_loss, recon_loss, delta_loss, smooth_loss


@torch.no_grad()
def evaluate_longlive(model, test_dataset, rank, batch_size, num_workers, device,
                      knobs, seed, max_batches=20):
    """Mean (total, recon, delta, smooth) on a fixed held-out subset."""
    model.eval()
    sampler = DistributedSampler(test_dataset, num_replicas=dist.get_world_size(),
                                 rank=rank, shuffle=True, seed=seed)
    loader = DataLoader(test_dataset, batch_size=batch_size, sampler=sampler,
                        num_workers=num_workers, pin_memory=True, drop_last=True)
    sums = torch.zeros(4, device=device)
    n = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        chunk = batch['frames'].to(device)
        losses = compute_losses(model, chunk, *knobs)
        sums += torch.stack([l.detach() for l in losses])
        n += 1
    sums /= max(n, 1)
    all_reduce(sums, op=ReduceOp.SUM)
    model.train()
    return (sums / dist.get_world_size()).tolist()


@hydra.main(version_base=None, config_path=CONFIG_ROOT, config_name='world_pretrain')
def main(cfg: DictConfig):
    assert torch.cuda.is_available(), "Pre-training currently requires at least one GPU."

    _, rank, device, _ = dist.init_distributed()
    seed = cfg.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)

    exp_name = make_exp_name(cfg)
    result_dir = os.path.join(RESULT_ROOT, exp_name)
    checkpoint_dir = os.path.join(result_dir, 'checkpoints')
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        save_run_info(result_dir, cfg)
        logger = create_logger(result_dir)
        logger.info(f"Experiment directory created at {result_dir}")
        wandb_run = init_wandb(cfg, exp_name, result_dir)
    else:
        logger = create_logger(None)
        wandb_run = None
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # ── Data: CONTIGUOUS latent clips — swap in the random_clip sampler and the
    # bare frame formatter (config surgery on the dataset blocks, so per-dataset
    # yamls need no LongLive-specific entries). Ensure frame_stride=1 from the
    # launcher — the temporal-delta window presumes adjacent-frame redundancy.
    OmegaConf.set_struct(cfg, False)
    datasets = {'train': [], 'test': []}
    for dn in cfg.datasets:
        stride = int(cfg[dn].sampler.random_clip.frame_stride)
        if stride != 1:
            logger.info(f"WARNING: {dn} random_clip.frame_stride={stride} != 1 — "
                        "the temporal-delta window presumes ADJACENT frames")
        for split in ['train', 'test']:
            if split in cfg[dn]:
                ds_cfg = OmegaConf.merge(cfg[dn][split])          # deep copy
                ds_cfg.formatter = cfg[dn].formatter.frame
                ds_cfg.base_dataset.sampler = cfg[dn].sampler.random_clip
                ds = instantiate(ds_cfg)
                datasets[split].append(ds)
                logger.info(f"Dataset: {dn} ({split}, random_clip), size: {len(ds)}")
    datasets['train'] = ConcatDataset(datasets['train'])
    datasets['test']  = ConcatDataset(datasets['test'])

    # ── Model: QueryViT encoder from the config (content mode), decoder dims
    # probed from the data (no shape knobs to drift).
    probe = datasets['train'][0]['frames']
    S, C, H, W = probe.shape
    ll = cfg.get('longlive', OmegaConf.create({}))
    hidden_dims  = [int(h) for h in ll.get('hidden_dims', [64, 128, 256])]
    delta_weight = float(ll.get('delta_weight', 1.0))
    delta_margin = float(ll.get('delta_margin', 0.85))
    delta_window = int(ll.get('delta_window', 3))
    smooth_weight = float(ll.get('smooth_weight', 1.0))
    knobs = (delta_margin, delta_weight, delta_window, smooth_weight)
    qv_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model.memory.query_vit, resolve=True))
    query_vit = instantiate(qv_cfg)
    assert (qv_cfg.img_height, qv_cfg.img_width, qv_cfg.in_chans) == (H, W, C), (
        f"QueryViT config ({qv_cfg.img_height}, {qv_cfg.img_width}, {qv_cfg.in_chans}-ch) "
        f"does not match the data's latents ({H}, {W}, {C}-ch)")
    n_gates = _open_adaln_gates(query_vit)
    logger.info(f"Opened {n_gates} zero-init adaLN gate tensors in the QueryViT "
                "(identity-at-init encoder has no escape gradient under the LongLive losses)")
    ae = LongLivePretrainModule(query_vit, C, H, W, hidden_dims).to(device)
    logger.info(f"LongLive module: clip S={S}, latents ({C}, {H}, {W}) -> "
                f"{query_vit.key_head.out_features}-dim QueryViT key; decoder hidden={hidden_dims}, "
                f"margin={delta_margin}, window={delta_window}, "
                f"weights (delta={delta_weight}, smooth={smooth_weight})")

    lr = float(cfg.lr)
    opt = torch.optim.AdamW(ae.parameters(), lr=lr, weight_decay=float(cfg.weight_decay), eps=1e-6)
    logger.info(f"Optimizer: AdamW on LatentAE, lr={lr}")
    warmup_steps = int(cfg.warmup_steps)
    scheduler = (
        torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.2, end_factor=1.0, total_iters=warmup_steps)
        if warmup_steps > 0 else None
    )

    # Resume — "memory" holds EXACTLY {query_vit.*} (the contrastive-pretrain
    # format: drop-in for PnPRetrieval.load_weights / build_retriever_keys);
    # the auxiliary decoder rides in its own "decoder" entry.
    latest_path = os.path.join(checkpoint_dir, "latest.pth.tar")
    train_steps = 0
    resume_path = latest_path if os.path.isfile(latest_path) else (cfg.from_checkpoint or None)
    if resume_path is not None and os.path.isfile(resume_path):
        logger.info(f"Resuming LongLive pre-training from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        mem_ckp = {k.replace('_orig_mod.', ''): v for k, v in ckpt["memory"].items()}
        logger.info(f"Loading QueryViT weights: {ae.query_vit.load_state_dict(
            {k.removeprefix('query_vit.'): v for k, v in mem_ckp.items()}, strict=True)}")
        if "decoder" in ckpt:
            ae.decoder.load_state_dict(ckpt["decoder"], strict=True)
        if "memory_opt" in ckpt:
            opt.load_state_dict(ckpt["memory_opt"])
        if "pretrain_steps" in ckpt:
            train_steps = ckpt["pretrain_steps"]
            if scheduler is not None:
                for _ in range(train_steps):
                    scheduler.step()

    # Content mode never touches the QueryViT's action/stride embedders, so
    # their params see no gradient — DDP must tolerate unused parameters.
    ae = DDP(ae, device_ids=[device], find_unused_parameters=True)
    logger.info(f"AE Parameters: {sum(p.numel() for p in ae.parameters()):,}")

    sampler = DistributedSampler(datasets['train'], num_replicas=dist.get_world_size(),
                                 rank=rank, shuffle=True, seed=cfg.global_seed)
    loader = DataLoader(datasets['train'], batch_size=cfg['batch_size'], shuffle=False,
                        sampler=sampler, num_workers=cfg['num_workers'], pin_memory=True,
                        drop_last=True, persistent_workers=True)
    logger.info(f"Train set: {len(datasets['train']):,} samples")

    ae.train()
    running = torch.zeros(4, device=device)
    n_steps = 0
    start_time = time()
    logger.info(f"Pre-training the LongLive AE for {cfg.iterations} iterations...")
    epoch = 0
    while train_steps < cfg.iterations:
        sampler.set_epoch(epoch)
        for batch in loader:
            if train_steps >= cfg.iterations:
                break
            chunk = batch['frames'].to(device)                       # (B, S, C, H, W)
            total, recon, delta, smooth = compute_losses(ae, chunk, *knobs)
            opt.zero_grad()
            total.backward()
            if cfg.grad_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(ae.parameters(), max_norm=cfg['grad_clip_val'])
            opt.step()
            if scheduler is not None:
                scheduler.step()

            running += torch.stack([total.detach(), recon.detach(), delta.detach(), smooth.detach()])
            n_steps += 1
            train_steps += 1

            if train_steps % cfg.log_every == 0:
                torch.cuda.synchronize()
                steps_per_sec = n_steps / (time() - start_time)
                avg = running / n_steps
                all_reduce(avg, op=ReduceOp.SUM)
                avg = (avg / dist.get_world_size()).tolist()
                logger.info(
                    f"(step={train_steps:07d}) Total: {avg[0]:.4f} "
                    f"(recon={avg[1]:.4f}, delta={avg[2]:.4f}, smooth={avg[3]:.4f}), "
                    f"Steps/Sec: {steps_per_sec:.2f}"
                )
                if wandb_run is not None:
                    wandb_run.log({'train/total': avg[0], 'train/recon': avg[1],
                                   'train/delta': avg[2], 'train/smooth': avg[3],
                                   'train/steps_per_sec': steps_per_sec}, step=train_steps)
                running = torch.zeros(4, device=device)
                n_steps = 0
                start_time = time()

            if train_steps % cfg.ckpt_every == 0 and train_steps > 0 and rank == 0:
                sd = ae.module.state_dict()
                ckpt = {
                    # exactly {query_vit.*} — the contrastive-pretrain format
                    "memory": {k: v for k, v in sd.items() if k.startswith("query_vit.")},
                    "decoder": ae.module.decoder.state_dict(),
                    "memory_opt": opt.state_dict(),
                    "pretrain_steps": train_steps,
                }
                torch.save(ckpt, f"{checkpoint_dir}/latest.pth.tar")
                torch.save(ckpt, f"{checkpoint_dir}/{train_steps:07d}.pth.tar")
                logger.info(f"Saved checkpoint at step {train_steps}")

            if train_steps % cfg.eval_every == 0 and train_steps > 0:
                t0 = time()
                val = evaluate_longlive(ae, datasets['test'], rank, cfg['batch_size'],
                                        cfg['num_workers'], device, knobs, cfg.global_seed)
                barrier()
                logger.info(f"(step={train_steps:07d}) Val Total: {val[0]:.4f} "
                            f"(recon={val[1]:.4f}, delta={val[2]:.4f}, smooth={val[3]:.4f}), "
                            f"Eval Time: {time()-t0:.2f}")
                if wandb_run is not None:
                    wandb_run.log({'eval/total': val[0], 'eval/recon': val[1],
                                   'eval/delta': val[2], 'eval/smooth': val[3]}, step=train_steps)
                start_time = time()
        epoch += 1

    if rank == 0 and wandb_run is not None:
        wandb_run.finish()
    barrier()
    destroy_process_group()


if __name__ == "__main__":
    main()
