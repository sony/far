"""Contrastive pre-training for the pnp_cl retriever (QueryViT).

This warm-starts the retriever with the frame-level contrastive (InfoNCE)
objective before full world-model training. It builds ONLY the QueryViT (via the
`model.memory.query_vit` sub-config) and wraps it in `ContrastivePretrainModule`,
which computes the InfoNCE loss directly (the memory module itself computes no
contrastive loss). No generator / tokenizer / diffusion / chunk-compressor /
beam search is built. Each step backwards the loss and steps the QueryViT
optimizer.

Entry config: `configs/world_pretrain.yaml` (datasets/samplers/formatters shared
with `world_train.yaml`; the InfoNCE knobs live under its `contrastive:` block).

Checkpoints save the module state (exactly `{query_vit.*}`, since the module's
buffers are non-persistent) under the `"memory"` key, so the result drops
straight into `train_latent.py` as a retriever init: set
`model.memory.ckpt_path=<this run>/checkpoints/latest.pth.tar` there —
`PnPRetrieval.load_weights` loads the `query_vit.*` weights (strict=False) and the
rest of the (retrieval-only) pnp_cl starts fresh.
"""
import math
import os
from datetime import datetime
from time import time

import hydra
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.distributed import all_reduce, barrier, destroy_process_group, ReduceOp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import far.distributed as dist
from far.constants import CONFIG_ROOT, RESULT_ROOT
from far.distributed import create_logger
from far.experiment import save_run_info, init_wandb
from far.memories.worldmem import pose_fov_overlap


_VALID_POSITIVE_PAIR_TYPES = ("temporal", "pose", "jepa")


class ContrastivePretrainModule(nn.Module):
    """Frame-level InfoNCE contrastive loss for pre-training pnp_cl's QueryViT.

    The ONLY trainable module it holds is the QueryViT (the only thing
    retriever pre-training trains); no chunk compressor / beam search / value
    path exists on this path.

    It is a drop-in replacement for the DDP-wrapped `PnPRetrieval` in the
    pre-training loop: `state_dict()` is exactly `{query_vit.*}` (pos_scale /
    pos_offset / last_cl_loss_* are NON-persistent buffers), so the checkpoint
    it writes still loads via `PnPRetrieval.load_weights`; and the whole
    contrastive runs in ONE `forward` (calling `self.query_vit` several times
    internally) so DDP's single-forward-per-backward contract holds.

    Past context (excluding the current frame) is chunked into `chunk_size`
    blocks — dropping the oldest `T_past mod chunk_size` frames exactly as
    pnp_cl did — and flattened into a pool. One QUERY per (sample, goal) is
    `QueryViT(current_frame, action, stride)`; positive/negative pool frames are
    encoded as content KEYs `QueryViT(frame)` (null action). Positives:
      - `temporal`: the `num_positives` most-recent valid pool frames.
      - `pose`:     per goal, the `num_positives` valid pool frames with highest
                    FOV overlap vs the goal pose (recovered from `action` via
                    `pos_scale`/`pos_offset`).
      - `jepa`:     the goal frame's own content key (1/goal).
    Each positive runs a per-positive InfoNCE against `num_negatives` shared
    random valid pool frames. Per-type means are stashed in `last_cl_loss_*`
    buffers for telemetry. Runs in fp32 (autocast disabled) like the original.
    """

    def __init__(
        self,
        query_vit: nn.Module,
        chunk_size: int,
        positive_pair_type,
        num_positives: int,
        num_negatives: int,
        cl_temperature: float,
        cl_pose_radius: float,
        cl_pose_fov_half_h: float,   # DEGREES (converted to radians here, as pnp_cl did)
        cl_pose_fov_half_v: float,   # DEGREES
        cl_pose_num_samples: int,
        local_pos_stats=None,
    ):
        super().__init__()
        positive_pair_type = tuple(positive_pair_type)
        unknown = [t for t in positive_pair_type if t not in _VALID_POSITIVE_PAIR_TYPES]
        assert not unknown, (
            f"unknown positive_pair_type entries {unknown}; valid: {_VALID_POSITIVE_PAIR_TYPES}"
        )
        assert num_positives >= 1, "num_positives must be ≥ 1"
        assert num_negatives >= 1, "num_negatives must be ≥ 1"
        assert cl_temperature > 0,  "cl_temperature must be > 0"
        self.query_vit           = query_vit
        self.chunk_size          = chunk_size
        self.positive_pair_type  = positive_pair_type
        self.num_positives       = num_positives
        self.num_negatives       = num_negatives
        self.cl_temperature      = cl_temperature
        self.cl_pose_radius      = cl_pose_radius
        self.cl_pose_fov_half_h  = math.radians(cl_pose_fov_half_h)
        self.cl_pose_fov_half_v  = math.radians(cl_pose_fov_half_v)
        self.cl_pose_num_samples = cl_pose_num_samples

        # Telemetry: per-positive-type InfoNCE split, one NON-persistent buffer
        # per CONFIGURED type — found by the train loop via the `last_cl_loss_`
        # suffix, exactly as pnp_cl exposed them.
        for _pt in positive_pair_type:
            self.register_buffer(f"last_cl_loss_{_pt}", torch.zeros(1), persistent=False)

        # Inverse-normalisation buffers (goal_pose[:3] = action[:3]*scale+offset),
        # identical to pnp_cl: identity when no stats supplied. NON-persistent so
        # `state_dict()` stays exactly `{query_vit.*}`.
        if local_pos_stats is not None:
            mn = torch.as_tensor(local_pos_stats["min"], dtype=torch.float32).reshape(-1)
            mx = torch.as_tensor(local_pos_stats["max"], dtype=torch.float32).reshape(-1)
            pos_scale  = (mx - mn) / 2
            pos_offset = (mx + mn) / 2
        else:
            pos_scale  = torch.ones(3)
            pos_offset = torch.zeros(3)
        self.register_buffer("pos_scale",  pos_scale,  persistent=False)
        self.register_buffer("pos_offset", pos_offset, persistent=False)

    def forward(self, context_frames, context_poses, context_times,
                action, stride, context_mask, target_frames, action_interact=None):
        # fp32 island (bf16 quantisation of QueryViT's residual stream can drift
        # keys/queries enough to destabilise training) — same as pnp_cl.forward.
        with torch.amp.autocast(device_type='cuda', enabled=False):
            B, num_goals = action.shape[:2]
            S = self.chunk_size

            # Past context (exclude the current frame at the last index).
            past_frames = context_frames[:, :-1]
            past_poses  = context_poses [:, :-1]
            past_times  = context_times [:, :-1]
            T_past = past_frames.shape[1]

            # Drop leading (oldest) frames so T_past is a multiple of chunk_size.
            N    = T_past // S
            drop = T_past - N * S
            past_frames = past_frames[:, drop:]
            past_poses  = past_poses [:, drop:]
            past_times  = past_times [:, drop:]

            past_frames = past_frames.reshape(B, N, S, *past_frames.shape[2:])
            past_poses  = past_poses .reshape(B, N, S, past_poses.shape[-1])
            past_times  = past_times .reshape(B, N, S)
            if context_mask is not None:
                past_mask = context_mask[:, :-1][:, drop:].reshape(B, N, S)

            # One query per (B, goal): same current frame, different action/stride.
            current_frame = context_frames[:, -1]
            current_frame_g = current_frame.unsqueeze(1).expand(B, num_goals, *current_frame.shape[1:])
            current_frame_g = current_frame_g.reshape(B * num_goals, *current_frame.shape[1:])
            action_g = action.reshape(B * num_goals, -1)
            stride_g = stride.reshape(B * num_goals)
            # Interaction labels condition the QUERY only, and only when the
            # encoder carries the pathway; the key encodes below stay pure
            # content (null action AND null interaction).
            interact_g = None
            if action_interact is not None and getattr(self.query_vit, "interact_dim", 0) > 0:
                interact_g = action_interact.reshape(B * num_goals, -1)
            queries  = self.query_vit(current_frame_g, action_g, stride_g, interact_g)

            # Flat (frame-level) view of the valid past pool.
            flat_pool_frames = past_frames.reshape(B, N * S, *past_frames.shape[3:])
            flat_pool_times  = past_times .reshape(B, N * S)
            flat_pool_poses  = past_poses .reshape(B, N * S, past_poses.shape[-1])
            if context_mask is not None:
                valid_frame = past_mask.reshape(B, N * S)
            else:
                valid_frame = torch.ones(B, N * S, dtype=torch.bool, device=context_frames.device)

            return self._infonce(
                queries, flat_pool_frames, flat_pool_times,
                flat_pool_poses, valid_frame, action, target_frames, num_goals,
            )

    def _infonce(self, queries, pool_frames, pool_times, pool_poses,
                 valid_frame, action, target_frames, num_goals):
        device = queries.device
        dtype  = queries.dtype
        B, T   = pool_frames.shape[0], pool_frames.shape[1]
        key_dim = queries.shape[-1]
        ng      = num_goals
        neg_inf = float("-inf")

        pos_scale  = self.pos_scale .to(device, action.dtype)
        pos_offset = self.pos_offset.to(device, action.dtype)
        goal_poses_raw = action.clone()
        goal_poses_raw[..., :3] = action[..., :3] * pos_scale + pos_offset

        def _encode(idx: torch.Tensor) -> torch.Tensor:
            K = idx.shape[1]
            bidx = torch.arange(B, device=device)[:, None].expand(B, K)
            fr   = pool_frames[bidx, idx]
            k    = self.query_vit(fr.reshape(B * K, *fr.shape[2:]))      # null action → KEY
            return k.float().reshape(B, K, key_dim)

        pos_keys_list   = []
        pos_type_counts = []
        pos_pool_idx    = []
        P = self.num_positives

        if "temporal" in self.positive_pair_type:
            t_score = pool_times.masked_fill(~valid_frame, neg_inf)
            k_t = min(P, T)
            top = t_score.topk(k_t, dim=1).indices
            keys_t = _encode(top)
            pos_keys_list.append(keys_t[:, None].expand(B, ng, k_t, key_dim))
            pos_pool_idx.append(top[:, None].expand(B, ng, k_t))
            pos_type_counts.append(("temporal", k_t))

        if "pose" in self.positive_pair_type:
            k_p = min(P, T)
            ov = torch.stack([
                pose_fov_overlap(
                    goal_poses_raw[:, g], pool_poses,
                    radius      = self.cl_pose_radius,
                    fov_half_h  = self.cl_pose_fov_half_h,
                    fov_half_v  = self.cl_pose_fov_half_v,
                    num_samples = self.cl_pose_num_samples,
                )
                for g in range(ng)
            ], dim=1)
            ov  = ov.masked_fill(~valid_frame[:, None], neg_inf)
            top = ov.topk(k_p, dim=-1).indices
            keys_p = _encode(top.reshape(B, ng * k_p)).reshape(B, ng, k_p, key_dim)
            pos_keys_list.append(keys_p)
            pos_pool_idx.append(top)
            pos_type_counts.append(("pose", k_p))

        if "jepa" in self.positive_pair_type:
            jk = self.query_vit(target_frames.reshape(B * ng, *target_frames.shape[2:]))
            pos_keys_list.append(jk.float().reshape(B, ng, 1, key_dim))
            pos_type_counts.append(("jepa", 1))

        if not pos_keys_list:
            return torch.zeros((), device=device, dtype=dtype)

        pos_keys = torch.cat(pos_keys_list, dim=2)
        P_total  = pos_keys.shape[2]

        weight = valid_frame.float()
        if pos_pool_idx:
            allp   = torch.cat([p.reshape(B, -1) for p in pos_pool_idx], dim=1)
            weight = weight.scatter(1, allp, 0.0)
        weight   = weight + 1e-9
        neg_idx  = torch.multinomial(weight, self.num_negatives, replacement=True)
        neg_keys = _encode(neg_idx)

        q  = F.normalize(queries.float().reshape(B, ng, key_dim), dim=-1)
        kp = F.normalize(pos_keys, dim=-1)
        kn = F.normalize(neg_keys, dim=-1)
        pos_logits = torch.einsum("bgd,bgpd->bgp", q, kp) / self.cl_temperature
        neg_logits = torch.einsum("bgd,bnd->bgn", q, kn) / self.cl_temperature
        all_logits = torch.cat(
            [
                pos_logits.unsqueeze(-1),
                neg_logits.unsqueeze(2).expand(B, ng, P_total, self.num_negatives),
            ],
            dim=-1,
        )
        per_pos = all_logits.logsumexp(dim=-1) - pos_logits
        with torch.no_grad():
            offset = 0
            for _pt, _cnt in pos_type_counts:
                getattr(self, f"last_cl_loss_{_pt}").copy_(
                    per_pos[..., offset:offset + _cnt].mean().reshape(1)
                )
                offset += _cnt
        return per_pos.mean().to(dtype)


def make_exp_name(cfg):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    datasets = '+'.join(cfg.datasets)
    return f"{timestamp}__pretrain_retriever__{cfg.run_name}__bs{cfg.batch_size}__{datasets}"


def _contrastive_loss(memory, batch, device, bf16):
    """One contrastive forward → scalar frame-level InfoNCE loss.

    Mirrors `train_latent.py`'s batch unpacking. `memory` is the
    `ContrastivePretrainModule` (which holds the QueryViT); its forward computes
    the InfoNCE loss directly (the chunking/pool setup that pnp_cl's `cl_only`
    path used now lives inside that module's forward)."""
    context_latent = batch["context_frames"].to(device, non_blocking=True)
    goal_latent    = batch["goal_frames"].to(device, non_blocking=True)
    action         = batch["goal_poses_nl"].to(device, non_blocking=True)
    stride         = batch["goal_times_nl"].to(device, non_blocking=True)
    context_poses  = batch["context_poses"].to(device, non_blocking=True)
    context_times  = batch["context_times"].to(device, non_blocking=True)
    context_mask   = batch["context_mask"].to(device, non_blocking=True)
    # Emitted only by backends that carry per-frame event counts (THOR); the
    # module ignores it unless the encoder was built with interact_dim > 0.
    action_interact  = batch.get("action_interact")
    if action_interact is not None:
        action_interact = action_interact.to(device, non_blocking=True)

    with torch.amp.autocast('cuda', enabled=bf16, dtype=torch.bfloat16):
        return memory(
            context_latent, context_poses, context_times, action, stride,
            context_mask, goal_latent, action_interact,
        )


@torch.no_grad()
def evaluate_contrastive(memory, test_dataset, rank, batch_size, num_workers, device, bf16, seed, max_batches=20):
    """Mean contrastive loss over a few test batches (retriever val metric).

    The module stays in `.train()` here — `@torch.no_grad` handles the
    no-gradient part and QueryViT has no dropout/BN, so train vs eval is
    otherwise identical."""
    memory.train()
    sampler = DistributedSampler(test_dataset, num_replicas=dist.get_world_size(),
                                 rank=rank, shuffle=True, seed=seed, drop_last=True)
    loader = DataLoader(test_dataset, batch_size=batch_size, sampler=sampler,
                        num_workers=num_workers, pin_memory=True, drop_last=True)
    total = torch.zeros((), device=device)
    count = torch.zeros((), device=device)
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        total += _contrastive_loss(memory, batch, device, bf16).detach().float()
        count += 1
    all_reduce(total, op=ReduceOp.SUM)
    all_reduce(count, op=ReduceOp.SUM)
    return (total / count).item() if count.item() > 0 else float('nan')


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

    # Build ONLY the QueryViT (the sole encoder retriever pre-training trains)
    # and wrap it in the contrastive pre-training module — no generator /
    # tokenizer / diffusion / compressor. `to_container(resolve=True)` resolves
    # the QueryViT sub-config's interpolations (e.g. action_dim=${action_dim}).
    # The module's `state_dict()` is exactly `{query_vit.*}`, so the checkpoints
    # it writes load straight into the full pnp_cl via `PnPRetrieval.load_weights`
    # for world-model training (`model.memory.ckpt_path=<run>/checkpoints/...`).
    qv_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model.memory.query_vit, resolve=True))
    query_vit = instantiate(qv_cfg)
    cl = cfg.contrastive
    local_pos_stats = (
        OmegaConf.to_container(cl.local_pos_stats, resolve=True)
        if cl.local_pos_stats is not None else None
    )
    memory = ContrastivePretrainModule(
        query_vit,
        chunk_size          = int(cl.chunk_size),
        positive_pair_type  = list(cl.positive_pair_type),
        num_positives       = int(cl.num_positives),
        num_negatives       = int(cl.num_negatives),
        cl_temperature      = float(cl.cl_temperature),
        cl_pose_radius      = float(cl.cl_pose_radius),
        cl_pose_fov_half_h  = float(cl.cl_pose_fov_half_h),
        cl_pose_fov_half_v  = float(cl.cl_pose_fov_half_v),
        cl_pose_num_samples = int(cl.cl_pose_num_samples),
        local_pos_stats     = local_pos_stats,
    )
    if not any(p.requires_grad for p in memory.parameters()):
        raise ValueError("Retriever pre-training requires a trainable QueryViT.")
    memory.to(device)

    lr = float(cfg.lr)
    weight_decay = float(cfg.weight_decay)
    opt = torch.optim.AdamW(memory.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-6)
    logger.info(f"Optimizer: AdamW on QueryViT only, lr={lr}")
    warmup_steps = int(cfg.warmup_steps)
    scheduler = (
        torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.2, end_factor=1.0, total_iters=warmup_steps)
        if warmup_steps > 0 else None
    )
    bf16 = bool(getattr(cfg, 'bfloat16', False))

    # Resume — custom keys so the same file ALSO loads cleanly into
    # `train_latent.py` (which reads only `memory` from it, ignoring the rest).
    latest_path = os.path.join(checkpoint_dir, "latest.pth.tar")
    train_steps = 0
    resume_path = latest_path if os.path.isfile(latest_path) else (cfg.from_checkpoint or None)
    if resume_path is not None and os.path.isfile(resume_path):
        logger.info(f"Resuming retriever pre-training from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        mem_ckp = {k.replace('_orig_mod.', ''): v for k, v in ckpt["memory"].items()}
        logger.info(f"Loading memory weights: {memory.load_state_dict(mem_ckp, strict=True)}")
        if "memory_opt" in ckpt:
            opt.load_state_dict(ckpt["memory_opt"])
        if "pretrain_steps" in ckpt:
            train_steps = ckpt["pretrain_steps"]
            if scheduler is not None:
                for _ in range(train_steps):
                    scheduler.step()

    memory = DDP(memory, device_ids=[device], find_unused_parameters=True)
    logger.info(f"Memory Parameters: {sum(p.numel() for p in memory.parameters()):,}")

    # Per-positive-type InfoNCE split buffers (pnp_cl) for telemetry.
    cl_loss_bufs = {}
    for _n, _b in memory.named_buffers():
        if 'last_cl_loss_' in _n:
            cl_loss_bufs[_n.rsplit('last_cl_loss_', 1)[1]] = _b

    datasets = {'train': [], 'test': []}
    for dn in cfg.datasets:
        for split in ['train', 'test']:
            if split in cfg[dn]:
                ds = instantiate(cfg[dn][split])
                datasets[split].append(ds)
                logger.info(f"Dataset: {dn} ({split}), size: {len(ds)}")
    datasets['train'] = ConcatDataset(datasets['train'])
    datasets['test']  = ConcatDataset(datasets['test'])

    sampler = DistributedSampler(datasets['train'], num_replicas=dist.get_world_size(),
                                 rank=rank, shuffle=True, seed=cfg.global_seed)
    loader = DataLoader(datasets['train'], batch_size=cfg['batch_size'], shuffle=False,
                        sampler=sampler, num_workers=cfg['num_workers'], pin_memory=True,
                        drop_last=True, persistent_workers=True)
    logger.info(f"Train set: {len(datasets['train']):,} samples")

    memory.train()
    running_cl = 0.0
    cl_steps = 0
    start_time = time()
    logger.info(f"Pre-training the retriever for {cfg.iterations} iterations...")
    epoch = 0
    while train_steps < cfg.iterations:
        sampler.set_epoch(epoch)
        for batch in loader:
            if train_steps >= cfg.iterations:
                break
            memory.train()
            cl_loss = _contrastive_loss(memory, batch, device, bf16)
            opt.zero_grad()
            cl_loss.backward()
            if cfg.grad_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(memory.parameters(), max_norm=cfg['grad_clip_val'])
            opt.step()
            if scheduler is not None:
                scheduler.step()

            running_cl += cl_loss.detach().item()
            cl_steps += 1
            train_steps += 1

            if train_steps % cfg.log_every == 0:
                torch.cuda.synchronize()
                steps_per_sec = cl_steps / (time() - start_time)
                samples_per_sec = (dist.get_world_size() * cfg['batch_size']
                                   * cfg['goals_per_obs'] * steps_per_sec)
                avg = torch.tensor(running_cl / cl_steps, device=device)
                all_reduce(avg, op=ReduceOp.SUM)
                avg = avg.item() / dist.get_world_size()
                extra = ''.join(f', cl/{pt}={float(b.item()):.4f}' for pt, b in cl_loss_bufs.items())
                logger.info(
                    f"(step={train_steps:07d}) CL Loss: {avg:.4f}, "
                    f"Steps/Sec: {steps_per_sec:.2f}, Samples/Sec: {samples_per_sec:.2f}{extra}"
                )
                if wandb_run is not None:
                    wlog = {'train/cl_loss': avg, 'train/steps_per_sec': steps_per_sec,
                            'train/samples_per_sec': samples_per_sec}
                    for pt, b in cl_loss_bufs.items():
                        wlog[f'train/cl_loss_{pt}'] = float(b.item())
                    wandb_run.log(wlog, step=train_steps)
                running_cl = 0.0
                cl_steps = 0
                start_time = time()

            if train_steps % cfg.ckpt_every == 0 and train_steps > 0 and rank == 0:
                ckpt = {
                    "memory": memory.module.state_dict(),   # loadable by train_latent.py
                    "memory_opt": opt.state_dict(),
                    "pretrain_steps": train_steps,
                }
                torch.save(ckpt, f"{checkpoint_dir}/latest.pth.tar")
                torch.save(ckpt, f"{checkpoint_dir}/{train_steps:07d}.pth.tar")
                logger.info(f"Saved checkpoint at step {train_steps}")

            if train_steps % cfg.eval_every == 0 and train_steps > 0:
                t0 = time()
                val_cl = evaluate_contrastive(memory, datasets['test'], rank,
                                              cfg['batch_size'], cfg['num_workers'], device, bf16, cfg.global_seed)
                barrier()
                logger.info(f"(step={train_steps:07d}) Val CL Loss: {val_cl:.4f}, Eval Time: {time()-t0:.2f}")
                if wandb_run is not None:
                    wandb_run.log({'eval/cl_loss': val_cl}, step=train_steps)
                start_time = time()   # don't count eval time in steps/sec
        epoch += 1

    if rank == 0 and wandb_run is not None:
        wandb_run.finish()
    barrier()
    destroy_process_group()


if __name__ == "__main__":
    main()
