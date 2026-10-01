import contextlib
import io
import os
import warnings
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
from time import time

import hydra
import lpips
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dreamsim import dreamsim

# The NGC container ships torchao 0.11 while peft>=0.18 demands >0.16; peft's
# LoRA dispatcher (reached when dreamsim loads its DINO LoRA adapters) raises
# ImportError on the mismatch even though torchao is never used here. Force the
# availability check off so the dispatcher skips torchao cleanly. Patches the
# name bound inside lora.torchao (imported via `from ... import`), not just the
# source in import_utils.
try:
    import peft.import_utils as _peft_import_utils
    import peft.tuners.lora.torchao as _peft_lora_torchao
    _peft_import_utils.is_torchao_available = lambda *a, **k: False
    _peft_lora_torchao.is_torchao_available = lambda *a, **k: False
except Exception:
    pass

import torch
import torch._dynamo
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.distributed import all_gather, all_reduce, barrier, destroy_process_group, ReduceOp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import far.distributed as dist
from far.constants import CONFIG_ROOT, RESULT_ROOT, hub_cache_dir
from far.data.transforms import tensor_to_image
from far.distributed import create_logger
from far.experiment import save_run_info, init_wandb


def make_exp_name(cfg):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    datasets = '+'.join(cfg.datasets)
    return f"{timestamp}__train_latent__{cfg.run_name}__bs{cfg.batch_size}__{datasets}"


def save_atomic(obj, path):
    """torch.save via a temp file + os.replace, so a preemption landing
    mid-write can never leave a torn checkpoint — `path` is always either the
    previous complete file or the new one (POSIX-atomic rename)."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        name = name.replace('_orig_mod.', '')
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


@torch.no_grad()
def detokenize_chunked(tokenizer, latents, chunk=32):
    # The VAE decoder's transient activations scale with batch (at SDXL/256px
    # a full eval_batch x num_goals decode is tens of GB — the step-5000 eval
    # OOM); slicing keeps eval memory flat in eval_batch_size.
    return torch.cat([tokenizer.detokenize(latents[i:i + chunk])
                      for i in range(0, latents.shape[0], chunk)])


@torch.no_grad()
def evaluate(model, eval_model, lpips_model, test_dataset, rank, batch_size, num_workers, device, save_dir, seed, step=None):
    # Memory in eval mode so any training-only behaviour (gated on
    # `self.training`) is disabled during evaluation. Re-enabled on the
    # training side of the loop.
    if model.core.memory is not None:
        model.core.memory.eval()
    model.core.last_agent_pred = None        # set by sample_latent when the agent-state protocol is active
    sampler = DistributedSampler(
        test_dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=seed,
    )
    loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    dreamsim_score = torch.tensor(0.).to(device)
    lpips_score    = torch.tensor(0.).to(device)
    psnr_score     = torch.tensor(0.).to(device)
    n_samples      = torch.tensor(0).to(device)

    for batch in loader:
        context_latent = batch["context_frames"].to(device)
        goal_latent = batch["goal_frames"].to(device)
        action = batch["goal_poses_nl"].to(device)
        stride = batch["goal_times_nl"].to(device).flatten(0, 1)
        context_poses = batch["context_poses"].to(device)
        context_times = batch["context_times"].to(device)
        context_mask = batch["context_mask"].to(device)
        # Pre-computed retriever keys for the (real) context frames — when
        # present, the memory retrieves over the same cached key space used at
        # training instead of re-encoding with the live QueryViT. None → encode.
        context_keys = batch.get("context_keys")
        if context_keys is not None:
            context_keys = context_keys.to(device)
        query_cue = batch.get("query_cue")
        if query_cue is not None:
            query_cue = query_cue.to(device)
        # Per-goal interaction multi-hot (THOR). Consumed by the retriever
        # (interact_dim > 0 on QueryViT) and/or the generator (interact_dim > 0
        # on cdit_mod_pose_fa); ignored by pose-only builds.
        action_interact = batch.get("action_interact")
        if action_interact is not None:
            action_interact = action_interact.to(device)
        # State-basis columns (consumed only by generators whose interact_dim
        # exceeds the count width — see LDWMCore._generator_interact).
        action_state = batch.get("action_state")
        if action_state is not None:
            action_state = action_state.to(device)
        # Second-agent detections (agent-state protocol): the pool's per-frame
        # states condition the predictor; the goals' targets score it below.
        context_agent_state = batch.get("context_agent_state")
        goal_agent_state = batch.get("goal_agent_state")
        if context_agent_state is not None:
            context_agent_state = context_agent_state.to(device)
            goal_agent_state = goal_agent_state.to(device)

        with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
            B = context_latent.shape[0]
            num_goals = goal_latent.shape[1]
            actions = action.flatten(0, 1)

            goal_pred_latent = model.core.sample_latent(
                context_latent, actions, stride, num_goals=num_goals,
                context_poses=context_poses, context_times=context_times,
                context_mask=context_mask, context_keys=context_keys, query_cue=query_cue,
                action_interact=action_interact, action_state=action_state,
                use_ddim=True,   # pairs with the "ddim20" eval diffusion swapped in by the caller
                context_agent_state=context_agent_state,
            )
            # Agent-state eval: the predictor's presence logit / position for every
            # (sample, goal), its target, and whether the pair is MEMORY-REQUIRED:
            # a reveal-anchored draw (the sampler's `event_anchored` branch) whose
            # current frame shows no agent, so the answer can only come from the
            # retrieved slots. Gathered across ranks after the loop.
            agent_rows = None
            ap = model.core.last_agent_pred
            if ap is not None and goal_agent_state is not None:
                tgt = goal_agent_state.flatten(0, 1)                                         # (B*G, 3)
                anchored = batch.get("event_anchored")
                anchored = (anchored.to(device).bool() if anchored is not None
                            else torch.zeros(B, dtype=torch.bool, device=device))
                memreq = (anchored & (context_agent_state[:, -1, 0] <= 0)).float()[:, None].expand(B, num_goals).reshape(-1, 1)
                agent_rows = torch.cat([ap.float()[:, :1], ap.float()[:, 1:3], tgt.float(), memreq], dim=1)   # (B*G, 7)

            goal_pred_pm = torch.clip(detokenize_chunked(model.tokenizer, goal_pred_latent), -1., 1.)
            goal_pm      = torch.clip(detokenize_chunked(model.tokenizer, goal_latent.flatten(0, 1)), -1., 1.)
            context_pm   = torch.clip(detokenize_chunked(model.tokenizer, context_latent[:, -1]), -1., 1.)
            goal_pred = goal_pred_pm * 0.5 + 0.5
            goal      = goal_pm * 0.5 + 0.5
            context   = context_pm.unsqueeze(1).expand(B, num_goals, *context_pm.shape[1:]).flatten(0, 1) * 0.5 + 0.5

            ds_res = eval_model(goal, goal_pred)
            lp_res = lpips_model(goal_pred_pm.float(), goal_pm.float()).flatten()
            mse    = ((goal_pred - goal) ** 2).flatten(1).mean(dim=1).clamp(min=1e-12)
            psnr   = -10.0 * torch.log10(mse)

            dreamsim_score += ds_res.sum()
            lpips_score    += lp_res.sum()
            psnr_score     += psnr.sum()
            n_samples      += len(ds_res)
        break

    if rank == 0:
        n = min(goal_pred.shape[0], 10)
        img_h, img_w = context.shape[-2], context.shape[-1]
        row_h = 12 / 3 * img_h / img_w  # match column width × image aspect ratio
        fig, axes = plt.subplots(n, 3, dpi=128, figsize=(12, row_h * n))
        if n == 1:
            axes = axes[None]
        axes[0, 0].set_title('Context', fontsize=14)
        axes[0, 1].set_title('Goal', fontsize=14)
        axes[0, 2].set_title('Goal Pred.', fontsize=14)
        for i in range(n):
            axes[i, 0].imshow(tensor_to_image(context[i]))
            axes[i, 1].imshow(tensor_to_image(goal[i]))
            axes[i, 2].imshow(tensor_to_image(goal_pred[i]))
            for ax in axes[i]:
                ax.axis('off')
        title_pad = 0.3 / (row_h * n)  # fixed pixel budget for titles, scaled to figure height
        plt.subplots_adjust(hspace=0.02, wspace=0.02, left=0.02, right=0.98, top=1 - title_pad, bottom=0.02)
        img_path = os.path.join(save_dir, f'{step:07d}.png')
        plt.savefig(img_path)
        plt.close()
    else:
        img_path = None

    all_reduce(dreamsim_score)
    all_reduce(lpips_score)
    all_reduce(psnr_score)
    all_reduce(n_samples)
    metrics = {
        'dreamsim': (dreamsim_score / n_samples).item(),
        'lpips':    (lpips_score / n_samples).item(),
        'psnr':     (psnr_score / n_samples).item(),
    }
    if agent_rows is not None:
        # Every rank holds the same number of rows (drop_last), so gather them
        # and score the pooled set identically on every rank.
        ws = dist.get_world_size()
        if ws > 1:
            parts = [torch.zeros_like(agent_rows) for _ in range(ws)]
            all_gather(parts, agent_rows.contiguous())
            rows = torch.cat(parts, 0)
        else:
            rows = agent_rows
        rows = rows.detach().cpu()
        logit, pos, tgt, memreq = rows[:, 0], rows[:, 1:3], rows[:, 3:6], rows[:, 6] > 0.5
        present = tgt[:, 0] > 0.5

        def _auc(s, y):
            # rank-based AUC (probability a positive outranks a negative); nan when one class is absent
            p, n = s[y], s[~y]
            if len(p) == 0 or len(n) == 0:
                return float('nan')
            return float(((p[:, None] > n[None]).float() + 0.5 * (p[:, None] == n[None]).float()).mean())
        metrics['agent_auc'] = _auc(logit, present)                                    # presence AUC over all (sample, goal)
        metrics['agent_auc_memreq'] = _auc(logit[memreq], present[memreq])              # ... on pairs the current frame cannot answer
        metrics['agent_acc'] = float(((logit > 0) == present).float().mean())
        metrics['agent_pos_err'] = (float(((pos[present] - tgt[present, 1:3]) ** 2).sum(-1).sqrt().mean())
                                    if bool(present.any()) else float('nan'))          # metres, goals with the agent in view
        metrics['agent_memreq_frac'] = float(memreq.float().mean())
        metrics['agent_presence_rate'] = float(present.float().mean())
    return metrics, img_path


@hydra.main(version_base=None, config_path=CONFIG_ROOT, config_name='world_train')
def main(cfg: DictConfig):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

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

    model = instantiate(cfg.model).to(device)
    generator = model.core.generator
    ema = deepcopy(generator).to(device)
    requires_grad(ema, False)
    requires_grad(model.tokenizer, False)

    # Memory ViTs (e.g. PnPRetrieval's compressor + query nets) carry trainable
    # parameters and must be optimised alongside the generator. Memories that
    # are parameter-free (TemporalRetrieval, WorldMemRetrieval) drop through.
    memory = model.core.memory
    memory_has_params = memory is not None and any(p.requires_grad for p in memory.parameters())

    lr = float(cfg.lr)
    weight_decay = float(cfg.weight_decay)
    # Per-component LRs (null → fall back to the generator `lr`):
    #   - memory (`lr_mem`): pnp_cl's QueryViT etc.
    lr_mem_raw = cfg.get('lr_mem') if hasattr(cfg, 'get') else getattr(cfg, 'lr_mem', None)
    lr_mem     = float(lr_mem_raw) if lr_mem_raw is not None else lr
    # One AdamW param group per component so each tracks its own LR (LinearLR
    # scales every group by the same warmup factor). Groups with equal LRs are
    # functionally identical to a single merged group.
    param_groups = [{'params': list(generator.parameters()), 'lr': lr, 'name': 'generator'}]
    if memory_has_params:
        param_groups.append({'params': list(memory.parameters()), 'lr': lr_mem, 'name': 'memory'})
    logger.info("Optimizer LR groups: " + ", ".join(f"{g['name']}={g['lr']}" for g in param_groups))
    opt = torch.optim.AdamW(param_groups, weight_decay=weight_decay, eps=1e-6)
    # Linear LR warmup from lr/5 → lr over `warmup_steps`; constant lr afterwards.
    # `warmup_steps <= 0` disables the schedule.
    warmup_steps = int(cfg.warmup_steps)
    if warmup_steps > 0:
        scheduler = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.2, end_factor=1.0, total_iters=warmup_steps,
        )
    else:
        scheduler = None
    bfloat_enable = bool(hasattr(cfg, 'bfloat16') and cfg.bfloat16)
    # NOTE: GradScaler is for fp16 only — bf16 has the same dynamic range as fp32
    # and does not need (and is destabilised by) the scale/unscale roundtrip.

    latest_path = os.path.join(checkpoint_dir, "latest.pth.tar")
    train_steps = 0
    if os.path.isfile(latest_path) or cfg.from_checkpoint:
        if os.path.isfile(latest_path) and cfg.from_checkpoint:
            raise ValueError("Resuming from checkpoint, this might override latest.pth.tar!!")
        latest_path = latest_path if os.path.isfile(latest_path) else cfg.from_checkpoint
        logger.info(f"Loading model from {latest_path}")
        latest_checkpoint = torch.load(latest_path, map_location=f'cuda:{device}', weights_only=False)

        if "model" in latest_checkpoint:
            model_ckp = {k.replace('_orig_mod.', ''): v for k, v in latest_checkpoint['model'].items()}
            logger.info(f"Loading model weights: {generator.load_state_dict(model_ckp, strict=True)}")
            model_ckp = {k.replace('_orig_mod.', ''): v for k, v in latest_checkpoint['ema'].items()}
            logger.info(f"Loading EMA model weights: {ema.load_state_dict(model_ckp, strict=True)}")
        else:
            update_ema(ema, generator, decay=0)

        if memory_has_params and "memory" in latest_checkpoint:
            mem_ckp = {k.replace('_orig_mod.', ''): v for k, v in latest_checkpoint['memory'].items()}
            logger.info(f"Loading memory weights: {memory.load_state_dict(mem_ckp, strict=True)}")

        if "opt" in latest_checkpoint:
            opt_ckp = {k.replace('_orig_mod.', ''): v for k, v in latest_checkpoint['opt'].items()}
            opt.load_state_dict(opt_ckp)
            logger.info("Loading optimizer params")

        if scheduler is not None:
            if "scheduler" in latest_checkpoint:
                scheduler.load_state_dict(latest_checkpoint['scheduler'])
                logger.info("Loading scheduler state")
            else:
                # Older checkpoint without scheduler state — fast-forward.
                for _ in range(latest_checkpoint["train_steps"]):
                    scheduler.step()

        if "train_steps" in latest_checkpoint:
            train_steps = latest_checkpoint["train_steps"]

        # RAG advantage-temperature EMAs (per_set/per_frame_loss_tau_ema) live
        # on LDWMCore itself — outside the generator/memory state dicts — so
        # without this they silently reset to init (0.1) on resume and the EM
        # loss runs mis-scaled while they re-converge (~a few k steps). Older
        # checkpoints lack the key and keep the legacy reset behaviour.
        if "core_buffers" in latest_checkpoint:
            for k, v in latest_checkpoint["core_buffers"].items():
                getattr(model.core, k).data.copy_(v.to(device))
            logger.info(f"Loading core buffers: {list(latest_checkpoint['core_buffers'])}")

    if cfg.torch_compile:
        # Disable dynamo's DDPOptimizer (the torch.compile×DDP graph splitter). It
        # breaks the generator into DDP-bucket-aligned subgraphs and compiles each
        # separately to overlap allreduce with the backward. With the variable
        # context length introduced by rag_step (it probes the reward at ctx=2 while
        # the generator step uses ctx=context_size), one of those split subgraphs
        # fails inductor codegen -> BackendCompilerFailed. Compiling the module as a
        # SINGLE graph is robust (it recompiles cleanly per context length), and the
        # optimizer's compute/comm overlap buys nothing at nproc_per_node=1 (there
        # is no cross-rank allreduce to hide).
        torch._dynamo.config.optimize_ddp = False
        generator = torch.compile(generator)
    # `find_unused_parameters=True`: both the generator and the memory have
    # parameters that legitimately don't appear in every forward (e.g. the
    # learned null embeddings used only when an input is None, or role/time
    # embeddings that are currently disabled). Without this flag DDP raises
    # "Parameter X did not receive grad" on the second iteration.
    generator = DDP(generator, device_ids=[device], find_unused_parameters=True)
    # Reassign so `model.core.compute_loss_latent` routes its `self.generator`
    # call through the DDP wrapper (otherwise DDP's gradient-sync hooks never
    # fire — the forward would go through the bare module).
    model.core.generator = generator
    logger.info(f"CDiT Parameters: {sum(p.numel() for p in generator.parameters()):,}")

    if memory_has_params:
        if cfg.torch_compile:
            memory = torch.compile(memory)
        memory = DDP(memory, device_ids=[device], find_unused_parameters=True)
        # Same reassignment rationale as for the generator above.
        model.core.memory = memory
        logger.info(f"Memory Parameters: {sum(p.numel() for p in memory.parameters()):,}")

    datasets = {'train': [], 'test': []}
    for dn in cfg.datasets:
        for split in ['train', 'test']:
            if split in cfg[dn]:
                dataset = instantiate(cfg[dn][split])
                datasets[split].append(dataset)
                logger.info(f"Dataset: {dn} ({split}), size: {len(dataset)}")
    logger.info(f"Combining {len(datasets['train'])} datasets.")
    datasets['train'] = ConcatDataset(datasets['train'])
    datasets['test'] = ConcatDataset(datasets['test'])

    sampler = DistributedSampler(
        datasets['train'],
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=cfg.global_seed,
    )
    loader = DataLoader(
        datasets['train'],
        batch_size=cfg['batch_size'],
        shuffle=False,
        sampler=sampler,
        num_workers=cfg['num_workers'],
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    logger.info(f"Dataset contains {len(datasets['train']):,} samples")

    if rank == 0:
        eval_model, _ = dreamsim(pretrained=True, cache_dir=hub_cache_dir())
        lpips_model = lpips.LPIPS(net='alex').to(device)
    else:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eval_model, _ = dreamsim(pretrained=True, cache_dir=hub_cache_dir())
            lpips_model = lpips.LPIPS(net='alex', verbose=False).to(device)
    requires_grad(lpips_model, False)
    lpips_model.eval()

    # In-training eval samples with DDIM-20 instead of the training diffusion's
    # full 1000-step DDPM (~14x cheaper eval at B-size; ~400s -> ~30s per eval).
    # Same schedule/params as training, only the respacing + sampler differ.
    # NOTE: eval metric LEVELS shift relative to pre-switch logs (deterministic
    # 20-step DDIM vs ancestral 1000-step DDPM) — compare curves within one
    # protocol only.
    eval_diffusion = instantiate(cfg.model.diffusion, timestep_respacing="ddim20")

    generator.train()
    ema.eval()
    samples_dir = os.path.join(result_dir, 'samples')
    if rank == 0:
        os.makedirs(samples_dir, exist_ok=True)

    # The generator always trains hard-EM on the single top set
    # (`compute_loss_latent`). When `rag_every > 0`, the retriever additionally
    # gets a forward-only, multi-noise RAG update (`rag_step`) every N steps;
    # `rag_every == 0` leaves the retriever to the per-step contrastive loss only.
    rag_every        = int(cfg.get('rag_every', 0) or 0)
    rag_noise_levels = int(cfg.get('rag_noise_levels', 4) or 4)
    local_rag        = bool(cfg.get('local_rag', False))
    local_rag_weight = float(cfg.get('local_rag_weight', 1.0))
    rag_mode         = str(cfg.get('rag_mode', 'diffusion') or 'diffusion')
    rag_ddim_steps   = int(cfg.get('rag_ddim_steps', 8) or 8)
    rag_mse_weight   = float(cfg.get('rag_mse_weight', 1.0))
    assert rag_mode in ('diffusion', 'ddim'), rag_mode
    if rag_every > 0:
        logger.info(
            f"RAG retriever update ({rag_mode}) every {rag_every} steps "
            + (f"(M={rag_noise_levels} stratified noise levels)."
               if rag_mode == 'diffusion' else
               f"(ddim{rag_ddim_steps} sampled reward: mask-weighted x0 MSE "
               f"x {rag_mse_weight}).")
        )

    log_steps = 0
    running_gen_loss = 0.0          # accumulator over `log_every` steps for the diffusion-piece loss
    running_ret_loss = 0.0          # accumulator over `log_every` steps for the soft-EM retriever loss
    running_rerank   = 0.0          # accumulator for the EM re-rank rate (argmax(q) != argmax(p))
    running_rank_corr = 0.0         # accumulator for Spearman ρ(prior, −reward) over the K sets
    running_local_ret = 0.0         # accumulator for the LOCAL (within-chunk) EM retriever loss
    running_local_rank_corr = 0.0   # accumulator for the LOCAL Spearman ρ(prior, −reward) over the S frames
    ret_loss_steps   = 0            # how many of those steps ran the EM retriever branch
    start_time = time()

    logger.info(f"Training for {cfg.iterations} iterations...")
    epoch = 0
    while train_steps < cfg.iterations:
        sampler.set_epoch(epoch)
        for batch in loader:
            if train_steps >= cfg.iterations:
                break
            # Re-enable training-only memory behaviour on every step in case the
            # previous iteration ran eval (which flipped memory to eval mode).
            if model.core.memory is not None:
                model.core.memory.train()
            context_latent = batch["context_frames"].to(device, non_blocking=True)
            goal_latent = batch["goal_frames"].to(device, non_blocking=True)
            action = batch["goal_poses_nl"].to(device, non_blocking=True)
            stride = batch["goal_times_nl"].to(device, non_blocking=True)
            context_poses = batch["context_poses"].to(device, non_blocking=True)
            context_times = batch["context_times"].to(device, non_blocking=True)
            context_mask = batch["context_mask"].to(device, non_blocking=True)
            # Pre-computed retriever keys (present only when the dataset has
            # `<clip>.keys.npy`). When supplied, pnp_cl uses them as cached
            # retrieval/scoring keys instead of encoding; None → encode.
            context_keys = batch.get("context_keys")
            if context_keys is not None:
                context_keys = context_keys.to(device, non_blocking=True)
            action_interact = batch.get("action_interact")
            if action_interact is not None:
                action_interact = action_interact.to(device, non_blocking=True)
            # State-basis columns for the generator (used only when its
            # interact_dim exceeds the count width — see _generator_interact).
            action_state = batch.get("action_state")
            if action_state is not None:
                action_state = action_state.to(device, non_blocking=True)
            query_cue = batch.get("query_cue")
            if query_cue is not None:
                query_cue = query_cue.to(device, non_blocking=True)
            # Per-goal change mask -> mask-weighted diffusion loss (no-op when
            # the dataset serves none, or model.mask_loss_lambda == 0).
            goal_loss_mask = batch.get("goal_loss_mask")
            if goal_loss_mask is not None:
                goal_loss_mask = goal_loss_mask.to(device, non_blocking=True)
            # Second-agent detections (THORDynBackend `agent_state`): the context
            # pool's per-frame states and the goals' targets. Consumed only by an
            # agent-state generator (`LDWMCore`'s agent-state protocol); no-op otherwise.
            context_agent_state = batch.get("context_agent_state")
            goal_agent_state = batch.get("goal_agent_state")
            if context_agent_state is not None:
                context_agent_state = context_agent_state.to(device, non_blocking=True)
                goal_agent_state = goal_agent_state.to(device, non_blocking=True)
            agent_active = context_agent_state is not None and model.core._agent_module() is not None

            with torch.amp.autocast('cuda', enabled=bfloat_enable, dtype=torch.bfloat16):
                # `opt_loss` is what we backward through — the top-set diffusion
                # loss (hard-EM generator step). `gen_loss == opt_loss` (kept as a
                # separate name for the clean diffusion-quality curve). The
                # retriever's RAG signal is trained lazily in `rag_step` every
                # `rag_every` steps.
                opt_loss, gen_loss = model.core.compute_loss_latent(
                    context_latent, goal_latent, action, stride,
                    context_poses=context_poses, context_times=context_times,
                    context_mask=context_mask,
                    context_keys=context_keys, query_cue=query_cue,
                    action_interact=action_interact, action_state=action_state,
                    goal_loss_mask=goal_loss_mask,
                    context_agent_state=context_agent_state, goal_agent_state=goal_agent_state,
                )

            opt.zero_grad()
            opt_loss.backward()
            # RAG: every `rag_every` steps, add the retriever's forward-only
            # multi-noise EM gradient before the optimizer step. The hard-EM
            # generator already trained on the top set above, so this only
            # accumulates the retriever's `L_ret` grad (the per-step contrastive
            # grad is already in there). Cheap: the K×M diffusion evals are
            # no_grad, only the retriever forward retains activations.
            did_rag = False
            if rag_every > 0 and train_steps % rag_every == 0:
                # rag_anchored_only: score / distil only the reveal-anchored draws (batch key from the sampler)
                rag_select = (batch['event_anchored'].to(device) if (bool(cfg.get('rag_anchored_only', False))
                              and batch.get('event_anchored') is not None) else None)
                with torch.amp.autocast('cuda', enabled=bfloat_enable, dtype=torch.bfloat16):
                    if rag_mode == 'ddim':
                        # Sampled-outcome reward. Unlike rag_step, this also
                        # conditions the generator on action_interact — a
                        # reveal sample cannot be rendered (hence rewarded)
                        # without it.
                        rag_loss = model.core.rag_step_ddim(
                            context_latent, goal_latent, action, stride,
                            context_poses=context_poses, context_times=context_times,
                            context_mask=context_mask,
                            context_keys=context_keys, query_cue=query_cue,
                            action_interact=action_interact, action_state=action_state,
                            goal_loss_mask=goal_loss_mask,
                            ddim_steps=rag_ddim_steps,
                            mse_weight=rag_mse_weight,
                            local_rag=local_rag, local_rag_weight=local_rag_weight,
                            context_agent_state=context_agent_state, goal_agent_state=goal_agent_state,
                            select=rag_select,
                        )
                    else:
                        rag_loss = model.core.rag_step(
                            context_latent, goal_latent, action, stride,
                            context_poses=context_poses, context_times=context_times,
                            context_mask=context_mask,
                            context_keys=context_keys, query_cue=query_cue,
                            action_interact=action_interact, action_state=action_state,
                            goal_loss_mask=goal_loss_mask,
                            num_noise_levels=rag_noise_levels,
                            local_rag=local_rag, local_rag_weight=local_rag_weight,
                            context_agent_state=context_agent_state, goal_agent_state=goal_agent_state,
                            select=rag_select,
                        )
                # A fully-frozen retriever (freeze_encoder + no adapters / no
                # metadata scorer) yields a rag_loss with no grad graph —
                # rag_step still runs for its telemetry (rank_corr, ret_loss,
                # tau_ema), but there is nothing to backward.
                if rag_loss is not None and rag_loss.requires_grad:
                    rag_loss.backward()
                    did_rag = True
            if cfg.grad_clip_val > 0:
                clip_params = list(generator.parameters())
                if memory_has_params:
                    clip_params += list(memory.parameters())
                torch.nn.utils.clip_grad_norm_(clip_params, max_norm=cfg['grad_clip_val'])
            opt.step()
            if scheduler is not None:
                scheduler.step()

            update_ema(ema, generator.module)

            running_gen_loss += gen_loss.detach().item()
            # RAG retriever loss `L_ret` is written by `rag_step` only — so
            # accumulate it on the steps the (lazy) rag_step actually fired.
            if did_rag:
                running_ret_loss += float(model.core.last_em_ret_loss.item())
                running_rerank   += float(model.core.last_em_rerank_rate.item())
                running_rank_corr += float(model.core.last_em_rank_corr.item())
                if local_rag:
                    running_local_ret += float(model.core.last_em_local_ret_loss.item())
                    running_local_rank_corr += float(model.core.last_em_local_rank_corr.item())
                ret_loss_steps   += 1
            log_steps += 1
            train_steps += 1
            if train_steps % cfg.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                samples_per_sec = dist.get_world_size() * context_latent.shape[0] * goal_latent.shape[1] * steps_per_sec

                # `gen_loss` is logged in every step → divide by `log_steps`.
                # All-reduce across ranks so the printed value is the global
                # mean rather than rank-0's local view.
                avg_gen_loss = torch.tensor(running_gen_loss / log_steps, device=device)
                all_reduce(avg_gen_loss, op=ReduceOp.SUM)
                avg_gen_loss = avg_gen_loss.item() / dist.get_world_size()

                # Same (sum, count) all-reduce for the soft-EM retriever loss so
                # `train/ret_loss` is the window-averaged, cross-rank mean (not a
                # last-step rank-0 snapshot). None when the EM branch never fired.
                ret_loss_sum_t   = torch.tensor(running_ret_loss, device=device)
                ret_loss_count_t = torch.tensor(ret_loss_steps,   device=device, dtype=torch.float32)
                all_reduce(ret_loss_sum_t,   op=ReduceOp.SUM)
                all_reduce(ret_loss_count_t, op=ReduceOp.SUM)
                avg_ret_loss = (
                    (ret_loss_sum_t / ret_loss_count_t).item()
                    if ret_loss_count_t.item() > 0 else None
                )
                # EM re-rank rate (argmax(q) != argmax(p)) — same firing steps as
                # ret_loss, so it shares `ret_loss_count_t` as the divisor.
                rerank_sum_t = torch.tensor(running_rerank, device=device)
                all_reduce(rerank_sum_t, op=ReduceOp.SUM)
                avg_rerank = (
                    (rerank_sum_t / ret_loss_count_t).item()
                    if ret_loss_count_t.item() > 0 else None
                )
                # Spearman ρ(prior, −reward) — same firing steps / divisor as ret_loss.
                rank_corr_sum_t = torch.tensor(running_rank_corr, device=device)
                all_reduce(rank_corr_sum_t, op=ReduceOp.SUM)
                avg_rank_corr = (
                    (rank_corr_sum_t / ret_loss_count_t).item()
                    if ret_loss_count_t.item() > 0 else None
                )
                # Local (within-chunk) EM retriever loss — same firing steps / divisor.
                local_ret_sum_t = torch.tensor(running_local_ret, device=device)
                all_reduce(local_ret_sum_t, op=ReduceOp.SUM)
                avg_local_ret = (
                    (local_ret_sum_t / ret_loss_count_t).item()
                    if (local_rag and ret_loss_count_t.item() > 0) else None
                )
                # Local Spearman ρ(prior, −reward) over the S frames — same divisor.
                local_rank_corr_sum_t = torch.tensor(running_local_rank_corr, device=device)
                all_reduce(local_rank_corr_sum_t, op=ReduceOp.SUM)
                avg_local_rank_corr = (
                    (local_rank_corr_sum_t / ret_loss_count_t).item()
                    if (local_rag and ret_loss_count_t.item() > 0) else None
                )

                # `gen_loss` (top-set diffusion loss) stays comparable across
                # all memory variants — it's the apples-to-apples training-
                # quality signal.
                extra = ''
                if rag_every > 0:
                    extra = f', tau_ema={float(model.core.per_set_loss_tau_ema.item()):.4f}'
                    if avg_ret_loss is not None:
                        extra += f', ret_loss={avg_ret_loss:.4f}'
                    if avg_rerank is not None:
                        extra += f', rerank={avg_rerank:.3f}'
                    if avg_rank_corr is not None:
                        extra += f', rank_corr={avg_rank_corr:+.3f}'
                    if avg_local_ret is not None:
                        extra += f', local_ret={avg_local_ret:.4f}'
                    if avg_local_rank_corr is not None:
                        extra += f', local_rank_corr={avg_local_rank_corr:+.3f}'
                    if local_rag:
                        extra += f', local_tau={float(model.core.per_frame_loss_tau_ema.item()):.4f}'
                    # ddim rag: cross-set spread of the sampled reward over the
                    # whole frame (unweighted) vs INSIDE the change region (this
                    # rank's last rag step). in_sp >> mse_sp = the candidate
                    # sets differ where the state changed, i.e. the retriever is
                    # being ranked on state — what the sampled reward is for.
                    if rag_mode == 'ddim' and ret_loss_steps > 0:
                        extra += (f', mse_sp={float(model.core.last_rag_mse_spread.item()):.4f}'
                                  f', in_sp={float(model.core.last_rag_in_mask_spread.item()):.4f}')
                # Mask-weighted diffusion loss: what fraction of goals carried a
                # usable change mask, and the realized in-mask share of the
                # loss (should sit near lambda/(1+lambda) on weighted goals).
                if float(getattr(model.core, 'mask_loss_lambda', 0.0)) > 0:
                    _mi = float(model.core.last_mask_in_mse.item())
                    _mo = float(model.core.last_mask_out_mse.item())
                    extra += (f', mask_w={float(model.core.last_mask_weighted_frac.item()):.2f}'
                              f', mask_share={float(model.core.last_mask_loss_share.item()):.2f}'
                              f', mask_in={_mi:.4f}, mask_out={_mo:.4f}'
                              f', in/out={_mi / max(_mo, 1e-9):.2f}')
                if agent_active:
                    extra += (f', agent_loss={float(model.core.last_agent_loss.item()):.4f}'
                              f', ret_sight={float(model.core.last_ret_sighting_frac.item()):.3f}'
                              f', ret_dt={float(model.core.last_ret_dt.item()):.1f}s')
                # `visual_scale` (α in the pnp_cl fused logit α·cos + meta) exists
                # only when the metadata cross-scorer is enabled. A single
                # DDP-synced scalar — no reduction needed. Its trajectory answers
                # "does content retrieval keep weight once geometry is learnable".
                _mem = model.core.memory
                _mem = getattr(_mem, 'module', _mem)
                _vs  = getattr(_mem, 'visual_scale', None)
                if _vs is not None:
                    extra += f', visual_scale={float(_vs.item()):.3f}'
                # z-gated fused mode (scoring="visual+meta_z"): `gate` is the
                # shared linear channel-gate (DDP-synced); `last_lambda` is
                # this rank's per-query effective λ~ from the memory's most
                # recent forward — a snapshot (like visual_scale), not a
                # window average. Its mean drifting from 0.5 answers "which
                # channel is the gate learning to trust", its spread whether
                # the gate actually varies per query (bounded to
                # [gate_eps, 1−gate_eps] by construction).
                _gate = getattr(_mem, 'gate', None)
                _lam  = getattr(_mem, 'last_lambda', None)
                if _gate is not None:
                    extra += f', |gate_w|={float(_gate.weight.detach().norm()):.3f}'
                    if _lam is not None:
                        extra += f', lambda={float(_lam.mean().item()):.3f}'

                logger.info(
                    f"(step={train_steps:07d}) "
                    f"Gen Loss: {avg_gen_loss:.4f}, "
                    f"Steps/Sec: {steps_per_sec:.2f}, Samples/Sec: {samples_per_sec:.2f}"
                    f"{extra}"
                )
                if wandb_run is not None:
                    wlog = {
                        # `gen_loss` is logged as `train/loss` (not
                        # `train/gen_loss`) to keep continuity with existing
                        # wandb dashboards. The value under this key is the
                        # top-set diffusion loss.
                        'train/loss': avg_gen_loss,
                        'train/steps_per_sec': steps_per_sec,
                        'train/samples_per_sec': samples_per_sec,
                    }
                    if rag_every > 0:
                        wlog['train/tau_ema'] = float(model.core.per_set_loss_tau_ema.item())
                        if avg_ret_loss is not None:
                            wlog['train/ret_loss'] = avg_ret_loss
                        if avg_rerank is not None:
                            wlog['train/rerank_rate'] = avg_rerank
                        if avg_rank_corr is not None:
                            wlog['train/rank_corr'] = avg_rank_corr
                        if avg_local_ret is not None:
                            wlog['train/local_ret_loss'] = avg_local_ret
                        if avg_local_rank_corr is not None:
                            wlog['train/local_rank_corr'] = avg_local_rank_corr
                        if local_rag:
                            wlog['train/local_tau_ema'] = float(model.core.per_frame_loss_tau_ema.item())
                        if rag_mode == 'ddim' and ret_loss_steps > 0:
                            wlog['train/rag_mse_spread'] = float(model.core.last_rag_mse_spread.item())
                            wlog['train/rag_in_mask_spread'] = float(model.core.last_rag_in_mask_spread.item())
                    if agent_active:
                        wlog['train/agent_loss'] = float(model.core.last_agent_loss.item())
                        wlog['train/ret_sighting_frac'] = float(model.core.last_ret_sighting_frac.item())   # retriever bootstrap: share of retrieved slots with A2 in view
                        wlog['train/ret_dt'] = float(model.core.last_ret_dt.item())                         # mean offset of the retrieved slots from the current frame (s)
                    if _vs is not None:
                        wlog['train/visual_scale'] = float(_vs.item())
                    if _gate is not None:
                        wlog['train/gate_weight_norm'] = float(_gate.weight.detach().norm())
                    if float(getattr(model.core, 'mask_loss_lambda', 0.0)) > 0:
                        wlog['train/mask_weighted_frac'] = float(model.core.last_mask_weighted_frac.item())
                        wlog['train/mask_loss_share'] = float(model.core.last_mask_loss_share.item())
                        _mi = float(model.core.last_mask_in_mse.item())
                        _mo = float(model.core.last_mask_out_mse.item())
                        wlog['train/mask_in_mse'] = _mi
                        wlog['train/mask_out_mse'] = _mo
                        wlog['train/mask_in_out_ratio'] = _mi / max(_mo, 1e-9)
                        if _lam is not None:
                            wlog['train/lambda_mean'] = float(_lam.mean().item())
                            wlog['train/lambda_std']  = float(_lam.std(correction=0).item())
                            wlog['train/lambda_min']  = float(_lam.min().item())
                            wlog['train/lambda_max']  = float(_lam.max().item())
                    wandb_run.log(wlog, step=train_steps)

                running_gen_loss = 0.0
                running_ret_loss = 0.0
                running_rerank   = 0.0
                running_rank_corr = 0.0
                running_local_ret = 0.0
                running_local_rank_corr = 0.0
                ret_loss_steps   = 0
                log_steps = 0
                start_time = time()

            # Split checkpoint cadence: `latest.pth.tar` — the cheap overwrite
            # that anchors preemption/resume — refreshes every `latest_every`
            # steps, while the numbered KEEPER checkpoints stay on the coarser
            # `ckpt_every` grid. Configs predating the knob (or setting it
            # null) fall back to the legacy single cadence. All writes are
            # ATOMIC (temp file + rename), so a preemption mid-write can never
            # corrupt an existing checkpoint.
            latest_every = int(cfg.get("latest_every", None) or cfg.ckpt_every)
            if train_steps > 0 and (train_steps % latest_every == 0
                                    or train_steps % cfg.ckpt_every == 0):
                if rank == 0:
                    checkpoint = {
                        "model": generator.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "train_steps": train_steps,
                        # LDWMCore-level PERSISTENT buffers (the RAG τ EMAs) —
                        # rank 0's copy, matching the telemetry convention.
                        # Non-persistent telemetry buffers (last_em_*) are
                        # excluded, mirroring state_dict semantics.
                        "core_buffers": {
                            k: v.detach().cpu()
                            for k, v in model.core.named_buffers(recurse=False)
                            if k not in model.core._non_persistent_buffers_set
                        },
                    }
                    if scheduler is not None:
                        checkpoint["scheduler"] = scheduler.state_dict()
                    if memory_has_params:
                        checkpoint["memory"] = memory.module.state_dict()
                    save_atomic(checkpoint, f"{checkpoint_dir}/latest.pth.tar")
                    if train_steps % cfg.ckpt_every == 0:
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pth.tar"
                        save_atomic(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                    else:
                        logger.info(f"Refreshed latest.pth.tar at step {train_steps}")

            if train_steps % cfg.eval_every == 0 and train_steps > 0:
                eval_start_time = time()
                model.core.generator, _gen = ema, model.core.generator
                model.core.diffusion, _diff = eval_diffusion, model.core.diffusion
                eval_batch_size = cfg.get("eval_batch_size") or cfg["batch_size"]
                metrics, img_path = evaluate(model, eval_model, lpips_model, datasets['test'], rank, eval_batch_size, cfg["num_workers"], device, samples_dir, cfg.global_seed, step=train_steps)
                model.core.diffusion = _diff
                model.core.generator = _gen
                barrier()
                logger.info(
                    f"(step={train_steps:07d}) DreamSim: {metrics['dreamsim']:.4f}, "
                    f"LPIPS: {metrics['lpips']:.4f}, PSNR: {metrics['psnr']:.2f}, "
                    + (f"Agent AUC: {metrics['agent_auc']:.3f} (memory-required {metrics['agent_auc_memreq']:.3f}, "
                       f"{metrics['agent_memreq_frac']:.0%} of pairs), acc {metrics['agent_acc']:.3f}, "
                       f"pos err {metrics['agent_pos_err']:.2f} m, " if 'agent_auc' in metrics else "")
                    + f"Eval Time: {time() - eval_start_time:.2f}"
                )
                if wandb_run is not None:
                    import wandb as wandb_lib
                    log_dict = {
                        'eval/dreamsim': metrics['dreamsim'],
                        'eval/lpips':    metrics['lpips'],
                        'eval/psnr':     metrics['psnr'],
                    }
                    for k_ in ('agent_auc', 'agent_auc_memreq', 'agent_acc', 'agent_pos_err',
                               'agent_memreq_frac', 'agent_presence_rate'):
                        if k_ in metrics:
                            log_dict[f'eval/{k_}'] = metrics[k_]
                    if img_path is not None:
                        log_dict['eval/samples'] = wandb_lib.Image(img_path)
                    wandb_run.log(log_dict, step=train_steps)

        epoch += 1

    logger.info("Done!")
    if wandb_run is not None:
        wandb_run.finish()
    destroy_process_group()


if __name__ == "__main__":
    main()
