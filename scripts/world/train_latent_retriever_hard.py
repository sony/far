"""Retriever (QueryViT) pre-training for STATE-bearing retrieval —
`train_latent_retriever.py` plus an OBJECT-CENTRIC positive term (and two
optional mechanisms, off by default).

WHY. The future-predictive (jepa) objective learns VIEW continuity — right for
navigation, where the future is answered by frames that look like it. On the
door-corridor corpus the future depends on the STATE of an entity (the second
agent crossed the doorway and stands at a corridor end), and the frames
carrying that state look nothing like the future view: a jepa tower ranks the
pose-identical EMPTY prologue view of the end above every agent-bearing frame.
Patching the jepa objective (long-horizon positives, pose-matched stale hard
negatives) does not fix this, because neither makes the crossing frames the
thing the deployment query is pulled toward, and long-horizon pairs ask an
empty doorway view to predict an unknowable state — solvable only by
memorising the episode.

OBJECT-CENTRIC POSITIVES (`obj_weight` > 0). Per-frame object presence
arrives from the backend's loss mask (`mask_suffix=.agent.npy`: the agent's
silhouette area per frame; on THOR a change-mask family plays the same role)
as `context_obj` (pool frames) — the formatter emits the mask AREA fraction,
0 = absent. For a query `(current, goal pose)` whose POOL contains the object:
  positives = the pool frames of the LATEST sighting (the last pulse — frames
              separated by < `obj_pulse_gap_s` walking back from the most
              recent presence frame), weighted by visible area, at most
              `obj_num_positives`;
  negatives = pool frames WITHOUT the object (`num_negatives`, random) — the
              pose-identical empty twins and their rotation-phase neighbours
              are all in here, no mining needed;
  neutral   = earlier sightings: never negatives (with alternating crossings
              they show the WRONG heading for the final state, so they must
              not be positives either).
The rule is EPISODE-level, not goal-frame-level: every eval-leg query of an
episode that has the object gets the same positives, whether or not the goal
frame shows the agent (the crossing is the right evidence for rendering an
EMPTY end too — it says the agent went elsewhere). A goal-frame-level rule
makes the same (scene, current, pose) map to opposite targets across episodes,
which is only solvable by memorisation. Presence frames are also excluded from
the jepa term's random negatives so the two terms never fight.
This trains the DEPLOYMENT direction directly — the query is the real one
(eval-leg current + reveal pose) and the positives are pool frames — and the
key must encode "agent present" across views (doorway crossing, at an end,
waiting at spawn): object-centric by construction. Scene identity is shared by
positives and every in-episode negative, so it is never discriminative.

KEPT, OFF BY DEFAULT: `long_horizon_p` in the sampler (mechanism 1) and the
pose-matched content-changed hard negatives (`num_hard_negatives` > 0,
mechanism 2); both measured harmful on the door-corridor corpus as
formulated.

Everything else is inherited: `state_dict()` is exactly `{query_vit.*}`, so
checkpoints load into pnp_cl via `PnPRetrieval.load_weights`. Knobs live under
`contrastive:` in `configs/world_pretrain.yaml`; `init_from` loads QueryViT
weights only (fresh optimizer / step 0). Telemetry: `obj_frac` (share of goals
with object positives), `obj_npos`, `obj_margin` (cosine: best positive minus
best negative — must go positive), `obj_loss`; plus the earlier `hard_*` /
`long_frac` / `sym` stats when those mechanisms are on.
"""
import math
import os
import sys
from datetime import datetime
from time import time

import hydra
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_latent_retriever import ContrastivePretrainModule          # noqa: E402  (base stage)


def _wrap(a: torch.Tensor) -> torch.Tensor:
    """Wrap an angle difference to (-pi, pi]."""
    return torch.atan2(torch.sin(a), torch.cos(a))


class StateContrastiveModule(ContrastivePretrainModule):
    """`ContrastivePretrainModule` + the object-centric positive term
    (+ optional pose-matched hard negatives / key-anchored symmetric term)."""

    def __init__(self, *args, obj_weight: float = 1.0, obj_num_positives: int = 8,
                 obj_pulse_gap_s: float = 2.0, obj_min_frac: float = 0.0, obj_last_only: bool = True,
                 num_hard_negatives: int = 0, hard_neg_weight: float = 4.0,
                 hard_pos_tol: float = 0.05, hard_yaw_tol_deg: float = 1.0,
                 hard_pitch_tol_deg: float = 1.0, hard_tau: float = 0.05,
                 hard_min_tokens: int = 8, hard_sym_weight: float = 1.0,
                 stride_clip=None, long_horizon_thr=None, fps: float = 10.0,
                 max_frame_offset: float = 10.0, **kwargs):
        super().__init__(*args, **kwargs)
        assert obj_weight >= 0 and obj_num_positives >= 1 and obj_pulse_gap_s > 0
        assert num_hard_negatives >= 0 and hard_neg_weight > 0 and hard_min_tokens >= 1
        self.obj_weight         = float(obj_weight)
        self.obj_num_positives  = int(obj_num_positives)
        self.obj_pulse_gap_s    = float(obj_pulse_gap_s)
        self.obj_min_frac       = float(obj_min_frac)
        self.obj_last_only      = bool(obj_last_only)
        self.num_hard_negatives = int(num_hard_negatives)
        self.hard_neg_weight    = float(hard_neg_weight)
        self.hard_pos_tol       = float(hard_pos_tol)
        self.hard_yaw_tol       = math.radians(hard_yaw_tol_deg)
        self.hard_pitch_tol     = math.radians(hard_pitch_tol_deg)
        self.hard_tau           = float(hard_tau)
        self.hard_min_tokens    = int(hard_min_tokens)
        self.hard_sym_weight    = float(hard_sym_weight)
        self.stride_clip        = None if stride_clip is None else float(stride_clip)
        self.long_horizon_thr   = None if long_horizon_thr is None else float(long_horizon_thr)   # FRAMES, current -> goal
        self.fps                = float(fps)
        self.max_frame_offset   = float(max_frame_offset)
        for _n in ("obj_loss", "obj_frac", "obj_npos", "obj_margin",
                   "hard_count", "hard_frac", "hard_margin", "long_frac", "sym"):
            self.register_buffer(f"last_stat_{_n}", torch.zeros(1), persistent=False)

    # ── forward: the base pool/query construction, plus the presence channel ──
    def forward(self, context_frames, context_poses, context_times,
                action, stride, context_mask, target_frames, action_interact=None,
                context_obj=None):
        with torch.no_grad():
            if self.long_horizon_thr is not None:
                horizon = stride.float() * self.max_frame_offset - context_times[:, -1:].float() * self.fps
                self.last_stat_long_frac.copy_((horizon > self.long_horizon_thr).float().mean().reshape(1))
        if self.stride_clip is not None:
            stride = stride.clamp(max=self.stride_clip)
        with torch.amp.autocast(device_type='cuda', enabled=False):
            B, num_goals = action.shape[:2]
            S = self.chunk_size
            past_frames = context_frames[:, :-1]
            past_poses  = context_poses [:, :-1]
            past_times  = context_times [:, :-1]
            T_past = past_frames.shape[1]
            N    = T_past // S
            drop = T_past - N * S
            past_frames = past_frames[:, drop:]
            past_poses  = past_poses [:, drop:]
            past_times  = past_times [:, drop:]
            if context_mask is not None:
                past_mask = context_mask[:, :-1][:, drop:]
            pool_obj = None
            if context_obj is not None:
                pool_obj = context_obj[:, :-1][:, drop:].float()                   # (B, N*S) area fraction

            current_frame = context_frames[:, -1]
            current_frame_g = current_frame.unsqueeze(1).expand(B, num_goals, *current_frame.shape[1:])
            current_frame_g = current_frame_g.reshape(B * num_goals, *current_frame.shape[1:])
            action_g = action.reshape(B * num_goals, -1)
            stride_g = stride.reshape(B * num_goals)
            interact_g = None
            if action_interact is not None and getattr(self.query_vit, "interact_dim", 0) > 0:
                interact_g = action_interact.reshape(B * num_goals, -1)
            queries = self.query_vit(current_frame_g, action_g, stride_g, interact_g)

            flat_pool_frames = past_frames
            flat_pool_times  = past_times
            flat_pool_poses  = past_poses
            if context_mask is not None:
                valid_frame = past_mask
            else:
                valid_frame = torch.ones(B, N * S, dtype=torch.bool, device=context_frames.device)
            return self._infonce(queries, flat_pool_frames, flat_pool_times, flat_pool_poses,
                                 valid_frame, action, target_frames, num_goals, pool_obj)

    # ── object positives: the LATEST sighting in the pool ──
    @torch.no_grad()
    def _object_positives(self, pool_obj, pool_times, valid_frame):
        """-> (pos_idx (B, P), pos_w (B, P) normalised area weights (0 where
        padded), pres (B, T) presence, last (B, T) last-sighting frames).
        Presence = valid & area > obj_min_frac. The last sighting is found by
        walking back from the most recent presence frame while consecutive
        presence frames are < obj_pulse_gap_s apart (a random pool subsample
        keeps within-pulse gaps far below the inter-pulse dwell)."""
        B, T = pool_obj.shape
        pres = valid_frame & (pool_obj > self.obj_min_frac)
        last = torch.zeros_like(pres)
        for b in range(B):
            idx = pres[b].nonzero().squeeze(1)
            if idx.numel() == 0:
                continue
            t = pool_times[b, idx]
            order = t.argsort(); idx, t = idx[order], t[order]
            if self.obj_last_only and idx.numel() > 1:
                cut = (t[1:] - t[:-1] >= self.obj_pulse_gap_s).nonzero()
                if cut.numel():
                    idx = idx[int(cut[-1]) + 1:]
            last[b, idx] = True
        P = self.obj_num_positives
        score = pool_obj.masked_fill(~last, -1.0)
        top = score.topk(P, dim=1)
        pos_idx = top.indices
        ok = top.values > -1.0
        w = top.values.clamp(min=0) * ok
        pos_w = w / w.sum(1, keepdim=True).clamp(min=1e-9)
        return pos_idx, pos_w, pres, last

    @torch.no_grad()
    def _mine_hard_negatives(self, pool_frames, pool_poses, valid_frame, goal_poses_raw, target_frames):
        B, T = pool_frames.shape[:2]
        ng = goal_poses_raw.shape[1]
        dpos   = (pool_poses[:, None, :, :3] - goal_poses_raw[:, :, None, :3]).norm(dim=-1)
        dyaw   = _wrap(pool_poses[:, None, :, 4] - goal_poses_raw[:, :, None, 4]).abs()
        dpitch = _wrap(pool_poses[:, None, :, 3] - goal_poses_raw[:, :, None, 3]).abs()
        pose_ok = ((dpos <= self.hard_pos_tol) & (dyaw <= self.hard_yaw_tol)
                   & (dpitch <= self.hard_pitch_tol) & valid_frame[:, None])
        counts = torch.zeros(B, ng, T, device=pool_frames.device, dtype=torch.long)
        for g in range(ng):
            d = ((pool_frames.float() - target_frames[:, g:g + 1].float()) ** 2).mean(2)
            counts[:, g] = (d > self.hard_tau).flatten(2).sum(-1)
        cand  = pose_ok & (counts >= self.hard_min_tokens)
        score = counts.float().masked_fill(~cand, float("-inf"))
        H = min(self.num_hard_negatives, T)
        top = score.topk(H, dim=-1)
        return top.indices, torch.isfinite(top.values)

    def _infonce(self, queries, pool_frames, pool_times, pool_poses,
                 valid_frame, action, target_frames, num_goals, pool_obj=None):
        device = queries.device
        dtype  = queries.dtype
        B, T   = pool_frames.shape[0], pool_frames.shape[1]
        key_dim = queries.shape[-1]
        ng      = num_goals
        neg_inf = float("-inf")
        tau     = self.cl_temperature

        pos_scale  = self.pos_scale .to(device, action.dtype)
        pos_offset = self.pos_offset.to(device, action.dtype)
        goal_poses_raw = action.clone()
        goal_poses_raw[..., :3] = action[..., :3] * pos_scale + pos_offset

        def _encode(idx: torch.Tensor) -> torch.Tensor:
            K = idx.shape[1]
            bidx = torch.arange(B, device=device)[:, None].expand(B, K)
            fr   = pool_frames[bidx, idx]
            k    = self.query_vit(fr.reshape(B * K, *fr.shape[2:]))      # null action -> KEY
            return k.float().reshape(B, K, key_dim)

        q = F.normalize(queries.float().reshape(B, ng, key_dim), dim=-1)

        # ── object positives first (their frames are neutral for the other terms) ──
        use_obj = self.obj_weight > 0 and pool_obj is not None
        if use_obj:
            opos_idx, opos_w, pres, last = self._object_positives(pool_obj, pool_times, valid_frame)
            has_obj = (opos_w.sum(1) > 0)                                  # (B,)
        else:
            pres = torch.zeros_like(valid_frame); has_obj = torch.zeros(B, dtype=torch.bool, device=device)

        pos_keys_list, pos_type_counts, pos_pool_idx = [], [], []
        P = self.num_positives
        jepa_slot = None
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
                pose_fov_overlap(goal_poses_raw[:, g], pool_poses, radius=self.cl_pose_radius,
                                 fov_half_h=self.cl_pose_fov_half_h, fov_half_v=self.cl_pose_fov_half_v,
                                 num_samples=self.cl_pose_num_samples)
                for g in range(ng)], dim=1)
            ov  = ov.masked_fill(~valid_frame[:, None], neg_inf)
            top = ov.topk(k_p, dim=-1).indices
            keys_p = _encode(top.reshape(B, ng * k_p)).reshape(B, ng, k_p, key_dim)
            pos_keys_list.append(keys_p)
            pos_pool_idx.append(top)
            pos_type_counts.append(("pose", k_p))
        if "jepa" in self.positive_pair_type:
            jk = self.query_vit(target_frames.reshape(B * ng, *target_frames.shape[2:]))
            jepa_slot = sum(c for _, c in pos_type_counts)
            pos_keys_list.append(jk.float().reshape(B, ng, 1, key_dim))
            pos_type_counts.append(("jepa", 1))

        loss = torch.zeros((), device=device)
        per_pos = None
        if pos_keys_list:
            pos_keys = torch.cat(pos_keys_list, dim=2)
            P_total  = pos_keys.shape[2]
            # Random negatives: valid, not a positive, and NOT an object-presence
            # frame (those are the object term's business — neutral here).
            weight = valid_frame.float()
            if pos_pool_idx:
                allp   = torch.cat([p.reshape(B, -1) for p in pos_pool_idx], dim=1)
                weight = weight.scatter(1, allp, 0.0)
            weight = weight.masked_fill(pres, 0.0) + 1e-9
            neg_idx  = torch.multinomial(weight, self.num_negatives, replacement=True)
            neg_keys = _encode(neg_idx)
            kp = F.normalize(pos_keys, dim=-1)
            kn = F.normalize(neg_keys, dim=-1)
            pos_logits = torch.einsum("bgd,bgpd->bgp", q, kp) / tau
            neg_logits = torch.einsum("bgd,bnd->bgn",  q, kn) / tau
            parts = [pos_logits.unsqueeze(-1), neg_logits.unsqueeze(2).expand(B, ng, P_total, self.num_negatives)]
            hard_ok = None
            if self.num_hard_negatives > 0:
                hard_idx, hard_ok = self._mine_hard_negatives(pool_frames, pool_poses, valid_frame, goal_poses_raw, target_frames)
                H = hard_idx.shape[-1]
                kh = F.normalize(_encode(hard_idx.reshape(B, ng * H)).reshape(B, ng, H, key_dim), dim=-1)
                hard_cos = torch.einsum("bgd,bghd->bgh", q, kh)
                hard_logits = (hard_cos / tau + math.log(self.hard_neg_weight)).masked_fill(~hard_ok, neg_inf)
                parts.append(hard_logits.unsqueeze(2).expand(B, ng, P_total, H))
            all_logits = torch.cat(parts, dim=-1)
            per_pos = all_logits.logsumexp(dim=-1) - pos_logits
            loss = loss + per_pos.mean()
            sym_loss = torch.zeros((), device=device)
            if hard_ok is not None:
                has_hard = hard_ok.any(-1)
                if self.hard_sym_weight > 0 and jepa_slot is not None and bool(has_hard.any()):
                    kj = kp[:, :, jepa_slot]
                    a_pos  = (kj * q).sum(-1) / tau
                    a_hard = (torch.einsum("bgd,bghd->bgh", kj, kh) / tau).masked_fill(~hard_ok, neg_inf)
                    sym = torch.cat([a_pos.unsqueeze(-1), a_hard], dim=-1).logsumexp(-1) - a_pos
                    sym_loss = (sym * has_hard).sum() / has_hard.sum().clamp(min=1)
                    loss = loss + self.hard_sym_weight * sym_loss
                with torch.no_grad():
                    self.last_stat_hard_count.copy_(hard_ok.float().sum(-1).mean().reshape(1))
                    self.last_stat_hard_frac.copy_(has_hard.float().mean().reshape(1))
                    self.last_stat_sym.copy_(sym_loss.detach().reshape(1))
                    if jepa_slot is not None and bool(has_hard.any()):
                        best_hard = hard_cos.masked_fill(~hard_ok, neg_inf).max(-1).values
                        self.last_stat_hard_margin.copy_(((pos_logits[..., jepa_slot] * tau - best_hard)[has_hard]).mean().reshape(1))
            with torch.no_grad():
                offset = 0
                for _pt, _cnt in pos_type_counts:
                    getattr(self, f"last_cl_loss_{_pt}").copy_(per_pos[..., offset:offset + _cnt].mean().reshape(1))
                    offset += _cnt

        # ── object-centric term ──
        obj_loss = torch.zeros((), device=device)
        if use_obj and bool(has_obj.any()):
            Po = opos_idx.shape[1]
            ko = F.normalize(_encode(opos_idx), dim=-1)                       # (B, Po, D)
            absent = (valid_frame & ~pres).float() + 1e-9                     # negatives: object-free pool frames
            oneg_idx = torch.multinomial(absent, self.num_negatives, replacement=True)
            kon = F.normalize(_encode(oneg_idx), dim=-1)                      # (B, Nn, D)
            o_pos = torch.einsum("bgd,bpd->bgp", q, ko) / tau                 # (B, ng, Po)
            o_neg = torch.einsum("bgd,bnd->bgn", q, kon) / tau                # (B, ng, Nn)
            o_all = torch.cat([o_pos.unsqueeze(-1), o_neg.unsqueeze(2).expand(B, ng, Po, self.num_negatives)], dim=-1)
            o_per = o_all.logsumexp(-1) - o_pos                               # (B, ng, Po)
            w = opos_w[:, None, :].expand(B, ng, Po)
            o_goal = (o_per * w).sum(-1)                                      # weighted over positives (B, ng)
            obj_loss = (o_goal * has_obj[:, None]).sum() / (has_obj.float().sum() * ng).clamp(min=1)
            loss = loss + self.obj_weight * obj_loss
            with torch.no_grad():
                okp = opos_w > 0
                best_pos = (o_pos * tau).masked_fill(~okp[:, None, :], neg_inf).max(-1).values      # (B, ng) cosine
                best_neg = (o_neg * tau).max(-1).values
                self.last_stat_obj_margin.copy_(((best_pos - best_neg)[has_obj]).mean().reshape(1))
                self.last_stat_obj_npos.copy_(okp.float().sum(1)[has_obj].mean().reshape(1))
        with torch.no_grad():
            self.last_stat_obj_loss.copy_(obj_loss.detach().reshape(1))
            self.last_stat_obj_frac.copy_(has_obj.float().mean().reshape(1))
        return loss.to(dtype)


HardNegContrastiveModule = StateContrastiveModule      # backward-compatible name


def make_exp_name(cfg):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    datasets = '+'.join(cfg.datasets)
    return f"{timestamp}__pretrain_retriever_hard__{cfg.run_name}__bs{cfg.batch_size}__{datasets}"


def _load_query_vit_weights(memory, path, logger):
    """Weights-only init of the QueryViT from any checkpoint carrying a
    `memory` state (a base pre-train run, or a world-model run's pnp_cl)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt["memory"] if isinstance(ckpt, dict) and "memory" in ckpt else ckpt
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if k.startswith("query_vit.")}
    assert sd, f"no query_vit.* weights in {path}"
    incompat = memory.load_state_dict(sd, strict=False)
    miss = [k for k in incompat.missing_keys if k.startswith("query_vit.")]
    logger.info(f"init_from {path}: loaded {len(sd)} query_vit tensors | missing(query_vit)={miss} "
                f"unexpected={list(incompat.unexpected_keys)}")


def _contrastive_loss(memory, batch, device, bf16):
    """One contrastive forward -> scalar loss (base unpacking + the presence channel)."""
    context_latent = batch["context_frames"].to(device, non_blocking=True)
    goal_latent    = batch["goal_frames"].to(device, non_blocking=True)
    action         = batch["goal_poses_nl"].to(device, non_blocking=True)
    stride         = batch["goal_times_nl"].to(device, non_blocking=True)
    context_poses  = batch["context_poses"].to(device, non_blocking=True)
    context_times  = batch["context_times"].to(device, non_blocking=True)
    context_mask   = batch["context_mask"].to(device, non_blocking=True)
    action_interact = batch.get("action_interact")
    if action_interact is not None:
        action_interact = action_interact.to(device, non_blocking=True)
    context_obj = batch.get("context_obj")
    if context_obj is not None:
        context_obj = context_obj.to(device, non_blocking=True)
    with torch.amp.autocast('cuda', enabled=bf16, dtype=torch.bfloat16):
        return memory(context_latent, context_poses, context_times, action, stride,
                      context_mask, goal_latent, action_interact, context_obj)


@torch.no_grad()
def evaluate_contrastive(memory, test_dataset, rank, batch_size, num_workers, device, bf16, seed, max_batches=20):
    """Mean loss over a few test batches, plus the mean of every `last_stat_*`
    buffer over those batches (so the object / hard-negative margins are
    measured on the VAL split, not only on training batches)."""
    memory.train()
    sampler = DistributedSampler(test_dataset, num_replicas=dist.get_world_size(),
                                 rank=rank, shuffle=True, seed=seed, drop_last=True)
    loader = DataLoader(test_dataset, batch_size=batch_size, sampler=sampler,
                        num_workers=num_workers, pin_memory=True, drop_last=True)
    total = torch.zeros((), device=device); count = torch.zeros((), device=device)
    mod = memory.module if hasattr(memory, "module") else memory
    stat_names = [n.rsplit("last_stat_", 1)[1] for n, _ in mod.named_buffers() if "last_stat_" in n]
    stats = {n: torch.zeros((), device=device) for n in stat_names}
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        total += _contrastive_loss(memory, batch, device, bf16).detach().float()
        for n in stat_names:
            stats[n] += getattr(mod, f"last_stat_{n}").float().reshape(())
        count += 1
    all_reduce(total, op=ReduceOp.SUM); all_reduce(count, op=ReduceOp.SUM)
    for n in stat_names:
        all_reduce(stats[n], op=ReduceOp.SUM)
    c = count.item()
    return ((total / count).item() if c > 0 else float('nan'),
            {n: (v / count).item() if c > 0 else float('nan') for n, v in stats.items()})


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

    qv_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model.memory.query_vit, resolve=True))
    query_vit = instantiate(qv_cfg)
    cl = cfg.contrastive
    local_pos_stats = (OmegaConf.to_container(cl.local_pos_stats, resolve=True)
                       if cl.local_pos_stats is not None else None)
    g = lambda k, d: cl[k] if k in cl and cl[k] is not None else d
    long_thr = float(cfg.len_traj_pred)
    memory = StateContrastiveModule(
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
        obj_weight          = float(g("obj_weight", 1.0)),
        obj_num_positives   = int(g("obj_num_positives", 8)),
        obj_pulse_gap_s     = float(g("obj_pulse_gap_s", 2.0)),
        obj_min_frac        = float(g("obj_min_frac", 0.0)),
        obj_last_only       = bool(g("obj_last_only", True)),
        num_hard_negatives  = int(g("num_hard_negatives", 0)),
        hard_neg_weight     = float(g("hard_neg_weight", 4.0)),
        hard_pos_tol        = float(g("hard_pos_tol", 0.05)),
        hard_yaw_tol_deg    = float(g("hard_yaw_tol_deg", 1.0)),
        hard_pitch_tol_deg  = float(g("hard_pitch_tol_deg", 1.0)),
        hard_tau            = float(g("hard_tau", 0.05)),
        hard_min_tokens     = int(g("hard_min_tokens", 8)),
        hard_sym_weight     = float(g("hard_sym_weight", 1.0)),
        stride_clip         = g("stride_clip", None),
        long_horizon_thr    = long_thr,
        fps                 = float(g("fps", 10.0)),
        max_frame_offset    = float(g("max_frame_offset", 10.0)),
    )
    if not any(p.requires_grad for p in memory.parameters()):
        raise ValueError("Retriever pre-training requires a trainable QueryViT.")
    logger.info(f"Object term: w={memory.obj_weight} P={memory.obj_num_positives} pulse_gap={memory.obj_pulse_gap_s}s "
                f"min_frac={memory.obj_min_frac} last_only={memory.obj_last_only} | hard negatives: H={memory.num_hard_negatives} "
                f"(w={memory.hard_neg_weight}, tol {memory.hard_pos_tol} m / {math.degrees(memory.hard_yaw_tol):.1f} deg, "
                f"tau={memory.hard_tau}, min_tokens={memory.hard_min_tokens}, sym_w={memory.hard_sym_weight}) | "
                f"stride_clip={memory.stride_clip} | long-horizon = horizon > {long_thr:.0f} frames")
    memory.to(device)

    lr = float(cfg.lr)
    weight_decay = float(cfg.weight_decay)
    opt = torch.optim.AdamW(memory.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-6)
    logger.info(f"Optimizer: AdamW on QueryViT only, lr={lr}")
    warmup_steps = int(cfg.warmup_steps)
    scheduler = (torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.2, end_factor=1.0, total_iters=warmup_steps)
                 if warmup_steps > 0 else None)
    bf16 = bool(getattr(cfg, 'bfloat16', False))

    latest_path = os.path.join(checkpoint_dir, "latest.pth.tar")
    train_steps = 0
    resume_path = latest_path if os.path.isfile(latest_path) else (cfg.from_checkpoint or None)
    init_from = g("init_from", None)
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
    elif init_from is not None:
        assert os.path.isfile(init_from), f"contrastive.init_from not found: {init_from}"
        _load_query_vit_weights(memory, init_from, logger)

    memory = DDP(memory, device_ids=[device], find_unused_parameters=True)
    logger.info(f"Memory Parameters: {sum(p.numel() for p in memory.parameters()):,}")

    cl_loss_bufs, stat_bufs = {}, {}
    for _n, _b in memory.named_buffers():
        if 'last_cl_loss_' in _n:
            cl_loss_bufs[_n.rsplit('last_cl_loss_', 1)[1]] = _b
        elif 'last_stat_' in _n:
            stat_bufs[_n.rsplit('last_stat_', 1)[1]] = _b

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
                        drop_last=True, persistent_workers=cfg['num_workers'] > 0)
    logger.info(f"Train set: {len(datasets['train']):,} samples")

    memory.train()
    running_cl, cl_steps, start_time, epoch = 0.0, 0, time(), 0
    logger.info(f"Pre-training the retriever (state / object-centric) for {cfg.iterations} iterations...")
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
                samples_per_sec = dist.get_world_size() * cfg['batch_size'] * cfg['goals_per_obs'] * steps_per_sec
                avg = torch.tensor(running_cl / cl_steps, device=device)
                all_reduce(avg, op=ReduceOp.SUM)
                avg = avg.item() / dist.get_world_size()
                extra = ''.join(f', cl/{pt}={float(b.item()):.4f}' for pt, b in cl_loss_bufs.items())
                extra += ''.join(f', {k}={float(b.item()):.3f}' for k, b in stat_bufs.items())
                logger.info(f"(step={train_steps:07d}) CL Loss: {avg:.4f}, Steps/Sec: {steps_per_sec:.2f}, "
                            f"Samples/Sec: {samples_per_sec:.2f}{extra}")
                if wandb_run is not None:
                    wlog = {'train/cl_loss': avg, 'train/steps_per_sec': steps_per_sec,
                            'train/samples_per_sec': samples_per_sec}
                    for pt, b in cl_loss_bufs.items():
                        wlog[f'train/cl_loss_{pt}'] = float(b.item())
                    for k, b in stat_bufs.items():
                        wlog[f'stat/{k}'] = float(b.item())
                    wandb_run.log(wlog, step=train_steps)
                running_cl, cl_steps, start_time = 0.0, 0, time()

            if train_steps % cfg.ckpt_every == 0 and train_steps > 0 and rank == 0:
                ckpt = {"memory": memory.module.state_dict(), "memory_opt": opt.state_dict(),
                        "pretrain_steps": train_steps}
                torch.save(ckpt, f"{checkpoint_dir}/latest.pth.tar")
                torch.save(ckpt, f"{checkpoint_dir}/{train_steps:07d}.pth.tar")
                logger.info(f"Saved checkpoint at step {train_steps}")

            if train_steps % cfg.eval_every == 0 and train_steps > 0:
                t0 = time()
                val_cl, val_stats = evaluate_contrastive(memory, datasets['test'], rank, cfg['batch_size'],
                                                         cfg['num_workers'], device, bf16, cfg.global_seed)
                barrier()
                vs = ''.join(f', {k}={v:.3f}' for k, v in val_stats.items() if k.startswith("obj_") or k in ("hard_margin", "sym"))
                logger.info(f"(step={train_steps:07d}) Val CL Loss: {val_cl:.4f}{vs}, Eval Time: {time()-t0:.2f}")
                if wandb_run is not None:
                    wandb_run.log({'eval/cl_loss': val_cl, **{f'eval/{k}': v for k, v in val_stats.items()}}, step=train_steps)
                start_time = time()
        epoch += 1

    if rank == 0 and wandb_run is not None:
        wandb_run.finish()
    barrier()
    destroy_process_group()


if __name__ == "__main__":
    main()
