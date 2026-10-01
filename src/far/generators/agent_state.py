"""Agent-state augmented generation (dynamic-agent corpus).

The observation tuple on the door-corridor corpus is (image, second-agent
detection). `AgentStatePredictor` reads the detections of the RETRIEVED context
slots and predicts the detection at the goal — `[vi, dx, dy]`, in view / position
in the goal frame — with a direct, timestep-free loss; `AgentStateCDiT` wraps it
together with the pose_fa DiT, which is conditioned on the prediction through
its adaLN vector (zero-init, detached), so the rendering does not re-make the
decision and step 0 is bit-identical to the bare DiT.

The predictor is attention pooling with a conditioned query over a SET of slot
tokens plus one learned null token: length-agnostic and permutation-invariant,
so the same module scores the EMDR² candidate sets (K sets x num_memories
slots), the local block (one frame + current) and any context size with no
special casing. The null token absorbs attention mass when the evidence is weak
(pooling over 1 slot and over 11 is calibrated alike) and doubles as the
abstain default.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from far.layers import ScalarEmbedding
from .cdit_mod_pose_fa import CDiT as PoseFaCDiT


def _mlp(d_in: int, d_hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.SiLU(), nn.Linear(d_hidden, d_out))


class AgentStatePredictor(nn.Module):
    """Predict the goal frame's agent detection from the context slots' detections.

    Inputs (N = batch of (sample, goal, set) rows, T = slots incl. the current frame):
        slot_state (N, T, state_dim)  `[vi, dx, dy, vi', dx', dy']` per slot
        slot_pose  (N, T, pose_dim)   slot pose relative to the current frame
        slot_time  (N, T)             slot time, seconds from the pool origin
        slot_mask  (N, T) bool | None True = real slot (sampler padding is False)
        action     (N, action_dim)    goal pose (the generator's `action`)
        stride     (N,)               goal time in `max_frame_offset` units
        cur_time   (N,)               current frame's time, seconds
    Output: (N, 3) `[logit vi, dx, dy]` in the goal frame's coordinates.
    """

    def __init__(self, state_dim: int = 6, pose_dim: int = 5, action_dim: int = 5,
                 dim: int = 128, heads: int = 4, pos_loss_weight: float = 1.0,
                 stride_to_seconds: float = 1.0, freq_emb: int = 64, query_to_slots: bool = True,
                 velocity_in_current_frame: bool = True):
        super().__init__()
        self.state_dim, self.pose_dim, self.action_dim = int(state_dim), int(pose_dim), int(action_dim)
        self.pos_loss_weight = float(pos_loss_weight)
        self.stride_to_seconds = float(stride_to_seconds)
        # `query_to_slots`: the query features (goal pose, goal time, current
        # time) are ALSO fed to every slot token, so "does this slot's velocity
        # point at the end being asked about" is a per-slot feature computed by
        # the slot MLP, not something the head has to reconstruct from a pooled
        # vector. Measured (probe_predictor_lengths): without it the predictor
        # scores the last and the previous crossing alike — it never reads the
        # direction and memorizes episodes instead.
        self.query_to_slots = bool(query_to_slots)
        # `velocity_in_current_frame`: append A2's velocity over the paired
        # observation, rotated by the slot's relative yaw into the CURRENT frame
        # (2 dims; zero unless both halves are in view). Parameter-free ego-motion
        # compensation from known poses — the slot's own frame is what a
        # detector reports, the current frame is where the goal direction lives,
        # and the decision is their dot product. Measured (probe_predictor_aug):
        # without it a velocity FLIP costs AUC 0.96 -> 0.57 (half the answer is a
        # per-scene lookup); with it 0.98 -> 0.16 (direction is read).
        self.velocity_in_current_frame = bool(velocity_in_current_frame)
        q_in = action_dim + dim                      # goal pose + goal time relative to the current frame
        self.time_emb = ScalarEmbedding(dim, frequency_embedding_size=freq_emb)
        self.slot_mlp = _mlp(state_dim + (2 if self.velocity_in_current_frame else 0) + pose_dim + dim
                             + (q_in if self.query_to_slots else 0), dim, dim)
        self.query_mlp = _mlp(q_in, dim, dim)
        self.null_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.head = _mlp(2 * dim, dim, 3)

    def forward(self, slot_state, slot_pose, slot_time, slot_mask, action, stride, cur_time):
        N, T = slot_state.shape[:2]
        # Times RELATIVE to the current frame: how long ago a slot was seen and
        # how far ahead the goal is are the task's time features; the absolute
        # pool-origin times carry no information about the goal but identify the
        # episode, and with ~800 training episodes the predictor memorizes them
        # (probe_predictor_lengths: train AUC 0.97 / test 0.72, direction unread).
        cur = cur_time.float()
        slot_dt = slot_time.float() - cur[:, None]
        goal_dt = stride.float() * self.stride_to_seconds - cur
        q_feat = torch.cat([action.float(), self.time_emb(goal_dt)], dim=-1)                  # (N, q_in)
        slot_state, slot_pose = slot_state.float(), slot_pose.float()
        slot_in = [slot_state]                                     # layout: [state, (v_cur), pose, time, (query)]
        if self.velocity_in_current_frame:
            both = (slot_state[..., 0] > 0) & (slot_state[..., 3] > 0)
            v = torch.where(both[..., None], slot_state[..., 1:3] - slot_state[..., 4:6], torch.zeros_like(slot_state[..., 1:3]))
            cy, sy = torch.cos(slot_pose[..., 4]), torch.sin(slot_pose[..., 4])
            slot_in.append(torch.stack([cy * v[..., 0] - sy * v[..., 1], sy * v[..., 0] + cy * v[..., 1]], dim=-1))
        slot_in += [slot_pose, self.time_emb(slot_dt)]
        if self.query_to_slots:
            slot_in.append(q_feat[:, None].expand(-1, T, -1))
        tok = self.slot_mlp(torch.cat(slot_in, dim=-1))                                       # (N, T, D)
        kv = torch.cat([tok, self.null_token.expand(N, -1, -1)], dim=1)                     # (N, T+1, D)
        pad = torch.zeros(N, T + 1, dtype=torch.bool, device=kv.device)
        if slot_mask is not None:
            pad[:, :T] = ~slot_mask.to(torch.bool)
        q = self.query_mlp(q_feat)[:, None]                                                  # (N, 1, D)
        pooled, _ = self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv), key_padding_mask=pad)
        return self.head(torch.cat([q, pooled], dim=-1)[:, 0])                               # (N, 3)

    def loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-sample `(N,)`: BCE on presence + `pos_loss_weight` x squared
        position error on goals where the agent IS in view. Per-sample so
        `rag_step` can charge it per candidate set / frame."""
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(pred[:, 0].float(), target[:, 0], reduction="none")
        pos = ((pred[:, 1:3].float() - target[:, 1:3]) ** 2).sum(-1) * target[:, 0]
        return bce + self.pos_loss_weight * pos


class AgentStateCDiT(nn.Module):
    """pose_fa DiT + `AgentStatePredictor` + zero-init adaLN conditioning on the
    prediction. Built with the DiT's own flat kwargs (so every existing
    `model.generator.*` override keeps working) plus the predictor knobs.

    Protocol for `LDWMCore` (§3.3): `predict_agent(...)` -> `(N, 3)`,
    `agent_loss(pred, target)` -> `(N,)`, and `forward(..., agent_state=)` with
    the (detached) prediction. Attributes the builder / core read
    (`context_size`, `interact_dim`, `in_channels`, ...) are delegated to the DiT.
    `load_weights` accepts either a checkpoint of this class or a FLAT pose_fa
    checkpoint (loaded strict into `dit`, predictor and conditioning fresh —
    the warm start: step 0 reproduces that checkpoint exactly)."""

    DELEGATED = ("context_size", "interact_dim", "in_channels", "out_channels", "patch_size",
                 "in_height", "in_width", "learn_sigma", "stride_to_seconds", "num_heads")

    def __init__(self, *, predictor_dim: int = 128, predictor_heads: int = 4,
                 pos_loss_weight: float = 1.0, state_dim: int = 6, **dit_kwargs):
        super().__init__()
        self.dit = PoseFaCDiT(**dit_kwargs)
        hidden = int(self.dit.x_embedder.proj.out_channels)
        self.predictor = AgentStatePredictor(
            state_dim=state_dim, pose_dim=int(dit_kwargs.get("pose_dim", 5)),
            action_dim=int(dit_kwargs.get("action_dim", 5)), dim=predictor_dim, heads=predictor_heads,
            pos_loss_weight=pos_loss_weight, stride_to_seconds=float(dit_kwargs.get("stride_to_seconds", 1.0)))
        # Prediction -> adaLN vector. Zero-init last layer: the DiT ignores the
        # state at step 0 and the pathway grows from a checkpoint that never saw it.
        # `agent_state` -> adaLN: `[±1, dx, dy]` under LDWMCore's binarised
        # protocol (`agent_cond_binary`), the raw `[logit, dx, dy]` otherwise.
        self.state_cond = _mlp(3, hidden, hidden)
        nn.init.zeros_(self.state_cond[2].weight)
        nn.init.zeros_(self.state_cond[2].bias)

    def __getattr__(self, name):
        if name in AgentStateCDiT.DELEGATED:
            return getattr(super().__getattr__("dit"), name)
        return super().__getattr__(name)

    def forward(self, x, t, action, context, stride, agent_state: torch.Tensor = None, **kw):
        cond_extra = self.state_cond(agent_state.float()) if agent_state is not None else None
        return self.dit(x, t, action, context, stride, cond_extra=cond_extra, **kw)

    def predict_agent(self, slot_state, slot_pose, slot_time, slot_mask, action, stride, cur_time):
        return self.predictor(slot_state, slot_pose, slot_time, slot_mask, action, stride, cur_time)

    def agent_loss(self, pred, target):
        return self.predictor.loss(pred, target)

    def load_weights(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["ema"] if isinstance(ckpt, dict) and "ema" in ckpt else ckpt
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
        if any(k.startswith("dit.") for k in sd):
            self.load_state_dict(sd, strict=True)
        else:                                       # flat pose_fa checkpoint -> warm start
            self.dit.load_state_dict(sd, strict=True)
