# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Adapted for FAR from Navigation World Models (NWM),
# https://github.com/facebookresearch/nwm (CC BY-NC 4.0).
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
# cdit_mod_pose_fa: the pose-conditioned full-attention CDiT.
# Frame/global alternating architecture plus
# per-token Plücker-ray pose conditioning (`far.layers.PluckerEmbedding`,
# ALWAYS on) across all three token streams: target rays toward the goal
# viewpoint (pose recovered from `action` via `local_pos_stats`), canonical
# rays for the current frame, and per-frame rays for the past context.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed

from far.layers import (
    PluckerEmbedding,
    ScalarEmbedding,
    build_2d_sincos_pos_embed,
    modulate,
    xavier_basic_init,
)


class FinalLayer(nn.Module):
    """The DiT-style adaLN-zero final projection."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear     = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # adaLN modulation runs in fp32 — see cdit_mod_score_xa for rationale.
        with torch.amp.autocast('cuda', enabled=False):
            shift, scale = self.adaLN_modulation(c.float()).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class GlobalDiTBlock(nn.Module):
    """DiT block with full self-attention over every token. No cross-attention.

    Plain DiT topology: pre-LN → self-attn → residual → pre-LN → MLP → residual,
    with adaLN-zero gating from the global conditioning vector. The 11 adaLN
    scalars used by `cdit_mod_score_xa.CDiTBlock` (3 self-attn + 5 cross-
    attn + 3 MLP) collapse to the standard DiT 6 (3 self-attn + 3 MLP)
    since cross-attention is gone.
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn  = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # adaLN in fp32 — bf16 rounds small scales (1 + scale ≈ 1) which silently
        # disables zero-init modulation early in training.
        with torch.amp.autocast('cuda', enabled=False):
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(c.float()).chunk(6, dim=1)
            )
        # Self-attention in an fp32 island. The QK^T softmax over (T+1)·P
        # tokens is bf16's worst-case operation: small logit differences
        # truncate to zero in the 8-bit mantissa, and accumulated rounding
        # shifts the softmax peak across keys, producing discontinuous
        # gradients. Same pattern as `pnp_cl.MaskedAttnBlock`. MLP stays in
        # bf16 (element-wise GELU + two matmuls; not precision-sensitive).
        h_mod = modulate(self.norm1(x), shift_msa, scale_msa)
        with torch.amp.autocast('cuda', enabled=False):
            attn_out = self.attn(h_mod.float())
        x = x + gate_msa.unsqueeze(1) * attn_out.to(x.dtype)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FrameDiTBlock(nn.Module):
    """DiT block with **intra-frame** self-attention only.

    VGGT (Wang et al., CVPR 2024) introduced alternating-attention transformer
    blocks where odd blocks run global attention over all tokens and even
    blocks restrict attention to within each per-frame group. The intra-frame
    block is much cheaper — O(F · P²) per layer vs O((F·P)²) for the global
    block (F = number of frames), an `F`× reduction in attention FLOPS — and
    forces fine-grained per-frame refinement that the global block then mixes
    across frames in the next layer.

    The per-frame patch count `P` is FIXED by the spatial grid; the number of
    frames `F` is INFERRED at forward time as `L // P`, so this block accepts a
    VARIABLE number of context frames — the sequence length `L` grows/shrinks
    with the context and `F` follows. `num_patches` (= P) is the only structural
    constant, and no parameter shape depends on the frame count, so a checkpoint
    trained at one context size runs unchanged at any other.

    Reshape mechanics:
      input  (N, F · P, H)      with F = L // P inferred from the input
      view   (N, F, P, H)
      flatten batch dim with frame dim → (N · F, P, H)
      run standard self-attention + MLP + adaLN within each (P, H) frame
      reshape back to (N, F · P, H)

    AdaLN conditioning is per-sample. Each of the `F` per-sample frame-groups
    shares the same conditioning vector, so we `repeat_interleave` each
    modulation along the batch axis to align with the reshape.
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_patches: int,
        mlp_ratio: float = 4.0,
        **block_kwargs,
    ):
        super().__init__()
        # Fixed per-frame patch count P. The frame count is inferred as L // P at
        # forward time, so a VARIABLE number of context frames is supported with
        # no parameter change (plain int, not part of the state_dict).
        self.num_patches = num_patches
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn  = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        N, L, H = x.shape
        P = self.num_patches
        assert L % P == 0, f"sequence length {L} not divisible by patches-per-frame {P}"
        F_ = L // P          # frame count inferred from the actual sequence → variable context OK

        # AdaLN modulation in fp32, then replicate each per-sample modulation
        # vector `F_` times along the batch axis to match the (N·F_, P, H)
        # reshape. `repeat_interleave(F_, dim=0)` places sample n's modulation
        # at new batch indices [n·F_, n·F_+F_) so frame f of sample n is
        # batch index n·F_+f after reshape — alignment is exact.
        with torch.amp.autocast('cuda', enabled=False):
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(c.float()).chunk(6, dim=1)
            )
        shift_msa = shift_msa.repeat_interleave(F_, dim=0)
        scale_msa = scale_msa.repeat_interleave(F_, dim=0)
        gate_msa  = gate_msa .repeat_interleave(F_, dim=0)
        shift_mlp = shift_mlp.repeat_interleave(F_, dim=0)
        scale_mlp = scale_mlp.repeat_interleave(F_, dim=0)
        gate_mlp  = gate_mlp .repeat_interleave(F_, dim=0)

        # Intra-frame attention: collapse frame axis into batch.
        # Attention runs in an fp32 island (same rationale as
        # `GlobalDiTBlock.forward`); the per-frame matrix is P×P (smaller
        # and less precision-sensitive than the global block's (T+1)·P
        # square, but we keep the pattern identical so both attention
        # ops in the cdit share the same numerical regime).
        x = x.reshape(N, F_, P, H).reshape(N * F_, P, H)
        h_mod = modulate(self.norm1(x), shift_msa, scale_msa)
        with torch.amp.autocast('cuda', enabled=False):
            attn_out = self.attn(h_mod.float())
        x = x + gate_msa.unsqueeze(1) * attn_out.to(x.dtype)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = x.reshape(N, F_, P, H).reshape(N, F_ * P, H)
        return x


class CDiT(nn.Module):
    """Diffusion model with a Transformer backbone — full-concat variant.

    The pose-conditioned full-attention (fa) CDiT: frame/global alternating
    blocks with per-token Plücker-ray pose conditioning always on, and
    WITHOUT chunk-score conditioning or null embeddings — `context_time` and
    `context_pose` are REQUIRED (`context_chunk_score` is accepted for API
    parity and ignored). Two architectural distinctions from the
    cross-attention variant:

      (1) **Full context concatenation**: target, current-frame, and ALL
          past-context frames are tokenised and concatenated along the token
          axis. The main DiT stream runs self-attention over a single
          `(N, (T+1)·P, H)` sequence — `P` target patches, `P` current-frame
          patches, and `(T-1)·P` past-context patches. The current frame is
          identified as `context[:, -1]` by the `LDWMCore.prepare_context`
          convention. Only the first `P` (target) tokens are unpatchified
          into the noise prediction; the other tokens serve as in-context
          conditioning during self-attention.
      (2) **No cross-attention, VGGT-style alternating self-attention**:
          blocks alternate between intra-frame (`FrameDiTBlock`, even depth
          indices) and global (`GlobalDiTBlock`, odd depth indices) attention.
          Frame blocks refine each per-frame group of P tokens independently
          and cheaply (O(P²) per frame); global blocks then mix across the
          (T+1) frames. There is no cross-attention, no separate KV stream,
          no `ScoredCrossAttention`, and no `log_w` bias.

    Cost trade-off vs cdit_mod_score_xa:
      - Global blocks (half of all blocks) attend over (T+1)·P tokens vs
        cdit_mod_score_xa's P self-attn main-stream + a (T-1)·P cross-attn
        KV pass.
      - Frame blocks (the other half) attend over just P tokens at a time,
        replicated across (T+1) frames in batch — total O((T+1)·P²) attention
        per layer, the same as one of cdit_mod_score_xa's small cross-
        attention passes.
      - For depth=12, T+1=5: global-only would cost 12·25·P² = 300 P²; with
        VGGT-style alternation it costs (6·25 + 6·5)·P² = 180 P² — a 40%
        attention-FLOPS saving relative to all-global, while keeping full
        cross-frame mixing capacity.

    Requires `context_size >= 2` so at least one past frame is included.
    `depth` should be even so the frame/global alternation pairs up; an odd
    depth places an unpaired frame block at the end (functional but slightly
    unusual — typically use even depths).
    """


    def __init__(
        self,
        in_height: int,
        in_width: int,
        in_channels: int,
        context_size: int = 4,
        patch_size: int = 2,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        learn_sigma: bool = True,
        action_dim: int = 3,
        pose_dim: int = 5,
        fov_half_h: float = 52.5,
        fov_half_v: float = 37.5,
        local_pos_stats: dict | None = None,
        stride_to_seconds: float = 1.0,
        interact_dim: int = 0,
    ):
        super().__init__()
        assert context_size >= 2, (
            f"cdit_mod_pose_fa requires context_size >= 2 — at least one past "
            f"frame must be present alongside the current frame; got "
            f"context_size={context_size}"
        )
        assert pose_dim == 5 and action_dim == 5, (
            "cdit_mod_pose_fa derives Plücker rays from [x, y, z, pitch, yaw] "
            "poses and treats `action` as the goal pose in the same format; "
            f"got pose_dim={pose_dim}, action_dim={action_dim}"
        )
        self.context_size = context_size
        self.stride_to_seconds = stride_to_seconds
        self.interact_dim      = int(interact_dim)
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.in_height = in_height
        self.in_width = in_width

        # Patch embedder shared by the noisy diffusion target AND the context
        # frames — context frames are clean `in_channels` latents (same modality
        # as the target). Sharing matters early in training: `x_embedder` gets a
        # strong, ungated gradient via the target residual → `final_layer` path
        # and is well-trained within a few steps, whereas a separate context
        # patcher would only see the adaLN-gated context-attention gradient and
        # lag badly — leaving the context poorly encoded for thousands of steps.
        self.x_embedder = PatchEmbed((in_height, in_width), patch_size, in_channels, hidden_size, bias=True)

        # 2D sincos spatial positional embedding (fixed, scaled by a learned
        # softplus scalar). Broadcast over every per-frame block of P tokens
        # in the main sequence.
        pos = build_2d_sincos_pos_embed(hidden_size, in_height // patch_size, in_width // patch_size)
        self.register_buffer('spatial_pos', torch.from_numpy(pos))  # (num_patches, H)
        self.pos_raw_scale = nn.Parameter(torch.tensor(-4.0))

        # Role tokens — one per stream. With every per-frame block of P
        # tokens sharing the same 2D-sincos spatial grid, the role tag is the
        # only signal that lets self-attention distinguish:
        #   - the noisy target patches (predict noise here),
        #   - the current-frame patches (clean, immediate observation),
        #   - the past-context patches (clean, retrieved frames; further
        #     differentiated from each other by `context_time_emb` and their
        #     Plücker-ray embeddings).
        # Separating current-frame from past-context lets the network learn
        # different attention patterns into each (typically: target patches
        # pull most strongly from current at the same spatial position, then
        # from past frames at correlated content).
        self.role_emb_target  = nn.Parameter(torch.zeros(hidden_size))
        self.role_emb_current = nn.Parameter(torch.zeros(hidden_size))
        self.role_emb_context = nn.Parameter(torch.zeros(hidden_size))

        # Optional INTERACTION channel (AI2-THOR): a multi-hot of the action
        # types occurring between the current frame and the target
        # (`action_interact`). Pose deltas cannot express "the container
        # opens", so without this the eval-leg reveal is unpredictable and the
        # rollout simply never opens the door — the memory probe never fires.
        # Injected through adaLN alongside the diffusion timestep (see
        # `forward`) — per-block and multiplicative, unlike the per-token
        # geometric conditions. Zero-init final layer: a pose-only checkpoint
        # is reproduced exactly at load, with or without the kwarg.
        if self.interact_dim > 0:
            self.interact_emb = nn.Sequential(
                nn.Linear(self.interact_dim, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )

        # ONE trajectory-time embedding shared by ALL frames (past, current,
        # target) so every frame sits on the same time axis and relative
        # offsets are directly comparable. The target's `stride` arrives in
        # max_frame_offset units; `stride_to_seconds` (= mfo / fps) converts
        # it onto the context frames' seconds axis (loopnav: 20/20 = 1.0).
        self.context_time_emb = ScalarEmbedding(hidden_size)

        # Plücker pose pathway: per-token ray embeddings for target / current /
        # past streams. `local_pos_stats` inverts the formatter's `_nl` position
        # normalisation on `action` (= goal_poses_nl) to recover the target
        # pose — same convention as WorldMemRetrieval. None = identity.
        self.plucker_emb = PluckerEmbedding(
            in_height // patch_size, in_width // patch_size,
            fov_half_h, fov_half_v, hidden_size,
        )
        if local_pos_stats is not None:
            mn = torch.as_tensor(list(local_pos_stats["min"]), dtype=torch.float32).reshape(-1)
            mx = torch.as_tensor(list(local_pos_stats["max"]), dtype=torch.float32).reshape(-1)
            pos_scale, pos_offset = (mx - mn) / 2, (mx + mn) / 2
        else:
            pos_scale, pos_offset = torch.ones(3), torch.zeros(3)
        self.register_buffer("plucker_pos_scale",  pos_scale,  persistent=False)
        self.register_buffer("plucker_pos_offset", pos_offset, persistent=False)

        # adaLN conditioning carries ONLY the diffusion timestep — the
        # transition query (goal pose, trajectory time) is fully conveyed
        # per-token (Plücker rays from `action`, shared time embedding from
        # `stride`), so the block modulation reduces to the vanilla-DiT
        # denoising-schedule signal.
        self.t_embedder = ScalarEmbedding(hidden_size)

        # Per-frame patch count P — fixed by the spatial grid, INDEPENDENT of the
        # number of context frames. FrameDiTBlock reshapes the sequence into
        # per-frame groups of P tokens and infers the frame count as L // P at
        # forward time, so the model accepts a VARIABLE number of context frames
        # at train/inference (e.g. context-size dropout) regardless of the
        # `context_size` it was built with. `context_size` shapes NO parameters
        # (which is why one checkpoint loads strict into any context_size);
        # `self.num_frames` is a nominal default only, no longer used by the blocks.
        num_patches = (in_height // patch_size) * (in_width // patch_size)
        self.num_frames = context_size + 1

        # VGGT-style alternating attention: even-index layers do intra-frame
        # attention, odd-index layers do global. With depth=12 this yields 6
        # frame blocks + 6 global blocks interleaved.
        self.blocks = nn.ModuleList([
            FrameDiTBlock(hidden_size, num_heads, num_patches=num_patches, mlp_ratio=mlp_ratio)
            if i % 2 == 0
            else GlobalDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for i in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def load_weights(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        self.load_state_dict(ckpt["ema"], strict=True)

    def initialize_weights(self):
        xavier_basic_init(self)

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        if self.interact_dim > 0:
            nn.init.normal_(self.interact_emb[0].weight, std=0.02)
            nn.init.constant_(self.interact_emb[0].bias, 0)
            nn.init.constant_(self.interact_emb[2].weight, 0)   # zero-init: no-op at step 0
            nn.init.constant_(self.interact_emb[2].bias, 0)
        nn.init.normal_(self.role_emb_target,              std=0.02)
        nn.init.normal_(self.role_emb_current,             std=0.02)
        nn.init.normal_(self.role_emb_context,             std=0.02)

        # Condition MLPs — small-normal first Linear, zero-init last so the
        # initial contribution to any token is zero.
        condition_mlps = [
            self.context_time_emb,
            self.plucker_emb,
        ]
        for emb in condition_mlps:
            nn.init.normal_(emb.mlp[0].weight, std=0.02)
            nn.init.normal_(emb.mlp[2].weight, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # adaLN-zero — every block's final modulation Linear is zero-init so
        # the residual gates start closed; the network behaves like an
        # identity at step 0 except for the noise-target prediction head.
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.patch_size
        h = self.in_height // p
        w = self.in_width // p
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    def forward(
        self,
        x: torch.Tensor,                                # (N, C, H, W) noisy target latent
        t: torch.Tensor,                                # (N,) diffusion timestep
        action: torch.Tensor,                           # (N, action_dim) pose delta
        context: torch.Tensor,                          # (N, context_size, C, H, W)
        stride: torch.Tensor,                           # (N,) relative time to target
        context_time: torch.Tensor = None,              # (N, context_size) seconds
        context_pose: torch.Tensor = None,              # (N, context_size, pose_dim)
        context_chunk_score: torch.Tensor = None,       # accepted for API parity; IGNORED
        action_interact: torch.Tensor = None,           # (N, interact_dim) multi-hot of action types in (current, target]
        cond_extra: torch.Tensor = None,                # (N, hidden) extra adaLN conditioning (AgentStateCDiT's predicted state)
    ) -> torch.Tensor:
        N, T = x.shape[0], context.shape[1]   # T = #context frames (current + past); MAY vary per call
        T_past = T - 1
        pos_scale = F.softplus(self.pos_raw_scale)
        assert context.shape[2] == self.in_channels, (
            f"context has {context.shape[2]} channels but in_channels="
            f"{self.in_channels}; this generator consumes plain `in_channels` "
            f"context frames (no value channels)"
        )

        # ── Tokenize noisy target + current-frame + past-context frames. Context
        # frames are clean `in_channels` latents — same modality as the target —
        # so they share `x_embedder` with it.
        current_frame = context[:, -1]                                       # (N, C, H, W)
        past_context  = context[:, :-1]                                      # (N, T-1, C, H, W)
        x_tok    = self.x_embedder(x)                                        # (N, P, H)
        curr_tok = self.x_embedder(current_frame)                            # (N, P, H)
        past_tok = self.x_embedder(past_context.flatten(0, 1)).unflatten(0, (N, T_past))   # (N, T-1, P, H)

        P = x_tok.shape[1]

        # Spatial positional embedding — same (P, H) grid broadcast onto every
        # per-frame block in the main sequence.
        x_tok    = x_tok    + pos_scale * self.spatial_pos
        curr_tok = curr_tok + pos_scale * self.spatial_pos
        past_tok = past_tok + pos_scale * self.spatial_pos

        # Role embeddings — distinct vectors for target / current / past so
        # self-attention can learn different read patterns into each stream.
        # Trajectory-time embeddings below further distinguish individual
        # frames within their shared role tags.
        x_tok    = x_tok    + self.role_emb_target
        curr_tok = curr_tok + self.role_emb_current
        past_tok = past_tok + self.role_emb_context

        # Required per-frame conditions — this variant has no null embeddings;
        # run it with `condition_on_context_time=true` and
        # `condition_on_context_pose=true` so LDWMCore forwards both.
        assert context_time is not None and context_pose is not None, (
            "cdit_mod_pose_fa requires context_time and context_pose "
            "(no null embeddings; set condition_on_context_time=true and "
            "condition_on_context_pose=true)"
        )

        # Trajectory-time embeddings (NOT the diffusion timestep), added
        # per-token to EVERY frame's token group through ONE shared embedding:
        # past + current frames at `context_time` (seconds from the pool
        # origin), the noisy target at `stride · stride_to_seconds` (same
        # origin, converted onto the same seconds axis) — so relative offsets
        # between any two frames are directly comparable. fp32 island —
        # sinusoidal args + MLP in fp32 to avoid bf16 truncation collapsing
        # nearby times.
        with torch.amp.autocast('cuda', enabled=False):
            ctx_time_emb = self.context_time_emb(context_time.float())      # (N, T, H)
            tgt_time_emb = self.context_time_emb(stride.float() * self.stride_to_seconds)  # (N, H)
        past_tok = past_tok + ctx_time_emb[:, :-1, None]                    # broadcast over P
        curr_tok = curr_tok + ctx_time_emb[:, -1, None]
        x_tok    = x_tok    + tgt_time_emb[:, None]

        # Plücker rays (per-token): target patches get rays toward the GOAL
        # viewpoint (pose recovered from `action` by inverting the `_nl`
        # normalisation), current patches the canonical rays (zero pose —
        # poses are local to the current frame), past patches their own
        # frames' rays. All in one fp32 island (trig + cross products).
        with torch.amp.autocast('cuda', enabled=False):
            act = action.float()
            tgt_pose = torch.cat(
                [act[:, :3] * self.plucker_pos_scale + self.plucker_pos_offset, act[:, 3:5]], dim=1,
            )
            x_tok    = x_tok    + self.plucker_emb(tgt_pose)                         # (N, P, H)
            curr_tok = curr_tok + self.plucker_emb(tgt_pose.new_zeros(N, 5))         # (N, P, H)
            past_tok = past_tok + self.plucker_emb(context_pose[:, :-1].float())     # (N, T-1, P, H)

        # ── Concatenate everything into ONE self-attention sequence ──────────
        # Order: [target patches, current-frame patches, past-frame patches].
        # The target patches sit at positions [0, P) so the final-layer
        # slice `x_tok[:, :P]` recovers exactly the noise prediction.
        past_tok = past_tok.flatten(1, 2)                                    # (N, (T-1)*P, H)
        x_tok    = torch.cat([x_tok, curr_tok, past_tok], dim=1)             # (N, (T+1)*P, H)

        # ── adaLN conditioning: diffusion timestep (+ optional interaction) ──
        # The interaction label is a GLOBAL, content-changing property of the
        # transition with no spatial structure — the case adaLN exists for. It
        # is re-injected at every block and gates the residual updates
        # multiplicatively, which a one-shot token addition cannot match: the
        # label is all-zero for ~91% of samples, and it has to drive a large
        # structural change (the container visibly opens), so weak
        # conditioning would just make the model hedge. Action/stride stay
        # per-token because THEY are geometric (Plücker rays); that rationale
        # never applied to a label with no spatial extent.
        with torch.amp.autocast('cuda', enabled=False):
            cond = self.t_embedder(t.float())
            if self.interact_dim > 0 and action_interact is not None:
                cond = cond + self.interact_emb(action_interact.float())
            if cond_extra is not None:
                cond = cond + cond_extra.float()

        # ── DiT denoising — pure self-attention over the full sequence ──────
        for block in self.blocks:
            x_tok = block(x_tok, cond)

        # Take only the target portion before the final adaLN-zero head; the
        # current/past tokens served as in-context conditioning and don't
        # need decoding.
        x_tok = x_tok[:, :P]
        x_tok = self.final_layer(x_tok, cond)
        return self.unpatchify(x_tok)
