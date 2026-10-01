"""Pnp_cl: QueryViT-only frame-level retrieval (action-conditioned, per goal).

`QueryViT` is the ONLY encoder. It maps a single frame → a `key_dim` vector;
with no action it is a pure-content KEY, with action+stride it is an action-
conditioned QUERY (same tower / `key_head`, so keys & queries share one cosine
space). The memory does no compression of its own: it returns the selected RAW
chunks, which are compressed downstream by the generator's FramePack
`ChunkEmbed` (`cdit_mod_chunk`).

Past context is split into chunks of `chunk_size` frames. Each chunk's KEY is
`QueryViT(middle_frame)` (null action). One QUERY per `(sample, goal)` is
`QueryViT(current_frame, action, stride)`. Chunks are ranked by their fused
per-frame logit (max over the chunk's frames) and the top `num_sets · num_memories`
are partitioned IN RANK ORDER into `num_sets` (K) sets of `num_memories`
chunks (set 0 = the best M chunks). Temporal diversity is enforced by the
chunking itself — distinct chunks are ≥ `chunk_size` frames apart — with no
pairwise-key diversity term or beam search. The per-set softmax scores feed
LDWMCore's RAG posterior over the K sets, so the retriever trains through the
selected keys + query.

`QueryViT` is a full-attention ViT — every token (frame patches and the
q_token) attends to all others; the q_token is read out as the key/query.

The retriever is warm-started by a SEPARATE contrastive (InfoNCE) pre-training
stage — `scripts/world/train_latent_retriever.py` — which trains the QueryViT on
its own (`ContrastivePretrainModule`); that loss is NOT computed here, and this
module always returns the 5-tuple LDWMCore consumes.
"""

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp, PatchEmbed

from far.layers import (
    ActionEmbedding,
    RoPE2D,
    ScalarEmbedding,
    modulate,
)


__all__ = [
    "LocalCandidates",
    "PnPRetrieval",
    "QueryViT",
]


class LocalCandidates(NamedTuple):
    """Per-(sample, goal) within-chunk candidates for local (frame-level) RAG.

    Emitted by `PnPRetrieval.forward(return_local=True)` ALONGSIDE the standard
    5-tuple, and consumed only by `LDWMCore.rag_step`'s local block. Holds the
    FULL `S`-frame chunks of every selected set (plain data) plus the
    grad-carrying per-frame prior over those `S` frames — so rag_step can score
    a within-chunk frame's diffusion reward and distil it into the retriever's
    per-frame scoring (which the chunk-max collapse otherwise leaves
    gradient-starved). Tensors carry the `(B, num_goals, K, M, S, ...)` axes.
    """
    chunks:      torch.Tensor   # (B, ng, K, M, S, C, H, W) full S-frame selected chunks (data)
    poses:       torch.Tensor   # (B, ng, K, M, S, pose_dim)
    times:       torch.Tensor   # (B, ng, K, M, S) absolute times
    frame_prior: torch.Tensor   # (B, ng, K, M, S) softmax over S of the per-frame query–key cosine (grad)


class MaskedRoPESelfAttention(nn.Module):
    """RoPE self-attention with an optional additive mask on the SDPA logits.

    Identical to `far.layers.attention.RoPESelfAttention` except for the
    `attn_mask` kwarg, which is forwarded to `F.scaled_dot_product_attention`.
    The learned attention sink appended to K/V (Darcet et al. 2023) lives
    outside the positional grid, so the mask shape is `(L, L + 1)` — the
    trailing sink column is always zero (every query can drain into it).
    """

    def __init__(self, hidden_size: int, num_heads: int, qkv_bias: bool = True, rope: nn.Module = None):
        super().__init__()
        assert hidden_size % num_heads == 0
        assert rope is not None, "rope module (RoPE3D) must be provided"
        self.num_heads = num_heads
        self.head_dim  = hidden_size // num_heads
        self.qkv       = nn.Linear(hidden_size, 3 * hidden_size, bias=qkv_bias)
        self.proj      = nn.Linear(hidden_size, hidden_size)
        self.rope      = rope
        self.bias_k = nn.Parameter(torch.empty(1, num_heads, 1, self.head_dim))
        self.bias_v = nn.Parameter(torch.empty(1, num_heads, 1, self.head_dim))
        nn.init.xavier_normal_(self.bias_k)
        nn.init.xavier_normal_(self.bias_v)

    def forward(self, x: torch.Tensor, *positions: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, L, _ = x.shape
        H, D = self.num_heads, self.head_dim
        qkv = self.qkv(x).reshape(B, L, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.rope(q, *positions)
        k = self.rope(k, *positions)
        sink_k = self.bias_k.to(k.dtype).expand(B, H, 1, D)
        sink_v = self.bias_v.to(v.dtype).expand(B, H, 1, D)
        k = torch.cat([k, sink_k], dim=2)
        v = torch.cat([v, sink_v], dim=2)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn = attn.transpose(1, 2).reshape(B, L, H * D)
        return self.proj(attn)


class MaskedViTAdaLNBlock(nn.Module):
    """ViT block with AdaLN-zero conditioning, axial-RoPE, and an attention mask.

    The same fp32 islands / dtype handling as `MaskedRoPESelfAttention` above; the
    only differences are the AdaLN-zero conditioning path and the
    self-attention forwarding an optional `attn_mask` through to
    `MaskedRoPESelfAttention`.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        rope: nn.Module = None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn  = MaskedRoPESelfAttention(hidden_size, num_heads=num_heads, qkv_bias=True, rope=rope)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        *positions: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # adaLN projection in fp32 — protects shift/scale/gate from bf16
        # quantisation. Gates are cast back to autocast dtype so the residual
        # stream stays in bf16; shifts/scales stay fp32 (modulate upcasts).
        if torch.is_autocast_enabled():
            ac_dtype = torch.get_autocast_dtype('cuda')
        else:
            ac_dtype = c.dtype
        with torch.amp.autocast(device_type='cuda', enabled=False):
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(c.float()).chunk(6, dim=1)
            )
        gate_msa = gate_msa.to(ac_dtype)
        gate_mlp = gate_mlp.to(ac_dtype)
        # Run self-attention in fp32 — same rationale as in `MaskedRoPESelfAttention`.
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        with torch.amp.autocast(device_type='cuda', enabled=False):
            attn_out = self.attn(h.float(), *positions, attn_mask=attn_mask)
        x = x + gate_msa.unsqueeze(1) * attn_out.to(x.dtype)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class QueryViT(nn.Module):
    """ViT compressor for the current frame + a learnable query token.

    2D-RoPE patches anchored at their grid index, q_token anchored at
    (0, 0), AdaLN-zero conditioning on action + rollout stride. Full
    (unmasked) attention: every token — frame patches and the q_token —
    attends to all others (and to the learned attention sink).

    Inputs:
        frame:    (B, C, H, W)      current frame
        action:   (B, action_dim)   pose delta toward the rollout goal
        stride:   (B,)              rollout stride (relative time into the future)
        interact: (B, interact_dim) optional multi-hot of the INTERACTION types
                  occurring between the current frame and the goal (THOR;
                  `action_interact` from the formatter). Kept OUT of `action`
                  on purpose: `cdit_mod_pose_fa` asserts action_dim == 5 and
                  decodes `action` into Plucker rays, and worldmem / the
                  metadata scorer invert it as a goal pose, so widening it
                  would break all three. Its embedding is zero-init, so a
                  pose-only checkpoint is reproduced exactly at step 0.

    Output:
        (B, key_dim) — a KEY (no action) or QUERY (with action+stride), projected
        by `key_head`; keys and queries share one cosine space.

    `query_adapter_hidden` (optional): when set, freezes the whole tower and
    adds a residual MLP head applied ONLY to the action-conditioned QUERY
    (the KEY path is untouched). The frozen encoder is stop-grad'd in `forward`
    (retains no activations), so only the head trains — a cheap way to adapt a
    pre-trained retriever to the diffusion task. Final layer is zero-init, so at
    step 0 the query equals the pre-trained encoder output.
    """

    def __init__(
        self,
        img_height: int = 18,
        img_width: int = 32,
        patch_size: int = 2,
        in_chans: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        action_dim: int = 3,
        mlp_ratio: float = 4.0,
        key_dim: int = 256,
        interact_dim: int = 0,
    ):
        super().__init__()
        # One learnable query token per (sample, goal).
        self.num_q_tokens = 1

        self.patch_embed = PatchEmbed((img_height, img_width), patch_size, in_chans, embed_dim, bias=True)

        # Patch grid (H', W') — used to build per-token (y, x) positions for 2D RoPE.
        self.H_grid = img_height // patch_size
        self.W_grid = img_width  // patch_size

        # Learnable query token.
        self.q_tokens = nn.Parameter(torch.zeros(1, self.num_q_tokens, embed_dim))

        # AdaLN conditioning: action + rollout stride.
        self.action_mlp = ActionEmbedding(embed_dim, action_dim=action_dim)
        self.stride_mlp = ScalarEmbedding(embed_dim)

        # Null embeddings substituted when `action` / `stride` are not passed.
        # With both null the tower produces a pure-content KEY; with both
        # supplied it produces an action-conditioned QUERY — same tower and
        # `key_head`, so keys and queries live in one comparable cosine space.
        self.null_action_emb = nn.Parameter(torch.zeros(embed_dim))
        self.null_stride_emb = nn.Parameter(torch.zeros(embed_dim))

        # Optional interaction-label pathway, ADDITIVE to the conditioning
        # (never folded into ActionEmbedding, which splits hidden_size across
        # its dims — appending labels there would silently shrink the pose
        # pathway's per-dim capacity). Zero-init final layer + a learned null
        # for the KEY path, so keys stay pure content and step 0 reproduces
        # the pose-only encoder exactly.
        self.interact_dim = int(interact_dim)
        if self.interact_dim > 0:
            self.interact_mlp = nn.Sequential(
                nn.Linear(self.interact_dim, embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim),
            )
            self.null_interact_emb = nn.Parameter(torch.zeros(embed_dim))

        # 2D RoPE shared across all blocks (positional buffers only).
        head_dim = embed_dim // num_heads
        rope = RoPE2D(head_dim)
        self.blocks = nn.ModuleList([
            MaskedViTAdaLNBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, rope=rope)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Project the q_token to key_dim — keys (null action) and queries
        # (with action) are compared by cosine similarity in this space.
        self.key_head = nn.Linear(embed_dim, key_dim)

        # NOTE: the query/key adapters AND the encoder freeze now live in
        # `PnPRetrieval` (applied to the QueryViT output), so they cover BOTH
        # cached + live-encoded keys/queries and QueryViT stays a pure encoder.
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_basic_init)

        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.patch_embed.proj.bias, 0)

        nn.init.normal_(self.q_tokens, std=0.02)
        nn.init.normal_(self.null_action_emb, std=0.02)
        nn.init.normal_(self.null_stride_emb, std=0.02)
        if self.interact_dim > 0:
            nn.init.normal_(self.interact_mlp[0].weight, std=0.02)
            nn.init.constant_(self.interact_mlp[0].bias, 0)
            nn.init.constant_(self.interact_mlp[2].weight, 0)   # zero-init: no-op at step 0
            nn.init.constant_(self.interact_mlp[2].bias, 0)
            # ZERO (not std=0.02 like the action/stride nulls): the null branch
            # is what the KEY path takes, so a nonzero init would shift every
            # key the moment interact_dim is enabled — silently invalidating
            # cached keys and breaking bit-identical reload of pose-only
            # checkpoints. Learnable, so it can still separate "no label" from
            # "label says nothing happened" during training.
            nn.init.zeros_(self.null_interact_emb)

        nn.init.normal_(self.stride_mlp.mlp[0].weight, std=0.02)
        nn.init.normal_(self.stride_mlp.mlp[2].weight, std=0.02)
        for emb in self.action_mlp.dim_embs:
            nn.init.normal_(emb.mlp[0].weight, std=0.02)
            nn.init.normal_(emb.mlp[2].weight, std=0.02)

        # adaLN-zero: identity init for the conditioning path
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def _build_positions(self, B: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Build (pos_y, pos_x) for the full [q_token, patches] sequence.

        Query token is anchored at (0, 0); patch tokens get their natural
        grid index. Returns two (B, L) tensors where L = num_q_tokens + H'*W'.
        """
        H_, W_ = self.H_grid, self.W_grid
        y  = torch.arange(H_, device=device, dtype=torch.float32)
        x  = torch.arange(W_, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing='ij')          # each (H', W')
        v_y, v_x = yy.reshape(-1), xx.reshape(-1)             # each (H'*W',)

        Q = self.num_q_tokens
        zeros = torch.zeros(Q, device=device, dtype=v_y.dtype)
        pos_y = torch.cat([zeros, v_y]).unsqueeze(0).expand(B, -1)
        pos_x = torch.cat([zeros, v_x]).unsqueeze(0).expand(B, -1)
        return pos_y, pos_x

    def forward(
        self,
        frame:  torch.Tensor,            # (B, C, H, W)
        action: torch.Tensor | None = None,   # (B, action_dim) or None → null (KEY)
        stride: torch.Tensor | None = None,   # (B,) or None → null
        interact: torch.Tensor | None = None, # (B, interact_dim) or None → null
    ) -> torch.Tensor:
        # Pure encoder. The adapters + the frozen-encoder no_grad now live in
        # `PnPRetrieval` (applied to the QueryViT output), so this just encodes.
        return self._encode_body(frame, action, stride, interact)

    def _encode_body(self, frame, action=None, stride=None, interact=None) -> torch.Tensor:
        """Shared encoder: [q_token, patches] → full-attention ViT → key_head;
        returns the q_token projection (B, key_dim). `forward` runs this under
        no_grad (stop-grad) when the query adapter is active."""
        B = frame.shape[0]
        frame_tokens = self.patch_embed(frame)              # (B, P, D)
        q_tokens     = self.q_tokens.expand(B, -1, -1)      # (B, Q, D)

        x = torch.cat([q_tokens, frame_tokens], dim=1)      # (B, Q + P, D)

        pos_y, pos_x = self._build_positions(B, x.device)

        # Conditioning in fp32 — drives every block's AdaLN. A missing action /
        # stride falls back to its learned null embedding (→ content KEY).
        with torch.amp.autocast(device_type='cuda', enabled=False):
            act_emb = self.action_mlp(action.float()) if action is not None else self.null_action_emb.expand(B, -1)
            str_emb = self.stride_mlp(stride.float()) if stride is not None else self.null_stride_emb.expand(B, -1)
            cond = act_emb + str_emb                                              # (B, D) fp32
            if self.interact_dim > 0:
                cond = cond + (self.interact_mlp(interact.float()) if interact is not None
                               else self.null_interact_emb.expand(B, -1))

        # Full (unmasked) attention — every token attends to all others.
        for blk in self.blocks:
            x = blk(x, cond, pos_y, pos_x)
        x = self.norm(x)

        # Query in fp32 — feeds the fp32 similarity scoring against the keys.
        with torch.amp.autocast(device_type='cuda', enabled=False):
            return self.key_head(x[:, :self.num_q_tokens].float()).squeeze(1)  # (B, key_dim) fp32


class MetadataCrossScorer(nn.Module):
    """Cross-scorer over pose/time metadata: a per-(frame, goal) logit.

    Unlike the visual bi-encoder (whose keys are expensive and cacheable), the
    geometrically meaningful quantity here is inherently a PAIR function — the
    goal's position in each context frame's egocentric coordinates, the
    relative heading, and the time offsets. These are computed in closed form,
    Fourier-featurised, and scored by a small MLP: no bilinear (dot-product)
    rank constraint, arbitrary sharpness — a learnable generalisation of the
    worldmem (FOV-overlap) and temporal (recency) score families.

    Inputs are the batch's local-frame quantities: context poses/times as fed
    to `PnPRetrieval.forward`, the goal pose recovered from `action`
    (= goal_poses_nl) by inverting the `_nl` normalisation via
    `local_pos_stats` (worldmem-style; None = identity), and the goal time
    from `stride` via `stride_to_seconds` (= max_frame_offset / fps).

    Zero-init last layer ⇒ all logits are 0 at init, so fusing this scorer
    into an existing ranking is a no-op at step 0. The output is tanh-bounded
    to [-1, 1] — the visual cosine's range — so the fused logit keeps the
    scale that downstream consumers (generator chunk-score embedding, softmax
    temperatures) were designed for.
    """

    def __init__(
        self,
        hidden: int = 128,
        num_layers: int = 2,
        num_bands: int = 6,
        local_pos_stats: dict | None = None,
        stride_to_seconds: float = 1.0,
        agent_state_dim: int = 0,
    ):
        super().__init__()
        assert num_layers >= 2, f"num_layers must be >= 2, got {num_layers}"
        self.stride_to_seconds = stride_to_seconds
        # AGENT-STATE CUE (dynamic-agent corpus): each context frame's
        # second-agent detection `[vi, dx, dy, vi', dx', dy']` (position in the
        # frame's own egocentric floor frame; primed = the paired earlier
        # observation), plus the pair's displacement rotated into the CURRENT
        # frame (2; the same derived feature the generator's predictor reads),
        # together with the CURRENT frame's own detection `[vi, dx, dy]` (the
        # query-side fact: is the agent in view right now, or must the memory
        # supply it), is projected by a ZERO-INIT linear map onto the first
        # hidden layer's pre-activation — so the scorer's ranking is unchanged
        # at step 0 and every metadata checkpoint loads (the new block is
        # expected-missing). Lets the retriever score frames by "an agent was
        # seen here, heading that way" instead of geometry alone. 0 = off.
        self.agent_state_dim = int(agent_state_dim)
        self.agent_proj = (nn.Linear(self.agent_state_dim + 2 + 3, hidden)
                           if self.agent_state_dim > 0 else None)
        if self.agent_proj is not None:
            nn.init.zeros_(self.agent_proj.weight)
            nn.init.zeros_(self.agent_proj.bias)
        if local_pos_stats is not None:
            mn = torch.as_tensor(list(local_pos_stats["min"]), dtype=torch.float32).reshape(-1)
            mx = torch.as_tensor(list(local_pos_stats["max"]), dtype=torch.float32).reshape(-1)
            pos_scale, pos_offset = (mx - mn) / 2, (mx + mn) / 2
        else:
            pos_scale, pos_offset = torch.ones(3), torch.zeros(3)
        self.register_buffer("pos_scale",  pos_scale,  persistent=False)
        self.register_buffer("pos_offset", pos_offset, persistent=False)
        # Octave Fourier bands: scalars span metres/seconds (~O(10)), so bands
        # 2^-3..2^2 cover coarse-to-fine structure.
        self.register_buffer("bands", 2.0 ** torch.arange(-3, -3 + num_bands), persistent=False)
        self._num_scalars = 10
        in_dim = self._num_scalars * (2 * num_bands + 1)
        # `num_layers` Linear layers with GELU between. Depth 2 builds the
        # exact legacy Sequential (param names mlp.0 / mlp.2), so existing
        # checkpoints strict-load at the default.
        layers = [nn.Linear(in_dim, hidden)]
        for _ in range(num_layers - 2):
            layers += [nn.GELU(approximate="tanh"), nn.Linear(hidden, hidden)]
        layers += [nn.GELU(approximate="tanh"), nn.Linear(hidden, 1)]
        self.mlp = nn.Sequential(*layers)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        poses:        torch.Tensor,   # (B, P, pose_dim) local [x, y, z, pitch, yaw]
        times:        torch.Tensor,   # (B, P) seconds from the pool origin
        action:       torch.Tensor,   # (B, ng, action_dim) goal pose, `_nl`-normalised
        stride:       torch.Tensor,   # (B, ng) goal time in max_frame_offset units
        current_time: torch.Tensor,   # (B,) the current frame's time (seconds)
        agent_state:  torch.Tensor | None = None,   # (B, P, agent_state_dim) per-frame second-agent detection (cue); None = off
        cur_state:    torch.Tensor | None = None,   # (B, 3) the CURRENT frame's own [vi, dx, dy] (query side); zeros when None
    ) -> torch.Tensor:                # (B, ng, P) per-(frame, goal) metadata logits
        poses, times, action, stride = poses.float(), times.float(), action.float(), stride.float()
        g_xyz = action[..., :3] * self.pos_scale + self.pos_offset            # (B, ng, 3)
        g_pitch, g_yaw = action[..., 3], action[..., 4]                       # (B, ng)
        d = g_xyz[:, :, None, :] - poses[:, None, :, :3]                      # (B, ng, P, 3)
        p_pitch, p_yaw = poses[..., 3], poses[..., 4]                         # (B, P)
        cp, sp = torch.cos(p_pitch), torch.sin(p_pitch)
        cy, sy = torch.cos(p_yaw),   torch.sin(p_yaw)
        fwd  = torch.stack([cp * cy, cp * sy, sp], dim=-1)                    # (B, P, 3)
        left = torch.stack([-sy, cy, torch.zeros_like(sy)], dim=-1)
        up   = torch.stack([-sp * cy, -sp * sy, cp], dim=-1)
        # Goal position in each context frame's egocentric coordinates.
        ego = torch.stack([(d * fwd[:, None]).sum(-1),
                           (d * left[:, None]).sum(-1),
                           (d * up[:, None]).sum(-1)], dim=-1)                # (B, ng, P, 3)
        dyaw   = g_yaw[:, :, None]   - p_yaw[:, None, :]                      # (B, ng, P)
        dpitch = g_pitch[:, :, None] - p_pitch[:, None, :]
        dist   = d.norm(dim=-1)
        dt_goal = stride[:, :, None] * self.stride_to_seconds - times[:, None, :]
        dt_cur  = (current_time[:, None] - times)[:, None, :].expand_as(dt_goal)
        x = torch.cat([ego,
                       dist.unsqueeze(-1),
                       torch.sin(dyaw).unsqueeze(-1),  torch.cos(dyaw).unsqueeze(-1),
                       torch.sin(dpitch).unsqueeze(-1), torch.cos(dpitch).unsqueeze(-1),
                       dt_goal.unsqueeze(-1), dt_cur.unsqueeze(-1)], dim=-1)  # (B, ng, P, 10)
        xb = x.unsqueeze(-1) * self.bands                                     # (B, ng, P, 10, bands)
        feats = torch.cat([x,
                           torch.sin(xb).flatten(-2),
                           torch.cos(xb).flatten(-2)], dim=-1)                # (B, ng, P, 10·(2·bands+1))
        # tanh-bounded to [-1, 1]: matches the visual cosine's range, so the
        # fused logit stays within the scale the downstream consumers assume
        # (the generator's CHUNK_SCORE_SCALE, the softmax temperatures, the
        # rank ordering). tanh(0)=0 with unit slope, so the zero-init
        # no-op start and its early gradients are unchanged; sharpness remains
        # expressible pre-tanh and via `softmax_temperature` / `visual_scale`.
        h = self.mlp[0](feats)                                                # (B, ng, P, hidden)
        if agent_state is not None and self.agent_proj is not None:
            # Agent-state cue: the frame's detection + the pair displacement
            # rotated into the CURRENT frame (the frame's yaw is relative to it),
            # zero unless both paired observations are in view — the predictor's
            # feature. Frame-side only (no goal term), broadcast over goals.
            st = agent_state.float()                                          # (B, P, S)
            both = (st[..., 0] > 0) & (st[..., 3] > 0)
            v = torch.where(both[..., None], st[..., 1:3] - st[..., 4:6], torch.zeros_like(st[..., 1:3]))
            v_cur = torch.stack([cy * v[..., 0] - sy * v[..., 1], sy * v[..., 0] + cy * v[..., 1]], dim=-1)
            cur = (cur_state.float() if cur_state is not None
                   else st.new_zeros(st.shape[0], 3))[:, None, :].expand(-1, st.shape[1], -1)   # (B, P, 3)
            h = h + self.agent_proj(torch.cat([st, v_cur, cur], dim=-1))[:, None]   # (B, 1, P, hidden) -> broadcast over ng
        for layer in self.mlp[1:]:
            h = layer(h)
        return h.squeeze(-1).tanh()                                           # (B, ng, P) in [-1, 1]


class PnPRetrieval(nn.Module):
    """Plug-and-play retrieval that returns the top-`num_sets` context sets.

    Past context (everything but the current frame) is split into chunks of
    `chunk_size` consecutive frames. Each chunk's KEY is `query_vit(middle_frame)`
    (null action → content key). A single query is encoded **per goal** by
    `query_vit(current_frame, action, stride)`. For each `(b, goal)`, chunks
    are ranked by their fused logit and the top `num_sets · num_memories` are
    partitioned in rank order into `num_sets` (=K) sets of `num_memories`
    chunks (set 0 = best). Temporal diversity comes from the chunking itself —
    distinct chunks are ≥ `chunk_size` frames apart. The selected
    chunks are returned as RAW frames (the generator's `ChunkEmbed` compresses
    them); per-frame absolute times/poses are returned alongside.

    Args:
        num_memories:        Set size — chunks per retrieved set.
        chunk_size:          Frames per chunk; MUST match the generator's chunk_size.
        query_vit:           QueryViT producing keys (no action) and queries (action).
        num_sets:            K — number of diverse sets to return per goal.
        scoring:             Ranking recipe for past frames:
                               "visual"      — QueryViT key–query cosine only
                                               (no scorer, no visual_scale
                                               parameter);
                               "visual+meta" — visual_scale · cos + meta_logit,
                                               with a LEARNABLE visual_scale
                                               (init 1; zero-init scorer ⇒
                                               exactly the visual ranking at
                                               step 0). Legacy fused form —
                                               every existing fused checkpoint
                                               strict-loads into it unchanged;
                               "visual+meta_z"
                                             — each channel standardized per
                                               query (masked mean/std over the
                                               pool's valid frames), then the
                                               bounded mix (1−λ~)·z_meta +
                                               λ~·z_visual. λ comes from ONE
                                               shared linear gate scoring each
                                               channel from the same DETACHED
                                               feature recipe — [raw top1,
                                               mean, std], a fixed one-hot
                                               source tag, and Δt-to-goal
                                               through per-channel (dt·tag)
                                               slots — with
                                               λ = softmax over the channel
                                               logits (= the visual weight),
                                               then squashed to
                                               λ~ = ε + (1−2ε)λ (ε = gate_eps,
                                               default 0.05) so both channels
                                               always keep ≥ ε weight. The
                                               zero-init gate ⇒ λ~ = 0.5
                                               everywhere at step 0; the
                                               λ-limits are ε-mixtures of the
                                               pure channel rankings;
                               "abl+visual+meta_z"
                                             — ABLATION of the learned gate:
                                               the same per-query standardized
                                               channels, fused with a FIXED
                                               neutral weight λ~ ≡ 0.5 (the
                                               gate's zero-init point). No
                                               fusion parameters exist in this
                                               mode; its gap to
                                               "visual+meta_z" isolates the
                                               LEARNED confidence weighting
                                               with standardization held
                                               fixed;
                               "meta"        — the metadata cross-scorer logit
                                               ALONE. No visual term in any
                                               ranking site and no fusion
                                               parameter — the trained
                                               metadata-only arm. (The visual
                                               encodings are still computed for
                                               telemetry/API parity; they just
                                               never enter a ranking.)
                             Each mode creates ONLY its own fusion parameters,
                             so checkpoints are mode-exact and strict-load.
        encoder_mode:        What the retriever loss may train in the encoder:
                               "frozen"   — QueryViT frozen, no adapter;
                               "adapter"  — QueryViT frozen + zero-init residual
                                            QUERY adapter (width adapter_hidden;
                                            identity at step 0);
                               "finetune" — full QueryViT trainable.
    """

    def __init__(
        self,
        chunk_size: int,
        query_vit: nn.Module,
        num_sets: int = 4,
        # Temperature τ for the per-set softmax `softmax(set_score / τ)` that
        # forms the retriever prior p_η over the K sets (and feeds the EM
        # posterior). Set scores are sums of `num_memories` cosine sims, so
        # they live in `[-num_memories, num_memories]`; τ=1.0 over that narrow
        # range gives a near-uniform prior (the EM retriever loss then sits at
        # ~log K with little gradient). Lower τ (e.g. 0.1) sharpens the prior
        # so it can express a real per-set preference.
        softmax_temperature: float = 1.0,
        scoring: str = "visual",
        metadata_hidden: int = 128,
        metadata_layers: int = 2,
        local_pos_stats: dict | None = None,
        stride_to_seconds: float = 1.0,
        encoder_mode: str = "frozen",
        adapter_hidden: int = 384,
        gate_eps: float = 0.05,
        query_source: str = "frame",
        query_conditioning: str = "action",
        metadata_agent_state: bool = False,
    ):
        super().__init__()
        assert scoring in ("visual", "visual+meta", "visual+meta_z",
                           "abl+visual+meta_z", "meta"), scoring
        # AGENT-STATE CUE for the metadata scorer (dynamic-agent corpus):
        # the dataset's per-frame second-agent detections (`context_agent_state`)
        # enter the cross-scorer through a zero-init block, so the retriever can
        # rank frames by "an agent was seen here, heading that way". Needs a
        # meta scoring mode; ignored under "visual" (no scorer).
        self.metadata_agent_state = bool(metadata_agent_state)
        assert encoder_mode in ("frozen", "adapter", "finetune"), encoder_mode
        # "action" (default): queries are encoded WITH (action, stride) — the
        # action-conditioned "what will I see there" query all pnp arms use.
        # "none": queries are CONTENT embeddings (null action/stride — the
        # same encode that produces keys): pure current-frame similarity
        # retrieval, the LongLive-RAG baseline semantics ("embedding of the
        # most recent completed latent", Alg. 1). No parameters depend on the
        # choice, so any checkpoint loads under either setting.
        assert query_conditioning in ("action", "none"), query_conditioning
        # "frame" (default): the query encoder reads the CURRENT latent frame.
        # "cue": it reads a per-step retrieval-cue window supplied at call time
        # (`query_cue`, e.g. the current binaural audio window) — the memory
        # never knows the modality. Cue mode requires CACHED pool keys
        # (context_keys): the pool side is never live-encoded from cues.
        assert query_source in ("frame", "cue"), query_source
        self.query_conditioning  = query_conditioning
        self.chunk_size          = chunk_size
        self.query_vit           = query_vit
        self.num_sets            = num_sets
        self.softmax_temperature = softmax_temperature
        self.scoring             = scoring
        self.query_source        = query_source

        # --- Encoder training mode (`encoder_mode`) ---------------------------
        # "adapter": zero-init residual MLP on the QueryViT output — adapts the
        # action-conditioned QUERY only. KEYS are used as-is (cached .keys.npy
        # or live-encoded); the key space stays the frozen pretrained one.
        # Zero-init ⇒ identity at step 0. "frozen"/"adapter" freeze the shared
        # encoder and run it under no_grad in `forward` (retains no
        # activations); "finetune" leaves the full QueryViT trainable.
        key_dim = query_vit.key_head.out_features
        self.query_proj = self._build_adapter(
            key_dim, adapter_hidden if encoder_mode == "adapter" else None)
        self._freeze_encoder = encoder_mode in ("frozen", "adapter")
        if self._freeze_encoder:
            for p in self.query_vit.parameters():
                p.requires_grad = False

        # --- Metadata cross-scorer (optional) ---------------------------------
        # Per-(frame, goal) pose/time logit fused with the visual cosine per
        # the fused `scoring` mode (class docstring). The fused logit drives
        # EVERYTHING downstream (chunk-max, rank-partition, set softmax / EM
        # prior, representative-frame choice, local-RAG frame prior) so global
        # and local RAG train against one ranking. Each mode creates ONLY its
        # own fusion parameters — "visual+meta" carries exactly the legacy
        # visual_scale, so every existing fused checkpoint strict-loads, and
        # "visual"/"meta" checkpoints carry no dead fusion parameter.
        if scoring != "visual":
            self.meta_scorer = MetadataCrossScorer(
                hidden=metadata_hidden,
                num_layers=metadata_layers,
                local_pos_stats=local_pos_stats,
                stride_to_seconds=stride_to_seconds,
                agent_state_dim=6 if self.metadata_agent_state else 0,
            )
        else:
            self.meta_scorer = None
        if scoring == "visual+meta":
            self.visual_scale = nn.Parameter(torch.tensor(1.0))
        elif scoring == "visual+meta_z":
            # ONE shared linear gate scores every channel from the same
            # feature recipe — [top1, mean, std, Δt-to-goal] + a FIXED one-hot
            # source tag (visual=[1,0], meta=[0,1]; the tag is what lets the
            # shared gate learn channel-specific trust) — one logit per
            # channel, λ = softmax over the logits. Zero-init ⇒ uniform
            # weights (λ = 0.5) at step 0. `gate_eps` keeps the squashed
            # λ~ = ε + (1−2ε)λ inside [ε, 1−ε]: the gate can lean, never
            # collapse a channel entirely.
            # Inputs (7): [top1, mean, std] of the channel's own score field,
            # the one-hot source tag (2), and Δt-to-goal × tag (2). Δt gets
            # PER-CHANNEL slots because any feature shared verbatim by all
            # channels cancels in the softmax's logit difference — the same
            # reason bias=False: a shared bias shifts every logit equally
            # (dead parameter); per-channel offsets live in the tag weights.
            self.gate = nn.Linear(7, 1, bias=False)
            nn.init.zeros_(self.gate.weight)
            self.gate_eps = float(gate_eps)
        # "abl+visual+meta_z" deliberately creates NO fusion parameters:
        # λ~ ≡ 0.5 over the same standardized channels (see forward) — the
        # trained-under-fixed-rule ablation of the learned gate.

    @staticmethod
    def _build_adapter(key_dim: int, hidden: int | None):
        """Zero-init residual MLP (key_dim→hidden→key_dim); identity at init. None → disabled."""
        if hidden is None:
            return None
        adapter = nn.Sequential(
            nn.Linear(key_dim, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, key_dim),
        )
        nn.init.zeros_(adapter[-1].weight)
        nn.init.zeros_(adapter[-1].bias)
        return adapter

    def _encode(self, *args):
        """QueryViT encode, wrapped in no_grad when the encoder is frozen (adapter
        mode) so it retains no activations. Adapters are applied by the caller."""
        if self._freeze_encoder:
            with torch.no_grad():
                return self.query_vit(*args)
        return self.query_vit(*args)

    @property
    def uses_agent_state(self) -> bool:
        """True when the metadata scorer carries the agent-state cue block —
        `LDWMCore.prepare_context` forwards `context_agent_state` only then."""
        return self.meta_scorer is not None and getattr(self.meta_scorer, "agent_proj", None) is not None

    def load_weights(self, ckpt_path):
        """Load retriever (QueryViT) weights from a checkpoint.

        Invoked by `instantiate_with_optional_ckpt` (see
        `far.builders.models.build_ldwm`) ONLY when the config supplies a
        non-null `ckpt_path`; a null path leaves the random init untouched.

        Accepts either a retriever pre-training checkpoint — a dict carrying a
        `"memory"` entry, as saved by `scripts/world/train_latent_retriever.py`
        — or a bare state_dict. Leading `module.` / `_orig_mod.` prefixes
        (DDP / torch.compile) are stripped so the load works regardless of how
        the checkpoint was produced.

        `strict=False`: a pre-training checkpoint has no `query_vit.query_proj.*`
        keys (the query adapter is added only for diffusion fine-tuning), so those
        are EXPECTED to be missing and the adapter keeps its zero-init. Any other
        missing/unexpected key is a real mismatch and is logged."""
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state = ckpt["memory"] if isinstance(ckpt, dict) and "memory" in ckpt else ckpt
        cleaned = {}
        for k, v in state.items():
            for pfx in ('_orig_mod.', 'module.'):
                if k.startswith(pfx):
                    k = k[len(pfx):]
            # Schema migration: adapters used to live inside QueryViT
            # (`query_vit.query_proj.*`); they are memory-level (`query_proj.*`)
            # now. Rename so old checkpoints' TRAINED adapters still load
            # (dropping one silently strips e.g. an adapter-era RAG fine-tune).
            if k.startswith('query_vit.query_proj.'):
                k = 'query_proj.' + k[len('query_vit.query_proj.'):]
            cleaned[k] = v
        incompat = self.load_state_dict(cleaned, strict=False)
        # Adapter / metadata-scorer params are fresh modules, never present in
        # retriever pre-training checkpoints — expected-missing, not warned.
        missing = [k for k in incompat.missing_keys
                   if not any(t in k for t in ('query_proj', 'meta_scorer', 'visual_scale',
                                               'gate'))]
        if missing or incompat.unexpected_keys:
            print(f"[PnPRetrieval.load_weights] unexpected={incompat.unexpected_keys} "
                  f"missing(non-adapter)={missing}")

    # Run the whole memory forward in eager. Its attention/projection blocks use
    # fp32 `autocast(enabled=False)` islands (e.g. `key_head`) that torch.compile
    # /inductor miscompiles — it drops the `.float()` cast and feeds a bf16
    # activation into the fp32 weight, raising "mat1 and mat2 must have the same
    # dtype, but got BFloat16 and Float". Keeping the memory out of dynamo avoids
    # the whole class of gather/reshape graph problems; the heavy generator stays
    # compiled.
    @torch._dynamo.disable
    def forward(
        self,
        context_frames: torch.Tensor,                # (B, T, C, H, W)         last index = current frame
        context_poses:  torch.Tensor,                # (B, T, pose_dim)        last index = current pose
        context_times:  torch.Tensor,                # (B, T) seconds          last index = current time
        action:         torch.Tensor,                # (B, num_goals, action_dim)
        stride:         torch.Tensor,                # (B, num_goals)
        context_mask:   torch.Tensor | None = None,  # (B, T) bool, False = padding
        context_keys:   torch.Tensor | None = None,  # (B, T, key_dim) pre-computed content keys → cached retrieval/scoring; None → encode
        query_cue:      torch.Tensor | None = None,  # (B, ...) current step's retrieval-cue window (query_source="cue")
        action_interact:  torch.Tensor | None = None,  # (B, num_goals, interact_dim) multi-hot of the interaction types in (current, goal] (THOR); QUERY side only — keys stay pure content
        context_agent_state: torch.Tensor | None = None,  # (B, T, 6) per-frame second-agent detections — the metadata scorer's cue (`metadata_agent_state`); ignored otherwise
        num_memories:   int = None,                  # frames per returned set (set size); supplied per-call by LDWMCore.prepare_context
        return_local:   bool = False,                # also return a LocalCandidates bundle (within-chunk frames + grad per-frame prior) for local RAG
    ) -> tuple:
        """Encode chunks once, then return the top-`num_sets` diverse context
        sets per (sample, goal).

        Returns the 5-tuple LDWMCore consumes. Each selected chunk is collapsed
        to ONE representative frame — its MAX key–query cosine frame — so the
        module returns `num_memories` frames per set (a per-frame retriever),
        consumed by the per-frame generators as `context_size = num_memories + 1`
        plain frames. `chunk_size` is kept purely as a temporal-diversity guard
        (selection is across chunks, so it can't pick many adjacent frames).
            selected_frames:       (B, num_goals, K, num_memories, C, H, W)
                the representative (max-scoring) frame of each selected chunk,
                for each of the K retrieved sets.
            selected_poses:        (B, num_goals, K, num_memories, pose_dim)
                poses of the representative frames.
            selected_times:        (B, num_goals, K, num_memories)
                ABSOLUTE times of the representative frames.
            selected_scores:       (B, num_goals, K) softmax over the K diverse-set
                scores (sums to 1 along K) — feeds LDWMCore's RAG soft-min.
            selected_chunk_scores: (B, num_goals, K, num_memories) the key–query
                cosine of each returned frame (its chunk's max score).
        """
        # Run the entire memory forward in fp32 — bf16's 8-bit mantissa can
        # accumulate enough quantisation error across QueryViT's residual stream
        # over many training steps to push keys/queries (and hence the softmax of
        # `rel - λ·div`) into pathological regimes that eventually NaN the
        # diffusion loss. The fp32 wrap also makes every `with
        # autocast(enabled=False)` island below redundant but harmless.
        with torch.amp.autocast(device_type='cuda', enabled=False):
            B, num_goals = action.shape[:2]
            M_set = num_memories               # chunks per set (set_size); per-call (rag_step → 1)
            Ksets = self.num_sets              # number of sets to return (K)
            S = self.chunk_size

            # ── Past context (exclude the current frame at the last index) ───
            past_frames = context_frames[:, :-1]                                # (B, T-1, C, H, W)
            past_poses  = context_poses [:, :-1]                                # (B, T-1, pose_dim)
            past_times  = context_times [:, :-1]                                # (B, T-1)
            T_past = past_frames.shape[1]

            # Drop leading (oldest) frames so T_past is a multiple of chunk_size.
            N    = T_past // S
            drop = T_past - N * S
            past_frames = past_frames[:, drop:]
            past_poses  = past_poses [:, drop:]
            past_times  = past_times [:, drop:]

            # Reshape into chunks: (B, N, S, ...).
            past_frames = past_frames.reshape(B, N, S, *past_frames.shape[2:])
            past_poses  = past_poses .reshape(B, N, S, past_poses.shape[-1])
            past_times  = past_times .reshape(B, N, S)
            # Agent-state cue for the metadata scorer: sliced exactly like the
            # poses (past frames, oldest `drop` removed, chunked); None when the
            # dataset serves none or the scorer was built without the block.
            past_state = None
            if context_agent_state is not None and self.uses_agent_state:
                past_state = context_agent_state[:, :-1][:, drop:].reshape(B, N, S, -1)

            # Content KEY (null action) of EVERY past frame, (B, N, S, key_dim):
            # gather cached when available, else encode all past frames once. The
            # flattened view `flat_pool_keys` (B, N*S, key_dim) is reused by the
            # per-frame FramePack scoring below.
            if context_keys is not None:
                pk = context_keys[:, :-1][:, drop:]                                # (B, N*S, key_dim)
            else:
                pk = self._encode(
                    past_frames.reshape(B * N * S, *past_frames.shape[3:])
                ).reshape(B, N * S, -1)                                            # (B, N*S, key_dim)
            past_keys      = pk.reshape(B, N, S, pk.shape[-1])                      # (B, N, S, key_dim)
            flat_pool_keys = pk

            # Per-chunk validity from context_mask: a chunk is usable if it holds at
            # least one real (non-padding) frame, so all-padding chunks (the sampler's
            # left-padding for short clips) are never selected. Fallback: a sample
            # with fewer than `M_set` valid chunks (so a full set can't be drawn from
            # valid chunks alone) reverts to allowing all chunks, avoiding -inf sets.
            if context_mask is not None:
                past_mask   = context_mask[:, :-1][:, drop:].reshape(B, N, S)
                valid_chunk = past_mask.any(dim=-1)                                # (B, N) bool
                valid_chunk = valid_chunk | (valid_chunk.sum(dim=1, keepdim=True) < M_set)
            else:
                valid_chunk = torch.ones(B, N, dtype=torch.bool, device=context_frames.device)

            # ── One query per (B, goal): same current frame, different action/stride ──
            if self.query_source == "cue":
                assert query_cue is not None, "query_source='cue' requires a query_cue input"
                assert context_keys is not None, \
                    "cue mode requires cached context_keys (pool cues are never live-encoded)"
                current_frame = query_cue                                        # (B, ...cue shape)
            else:
                current_frame = context_frames[:, -1]                            # (B, C, H, W)
            current_frame_g = current_frame.unsqueeze(1).expand(B, num_goals, *current_frame.shape[1:])
            current_frame_g = current_frame_g.reshape(B * num_goals, *current_frame.shape[1:])
            action_g = action.reshape(B * num_goals, -1)                          # (B*num_goals, action_dim)
            stride_g = stride.reshape(B * num_goals)                              # (B*num_goals,)
            # Interaction labels condition the QUERY only ("I am about to open
            # something"); keys stay pure content so they remain cacheable and
            # mutually comparable. Ignored when the encoder was built without
            # the pathway (interact_dim=0) or in content-query mode.
            interact_g = None
            if action_interact is not None and getattr(
                    self.query_vit, "interact_dim", 0) > 0:
                interact_g = action_interact.reshape(B * num_goals, -1)
            if self.query_conditioning == "none":
                # Content query (LongLive baseline): same encode as the keys.
                queries = self._encode(current_frame_g)                           # (B*num_goals, key_dim)
            else:
                queries = self._encode(current_frame_g, action_g, stride_g, interact_g)
            if self.query_proj is not None:                                       # zero-init residual query adapter
                queries = queries + self.query_proj(queries.float())

            # ── Top-K·M chunk retrieval (fp32) ──────────────────────────────────
            # Score each chunk by the MAXIMUM fused per-frame logit over its S
            # frames. The best frame is query-dependent, so the per-chunk score
            # is per goal.
            BG  = B * num_goals
            q   = F.normalize(queries.float(), dim=-1)                            # (BG, key_dim)
            pkn = F.normalize(past_keys.float(), dim=-1)                          # (B, N, S, key_dim)
            frame_sims = torch.einsum('bgd,bnsd->bgns',
                                      q.reshape(B, num_goals, -1), pkn)           # (B, ng, N, S)
            # Per-frame ranking logit per the `scoring` recipe: plain cosine
            # ("visual"), the metadata logit alone ("meta"), the legacy
            # visual_scale·cos + meta ("visual+meta"), or per-query STANDARDIZED
            # channels mixed by the bounded metadata-confidence gate
            # ("visual+meta_z"). `fuse_z` closes over the pool statistics and λ
            # so the SAME transform is reused at the rep-frame and local-RAG
            # fusion sites below — every downstream consumer sees one
            # consistent ranking.
            fuse_z = None
            if self.meta_scorer is not None:
                meta = self.meta_scorer(
                    past_poses.reshape(B, N * S, -1), past_times.reshape(B, N * S),
                    action, stride, context_times[:, -1],
                    agent_state=(past_state.reshape(B, N * S, -1) if past_state is not None else None),
                    cur_state=(context_agent_state[:, -1, :3] if past_state is not None else None),
                ).reshape(B, num_goals, N, S)                                     # (B, ng, N, S) grad-carrying
                if self.scoring == "meta":
                    frame_logits = meta
                elif self.scoring == "visual+meta":
                    frame_logits = self.visual_scale * frame_sims + meta
                else:                                                             # "[abl+]visual+meta_z"
                    fmask = (past_mask if context_mask is not None else
                             torch.ones(B, N, S, dtype=torch.bool, device=meta.device))
                    fm  = fmask.unsqueeze(1).float()                              # (B, 1, N, S)
                    cnt = fm.sum((-2, -1)).clamp(min=1.0)                         # (B, 1)

                    def _stats(x):                                                # masked per-query mean/std
                        # Variance is clamped BEFORE the sqrt: the zero-init
                        # meta scorer outputs identically 0 (tanh(0)) at step
                        # 0, so its variance is EXACTLY 0 — and sqrt(0) has an
                        # infinite backward that a post-sqrt clamp cannot mask
                        # (autograd's 0·inf = NaN poisons the scorer's grads).
                        # The 1e-6 variance floor (sd ≥ 1e-3) also bounds the
                        # near-degenerate 1/sd gradient at 1e3, so the meta
                        # channel RAMPS IN as its spread grows past the floor
                        # instead of detonating Adam's second moment at step 1;
                        # trained spreads (meta tanh ~O(0.1-1), cos ~O(0.05))
                        # sit far above it, where the floor is inactive.
                        mu = (x * fm).sum((-2, -1)) / cnt                         # (B, ng)
                        sd = ((x - mu[..., None, None]).pow(2) * fm).sum((-2, -1)) \
                            .div(cnt).clamp(min=1e-6).sqrt()
                        return mu, sd

                    mu_v, sd_v = _stats(frame_sims)
                    mu_m, sd_m = _stats(meta)
                    # SYMMETRIC shared gate: every channel is described by the
                    # SAME feature recipe over its raw score field — [top1,
                    # mean, std, Δt-to-goal] — plus a FIXED one-hot source tag
                    # (visual=[1,0], meta=[0,1]) that lets the single linear
                    # gate learn channel-specific trust; one logit per channel,
                    # λ = softmax over the two logits (= the visual weight).
                    # Features are DETACHED: λ's gradient trains the gate
                    # alone — neither scorer can inflate its own confidence to
                    # steer the mix (the visual_scale confidence-for-competence
                    # failure, closed off). λ is then squashed to
                    # λ~ = ε + (1−2ε)·λ (ε = gate_eps), so BOTH channels
                    # always keep ≥ ε weight — the gate can lean, never
                    # collapse (the exact-nesting λ-limits become ε-mixtures).
                    dt_goal = (stride * self.meta_scorer.stride_to_seconds
                               - context_times[:, -1:]).detach() \
                        .to(frame_sims.dtype)                                     # (B, ng) seconds

                    def _gate_logit(field, mu, sd, tag):
                        top1 = field.masked_fill(~fmask.unsqueeze(1),
                                                 float("-inf")).amax((-2, -1))
                        f = torch.stack([top1.detach(), mu.detach(), sd.detach()],
                                        dim=-1)                                   # (B, ng, 3)
                        onehot = f.new_zeros(*f.shape[:-1], 2)
                        onehot[..., tag] = 1.0
                        # Δt-to-goal enters through PER-CHANNEL slots (dt·tag):
                        # a shared dt slot would cancel in u_v − u_m.
                        return self.gate(torch.cat(
                            [f, onehot, dt_goal[..., None] * onehot], -1)).squeeze(-1)

                    if self.scoring == "abl+visual+meta_z":
                        # ABLATION — fixed NEUTRAL fusion: λ~ ≡ 0.5, the
                        # learned gate's zero-init point. Equal-weight sum of
                        # the standardized channels; `_gate_logit` above is
                        # never called (no `gate` exists in this mode).
                        lam = torch.full_like(mu_v, 0.5)
                    else:
                        u_v = _gate_logit(frame_sims, mu_v, sd_v, 0)
                        u_m = _gate_logit(meta, mu_m, sd_m, 1)
                        lam = torch.sigmoid(u_v - u_m)                            # 2-way softmax
                        lam = self.gate_eps + (1.0 - 2.0 * self.gate_eps) * lam   # λ~ ∈ [ε, 1−ε]
                    # Telemetry snapshot (plain attribute — never enters the
                    # state dict): the per-query effective λ~ of the LAST
                    # forward, read by train_latent's log block as train/lambda_*.
                    self.last_lambda = lam.detach()

                    def fuse_z(cos, met):
                        shp = (B, num_goals) + (1,) * (cos.dim() - 2)
                        zv = (cos - mu_v.reshape(shp)) / sd_v.reshape(shp)
                        zm = (met - mu_m.reshape(shp)) / sd_m.reshape(shp)
                        l  = lam.reshape(shp)
                        return (1 - l) * zm + l * zv

                    frame_logits = fuse_z(frame_sims, meta)
            else:
                meta = None
                frame_logits = frame_sims
            chunk_logits = frame_logits.max(dim=-1).values                        # (B, ng, N) fused chunk relevance
            # ── Rank-partition set selection (no beam, no pairwise diversity) ──
            # Chunks are ranked by fused relevance (invalid chunks → -inf) and
            # the top K·M are partitioned IN RANK ORDER: set 0 = the best M
            # chunks, set 1 = the next M, ... Temporal diversity is enforced by
            # the chunking itself (distinct chunks are ≥ chunk_size frames
            # apart). Pools with fewer than K·M chunks wrap around the ranking
            # (duplicate chunks across — never within — sets).
            valid_bg = valid_chunk.unsqueeze(1).expand(B, num_goals, N).reshape(BG, N)
            rel_bg   = chunk_logits.reshape(BG, N).masked_fill(~valid_bg, float("-inf"))
            order    = rel_bg.argsort(dim=-1, descending=True)                    # (BG, N)
            take     = torch.arange(Ksets * M_set, device=order.device) % N       # wraps when K·M > N
            final_sets = order[:, take].reshape(B, num_goals, Ksets, M_set)       # chunk indices

            # ── Each set's score and the softmax over the K sets — computed
            # from the actually-returned chunks; differentiable in q / keys
            # (and the metadata scorer).
            fs_bg  = final_sets.reshape(BG, Ksets, M_set)
            rel_per_chunk = chunk_logits.reshape(BG, N).gather(
                1, fs_bg.reshape(BG, Ksets * M_set)).reshape(BG, Ksets, M_set)     # (BG, K, M) fused chunk logits
            rel    = rel_per_chunk.sum(dim=-1)                                    # (BG, K) set-level relevance
            set_score  = rel                                                      # (BG, K)
            # Temperature-scaled softmax (τ = `softmax_temperature`): τ<1 sharpens
            # the prior so the K best sets actually rank; over set scores in
            # [-M, M], τ=1 is nearly uniform and the per-set loss weighting / EM
            # posterior carry almost no signal.
            sel_scores = (set_score / self.softmax_temperature).softmax(dim=-1)   # (BG, K)
            sel_scores       = sel_scores.reshape(B, num_goals, Ksets)            # (B, num_goals, K)
            # ── Gather the selected chunks (keep the S-frame axis), then collapse
            # each chunk to its single MAX key–query cosine frame. `chunk_size`
            # still groups temporally-adjacent frames so the selection can't pick
            # many neighbours (a temporal-diversity guard); returning one frame per
            # chunk makes pnp_cl a per-frame retriever (num_memories frames/set),
            # so the per-frame generators (cdit_mod_score_{xa,fa}) consume it as
            # `context_size = num_memories + 1` plain frames.
            bidx = torch.arange(B, device=final_sets.device)[:, None, None, None] # (B,1,1,1)
            sel_chunks = past_frames[bidx, final_sets]                            # (B, ng, K, M, S, C, H, W)
            sel_p      = past_poses [bidx, final_sets]                            # (B, ng, K, M, S, pose_dim)
            sel_t      = past_times [bidx, final_sets]                            # (B, ng, K, M, S)

            # Per-frame relevance (QueryViT key–query cosine) for every frame of
            # the selected chunks; argmax over the S frames picks each chunk's
            # representative. Under no_grad — the score only RANKS frames (argmax is
            # non-differentiable); the retriever trains via `sel_scores`, whose set
            # relevance already uses these same per-chunk max-frame keys (`rel`).
            with torch.no_grad():
                if flat_pool_keys is not None:
                    # chunk c's S frames live at flat indices [c*S, c*S+S).
                    fidx = (final_sets.unsqueeze(-1) * S
                            + torch.arange(S, device=final_sets.device))          # (B, ng, K, M, S)
                    fidx = fidx.reshape(B, num_goals, Ksets * M_set * S)
                    bsel = torch.arange(B, device=final_sets.device)[:, None, None]
                    sel_keys = flat_pool_keys[bsel, fidx].reshape(
                        B, num_goals, Ksets, M_set, S, -1)                         # (B, ng, K, M, S, key_dim) — cached
                else:
                    sel_keys = self.query_vit(
                        sel_chunks.reshape(-1, *sel_chunks.shape[5:])
                    ).reshape(B, num_goals, Ksets, M_set, S, -1)                   # (B, ng, K, M, S, key_dim)
                qn = F.normalize(queries.reshape(B, num_goals, -1), dim=-1)        # (B, ng, key_dim)
                kn = F.normalize(sel_keys, dim=-1)                                 # (B, ng, K, M, S, key_dim)
                frame_scores = torch.einsum('bgd,bgkmsd->bgkms', qn, kn)          # (B, ng, K, M, S)
                if meta is not None:
                    meta_sel = meta.reshape(B, num_goals, N * S).gather(
                        2, fidx).reshape(B, num_goals, Ksets, M_set, S)            # (B, ng, K, M, S)
                    if self.scoring == "meta":
                        frame_scores = meta_sel
                    elif self.scoring == "visual+meta":
                        frame_scores = self.visual_scale * frame_scores + meta_sel
                    else:                                                          # "visual+meta_z"
                        frame_scores = fuse_z(frame_scores, meta_sel)
                rep_score, rep_idx = frame_scores.max(dim=-1)                     # (B, ng, K, M) each

            # Collapse each selected chunk to its representative (max-scoring) frame.
            bi2 = torch.arange(B,         device=rep_idx.device)[:, None, None, None]
            gi2 = torch.arange(num_goals, device=rep_idx.device)[None, :, None, None]
            ki2 = torch.arange(Ksets,     device=rep_idx.device)[None, None, :, None]
            mi2 = torch.arange(M_set,     device=rep_idx.device)[None, None, None, :]
            sel_frames       = sel_chunks[bi2, gi2, ki2, mi2, rep_idx]            # (B, ng, K, M, C, H, W)
            sel_poses        = sel_p     [bi2, gi2, ki2, mi2, rep_idx]            # (B, ng, K, M, pose_dim)
            sel_times        = sel_t     [bi2, gi2, ki2, mi2, rep_idx]            # (B, ng, K, M)
            sel_chunk_scores = rep_score                                         # (B, ng, K, M) one per returned frame

            if not return_local:
                return sel_frames, sel_poses, sel_times, sel_scores, sel_chunk_scores

            # ── Local-RAG bundle (opt-in; pnp_cl only) ──────────────────────────
            # The chunk-max collapse above discards every non-representative frame
            # and trains the per-frame keys only through the argmax. For local
            # (within-chunk) RAG we expose the FULL S-frame chunks plus a
            # GRAD-carrying per-frame prior, so rag_step can rank frames *inside* a
            # chunk by their diffusion reward and distil it into the keys. The
            # per-frame cosine is recomputed OUTSIDE the no_grad ranking block above
            # so its gradient reaches the query/key adapters; the chunk tensors
            # (`sel_chunks`/`sel_p`/`sel_t`) are plain indexed data.
            fidx = (final_sets.unsqueeze(-1) * S
                    + torch.arange(S, device=final_sets.device))                 # (B, ng, K, M, S)
            fidx = fidx.reshape(B, num_goals, Ksets * M_set * S)
            bsel = torch.arange(B, device=final_sets.device)[:, None, None]
            loc_keys = flat_pool_keys[bsel, fidx].reshape(
                B, num_goals, Ksets, M_set, S, -1)                               # (B, ng, K, M, S, key_dim)
            qn_l = F.normalize(queries.reshape(B, num_goals, -1), dim=-1)         # (B, ng, key_dim) — grad via query_proj
            kn_l = F.normalize(loc_keys, dim=-1)
            loc_cos = torch.einsum('bgd,bgkmsd->bgkms', qn_l, kn_l)              # (B, ng, K, M, S) grad-carrying
            if meta is not None:
                meta_loc = meta.reshape(B, num_goals, N * S).gather(
                    2, fidx).reshape(B, num_goals, Ksets, M_set, S)               # grad-carrying (recomputed outside no_grad)
                if self.scoring == "meta":
                    loc_cos = meta_loc
                elif self.scoring == "visual+meta":
                    loc_cos = self.visual_scale * loc_cos + meta_loc
                else:                                                             # "visual+meta_z"
                    loc_cos = fuse_z(loc_cos, meta_loc)
            # Per-frame prior over the S frames of each chunk (same temperature as
            # the global set prior).
            frame_prior = (loc_cos / self.softmax_temperature).softmax(dim=-1)   # (B, ng, K, M, S) — sums to 1 over S
            local = LocalCandidates(chunks=sel_chunks, poses=sel_p, times=sel_t,
                                    frame_prior=frame_prior)
            return sel_frames, sel_poses, sel_times, sel_scores, sel_chunk_scores, local
