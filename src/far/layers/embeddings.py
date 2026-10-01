"""Shared embedding and conditioning primitives.

Used by the generators (`far.generators.cdit_mod_pose_fa`) and the memory ViTs (`far.memories.pnp_cl`). Anything
that was duplicated across those files lives here.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN-zero feature modulation: x · (1 + scale) + shift, with broadcast.

    The (1 + scale) expression is the classic bf16 precision trap — values of
    `scale` smaller than ~0.008 round to 0 against `1.0` in bf16 (only 7 mantissa
    bits), zeroing out the modulation entirely. Compute the modulation in fp32
    and cast the result back to x's dtype to preserve precision while keeping
    the rest of the residual stream in low precision.
    """
    out = x.float() * (1.0 + scale.unsqueeze(1).float()) + shift.unsqueeze(1).float()
    return out.to(x.dtype)


def xavier_basic_init(module: nn.Module) -> None:
    """Xavier-uniform init for every nn.Linear in `module`; biases zeroed."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def build_2d_sincos_pos_embed(embed_dim: int, h: int, w: int) -> np.ndarray:
    """Build a 2D sincos positional embedding for an h x w patch grid.

    Returns:
        (h * w, embed_dim) float32 array.
    """
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D sincos embedding"
    grid_h = np.arange(h, dtype=np.float32)
    grid_w = np.arange(w, dtype=np.float32)
    grid_w, grid_h = np.meshgrid(grid_w, grid_h)  # (h, w) each
    omega = 1.0 / 10000 ** (np.arange(embed_dim // 4, dtype=np.float64) / (embed_dim / 4.0))
    out_h = np.outer(grid_h.reshape(-1), omega)  # (h*w, embed_dim/4)
    out_w = np.outer(grid_w.reshape(-1), omega)  # (h*w, embed_dim/4)
    return np.concatenate(
        [np.sin(out_h), np.cos(out_h), np.sin(out_w), np.cos(out_w)],
        axis=1,
    ).astype(np.float32)


class ScalarEmbedding(nn.Module):
    """Embeds a scalar tensor of arbitrary shape (...) into (..., hidden_size).

    Applies sinusoidal frequency encoding then a 2-layer MLP. Reshapes
    internally so any batch / sequence leading dimensions are preserved.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        shape = t.shape
        half  = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32) / half
        ).to(t.device)
        args  = t.float().reshape(-1)[:, None] * freqs[None]                    # (M, half)
        emb   = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)           # (M, freq_emb_size)
        return self.mlp(emb).reshape(*shape, -1)                                # (..., hidden_size)


class ActionEmbedding(nn.Module):
    """Embeds a length-`action_dim` action vector into a hidden_size representation.

    Each dimension gets its own `ScalarEmbedding`; the per-dim outputs are
    concatenated along the last axis.
    """

    def __init__(self, hidden_size: int, action_dim: int = 3, frequency_embedding_size: int = 256):
        super().__init__()
        hsize = hidden_size // action_dim
        sizes = [hsize] * (action_dim - 1) + [hidden_size - hsize * (action_dim - 1)]
        self.dim_embs = nn.ModuleList([
            ScalarEmbedding(s, frequency_embedding_size) for s in sizes
        ])

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        assert action.shape[-1] == len(self.dim_embs), (
            f"ActionEmbedding expected {len(self.dim_embs)}-dim action, got {action.shape[-1]}"
        )
        return torch.cat([emb(action[..., i]) for i, emb in enumerate(self.dim_embs)], dim=-1)


class TubeletEmbed(nn.Module):
    """3D patch embedding for video — Conv3d with stride == kernel == tubelet_size."""

    def __init__(self, in_chans: int = 3, embed_dim: int = 768, tubelet_size=(2, 16, 16)):
        super().__init__()
        self.tubelet_size = tubelet_size
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=tubelet_size,
            stride=tubelet_size,
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        # video: [B, T, C, H, W] → [B, N, D]
        video = video.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        x = self.proj(video)                  # [B, D, T', H', W']
        x = x.flatten(2).transpose(1, 2)      # [B, N, D]
        return x


class PoseEmbedding(nn.Module):
    """Projects a pose vector (..., pose_dim) → (..., hidden_size) via a 2-layer MLP."""

    def __init__(self, pose_dim: int, hidden_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(pose_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        return self.mlp(pose)


class PluckerEmbedding(nn.Module):
    """Per-token Plücker-ray embedding for a camera at a given local pose.

    Precomputes the token grid's camera-frame ray directions from a fixed
    horizontal/vertical FOV (pinhole; token centers). `forward(pose)` rotates
    them into the shared reference frame using the pose's pitch/yaw — the same
    viewing-direction convention as `far.data.transforms.global_to_local_pose`
    (`d = [cos(p)cos(y), cos(p)sin(y), sin(p)]`, camera basis forward/left/up)
    — and pairs each direction with the ray moment `o × d` (`o` = camera
    position in the shared frame). The 6-dim Plücker coordinates are projected
    to `hidden_size` by a 2-layer MLP.

    Pose columns: `[x, y, z, pitch, yaw]` (angles in radians, positions in the
    shared frame's units). A zero pose (the CURRENT frame under the formatter's
    local convention) yields the canonical camera rays with zero moment.
    """

    def __init__(self, grid_h: int, grid_w: int, fov_half_h: float, fov_half_v: float, hidden_size: int):
        super().__init__()
        # Token-center image-plane coordinates in [-1, 1]; u → right, v → down.
        v = (torch.arange(grid_h, dtype=torch.float32) + 0.5) / grid_h * 2 - 1
        u = (torch.arange(grid_w, dtype=torch.float32) + 0.5) / grid_w * 2 - 1
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        th = math.tan(math.radians(fov_half_h))
        tv = math.tan(math.radians(fov_half_v))
        # Camera frame (matches global_to_local_pose): x forward, y left, z up.
        # Image-right = -left, image-down = -up.
        dirs = torch.stack([torch.ones_like(uu), -uu * th, -vv * tv], dim=-1)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
        self.register_buffer("cam_dirs", dirs.reshape(-1, 3), persistent=False)  # (P, 3)
        self.mlp = nn.Sequential(
            nn.Linear(6, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        """(..., 5) pose → (..., P, hidden_size) per-token Plücker embedding."""
        cp, sp = torch.cos(pose[..., 3]), torch.sin(pose[..., 3])
        cy, sy = torch.cos(pose[..., 4]), torch.sin(pose[..., 4])
        fwd  = torch.stack([cp * cy, cp * sy, sp], dim=-1)
        left = torch.stack([-sy, cy, torch.zeros_like(sy)], dim=-1)
        up   = torch.stack([-sp * cy, -sp * sy, cp], dim=-1)
        basis = torch.stack([fwd, left, up], dim=-2)                    # (..., 3[basis], 3[xyz])
        d = torch.einsum("pc,...cx->...px", self.cam_dirs, basis)       # (..., P, 3)
        o = pose[..., None, :3].expand_as(d)
        m = torch.cross(o, d, dim=-1)
        return self.mlp(torch.cat([d, m], dim=-1))                      # (..., P, hidden)


class GlobalCond(nn.Module):
    """Fuse a variable number of scalar / vector inputs into one cond vector.

    Each input is independently sinusoidally encoded into `freq_embed_size`
    channels (per scalar dimension), the per-dim encodings are concatenated
    along the last axis, and a 2-layer MLP + final LayerNorm projects to
    hidden_size. The final LayerNorm bounds the magnitude of the conditioning
    vector regardless of the input distribution.

    Args:
        hidden_size:      Output channel dim.
        num_scalars:      Total scalar count across all forward inputs (e.g.
                          for `forward(t, stride, action)` with action_dim=3,
                          num_scalars = 1 + 1 + 3 = 5).
        freq_embed_size:  Number of sinusoidal channels per scalar dimension.
    """

    def __init__(self, hidden_size: int, num_scalars: int, freq_embed_size: int = 256):
        super().__init__()
        self.freq_embed_size = freq_embed_size
        concat_size = num_scalars * freq_embed_size
        self.fusion_mlp = nn.Sequential(
            nn.Linear(concat_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.LayerNorm(hidden_size),
        )

    def _freq_embed(self, t: torch.Tensor) -> torch.Tensor:
        half  = self.freq_embed_size // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32) / half
        ).to(t.device)
        shape = t.shape
        args  = t.float().reshape(-1)[:, None] * freqs[None]
        emb   = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return emb.reshape(*shape, -1)

    def forward(self, *scalars: torch.Tensor) -> torch.Tensor:
        """Each input may be (N,) — a single scalar per batch — or (N, D) — D scalars per batch."""
        embs = []
        for s in scalars:
            e = self._freq_embed(s)             # (N, F) or (N, D, F)
            if e.dim() > 2:
                e = e.flatten(1)                # (N, D * F)
            embs.append(e)
        concat = torch.cat(embs, dim=-1)
        return self.fusion_mlp(concat)


class ChunkEmbed(nn.Module):
    """FramePack-style frame-level chunk compressor (arXiv:2504.12626).

    Compresses a chunk of `chunk_size` consecutive latent frames into a FIXED
    budget of `budget_mult · P_c` tokens, where `P_c` is the padded patch grid
    `(H_pad/patch)·(W_pad/patch)` (= "the number of tokens in one frame").
    Frames are ranked by recency (most-recent first); rank `k` is patchified by
    its OWN strided conv into a geometrically smaller grid, and the last two
    ranks share the smallest grid. The token budget shifts where the geometric
    chain starts:
      - `budget_mult=1` → most-recent frame gets `P_c/2`, then `P_c/4`, … summing
        to `P_c` (one frame's worth of tokens).
      - `budget_mult=2` → most-recent frame keeps the FULL grid `P_c`
        (uncompressed), then `P_c/2`, `P_c/4`, … summing to `2·P_c`.
    Each rank's tokens carry a (downsampled) 2D-sincos positional embedding for
    its own grid plus the frame's absolute-time embedding; there is NO relative-
    time term. The output tokens are ready to drop straight into a DiT context
    stream.

    The latent frame (`in_height × in_width`) is zero-padded (bottom/right) up to
    the next power of two per axis (e.g. 18×32 → 32×32) so the patch grid
    (`H_pad/patch × W_pad/patch`, e.g. 8×8 = 64) halves cleanly via strided
    convs; this caps `chunk_size` at `1 + log2(P_c)` for `budget_mult=1`
    (e.g. 7 for `P_c=64`) and one more for `budget_mult=2` (e.g. 8).
    """

    def __init__(
        self,
        in_height: int = 18,
        in_width: int = 32,
        patch_size: int = 4,
        in_channels: int = 16,
        embed_dim: int = 384,
        chunk_size: int = 7,
        budget_mult: int = 2,
        pose_dim: int = 5,
    ):
        super().__init__()
        assert chunk_size >= 2, f"chunk_size must be >= 2, got {chunk_size}"
        H_pad = 1 << (in_height - 1).bit_length()
        W_pad = 1 << (in_width  - 1).bit_length()
        assert H_pad % patch_size == 0 and W_pad % patch_size == 0, (
            f"padded grid {H_pad}x{W_pad} not divisible by patch_size {patch_size}"
        )
        self.in_height, self.in_width = in_height, in_width
        self.H_pad, self.W_pad = H_pad, W_pad
        self.pad_h, self.pad_w = H_pad - in_height, W_pad - in_width
        self.patch_size = patch_size
        self.chunk_size = chunk_size
        self.budget_mult = budget_mult

        H_grid, W_grid = H_pad // patch_size, W_pad // patch_size
        self.P_c = H_grid * W_grid                                # tokens in one (padded) frame
        grids = self._make_grids(H_grid, W_grid, chunk_size, budget_mult)
        self.num_tokens = sum(h * w for h, w in grids)           # = budget_mult * P_c
        assert self.num_tokens == budget_mult * self.P_c, (grids, self.P_c, budget_mult)
        self.grids = grids

        # One strided conv per rank — kernel == stride downsamples the padded
        # 32×32 frame to that rank's (h_k, w_k) grid, so the n_k tokens preserve
        # spatial locality (standard ViT patchifier). Per-rank 2D-sincos
        # positional embeddings, concatenated in rank order so a single add
        # aligns each rank's positions with its tokens.
        self.convs = nn.ModuleList()
        pos_list = []
        for (h, w) in grids:
            self.convs.append(nn.Conv2d(
                in_channels, embed_dim,
                kernel_size=(H_pad // h, W_pad // w),
                stride=(H_pad // h, W_pad // w),
            ))
            pos_list.append(torch.from_numpy(build_2d_sincos_pos_embed(embed_dim, h, w)))
        self.register_buffer('pos_embed', torch.cat(pos_list, dim=0).float())   # (num_tokens, D)

        self.abs_time_emb = ScalarEmbedding(embed_dim)
        # Per-frame pose embedding — used only when `chunk_poses` is supplied to
        # forward. With `chunk_poses=None` it is never called, so the output is
        # byte-identical to the time-only path.
        self.pose_emb = PoseEmbedding(pose_dim, embed_dim)

    @staticmethod
    def _make_grids(H_grid: int, W_grid: int, S: int, budget_mult: int = 1) -> list:
        """Per-rank (h, w) grids along the full→coarse halving chain of the grid
        (prefer the width axis, then height), last two ranks sharing the smallest
        grid. `budget_mult` selects where the chain starts: 1 → most-recent frame
        at `P_c/2` (counts sum to `P_c`); 2 → most-recent frame at the full grid
        `P_c` (counts sum to `2·P_c`)."""
        assert budget_mult in (1, 2), f"budget_mult must be 1 or 2, got {budget_mult}"
        # Full halving chain from the full grid: token counts [P_c, P_c/2, …, 1].
        chain = [(H_grid, W_grid)]
        h, w = H_grid, W_grid
        while w > 1 or h > 1:
            if w > 1 and w % 2 == 0:
                w //= 2
            elif h > 1 and h % 2 == 0:
                h //= 2
            else:
                break
            chain.append((h, w))
        start = 2 - budget_mult          # mult=1 → start at chain[1] (P_c/2); mult=2 → chain[0] (P_c)
        max_S = len(chain) - start + 1   # +1 for the repeated last rank
        if not (2 <= S <= max_S):
            raise ValueError(
                f"chunk_size={S} out of range for a {H_grid}x{W_grid} grid at "
                f"budget_mult={budget_mult}: valid chunk_size is 2..{max_S}"
            )
        grids = list(chain[start:start + (S - 1)])
        grids.append(grids[-1])          # last two ranks share the smallest grid
        return grids

    def forward(
        self,
        chunk_frames: torch.Tensor,
        chunk_times: torch.Tensor,
        chunk_poses: torch.Tensor | None = None,
        chunk_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """chunk_frames: (N', S, C, in_h, in_w); chunk_times: (N', S);
        chunk_poses: (N', S, pose_dim) or None; chunk_scores: (N', S) per-frame
        relevance or None.
        Returns (N', num_tokens, embed_dim) with num_tokens = budget_mult * P_c."""
        Np, S, C, H, W = chunk_frames.shape
        x = chunk_frames.reshape(Np * S, C, H, W)
        x = F.pad(x, (0, self.pad_w, 0, self.pad_h))            # (Np*S, C, H_pad, W_pad)
        x = x.reshape(Np, S, C, self.H_pad, self.W_pad)

        # Rank frames: by per-frame relevance score when provided (highest =
        # rank 0 = most tokens), else by recency (largest time = rank 0). For a
        # RETRIEVED chunk, recency within the chunk is arbitrary, so a relevance
        # score concentrates the FramePack token budget on the frame that matters.
        rank_key = chunk_scores if chunk_scores is not None else chunk_times
        order = torch.argsort(rank_key, dim=1, descending=True)             # (Np, S)
        order_f = order.view(Np, S, 1, 1, 1).expand(-1, -1, C, self.H_pad, self.W_pad)
        x = torch.gather(x, 1, order_f)                                      # ranked frames
        sorted_times = torch.gather(chunk_times, 1, order)                  # (Np, S)
        with torch.amp.autocast(device_type='cuda', enabled=False):
            te = self.abs_time_emb(sorted_times.float())                    # (Np, S, D)
            # Per-frame pose embedding (same rank order). None → time-only path.
            pe = None
            if chunk_poses is not None:
                order_p = order.unsqueeze(-1).expand(-1, -1, chunk_poses.shape[-1])
                sorted_poses = torch.gather(chunk_poses, 1, order_p)        # (Np, S, pose_dim)
                pe = self.pose_emb(sorted_poses.float())                    # (Np, S, D)

        tok = []
        for k, conv in enumerate(self.convs):
            tk  = conv(x[:, k]).flatten(2).transpose(1, 2)                  # (Np, n_k, D)
            emb = te[:, k] if pe is None else te[:, k] + pe[:, k]           # (Np, D)
            tok.append(tk + emb.unsqueeze(1).to(tk.dtype))                 # + abs time (+ pose)
        tokens = torch.cat(tok, dim=1)                                      # (Np, P_c, D)
        return tokens + self.pos_embed.to(tokens.dtype)                    # + downsampled positional
