"""Attention modules used across generators and memory ViTs.

- `RoPESelfAttention`: SDPA self-attention that rotates Q and K with an axial
  RoPE module before the dot product. The `rope` module owns the head_dim
  split, so the same wrapper covers 2D / 3D positions.
- `RoPECrossAttention`: SDPA cross-attention that rotates Q with the query
  positions and K with the key positions, so the dot product encodes their
  relative offset on each axis.
- `ScoredCrossAttention`: cross-attention with a per-key additive log-weight
  bias on the attention logits, used by the score-aware CDiT variant.

`RoPESelfAttention` and `RoPECrossAttention` both append a learnable
`bias_k`/`bias_v` attention sink to K/V (outside the positional grid).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RoPESelfAttention(nn.Module):
    """Self-attention with axial RoPE applied to Q and K.

    The `rope` module owns the head_dim split and is invoked as
    `rope(x, *positions)` — so this class works for `RoPE2D` or `RoPE3D`
    depending on which rope is supplied.

    Includes a learnable attention sink (`bias_k` / `bias_v`) appended to K and
    V post-projection, matching `nn.MultiheadAttention(add_bias_kv=True)`. The
    sink lives outside the positional grid (no RoPE rotation), giving queries a
    fixed learned position to dump excess attention mass onto — see Darcet et
    al. 2023, "Vision Transformers Need Registers".
    """

    def __init__(self, hidden_size: int, num_heads: int, qkv_bias: bool = True, rope: nn.Module = None):
        super().__init__()
        assert hidden_size % num_heads == 0
        assert rope is not None, "rope module (RoPE2D/RoPE3D) must be provided"
        self.num_heads = num_heads
        self.head_dim  = hidden_size // num_heads
        self.qkv       = nn.Linear(hidden_size, 3 * hidden_size, bias=qkv_bias)
        self.proj      = nn.Linear(hidden_size, hidden_size)
        self.rope      = rope
        # Learnable attention-sink K/V — already in per-head form so they can
        # be concatenated to the rotated Q-K-V tensors without re-reshape.
        self.bias_k = nn.Parameter(torch.empty(1, num_heads, 1, self.head_dim))
        self.bias_v = nn.Parameter(torch.empty(1, num_heads, 1, self.head_dim))
        nn.init.xavier_normal_(self.bias_k)
        nn.init.xavier_normal_(self.bias_v)

    def forward(self, x: torch.Tensor, *positions: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        H, D = self.num_heads, self.head_dim
        qkv = self.qkv(x).reshape(B, L, 3, H, D).permute(2, 0, 3, 1, 4)         # (3, B, H, L, D)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.rope(q, *positions)
        k = self.rope(k, *positions)
        # Append sink K/V after rotation — sits "outside" the positional grid,
        # so it receives no RoPE phase shift regardless of axis count.
        sink_k = self.bias_k.to(k.dtype).expand(B, H, 1, D)
        sink_v = self.bias_v.to(v.dtype).expand(B, H, 1, D)
        k = torch.cat([k, sink_k], dim=2)
        v = torch.cat([v, sink_v], dim=2)
        attn = F.scaled_dot_product_attention(q, k, v)                          # (B, H, L, D)
        attn = attn.transpose(1, 2).reshape(B, L, H * D)
        return self.proj(attn)


class RoPECrossAttention(nn.Module):
    """Cross-attention with axial RoPE on Q and K, plus a learnable sink.

    Q is rotated with `q_positions`, K with `k_positions`; the dot product then
    encodes the relative (t, y, x) offset between query and key tokens. A
    learnable `bias_k`/`bias_v` sink is appended to K/V (outside the grid, no
    rotation), matching `RoPESelfAttention`.
    """

    def __init__(self, hidden_size: int, num_heads: int, qkv_bias: bool = True, rope: nn.Module = None):
        super().__init__()
        assert hidden_size % num_heads == 0
        assert rope is not None, "rope module (RoPE2D/RoPE3D) must be provided"
        self.num_heads = num_heads
        self.head_dim  = hidden_size // num_heads
        self.q_proj    = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.k_proj    = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.v_proj    = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.out_proj  = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.rope      = rope
        self.bias_k = nn.Parameter(torch.empty(1, num_heads, 1, self.head_dim))
        self.bias_v = nn.Parameter(torch.empty(1, num_heads, 1, self.head_dim))
        nn.init.xavier_normal_(self.bias_k)
        nn.init.xavier_normal_(self.bias_v)

    def forward(self, q_tokens, kv_tokens, q_positions, k_positions):
        B, Lq, _ = q_tokens.shape
        Lk = kv_tokens.shape[1]
        H, D = self.num_heads, self.head_dim
        q = self.q_proj(q_tokens ).reshape(B, Lq, H, D).transpose(1, 2)   # (B, H, Lq, D)
        k = self.k_proj(kv_tokens).reshape(B, Lk, H, D).transpose(1, 2)
        v = self.v_proj(kv_tokens).reshape(B, Lk, H, D).transpose(1, 2)
        q = self.rope(q, *q_positions)
        k = self.rope(k, *k_positions)
        # Sink: append a learnable, position-free K/V slot.
        sink_k = self.bias_k.to(k.dtype).expand(B, H, 1, D)
        sink_v = self.bias_v.to(v.dtype).expand(B, H, 1, D)
        k = torch.cat([k, sink_k], dim=2)
        v = torch.cat([v, sink_v], dim=2)
        attn = F.scaled_dot_product_attention(q, k, v)                    # (B, H, Lq, D)
        attn = attn.transpose(1, 2).reshape(B, Lq, H * D)
        return self.out_proj(attn)


class ScoredCrossAttention(nn.Module):
    """Cross-attention with a per-key additive log-weight on the attention logits.

    For nonneg per-context-frame importance weights w_j (summing to 1 across
    context frames), the attention becomes:

        a_ij = softmax_j( q_i^T k_j / sqrt(d) + log w_j )

    which is equivalent to multiplying the unnormalised attention probability
    by w_j. Useful for injecting retrieval probabilities, confidences,
    recency weights, or learned token scores.
    """

    def __init__(self, hidden_size: int, num_heads: int, qkv_bias: bool = True,
                 add_bias_kv: bool = True):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = hidden_size // num_heads
        self.q_proj    = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.k_proj    = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.v_proj    = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.out_proj  = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        # Learnable "attention-sink" pair appended post-projection to K and V,
        # matching `nn.MultiheadAttention(add_bias_kv=True)`. Gives queries a
        # fixed learned position to dump excess attention mass onto, which is
        # well-documented to stabilise transformer convergence. `add_bias_kv=
        # False` disables it (queries must spend all attention on real keys) —
        # the default True matches checkpoints saved before the flag existed.
        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty(1, 1, hidden_size))
            self.bias_v = nn.Parameter(torch.empty(1, 1, hidden_size))
            nn.init.xavier_normal_(self.bias_k)
            nn.init.xavier_normal_(self.bias_v)
        else:
            self.bias_k = None
            self.bias_v = None

    def forward(
        self,
        query: torch.Tensor,                # (B, Lq, H)
        key:   torch.Tensor,                # (B, Lk, H)
        value: torch.Tensor,                # (B, Lk, H)
        log_w: torch.Tensor | None = None,  # (B, Lk) per-key additive bias on logits
    ) -> torch.Tensor:
        B, Lq, _ = query.shape
        Lk = key.shape[1]
        H, D = self.num_heads, self.head_dim
        q = self.q_proj(query)                              # (B, Lq, hidden_size)
        k = self.k_proj(key)
        v = self.v_proj(value)
        # Append learnable sink position to K/V post-projection (matches MHA).
        if self.bias_k is not None:
            k = torch.cat([k, self.bias_k.to(k.dtype).expand(B, 1, -1)], dim=1)
            v = torch.cat([v, self.bias_v.to(v.dtype).expand(B, 1, -1)], dim=1)
            Lk = Lk + 1

        q = q.reshape(B, Lq, H, D).transpose(1, 2)          # (B, H, Lq, D)
        k = k.reshape(B, Lk, H, D).transpose(1, 2)
        v = v.reshape(B, Lk, H, D).transpose(1, 2)

        # SDPA's `attn_mask` is added to the logits before softmax.
        # Broadcast (B, Lk) → (B, 1, 1, Lk) so the same per-key bias applies
        # across all heads and all query positions. Pad log_w with a neutral
        # entry (log 1 = 0) for the appended sink position so it remains
        # unweighted relative to the real keys.
        if log_w is not None and self.bias_k is not None:
            log_w = F.pad(log_w, (0, 1), value=0.0)
        attn_mask = log_w.to(q.dtype).view(B, 1, 1, Lk) if log_w is not None else None
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)  # (B, H, Lq, D)
        attn = attn.transpose(1, 2).reshape(B, Lq, H * D)
        return self.out_proj(attn)
