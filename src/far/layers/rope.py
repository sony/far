"""Axial Rotary Position Embedding (RoPE) modules.

Provides 2D / 3D axial RoPE that splits `head_dim` across the position axes.
The same `rope` module instance is passed to a RoPE-aware attention (see
`far.layers.attention`); the call site supplies as many positional tensors
as the rope module expects (2 for RoPE2D, 3 for RoPE3D).
"""
import math

import torch
import torch.nn as nn


def _build_freqs(dim_axis: int, base: float) -> torch.Tensor:
    """Build the (dim_axis / 2) sinusoidal frequencies for one axis."""
    half = dim_axis // 2
    return torch.exp(-math.log(base) * torch.arange(half, dtype=torch.float32) / half)


def _rotate(x_slice: torch.Tensor, pos: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply 1D RoPE to the last dim of `x_slice`.

    Phase (`pos · freq`), sin/cos, and the rotation arithmetic all run in fp32
    even under bf16 autocast. Time-axis phases in particular can grow to large
    values where bf16 mantissa precision (~0.008) destroys the small relative
    phase differences that distinguish nearby frames.

    Args:
        x_slice: (B, H, L, D_axis)
        pos:     (B, L)
        freqs:   (D_axis / 2,)
    """
    out_dtype = x_slice.dtype
    half      = x_slice.shape[-1] // 2
    x1, x2    = x_slice[..., :half], x_slice[..., half:]
    with torch.amp.autocast(device_type='cuda', enabled=False):
        angles = pos.float().unsqueeze(-1) * freqs.float()      # (B, L, D_axis/2)
        cos    = angles.cos().unsqueeze(1)                      # (B, 1, L, D_axis/2)
        sin    = angles.sin().unsqueeze(1)
        x1f, x2f = x1.float(), x2.float()
        rotated = torch.cat([x1f * cos - x2f * sin, x1f * sin + x2f * cos], dim=-1)
    return rotated.to(out_dtype)


class RoPE2D(nn.Module):
    """Axial 2D RoPE for (y, x) tokens.

    Splits `head_dim` evenly between the y and x axes; each slice is rotated
    independently by its corresponding 1D RoPE.
    """

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 4 == 0, "head_dim must be divisible by 4 for axial 2D RoPE"
        D_y = head_dim // 2
        D_x = head_dim - D_y
        assert D_y % 2 == 0 and D_x % 2 == 0
        self.D_y, self.D_x = D_y, D_x
        self.register_buffer("freqs_y", _build_freqs(D_y, base), persistent=False)
        self.register_buffer("freqs_x", _build_freqs(D_x, base), persistent=False)

    def forward(self, x: torch.Tensor, pos_y: torch.Tensor, pos_x: torch.Tensor) -> torch.Tensor:
        x_y = _rotate(x[..., :self.D_y], pos_y, self.freqs_y)
        x_x = _rotate(x[..., self.D_y:], pos_x, self.freqs_x)
        return torch.cat([x_y, x_x], dim=-1)


class RoPE3D(nn.Module):
    """Axial 3D RoPE for (t, y, x) tokens.

    Splits `head_dim` into three slices — D_t, D_y, D_x — and applies
    independent 1D RoPE on each slice using the corresponding axis position.
    Following WorldMem (https://github.com/xizaoqu/WorldMem), the time axis
    gets a smaller slice (defaults to head_dim // 4) than the spatial axes.
    """

    def __init__(self, head_dim: int, base: float = 10000.0, dt_ratio: float = 0.25):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even"
        D_t = max(2, (int(head_dim * dt_ratio) // 2) * 2)
        D_rest = head_dim - D_t
        D_y = (D_rest // 4) * 2
        D_x = D_rest - D_y
        assert D_t > 0 and D_y > 0 and D_x > 0
        assert all(d % 2 == 0 for d in (D_t, D_y, D_x))
        assert D_t + D_y + D_x == head_dim
        self.D_t, self.D_y, self.D_x = D_t, D_y, D_x
        self.register_buffer("freqs_t", _build_freqs(D_t, base), persistent=False)
        self.register_buffer("freqs_y", _build_freqs(D_y, base), persistent=False)
        self.register_buffer("freqs_x", _build_freqs(D_x, base), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        pos_t: torch.Tensor,
        pos_y: torch.Tensor,
        pos_x: torch.Tensor,
    ) -> torch.Tensor:
        D_t, D_y = self.D_t, self.D_y
        x_t = _rotate(x[..., :D_t],            pos_t, self.freqs_t)
        x_y = _rotate(x[..., D_t:D_t + D_y],   pos_y, self.freqs_y)
        x_x = _rotate(x[..., D_t + D_y:],      pos_x, self.freqs_x)
        return torch.cat([x_t, x_y, x_x], dim=-1)


