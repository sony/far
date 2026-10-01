"""
ViT-based VAE tokenizer. Portions adapted from
    - VQGAN / taming-transformers: https://github.com/CompVis/taming-transformers (MIT)
    - MAE: https://github.com/facebookresearch/mae (CC BY-NC 4.0)
See THIRD_PARTY_NOTICES.md.
"""

import functools
import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.vision_transformer import Mlp
from timm.layers.helpers import to_2tuple
from .rotary_embedding import RotaryEmbedding, apply_rotary_emb

VAE_SCALE = 0.07843137255

_NORM_LAYER = functools.partial(nn.LayerNorm, eps=1e-6)


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0.0)
        nn.init.constant_(m.weight, 1.0)


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        img_height=256,
        img_width=256,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        img_size = (img_height, img_width)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x, random_sample=False):
        B, C, H, W = x.shape
        assert random_sample or (H == self.img_size[0] and W == self.img_size[1]), \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = rearrange(x, "B C H W -> B (H W) C")
        else:
            x = rearrange(x, "B C H W -> B H W C")
        x = self.norm(x)
        return x


class DiagonalGaussianDistribution:
    def __init__(self, parameters, deterministic=False, dim=1):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=dim)
        if dim == 1:
            self.dims = [1, 2, 3]
        elif dim == 2:
            self.dims = [1, 2]
        else:
            raise NotImplementedError
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        return self.mean + self.std * torch.randn_like(self.mean)

    def mode(self):
        return self.mean

    def kl(self) -> torch.Tensor:
        return 0.5 * torch.sum(
            self.mean ** 2 + self.var - 1.0 - self.logvar, dim=self.dims
        ).mean()


class Attention(nn.Module):
    def __init__(self, dim, num_heads, frame_height, frame_width, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.frame_height = frame_height
        self.frame_width = frame_width

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        rotary_freqs = RotaryEmbedding(
            dim=head_dim // 4,
            freqs_for="pixel",
            max_freq=frame_height * frame_width,
        ).get_axial_freqs(frame_height, frame_width)
        self.register_buffer("rotary_freqs", rotary_freqs, persistent=False)

    def forward(self, x):
        B, N, C = x.shape
        assert N == self.frame_height * self.frame_width

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = rearrange(q, "b (H W) (h d) -> b h H W d", H=self.frame_height, W=self.frame_width, h=self.num_heads)
        k = rearrange(k, "b (H W) (h d) -> b h H W d", H=self.frame_height, W=self.frame_width, h=self.num_heads)
        v = rearrange(v, "b (H W) (h d) -> b h H W d", H=self.frame_height, W=self.frame_width, h=self.num_heads)

        q = apply_rotary_emb(self.rotary_freqs, q)
        k = apply_rotary_emb(self.rotary_freqs, k)

        q = rearrange(q, "b h H W d -> b h (H W) d")
        k = rearrange(k, "b h H W d -> b h (H W) d")
        v = rearrange(v, "b h H W d -> b h (H W) d")

        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b h N d -> b N (h d)")
        return self.proj(x)


class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, frame_height, frame_width, mlp_ratio=4.0, qkv_bias=False,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads, frame_height, frame_width, qkv_bias=qkv_bias)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ── Sub-modules ───────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(
        self,
        input_height: int = 256,
        input_width: int = 256,
        patch_size: int = 16,
        dim: int = 768,
        depth: int = 6,
        heads: int = 12,
        latent_dim: int = 16,
        use_variational: bool = True,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.seq_h = input_height // patch_size
        self.seq_w = input_width // patch_size
        self.latent_dim = latent_dim
        self.use_variational = use_variational

        self.patch_embed = PatchEmbed(input_height, input_width, patch_size, 3, dim)
        self.blocks = nn.ModuleList([
            AttentionBlock(dim, heads, self.seq_h, self.seq_w, mlp_ratio, qkv_bias=True, norm_layer=_NORM_LAYER)
            for _ in range(depth)
        ])
        self.norm = _NORM_LAYER(dim)
        self.quant_conv = nn.Linear(dim, (2 if use_variational else 1) * latent_dim)

        self.apply(_init_weights)
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def forward(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        x = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        moments = self.quant_conv(x)
        if not self.use_variational:
            moments = torch.cat((moments, torch.zeros_like(moments)), dim=2)
        return DiagonalGaussianDistribution(moments, deterministic=not self.use_variational, dim=2)


class Decoder(nn.Module):
    def __init__(
        self,
        input_height: int = 256,
        input_width: int = 256,
        patch_size: int = 16,
        dim: int = 768,
        depth: int = 6,
        heads: int = 12,
        latent_dim: int = 16,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.seq_h = input_height // patch_size
        self.seq_w = input_width // patch_size
        self.patch_size = patch_size
        self.input_height = input_height
        self.input_width = input_width
        self.patch_dim = 3 * patch_size ** 2

        self.post_quant_conv = nn.Linear(latent_dim, dim)
        self.blocks = nn.ModuleList([
            AttentionBlock(dim, heads, self.seq_h, self.seq_w, mlp_ratio, qkv_bias=True, norm_layer=_NORM_LAYER)
            for _ in range(depth)
        ])
        self.norm = _NORM_LAYER(dim)
        self.predictor = nn.Linear(dim, self.patch_dim)

        self.apply(_init_weights)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        x = x.reshape(bsz, self.seq_h, self.seq_w, self.patch_dim).permute(0, 3, 1, 2)
        x = x.reshape(bsz, 3, self.patch_size, self.patch_size, self.seq_h, self.seq_w).permute(0, 1, 4, 2, 5, 3)
        return x.reshape(bsz, 3, self.input_height, self.input_width)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.post_quant_conv(z)
        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z)
        z = self.predictor(z)
        return self.unpatchify(z)


# ── Top-level model ───────────────────────────────────────────────────────────

class ViTVAE(nn.Module):
    """
    ViT-based Variational Autoencoder.

    Composes an Encoder and Decoder with a variational bottleneck. Exposes
    tokenize/detokenize to serve as a drop-in tokenizer for LDWM, reshaping the
    patch-sequence latent to a spatial tensor (B, latent_dim, seq_h, seq_w).
    """

    def __init__(self, encoder: Encoder, decoder: Decoder, kl_weight: float = 1e-6, lpips_weight: float = 0.0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.kl_weight = kl_weight
        self.lpips_weight = lpips_weight
        if lpips_weight > 0:
            self.lpips_fn = lpips.LPIPS(net='vgg', verbose=False)
            for p in self.lpips_fn.parameters():
                p.requires_grad_(False)
        else:
            self.lpips_fn = None

    def train(self, mode: bool = True):
        super().train(mode)
        if self.lpips_fn is not None:
            self.lpips_fn.eval()
        return self

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """Encode pixel images to a spatial latent tensor.

        Args:
            x: (N, C, H, W) images in [-1, 1].
        Returns:
            (N, latent_dim, seq_h, seq_w) scaled latent tensor (posterior mean).
        """
        z = self.encoder(x).mode()
        B = x.shape[0]
        z = z.reshape(B, self.encoder.seq_h, self.encoder.seq_w, self.encoder.latent_dim).permute(0, 3, 1, 2)
        return z * VAE_SCALE

    def detokenize(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a spatial latent tensor to pixel images.

        Args:
            z: (N, latent_dim, seq_h, seq_w) scaled latent tensor.
        Returns:
            (N, C, H, W) images in [-1, 1].
        """
        B, latent_dim, seq_h, seq_w = z.shape
        z_seq = (z / VAE_SCALE).permute(0, 2, 3, 1).reshape(B, seq_h * seq_w, latent_dim)
        return self.decoder(z_seq)

    def compute_recon_loss(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Compute reconstruction + KL + LPIPS loss for a batch of images.

        Args:
            x: (B, C, H, W) input images in [-1, 1].
        Returns:
            x_recon:    (B, C, H, W) reconstructed images in [-1, 1].
            loss:       Scalar total loss.
            loss_dict:  Dict with keys 'l1', 'kl', 'lpips' (unweighted values).
        """
        posterior = self.encoder(x)
        z = posterior.sample()
        x_recon = self.decoder(z)
        l1 = F.l1_loss(x_recon, x)
        kl = posterior.kl()
        loss = l1 + self.kl_weight * kl
        lpips_val = torch.tensor(0.0, device=x.device)
        if self.lpips_fn is not None:
            lpips_val = self.lpips_fn(x_recon, x.detach()).clamp(min=0).mean()
            loss = loss + self.lpips_weight * lpips_val
        return x_recon, loss, {"l1": l1.detach(), "kl": kl.detach(), "lpips": lpips_val.detach()}

    def load_weights(self, ckpt_path):
        self.load_state_dict(torch.load(ckpt_path))

    @property
    def latent_size(self) -> tuple[int, int, int]:
        return (self.encoder.latent_dim, self.encoder.seq_h, self.encoder.seq_w)
