import torch
import torch.nn as nn
from diffusers.models import AutoencoderKL

VAE_SCALE = 0.18215


class SD_VAE(nn.Module):

    def __init__(self, image_height: int, image_width: int,
                 vae_scale: float = VAE_SCALE):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        # Latent scaling factor. Default = SD1.x's 0.18215 (existing configs /
        # checkpoints unchanged); the SDXL autoencoder uses 0.13025
        # (configs/model/tokenizer/sdxl_vae.yaml). Pre-computed .npy latents
        # must be saved at THIS scale (tokenize() output).
        self.vae_scale = float(vae_scale)
        self.backbone = AutoencoderKL()

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """Encode pixel images to scaled VAE latents.

        Args:
            x: (N, C, H, W) pixel images normalized to [-1, 1].
        Returns:
            (N, 4, H/8, W/8) scaled latent tensors.
        """
        return self.backbone.encode(x).latent_dist.sample().mul_(self.vae_scale)

    def detokenize(self, z: torch.Tensor) -> torch.Tensor:
        """Decode scaled VAE latents back to pixel images.

        Args:
            z: (N, 4, H/8, W/8) scaled latent tensors.
        Returns:
            (N, C, H, W) pixel images in [-1, 1].
        """
        return self.backbone.decode(z / self.vae_scale).sample
    
    def load_weights(self, ckpt_path):
        self.backbone = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path=ckpt_path,
            local_files_only=False,
        )
    
    @property
    def latent_size(self) -> tuple[int, int, int]:
        return (4, self.image_height // 8, self.image_width // 8)
