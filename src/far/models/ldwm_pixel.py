import torch
import torch.nn as nn

from .ldwm_core import LDWMCore


class PixelLDWM(nn.Module):
    """
    Pixel-space wrapper around LDWMCore.

    Handles tokenize/detokenize so that callers work entirely in pixel space
    while the diffusion process runs in the compressed latent space owned by
    LDWMCore.
    """

    def __init__(self, tokenizer: nn.Module, core: LDWMCore):
        super().__init__()
        self.tokenizer = tokenizer
        self.core = core

    # ------------------------------------------------------------------
    # Convenience pass-throughs
    # ------------------------------------------------------------------

    @property
    def generator(self):
        return self.core.generator

    @property
    def diffusion(self):
        return self.core.diffusion

    # ------------------------------------------------------------------
    # Pixel-space API
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        context: torch.Tensor,
        goal: torch.Tensor,
        action: torch.Tensor,
        stride: torch.Tensor,
        context_poses: torch.Tensor = None,
        context_times: torch.Tensor = None,
        context_mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode frames to latents and forward to `LDWMCore.compute_loss_latent`.

        Args:
            context:       (B, context_size, C, H, W) conditioning pixel frames in [-1, 1].
            goal:          (B, num_goals, C, H, W) target pixel frames in [-1, 1].
            action:        (B, num_goals, action_dim) actions for each goal frame
                           — `goal_poses_nl` in the formatter's nomenclature;
                           pnp_cl recovers the raw goal pose for its `'pose'`
                           positives by inverting the `_nl` normalisation
                           internally (same pattern as worldmem).
            stride:        (B, num_goals) relative times for each goal frame.
            context_poses: (B, context_size, pose_dim) per-frame poses; required
                           for memory-based context selection.
            context_times: (B, context_size) per-frame timestamps in seconds.

        Returns:
            `(opt_loss, gen_loss)` — two scalar tensors.
            See `LDWMCore.compute_loss_latent` for semantics.
        """
        B, context_size = context.shape[:2]
        num_goals = goal.shape[1]

        with torch.no_grad():
            context_latent = self.tokenizer.tokenize(context.flatten(0, 1)).unflatten(0, (B, context_size))
            goal_latent = self.tokenizer.tokenize(goal.flatten(0, 1)).unflatten(0, (B, num_goals))

        return self.core.compute_loss_latent(context_latent=context_latent, goal_latent=goal_latent, action=action,
                                            stride=stride, context_poses=context_poses, context_times=context_times,
                                            context_mask=context_mask)

    def sample(
        self,
        context: torch.Tensor,
        action: torch.Tensor,
        stride: torch.Tensor,
        num_goals: int = 1,
        progress: bool = False,
        context_poses: torch.Tensor = None,
        context_times: torch.Tensor = None,
        context_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Encode conditioning frames, run reverse diffusion, and decode to pixels.

        Args:
            context:       (B, context_size, C, H, W) conditioning pixel frames in [-1, 1].
            action:        (B * num_goals, action_dim) actions toward each goal frame.
            stride:        (B * num_goals,) relative times for each goal frame.
            num_goals:     number of future frames to generate per batch element.
            progress:      show tqdm progress bar during sampling.
            context_poses: (B, context_size, pose_dim) per-frame poses.
            context_times: (B, context_size) per-frame timestamps in seconds.

        Returns:
            (B * num_goals, C, H, W) generated pixel frames clipped to [-1, 1].
        """
        B, context_size = context.shape[:2]

        with torch.no_grad():
            context_latent = self.tokenizer.tokenize(context.flatten(0, 1)).unflatten(0, (B, context_size))

        samples = self.core.sample_latent(context_latent, action, stride, num_goals, progress,
                                          context_poses=context_poses, context_times=context_times,
                                          context_mask=context_mask)
        return torch.clip(self.tokenizer.detokenize(samples), -1., 1.)

    def sample_ar(
        self,
        context: torch.Tensor,
        action: torch.Tensor,
        stride: float,
        context_poses: torch.Tensor = None,
        context_times: torch.Tensor = None,
        context_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Autoregressively sample T future frames, feeding each prediction back as context.

        Args:
            context:       (B, context_size, C, H, W) conditioning frames in [-1, 1].
            action:        (B, T, action_dim) per-step pose deltas in local coords.
            stride:        Rollout stride normalised by max_frame_offset.
            context_poses: (B, context_size, pose_dim) per-frame poses.
            context_times: (B, context_size) per-frame timestamps in seconds.

        Returns:
            (B, T, C, H, W) predicted frames in [-1, 1].
        """
        B, context_size = context.shape[:2]

        with torch.no_grad():
            context_latent = self.tokenizer.tokenize(context.flatten(0, 1)).unflatten(0, (B, context_size))

        traj = self.core.sample_ar_latent(context_latent, action, stride,
                                          context_poses=context_poses, context_times=context_times,
                                          context_mask=context_mask)

        T = traj.shape[1]
        pixel_traj = self.tokenizer.detokenize(traj.flatten(0, 1)).unflatten(0, (B, T))
        return torch.clip(pixel_traj, -1., 1.)
