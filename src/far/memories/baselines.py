import torch
import torch.nn as nn


class TemporalRetrieval(nn.Module):
    """Select the K most recent evenly-spaced past context frames.

    Follows the same API as WorldMemRetrieval: the last temporal index is
    treated as the current state and is excluded from retrieval. For each
    batch element we build K target times spaced ``interval`` seconds apart
    walking backward from an anchor:

        anchor   = min(current_time, last_valid_past_time + interval)
        target_k = anchor - (k + 1) * interval           for k = 0 .. K-1

    and pick the past frame whose timestamp is closest to each target. When
    the pool is contiguous up to the current frame (training; the last valid
    frame is one frame old) the anchor IS the current time. When the pool
    stops earlier (e.g. the rollout's exploration-only pool) the ladder starts
    at the most recent valid frame instead of snapping every target to it.
    With ``interval`` small enough that consecutive targets fall on
    consecutive frames, this reduces to "K most recent past frames".

    Args:
        num_memories: Number of past frames to return (K).
        interval:     Spacing between consecutive target frames, in seconds.
    """

    def __init__(self, interval: float = 0.0):
        super().__init__()
        assert interval >= 0.0
        self.interval = float(interval)

    @torch.no_grad()
    def forward(
        self,
        context_frames: torch.Tensor,                # (B, T, C, H, W)  last index = current frame
        context_poses:  torch.Tensor,                # (B, T, 5)         last index = current pose
        context_times:  torch.Tensor,                # (B, T)  seconds   last index = current time
        action:         torch.Tensor,                # (B, num_goals, action_dim) — unused, only needs num_goals
        stride:         torch.Tensor,                # (B, num_goals)              — unused
        context_mask:   torch.Tensor | None = None,  # (B, T) bool, False = padding
        context_keys:   torch.Tensor | None = None,  # accepted for API parity; unused
        return_local:   bool = False,                # accepted for API parity; unused
        num_memories:   int = None,                  # frames to return per goal; supplied by LDWMCore.prepare_context
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        """Return K evenly-spaced past frames, broadcast across num_goals.

        Returns the unified 5-tuple shape shared with `WorldMemRetrieval` and
        `pnp_cl.PnPRetrieval` (eval mode):
            selected_frames:       (B, num_goals, K, C, H, W)
            selected_poses:        (B, num_goals, K, 5)
            selected_times:        (B, num_goals, K)
            selected_scores:       None — no set-level ranking signal.
            selected_chunk_scores: None — no per-chunk relevance signal.
        """
        del stride, context_keys, return_local                                     # API parity only
        num_goals = action.shape[1]
        past_frames = context_frames[:, :-1]   # (B, T-1, C, H, W)
        past_poses  = context_poses[:, :-1]    # (B, T-1, 5)
        past_times  = context_times[:, :-1]    # (B, T-1)
        current_times = context_times[:, -1]   # (B,)
        B, T_past = past_frames.shape[:2]
        K = num_memories
        device = context_frames.device

        # Trivial case: no history at all → repeat index 0 (matches the prior
        # padding behaviour; assumes T_past >= 1, which the sampler guarantees).
        if T_past == 0:
            idx = torch.zeros(K, dtype=torch.long, device=device)
            sel_frames, sel_poses, sel_times = past_frames[:, idx], past_poses[:, idx], past_times[:, idx]
        else:
            # Target times, most-recent-first:
            #   k = 0     → current - 1 * interval
            #   k = K - 1 → current - K * interval
            pad = None if context_mask is None else ~context_mask[:, :-1]                        # (B, T_past)
            # Anchor: the current time while the pool reaches the present, else the
            # most recent VALID past time + interval (so target_0 lands on it).
            pt = past_times.float() if pad is None else past_times.float().masked_fill(pad, float('-inf'))
            last_valid = pt.max(dim=1).values                                                     # (B,)
            last_valid = torch.where(torch.isfinite(last_valid), last_valid, current_times.float())
            anchor = torch.minimum(current_times.float(), last_valid + self.interval)             # (B,)
            offsets = self.interval * torch.arange(1, K + 1, device=device, dtype=torch.float32)  # (K,)
            target_times = anchor.unsqueeze(1) - offsets.unsqueeze(0)                            # (B, K)

            # |past_time - target_time| over (B, T_past, K). Mask padding so it's
            # never the closest match.
            dist = (past_times.float().unsqueeze(2) - target_times.unsqueeze(1)).abs()            # (B, T_past, K)
            if pad is not None:
                dist = dist.masked_fill(pad.unsqueeze(2), float('inf'))
                # Fallback: any sample with zero valid past frames maps every target to index 0.
                no_valid = pad.all(dim=1)
                if no_valid.any():
                    dist[no_valid, 0, :] = 0.0

            sel_idx = dist.argmin(dim=1)            # (B, K)
            B_range = torch.arange(B, device=device)
            sel_frames = torch.stack([past_frames[B_range, sel_idx[:, k]] for k in range(K)], dim=1)
            sel_poses  = torch.stack([past_poses[B_range,  sel_idx[:, k]] for k in range(K)], dim=1)
            sel_times  = torch.stack([past_times[B_range,  sel_idx[:, k]] for k in range(K)], dim=1)

        # Action-independent: broadcast the same selection across all goals.
        # Trailing `None, None` are the unified set-/chunk-score placeholders.
        return (
            sel_frames.unsqueeze(1).expand(B, num_goals, *sel_frames.shape[1:]),
            sel_poses .unsqueeze(1).expand(B, num_goals, *sel_poses .shape[1:]),
            sel_times .unsqueeze(1).expand(B, num_goals, *sel_times .shape[1:]),
            None,
            None,
        )

