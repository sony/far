"""
WorldMem memory retrieval module.

Reference: https://github.com/xizaoqu/WorldMem (Yu et al., 2025) — specifically
the `_generate_condition_indices` greedy retrieval in
`algorithms/worldmem/df_video.py`. This is an independent re-implementation of
that algorithm as a baseline (no WorldMem code is copied); WorldMem itself is
released under the S-Lab License 1.0 (non-commercial). See THIRD_PARTY_NOTICES.md.

We select past frames whose FOV overlaps the FOV of the **frame to be
generated** (the goal), greedily picking the past frame with the highest
marginal coverage of the goal's currently-uncovered visibility region (with a
small recency tiebreaker), then subtracting that frame's footprint before the
next pick — same algorithm as the WorldMem repo, but the reference FOV is the
goal pose rather than the current pose.

Where does the goal pose come from?
-----------------------------------
The memory module receives `action` from the LDWM training loop, which is the
formatter's `goal_poses_nl`: the goal frame's local pose with the position dims
normalised to [-1, 1] via the dataspec's `local_pos_stats`. We invert that
normalisation here (using the same stats, passed in at construction) to recover
the goal pose in the SAME local coordinate system as `context_poses`. Pitch and
yaw are not normalised by the formatter and pass through unchanged.

  goal_pose = action;   goal_pose[:3] ← action[:3] * pos_scale + pos_offset

  where pos_scale  = (max - min) / 2,  pos_offset = (max + min) / 2
  matches `unnormalize_data`'s inverse of `normalize_data`.

If `local_pos_stats` is left as the identity default, `action` is treated as
already in `context_poses` units (e.g. the demo notebook that passes zero
actions). Per-goal retrieval is then performed independently for each goal.

Pose / coordinate conventions
-----------------------------
Both FAR and WorldMem store pose as `[x, y, z, pitch, yaw]` in that order
(pitch at index 3, yaw at index 4). They differ only in the world axes:
  - WorldMem (Minecraft):  +z forward, +y up, +x lateral.
                          azimuth = atan2(x, z), elevation = atan2(y, hypot(x,z));
                          all angles in DEGREES.
  - FAR (LoopNav, etc.):  +x forward, +z up, +y lateral.
                          azimuth = atan2(y, x), elevation = atan2(z, hypot(x,y));
                          all angles in RADIANS.
FAR's `_is_inside_fov` already encodes FAR's convention, so the FOV-overlap
math reads `[x, y, z, pitch, yaw]` directly. The only change relative to the
original WorldMem repo is the reference pose used to define the target
visibility region: **goal pose**, not current pose.
"""

import math
import torch
import torch.nn as nn


def _generate_points_in_sphere(
    n_points: int, radius: float, device: torch.device
) -> torch.Tensor:
    """Uniformly sample points within a sphere of given radius.

    Returns:
        (N, 3) float tensor.
    """
    r     = radius * torch.rand(n_points, device=device).pow(1 / 3)
    phi   = 2 * math.pi * torch.rand(n_points, device=device)
    theta = (1 - 2 * torch.rand(n_points, device=device)).acos()
    return torch.stack([
        r * theta.sin() * phi.cos(),
        r * theta.sin() * phi.sin(),
        r * theta.cos(),
    ], dim=1)  # (N, 3)


def _is_inside_fov(
    points:    torch.Tensor,  # (N, B, 3)
    center:    torch.Tensor,  # (B, 3)
    pitch:     torch.Tensor,  # (B,)  radians
    yaw:       torch.Tensor,  # (B,)  radians
    fov_half_h: float,        # radians
    fov_half_v: float,        # radians
) -> torch.Tensor:            # (N, B) bool
    """Return which points fall within each camera's FOV cone.

    Matches the dataset's viewing-direction convention
    d = [cos(pitch)cos(yaw), cos(pitch)sin(yaw), sin(pitch)]: at (yaw=0, pitch=0)
    the camera looks along +x, with +y lateral and +z up. Yaw is therefore the
    azimuth in the x-y plane (from +x) and pitch the elevation toward +z. All
    angles are in radians.
    """
    vec = points - center[None]          # (N, B, 3)
    x, y, z = vec[..., 0], vec[..., 1], vec[..., 2]

    azimuth   = torch.atan2(y, x)                          # (N, B)  x-y plane, from +x
    elevation = torch.atan2(z, (x ** 2 + y ** 2).sqrt())   # (N, B)  toward +z (up)

    two_pi  = 2 * math.pi
    diff_az = (azimuth   - yaw[None]).abs()   % two_pi
    diff_el = (elevation - pitch[None]).abs() % two_pi
    diff_az = torch.where(diff_az > math.pi, two_pi - diff_az, diff_az)
    diff_el = torch.where(diff_el > math.pi, two_pi - diff_el, diff_el)

    return (diff_az < fov_half_h) & (diff_el < fov_half_v)


def pose_fov_overlap(
    ref_poses:       torch.Tensor,        # (B, 5)            — single reference per batch element
    candidate_poses: torch.Tensor,        # (B, T, 5)         — T candidates per batch element
    radius:          float,
    fov_half_h:      float,               # RADIANS
    fov_half_v:      float,               # RADIANS
    num_samples:     int = 10_000,
) -> torch.Tensor:                        # (B, T)            — per-candidate overlap fraction
    """For each `(b, t)`, fraction of the `b`-th reference camera's visibility-
    sphere points (uniformly sampled inside a `radius`-ball around
    `ref_poses[b]`'s position) that also fall inside the FOV of
    `candidate_poses[b, t]`. Higher = candidate covers more of the same scene
    region the reference camera sees.

    This is the **static, non-greedy** version of the FOV-overlap score that
    `WorldMemRetrieval._greedy_select` computes inside its iterations. Use
    this when you only need a one-shot pose-similarity ranking — for
    example, picking pose-based positive context chunks in
    `far.memories.pnp_cl.PnPRetrieval` for the InfoNCE loss.

    Pose format: `[x, y, z, pitch, yaw]` with angles in radians and
    positions in dataset units (matches `context_poses` and `goal_poses`
    from `ContextPoseFormatter`).
    """
    B, T, _ = candidate_poses.shape
    device  = ref_poses.device

    # Visibility sphere centered at each reference position. `points` here
    # has shape (N, B, 3): the same N samples shifted to each batch element's
    # reference origin. `_is_inside_fov` accepts this layout natively.
    points = _generate_points_in_sphere(num_samples, radius, device)               # (N, 3)
    points = points[:, None].expand(-1, B, -1) + ref_poses[None, :, :3]            # (N, B, 3)

    # Reference FOV mask: which sphere points are inside the reference camera's view.
    in_fov_ref = _is_inside_fov(
        points,
        ref_poses[:, :3],
        ref_poses[:, 3],
        ref_poses[:, 4],
        fov_half_h, fov_half_v,
    )                                                                               # (N, B) bool

    # For each candidate frame, evaluate ITS FOV against the same sphere
    # points. The inner loop is `T` cheap GPU ops, much smaller than the
    # alternative (expanding `points` to `(N, B*T, 3)` blows up memory at
    # the `num_samples × B × T` scale).
    in_fov_cand = torch.stack([
        _is_inside_fov(
            points,
            candidate_poses[:, t, :3],
            candidate_poses[:, t, 3],
            candidate_poses[:, t, 4],
            fov_half_h, fov_half_v,
        )
        for t in range(T)
    ])                                                                              # (T, N, B) bool

    intersection = (in_fov_ref[None] & in_fov_cand).sum(1).float()                 # (T, B)
    total        = in_fov_ref.sum(0).float().clamp(min=1)                          # (B,)
    return (intersection / total[None]).T                                          # (B, T)


class WorldMemRetrieval(nn.Module):
    """Select relevant context frames from memory based on camera FOV overlap
    with the **goal** frame (the frame to be generated), greedily, with a
    recency tiebreaker.

    For each past frame the confidence is
        overlap_ratio + recency_penalty,
    where `overlap_ratio` is the fraction of the goal's currently-uncovered
    visibility-sphere points falling inside that past frame's FOV, and the
    recency penalty is proportional to elapsed time since the past frame
    (normalised by the oldest delta in the batch). Each iteration picks the
    highest-confidence past frame and subtracts its footprint from `remaining`,
    so the next pick covers the unattended part of the goal's view.

    Selection is **per goal**: each `(b, goal)` runs its own greedy against the
    goal-pose-centered visibility sphere.

    Pose format: [x, y, z, pitch, yaw]  (radians; positions in dataset units).

    Args:
        num_memories:    Number of context frames to return (K).
        num_samples:     Points sampled inside the visibility sphere.
        radius:          Sphere radius around the goal camera position.
        fov_half_h:      Horizontal half-FOV in degrees (default 52.5 → 105° total).
        fov_half_v:      Vertical   half-FOV in degrees (default 37.5 →  75° total).
        recency_factor:  Maximum temporal penalty applied to the oldest frame.
        local_pos_stats: Dataset's `{min, max}` position normalisation stats,
            used to invert the formatter's `_nl` normalisation so the goal
            pose recovered from `action` is in the same units as `context_poses`.
            If None, `action` is assumed to already be in `context_poses` units
            (e.g. demo callers that pass an already-unnormalised action or zeros).
    """

    def __init__(
        self,
        num_samples:     int   = 10_000,
        radius:          float = 30.0,
        fov_half_h:      float = 52.5,
        fov_half_v:      float = 37.5,
        recency_factor:  float = 0.2,
        local_pos_stats: dict | None = None,
    ):
        super().__init__()
        self.num_samples    = num_samples
        self.radius         = radius
        # Config provides half-FOV in degrees (intuitive); store radians since
        # the pose angles compared in `_is_inside_fov` are in radians.
        self.fov_half_h     = math.radians(fov_half_h)
        self.fov_half_v     = math.radians(fov_half_v)
        self.recency_factor = recency_factor

        # Inverse-normalisation of action positions: x = x_nl * pos_scale + pos_offset
        # matches `unnormalize_data((x_nl), {min, max})`. Default = identity
        # (action treated as already in context_poses units).
        if local_pos_stats is not None:
            mn = torch.as_tensor(local_pos_stats["min"], dtype=torch.float32).reshape(-1)
            mx = torch.as_tensor(local_pos_stats["max"], dtype=torch.float32).reshape(-1)
            pos_scale  = (mx - mn) / 2
            pos_offset = (mx + mn) / 2
        else:
            pos_scale  = torch.ones(3)
            pos_offset = torch.zeros(3)
        self.register_buffer("pos_scale",  pos_scale,  persistent=False)
        self.register_buffer("pos_offset", pos_offset, persistent=False)

    @torch.no_grad()
    def forward(
        self,
        context_frames: torch.Tensor,                # (B, T, C, H, W)  last index = current frame
        context_poses:  torch.Tensor,                # (B, T, 5)         last index = current pose
        context_times:  torch.Tensor,                # (B, T)  seconds   last index = current time
        action:         torch.Tensor,                # (B, num_goals, action_dim) = goal_poses_nl
        stride:         torch.Tensor,                # (B, num_goals)              — unused
        context_mask:   torch.Tensor | None = None,  # (B, T) bool, False = padding
        context_keys:   torch.Tensor | None = None,  # accepted for API parity; unused
        return_local:   bool = False,                # accepted for API parity; unused
        num_memories:   int = None,                  # frames to return per goal; supplied by LDWMCore.prepare_context
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, torch.Tensor]:
        """Select K past context frames per goal via goal-FOV-overlap scoring.

        The goal pose is recovered from `action` (= formatter's `goal_poses_nl`):
        positions un-normalised via `local_pos_stats`, pitch / yaw passed through.

        Returns:
            selected_frames:       (B, num_goals, K, C, H, W)
            selected_poses:        (B, num_goals, K, 5)
            selected_times:        (B, num_goals, K)
            selected_scores:       None — WorldMem produces a single set per goal,
                so set-level scoring is trivial (one set).
            selected_chunk_scores: (B, num_goals, K) per-frame importance — each
                slot holds the `confidence = overlap + recency_penalty` value
                that frame had at the moment it was greedily selected. Fed to
                generators that consume `context_chunk_score` (cdit_mod_score_{xa,fa})
                so the model can reweight each retrieved memory by its FOV-
                overlap relevance. The "not enough history" fallback returns
                zeros for these scores (no relevance signal exists when slots
                are just padding repeats of the oldest frame).
        """
        del stride, context_keys, return_local                                    # API parity only
        num_goals = action.shape[1]
        B, T, *_  = context_frames.shape
        K         = num_memories
        device    = context_poses.device

        # Split current state from retrievable past frames.
        current_time = context_times[:, -1]     # (B,)
        past_frames  = context_frames[:, :-1]   # (B, T-1, C, H, W)
        past_poses   = context_poses[:, :-1]    # (B, T-1, 5)
        past_times   = context_times[:, :-1]    # (B, T-1)
        T_past = T - 1

        # Not enough history: fill remaining slots with the first (oldest)
        # frame and broadcast across goals. No real ranking signal exists
        # here, so chunk scores are returned as zeros.
        if T_past <= K:
            pad = K - T_past
            idx = torch.tensor(list(range(T_past)) + [0] * pad, device=device)  # (K,)
            sel_frames = past_frames[:, idx]  # (B, K, C, H, W)
            sel_poses  = past_poses[:, idx]   # (B, K, 5)
            sel_times  = past_times[:, idx]   # (B, K)
            sel_chunk_scores = torch.zeros(B, num_goals, K, device=device, dtype=past_poses.dtype)
            return (
                sel_frames.unsqueeze(1).expand(B, num_goals, *sel_frames.shape[1:]),
                sel_poses .unsqueeze(1).expand(B, num_goals, *sel_poses .shape[1:]),
                sel_times .unsqueeze(1).expand(B, num_goals, *sel_times .shape[1:]),
                None,
                sel_chunk_scores,
            )

        # ── Recover the goal pose in `context_poses` units from `action` ─────
        # action shape (B, num_goals, 5) = [x_nl, y_nl, z_nl, pitch, yaw]. Cast
        # the inverse-normalisation buffers to action's device so the module
        # works whether or not the caller invoked `.to(device)` on it.
        pos_scale  = self.pos_scale .to(action.device)
        pos_offset = self.pos_offset.to(action.device)
        goal_poses = action.clone()
        goal_poses[..., :3] = action[..., :3] * pos_scale + pos_offset

        # ── Per-goal greedy: flatten (B, num_goals) into a single batch axis.
        # Past frames / times are duplicated across goals (same memory pool per
        # batch element) but `points` and `in_fov_*` vary because the sphere is
        # centered on each goal's pose.
        BG = B * num_goals

        goal_poses_bg   = goal_poses.reshape(BG, 5)                                        # (BG, 5)
        past_frames_bg  = past_frames.unsqueeze(1).expand(B, num_goals, *past_frames.shape[1:]).reshape(BG, *past_frames.shape[1:])
        past_poses_bg   = past_poses .unsqueeze(1).expand(B, num_goals, T_past, 5).reshape(BG, T_past, 5)
        past_times_bg   = past_times .unsqueeze(1).expand(B, num_goals, T_past).reshape(BG, T_past)
        current_time_bg = current_time.unsqueeze(1).expand(B, num_goals).reshape(BG)        # (BG,)
        if context_mask is not None:
            context_mask_bg = context_mask.unsqueeze(1).expand(B, num_goals, T).reshape(BG, T)
        else:
            context_mask_bg = None

        sel_frames_bg, sel_poses_bg, sel_times_bg, sel_conf_bg = self._greedy_select(
            ref_poses=goal_poses_bg,
            past_frames=past_frames_bg, past_poses=past_poses_bg, past_times=past_times_bg,
            current_time=current_time_bg, context_mask=context_mask_bg,
            num_memories=num_memories,
        )

        return (
            sel_frames_bg.reshape(B, num_goals, *sel_frames_bg.shape[1:]),
            sel_poses_bg .reshape(B, num_goals, *sel_poses_bg .shape[1:]),
            sel_times_bg .reshape(B, num_goals, *sel_times_bg .shape[1:]),
            None,
            sel_conf_bg  .reshape(B, num_goals, K),
        )

    # ── Internal: WorldMem-style greedy selection at a single batch axis ────
    @torch.no_grad()
    def _greedy_select(
        self,
        ref_poses:     torch.Tensor,        # (B, 5) reference pose used to define the target FOV
        past_frames:   torch.Tensor,        # (B, T_past, C, H, W)
        past_poses:    torch.Tensor,        # (B, T_past, 5)
        past_times:    torch.Tensor,        # (B, T_past)
        current_time:  torch.Tensor,        # (B,)
        context_mask:  torch.Tensor | None, # (B, T) bool — full mask incl. current
        num_memories:  int,                 # K — frames to select (forwarded from forward())
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run WorldMem's greedy FOV-overlap selection over a single batch axis.

        The visibility sphere is centered on each `ref_poses` position and
        intersected with that pose's FOV → `in_fov_target`. We then greedily
        pick K past frames whose FOV most covers the as-yet-uncovered target
        sphere (with a recency tiebreaker), removing each pick's footprint
        before the next iteration.

        Also returns the per-iteration confidence value (`overlap + recency`)
        at the moment of selection — these are the per-chunk importance
        scores forwarded to `cdit_mod_score_{xa,fa}` via `context_chunk_score`.
        """
        B, T_past = past_times.shape
        K         = num_memories
        N         = self.num_samples
        device    = past_poses.device

        # Visibility sphere centered at the reference (goal) position.
        points = _generate_points_in_sphere(N, self.radius, device)                # (N, 3)
        points = points[:, None].expand(-1, B, -1) + ref_poses[None, :, :3]        # (N, B, 3)

        # Target FOV: which sphere points are visible from the reference pose.
        in_fov_target = _is_inside_fov(
            points,
            ref_poses[:, :3],
            ref_poses[:, 3],
            ref_poses[:, 4],
            self.fov_half_h, self.fov_half_v,
        )                                                                          # (N, B) bool

        # FOV mask for every past frame, evaluated against the SAME sphere.
        in_fov_past = torch.stack([
            _is_inside_fov(
                points,
                past_poses[:, t, :3],
                past_poses[:, t, 3],
                past_poses[:, t, 4],
                self.fov_half_h, self.fov_half_v,
            )
            for t in range(T_past)
        ])                                                                         # (T_past, N, B)

        # Recency penalty — proportional to elapsed time since each past frame.
        time_delta = (current_time[:, None] - past_times).float()                  # (B, T_past)
        max_delta  = time_delta.amax(dim=1, keepdim=True).clamp(min=1e-6)          # (B, 1)
        t_penalty  = (-self.recency_factor * time_delta / max_delta).T             # (T_past, B)

        padding_mask = None
        if context_mask is not None:
            padding_mask = ~context_mask[:, :-1].T                                 # (T_past, B)

        remaining = in_fov_target.clone()                                          # (N, B) — shrinks
        B_range   = torch.arange(B, device=device)
        selected: list[torch.Tensor] = []                                          # list of K (B,) longs
        sel_conf: list[torch.Tensor] = []                                          # list of K (B,) floats — confidence at pick

        for _ in range(K):
            # Fraction of currently-uncovered target points each past frame covers.
            intersection = (remaining[None] & in_fov_past).sum(1).float()          # (T_past, B)
            total        = remaining.sum(0).float().clamp(min=1)                   # (B,)
            overlap      = intersection / total[None]                              # (T_past, B)

            confidence = overlap + t_penalty                                       # (T_past, B)

            if padding_mask is not None:
                confidence[padding_mask] = -1e10
            for prev in selected:
                confidence.scatter_(0, prev[None], -1e10)

            best = confidence.argmax(dim=0)                                        # (B,)
            selected.append(best)
            # Capture the actual confidence (post-mask) of the picked frame.
            # `gather` on dim 0 with index of shape (1, B) → (1, B); squeeze → (B,).
            best_conf = confidence.gather(0, best.unsqueeze(0)).squeeze(0)         # (B,)
            sel_conf.append(best_conf)

            # Remove picked frame's visibility footprint from `remaining`.
            ctx_BT_N = in_fov_past.permute(2, 0, 1).long()                         # (B, T_past, N)
            picked   = ctx_BT_N.gather(
                1, best[:, None, None].expand(-1, 1, N)
            ).squeeze(1)                                                            # (B, N)
            remaining = remaining & ~picked.permute(1, 0).bool()                   # (N, B)

        selected = torch.stack(selected, dim=0)                                    # (K, B)

        sel_frames = torch.stack(
            [past_frames[B_range, selected[k]] for k in range(K)], dim=1
        )  # (B, K, C, H, W)
        sel_poses = torch.stack(
            [past_poses[B_range, selected[k]] for k in range(K)], dim=1
        )  # (B, K, 5)
        sel_times = torch.stack(
            [past_times[B_range, selected[k]] for k in range(K)], dim=1
        )  # (B, K)
        sel_confidence = torch.stack(sel_conf, dim=1)  # (B, K)
        return sel_frames, sel_poses, sel_times, sel_confidence
