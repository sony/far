import numpy as np
import torch

from .spec import DataSpec
from .transforms import (
    normalize_data,
    global_to_local_pose,
    indices_to_seconds,
)


class FrameFormatter:
    """Returns all frames and video_id from a VideoDataset item."""

    def __call__(self, item: dict) -> dict:
        return {
            "frames": item["frames"],
            "video_id": item["video_id"],
        }


class ContextPoseFormatter:
    """
    Splits frames/pose from VideoDataset into context and goal parts.

    Returns both unnormalized and normalized pose and time for each split:
    - *_poses:     local-frame pose, xyz in waypoint units (not normalized).
    - *_poses_nl:  local-frame pose, xyz normalized to [-1, 1] via local_pos_stats.
    - *_times:     relative time in seconds from the first context frame.
    - *_times_nl:  relative frame offset normalized by max_frame_offset.

    Output pose columns: [x_local, y_local, z_local, pitch, yaw_local]
    """

    def __init__(self, spec: DataSpec) -> None:
        if spec.fps is None:
            raise ValueError("DataSpec.fps is required for ContextPoseFormatter.")
        self.meters_per_waypoint = spec.meters_per_waypoint
        self.local_pos_stats = {k: v.unsqueeze(0).numpy() for k, v in spec.local_pos_stats.items()}
        self.max_frame_offset = spec.max_frame_offset
        self.fps = spec.fps

    def __call__(self, item: dict) -> dict:
        frames = item["frames"]
        pose = item["pose"].numpy().astype(np.float64)   # (T, 5) [x, y, z, pitch, yaw]
        n = len(item["context_indices"])
        g = len(item["goal_indices"])

        local_pose = global_to_local_pose(pose, pose[n - 1])
        local_pose[:, :3] /= self.meters_per_waypoint

        local_pose_nl = local_pose.copy()
        local_pose_nl[:, :3] = normalize_data(local_pose_nl[:, :3], self.local_pos_stats)

        all_poses    = torch.as_tensor(local_pose,    dtype=torch.float32)
        all_poses_nl = torch.as_tensor(local_pose_nl, dtype=torch.float32)

        context_indices = item["context_indices"]
        goal_indices    = item["goal_indices"]
        origin = context_indices[0]
        ctx_shifted  = torch.tensor(context_indices, dtype=torch.float32) - origin
        goal_shifted = torch.tensor(goal_indices,    dtype=torch.float32) - origin

        context_mask = item.get("context_mask")
        if context_mask is not None:
            context_mask = torch.tensor(context_mask, dtype=torch.bool)

        out = {
            "context_frames":    frames[:n],
            "context_poses":     all_poses[:n],
            "context_poses_nl":  all_poses_nl[:n],
            "context_times":     indices_to_seconds(ctx_shifted,  self.fps),
            "context_times_nl":  ctx_shifted / self.max_frame_offset,
            "context_mask":      context_mask,
            "goal_frames":       frames[n:n + g],
            "goal_poses":        all_poses[n:n + g],
            "goal_poses_nl":     all_poses_nl[n:n + g],
            "goal_times":        indices_to_seconds(goal_shifted, self.fps),
            "goal_times_nl":     goal_shifted / self.max_frame_offset,
            "pose_fields":       item["pose_fields"],
            "video_id":          item["video_id"],
        }

        # Optional pre-computed retriever keys for the CONTEXT pool (used by the
        # retriever as cached lookups). Present only when the backend loaded a
        # `<clip>.keys.npy`, so datasets without them collate exactly as before.
        keys = item.get("keys")
        if keys is not None:
            out["context_keys"] = keys[:n]
        # Optional retrieval-cue window of the CURRENT frame (pool tail), e.g.
        # the binaural audio window — consumed by pnp_cl's query_source="cue"
        # as the live query-encoder input. Modality-agnostic by design.
        cues = item.get("cues")
        if cues is not None:
            out["query_cue"] = cues[n - 1]
        # Optional per-goal INTERACTION multi-hot: which action types occur in
        # the interval (current, goal]. The backend supplies CUMULATIVE
        # per-frame counts, so differencing the two sampled endpoints recovers
        # the interval exactly — the intermediate frames are never sampled.
        # Present only for backends that emit them (THOR), so every other
        # dataset collates unchanged.
        events = item.get("events")
        if events is not None:
            out["action_interact"] = (events[n:n + g] - events[n - 1]).clamp(0, 1)
        # Optional per-goal STATE-style interaction label (THOR):
        # [door fraction AT the goal, 0, pickup begins in (current, goal],
        # put begins in (current, goal]]. The door column is an ABSOLUTE
        # state (idempotent — a repeated label cannot toggle the door, unlike
        # the event flags above); pickup/put are onset counts, so a window
        # overlapping a span mid-way is never labelled twice. Its leading
        # columns are APPENDED to `action_interact` for generators built with
        # interact_dim > 4 (`LDWMCore._generator_interact`); the retriever
        # keeps the plain counts.
        st = item.get("interact_state")
        if st is not None:
            a_state = st[n:n + g].clone()
            a_state[:, 2:] = (st[n:n + g, 2:] - st[n - 1, 2:]).clamp(0, 1)
            out["action_state"] = a_state
        # Optional per-GOAL change mask on the latent grid: 1 where the room
        # differs from the same viewpoint earlier in the episode. Only the goal
        # rows matter — it weights the diffusion loss on the frame being
        # predicted (`LDWMCore.mask_loss_lambda`). Absent for every other
        # backend, so their batches collate unchanged.
        lm = item.get("loss_mask")
        if lm is not None:
            out["goal_loss_mask"] = lm[n:n + g]
            # Per-frame object PRESENCE / size = the loss-mask area fraction per
            # frame (agent-silhouette masks: the second agent on screen; change
            # masks: a changed region at this view). Emitted for CONTEXT frames
            # too — the object-centric retriever pre-training
            # (train_latent_retriever_hard.py) picks its positives from them.
            out["context_obj"] = lm[:n].reshape(n, -1).float().mean(1)
            out["goal_obj"]    = lm[n:n + g].reshape(g, -1).float().mean(1)
        # Optional SECOND-AGENT DETECTION per frame (THORDynBackend
        # `agent_state`): `[vi, dx, dy, vi', dx', dy']` in the frame's OWN
        # egocentric floor coordinates (primed = the frame `agent_state_pair_s`
        # earlier). Frame-intrinsic, so nothing is transformed here — only the
        # positions are put in waypoint units like the poses. The CONTEXT rows
        # are the observation the predictor reads (`context_agent_state`, 6);
        # the GOAL rows' own triple is its regression target
        # (`goal_agent_state`, 3) in the goal frame's coordinates.
        ast = item.get("agent_state")
        if ast is not None:
            ctx_state = ast[:n].clone().float()
            ctx_state[:, 1:3] /= self.meters_per_waypoint
            ctx_state[:, 4:6] /= self.meters_per_waypoint
            goal_state = ast[n:n + g, :3].clone().float()
            goal_state[:, 1:3] /= self.meters_per_waypoint
            out["context_agent_state"] = ctx_state
            out["goal_agent_state"] = goal_state
            # Whether this draw came from the sampler's event-anchored branch (the
            # reveal pairs that must be answered from memory) — the eval splits the
            # predictor's metrics on it.
            out["event_anchored"] = torch.tensor(bool(item.get("event_anchored", False)))
        return out



class ContextPoseFormatter2D:
    """
    2D variant of ContextPoseFormatter.

    Same as ContextPoseFormatter but returns pose columns [x_local, y_local, yaw_local],
    discarding z and pitch.
    """

    def __init__(self, spec: DataSpec) -> None:
        if spec.fps is None:
            raise ValueError("DataSpec.fps is required for ContextPoseFormatter2D.")
        self.meters_per_waypoint = spec.meters_per_waypoint
        self.local_pos_stats = {k: v.unsqueeze(0).numpy() for k, v in spec.local_pos_stats.items()}
        self.max_frame_offset = spec.max_frame_offset
        self.fps = spec.fps

    def __call__(self, item: dict) -> dict:
        frames = item["frames"]
        pose = item["pose"].numpy().astype(np.float64)   # (T, 5) [x, y, z, pitch, yaw]
        n = len(item["context_indices"])
        g = len(item["goal_indices"])

        local_pose = global_to_local_pose(pose, pose[n - 1])
        local_pose[:, :3] /= self.meters_per_waypoint

        local_pose_nl = local_pose.copy()
        local_pose_nl[:, :2] = normalize_data(local_pose_nl[:, :2], self.local_pos_stats)

        all_poses    = torch.as_tensor(local_pose[:, [0, 1, 4]],    dtype=torch.float32)
        all_poses_nl = torch.as_tensor(local_pose_nl[:, [0, 1, 4]], dtype=torch.float32)

        context_indices = item["context_indices"]
        goal_indices    = item["goal_indices"]
        origin = context_indices[0]
        ctx_shifted  = torch.tensor(context_indices, dtype=torch.float32) - origin
        goal_shifted = torch.tensor(goal_indices,    dtype=torch.float32) - origin

        context_mask = item.get("context_mask")
        if context_mask is not None:
            context_mask = torch.tensor(context_mask, dtype=torch.bool)

        out = {
            "context_frames":    frames[:n],
            "context_poses":     all_poses[:n],
            "context_poses_nl":  all_poses_nl[:n],
            "context_times":     indices_to_seconds(ctx_shifted,  self.fps),
            "context_times_nl":  ctx_shifted / self.max_frame_offset,
            "context_mask":      context_mask,
            "goal_frames":       frames[n:n + g],
            "goal_poses":        all_poses[n:n + g],
            "goal_poses_nl":     all_poses_nl[n:n + g],
            "goal_times":        indices_to_seconds(goal_shifted, self.fps),
            "goal_times_nl":     goal_shifted / self.max_frame_offset,
            "pose_fields":       ["x", "y", "yaw"],
            "video_id":          item["video_id"],
        }
        return out



