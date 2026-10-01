import torch

from far.data.spec import DataSpec


def build_dataspec(local_pos_stats, meters_per_waypoint, max_frame_offset, len_traj_pred,
                   planner_mu_init=None, planner_sigma_init=None,
                   pose_fields=None, pose_dim=None, pos_dim=None,
                   angle_dim=None, pose_mode=None, fps=None) -> DataSpec:
    return DataSpec(
        local_pos_stats={k: torch.tensor(v) for k, v in local_pos_stats.items()},
        meters_per_waypoint=meters_per_waypoint,
        max_frame_offset=max_frame_offset,
        len_traj_pred=len_traj_pred,
        planner_mu_init=planner_mu_init,
        planner_sigma_init=planner_sigma_init,
        pose_fields=pose_fields,
        pose_dim=pose_dim,
        pos_dim=pos_dim,
        angle_dim=angle_dim,
        pose_mode=pose_mode,
        fps=fps,
    )
