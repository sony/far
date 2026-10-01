from dataclasses import dataclass


@dataclass
class DataSpec:
    local_pos_stats: dict    # {'min': Tensor, 'max': Tensor} — xy normalization stats
    meters_per_waypoint: float
    max_frame_offset: int
    len_traj_pred: int
    planner_mu_init: list | None = None     # CEM initial Gaussian mean (None if not used)
    planner_sigma_init: list | None = None  # CEM initial Gaussian std  (None if not used)
    pose_fields: list | None = None         # ordered field names of the raw pose vector
    pose_dim: int | None = None             # total pose dimensionality (always 5)
    pos_dim: int | None = None              # number of spatial position dimensions (2 or 3)
    angle_dim: int | None = None            # number of angle dimensions (1 yaw-only or 2 pitch+yaw)
    pose_mode: str | None = None            # "2d" or "3d"
    fps: float | None = None               # native frame rate of the dataset (frames per second)
