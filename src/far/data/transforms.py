# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Adapted for FAR from Navigation World Models (NWM),
# https://github.com/facebookresearch/nwm (CC BY-NC 4.0).
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms


# ── Image transform pipeline ──────────────────────────────────────────────────

IMAGE_ASPECT_RATIO = (4 / 3)  # all images are center-cropped to 4:3 in training

class CenterCropAR:
    def __init__(self, ar: float = IMAGE_ASPECT_RATIO):
        self.ar = ar

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            h, w = img.shape[-2], img.shape[-1]
        else:
            w, h = img.size  # PIL Image
        if w > h:
            img = TF.center_crop(img, (h, int(h * self.ar)))
        else:
            img = TF.center_crop(img, (int(w / self.ar), w))
        return img

class ToFloat:
    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        return video.float() / 255.0

def make_centercrop_resize(image_height: int, image_width: int):
    return transforms.Compose([
        CenterCropAR(),
        transforms.Resize((image_height, image_width)),
        ToFloat(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])

def make_resize(image_height: int, image_width: int):
    return transforms.Compose([
        transforms.Resize((image_height, image_width)),
        ToFloat(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])

unnormalize_img = transforms.Normalize(
    mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
    std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
)

def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert a (C, H, W) float tensor in [0, 1] to a (H, W, C) uint8 numpy array."""
    return (tensor.permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)

# ── Action normalization ──────────────────────────────────────────────────────

def normalize_data(data, stats):
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    return ndata * 2 - 1

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'].to(ndata) - stats['min'].to(ndata)) + stats['min'].to(ndata)
    return data


# ── Trajectory geometry ───────────────────────────────────────────────────────

def angle_difference(theta1, theta2):
    delta_theta = theta2 - theta1
    return delta_theta - 2 * np.pi * np.floor((delta_theta + np.pi) / (2 * np.pi))


def global_to_local_pose(world_pose: np.ndarray, world_pose0: np.ndarray) -> np.ndarray:
    """Transform a world-frame trajectory into local-frame pose.

    Args:
        world_pose:  (T, 5) array [x, y, z, pitch, yaw] in world frame (angles in radians).
        world_pose0: (5,)   array [x, y, z, pitch, yaw] of the reference pose.

    Returns:
        (T, 5) array [x_local, y_local, z_local, pitch_local, yaw_local].
        xyz are translated and rotated into the local frame defined by world_pose0's
        pitch and yaw. The rotation is consistent with the viewing-direction convention
        d = [cos(pitch)cos(yaw), cos(pitch)sin(yaw), sin(pitch)].
        pitch and yaw are expressed as angle differences from the reference.
    """
    yaw0   = float(world_pose0[4])
    pitch0 = float(world_pose0[3])
    cy, sy = np.cos(yaw0),   np.sin(yaw0)
    cp, sp = np.cos(pitch0), np.sin(pitch0)

    # World-to-local rotation matrix (R @ delta gives local-frame coordinates).
    R = np.array([
        [ cp * cy,  cp * sy, sp],
        [-sy,        cy,      0.],
        [-sp * cy, -sp * sy, cp],
    ])

    local_pose = np.empty_like(world_pose, dtype=np.float64)
    local_pose[:, :3] = (world_pose[:, :3] - world_pose0[:3]) @ R.T
    local_pose[:, 3]  = angle_difference(pitch0, world_pose[:, 3])
    local_pose[:, 4]  = angle_difference(yaw0,   world_pose[:, 4])

    return local_pose


def indices_to_seconds(indices: torch.Tensor, fps: float) -> torch.Tensor:
    """Convert relative frame indices to time in seconds.

    Args:
        indices: (...) frame offsets from the origin frame.
        fps:     Frame rate of the dataset in frames per second.

    Returns:
        (...) relative time in seconds from the origin frame.
    """
    return indices / fps


def pos_to_PoseTrajectory3D(positions: torch.Tensor):
    """Convert a (T, 2) or (T, 3) position tensor into an evo PoseTrajectory3D.

    Orientations are set to identity and timestamps to integer frame indices.
    """
    from evo.core.trajectory import PoseTrajectory3D
    T = positions.shape[0]
    pos_xyz = torch.zeros((T, 3))
    pos_xyz[:, :positions.shape[1]] = positions
    orientations_quat_wxyz = torch.zeros((T, 4))
    orientations_quat_wxyz[:, -1] = 1
    timestamps = torch.arange(T, dtype=torch.float64)
    return PoseTrajectory3D(
        positions_xyz=pos_xyz,
        orientations_quat_wxyz=orientations_quat_wxyz,
        timestamps=timestamps,
    )
