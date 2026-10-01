# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Adapted for FAR from Navigation World Models (NWM),
# https://github.com/facebookresearch/nwm (CC BY-NC 4.0).
import os
import subprocess
import sys

from omegaconf import OmegaConf


def save_run_info(result_dir, cfg):
    """Save config, git state, environment snapshot, and launch command to result_dir."""
    OmegaConf.save(cfg, os.path.join(result_dir, 'config.yaml'))

    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], text=True).strip()
        git_info = f"branch: {branch}\ncommit: {commit}\n"
    except subprocess.CalledProcessError:
        git_info = "git info unavailable\n"
    with open(os.path.join(result_dir, 'git.txt'), 'w') as f:
        f.write(git_info)

    try:
        env = subprocess.check_output(['pip', 'freeze'], text=True)
    except subprocess.CalledProcessError:
        env = "pip freeze unavailable\n"
    with open(os.path.join(result_dir, 'env.txt'), 'w') as f:
        f.write(env)

    with open(os.path.join(result_dir, 'command.txt'), 'w') as f:
        f.write(' '.join(sys.argv) + '\n')


def init_wandb(cfg, run_name: str, result_dir: str):
    """Initialize a wandb run on rank 0. Returns the run object, or None if disabled."""
    if not hasattr(cfg, 'wandb') or cfg.wandb.get('mode', 'online') == 'disabled':
        return None
    import wandb
    wcfg = cfg.wandb
    tags = list(wcfg.tags) if wcfg.get('tags') else None
    return wandb.init(
        project=wcfg.project,
        entity=wcfg.get('entity', None),
        mode=wcfg.get('mode', 'online'),
        name=wcfg.get('name', None) or run_name,
        group=wcfg.get('group', None),
        tags=tags,
        dir=result_dir,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
