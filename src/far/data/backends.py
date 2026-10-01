from __future__ import annotations

import json
import pickle
from pathlib import Path

from typing import Any
from PIL import Image
import numpy as np
import torch
import av
import io


def _decode_all_av_frames(
    fileobj: io.BytesIO,
    *,
    strict: bool = True,
    name: str = "?",
) -> torch.Tensor:
    """
    Decode all video frames from an AVI byte stream.

    Returns:
        video: uint8 tensor [T, C, H, W]
    """
    frames = []
    with av.open(fileobj) as container:
        for frame in container.decode(video=0):
            arr = frame.to_ndarray(format="rgb24")  # [H, W, C], uint8
            frames.append(torch.from_numpy(arr).permute(2, 0, 1).contiguous())

    if not frames:
        if strict:
            raise ValueError(f"No frames decoded from {name}")
        return torch.empty((0, 3, 0, 0), dtype=torch.uint8)

    return torch.stack(frames, dim=0)  # [T, C, H, W]


def _parse_traj_json(raw: bytes | str | Any) -> list[dict[str, Any]]:
    if isinstance(raw, bytes):
        return json.loads(raw.decode("utf-8"))
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


class WebDatasetBackend:
    """
    Streaming backend for WebDataset shards.

    Each WebDataset sample is expected to contain:
        <key>.avi
        <key>.json

    The .json is assumed to be a list of per-frame trajectory dicts with:
        { "frame_count": ..., "x": ..., "y": ..., "z": ..., "yaw": ... }
    """

    pose_fields = ["x", "y", "z", "pitch", "yaw"]

    def __init__(self, strict: bool = True) -> None:
        self.strict = strict

    def decode_sample(self, sample: dict[str, Any]) -> tuple[str, torch.Tensor, torch.Tensor]:
        clip_key = str(sample.get("__key__", "?"))

        available = [k for k in sample if not k.startswith("__")]
        if "avi" not in sample:
            raise KeyError(f"Missing 'avi' field in sample {clip_key}. Available keys: {available}")
        if "json" not in sample:
            raise KeyError(f"Missing 'json' field in sample {clip_key}. Available keys: {available}")

        frames = _decode_all_av_frames(
            io.BytesIO(sample["avi"]),
            strict=self.strict,
            name=clip_key,
        )
        num_video_frames = int(frames.shape[0])

        traj_data = _parse_traj_json(sample["json"])
        if not isinstance(traj_data, list) or len(traj_data) == 0:
            raise ValueError(f"Invalid trajectory data in sample {clip_key}")

        usable = min(num_video_frames, len(traj_data))
        if usable <= 0:
            raise ValueError(f"No usable aligned frames in sample {clip_key}")

        frames = frames[:usable]

        pose = torch.zeros(usable, 5, dtype=torch.float32)
        for i, entry in enumerate(traj_data[:usable]):
            pose[i, 0] = entry["x"]
            pose[i, 1] = entry["y"]
            pose[i, 2] = entry["z"]
            # pose[i, 3] = 0.0  pitch — not present in data, zero-imputed
            pose[i, 4] = entry["yaw"]

        return clip_key, frames, pose


def _standardize_loopnav_pose(pose: torch.Tensor) -> torch.Tensor:
    """Map raw LoopNav (Minecraft) pose axes to the z-up standard frame.

    The sidecar JSON logs Minecraft world coordinates: `y` is HEIGHT, the
    horizontal plane is (x, z), and the viewing/walking direction is
    `(-sin yaw, -cos yaw)` in (x, z) (verified empirically: mean cosine with
    walking motion +0.999). The pose stack (`global_to_local_pose`, worldmem's
    FOV math, `PluckerEmbedding`) instead assumes z-up with viewing direction
    `(cos yaw, sin yaw)` in the (x, y) horizontal plane. Remap:

        [x_std, y_std, z_std] = [-z_raw, -x_raw, y_raw]     (right-handed)

    after which forward = (cos yaw, sin yaw) in (x_std, y_std), z_std is up,
    and `global_to_local_pose` becomes a true rigid re-anchoring. pitch/yaw
    columns pass through unchanged.
    """
    out = pose.clone()
    out[:, 0] = -pose[:, 2]
    out[:, 1] = -pose[:, 0]
    out[:, 2] = pose[:, 1]
    return out


class LoopNavBackend:
    pose_fields = ["x", "y", "z", "pitch", "yaw"]

    def __init__(self, strict: bool = True, standard_frame: bool = False) -> None:
        # `standard_frame=True` remaps raw Minecraft pose axes to the z-up
        # standard frame (see `_standardize_loopnav_pose`). Default False —
        # checkpoints trained on the raw convention must keep receiving it.
        self.strict = strict
        self.standard_frame = standard_frame

        if av is None:
            raise ImportError(
                "PyAV is required for AviBackend. Install it with `pip install av`."
            )

    def load_clip(self, sample: dict) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Load all frames and poses for a clip.

        Returns:
            video:          Tensor of shape [T, C, H, W], dtype=torch.uint8
            pose:           Tensor of shape [T, 5] [x, y, z, pitch, yaw]; pitch zero-imputed
            curr_frame_idx: index of the first frame where goal["rangeSq"]==1, i.e. the
                            last frame before the agent starts its return leg to A.
                            -1 if rangeSq never appears in this clip.
        """
        path = Path(sample["path"])
        if not path.is_file():
            raise RuntimeError(f"AVI file does not exist: {path}")

        with open(path.with_suffix(".json")) as f:
            traj_data = json.load(f)

        with open(path, "rb") as f:
            video = _decode_all_av_frames(io.BytesIO(f.read()), strict=self.strict, name=str(path))

        usable = min(int(video.shape[0]), len(traj_data))
        video = video[:usable]
        pose = torch.zeros(usable, 5, dtype=torch.float32)
        curr_frame_idx = -1
        for i, entry in enumerate(traj_data[:usable]):
            pose[i, 0] = entry["x"]
            pose[i, 1] = entry["y"]
            pose[i, 2] = entry["z"]
            # pose[i, 3] = 0.0  pitch — not present in data, zero-imputed
            pose[i, 4] = entry["yaw"]
            if curr_frame_idx == -1 and entry.get("goal", {}).get("rangeSq") == 1:
                curr_frame_idx = i
        if self.standard_frame:
            pose = _standardize_loopnav_pose(pose)
        return video, pose, curr_frame_idx


class LatentLoopNavBackend:
    """Backend for precomputed ViTVAE latents saved as .npy + sidecar .json."""

    pose_fields = ["x", "y", "z", "pitch", "yaw"]

    def __init__(self, key_suffix: str | None = ".keys.npy", standard_frame: bool = False) -> None:
        # Sidecar suffix for pre-computed retriever keys: `<clip>.npy` →
        # `<clip><key_suffix>`. Lets a run SELECT among several coexisting key
        # sets (e.g. from different retriever checkpoints), each generated by
        # `build_retriever_keys.py --key-suffix <suffix>`. Set to null/"" to
        # disable key loading entirely (forces on-the-fly QueryViT encoding).
        self.key_suffix = key_suffix
        # `standard_frame=True` remaps raw Minecraft pose axes to the z-up
        # standard frame (see `_standardize_loopnav_pose`). Default False —
        # checkpoints trained on the raw convention must keep receiving it.
        self.standard_frame = standard_frame

    def load_clip(self, sample: dict) -> tuple:
        """Load all latent frames and poses for a clip.

        Returns:
            latents:       Tensor of shape [T, latent_dim, seq_h, seq_w], dtype=torch.float32
            pose:          Tensor of shape [T, 5] [x, y, z, pitch, yaw]; pitch zero-imputed
            curr_frame_idx: index of the first frame where goal["rangeSq"]==1, i.e. the
                            last frame before the agent starts its return leg to A.
                            -1 if rangeSq never appears in this clip.
            keys:          Tensor [T, key_dim] of pre-computed frozen-QueryViT content
                           keys (`<clip><key_suffix>`, built by
                           `scripts/world/build_retriever_keys.py`), or None when
                           `key_suffix` is unset or the file is absent. When present,
                           the retriever uses them as cached lookups instead of
                           encoding frames at train time.
        """
        path = Path(sample["path"])
        if not path.is_file():
            raise RuntimeError(f"Latent file does not exist: {path}")

        with open(path.with_suffix(".json")) as f:
            traj_data = json.load(f)

        latents = torch.from_numpy(np.load(path, mmap_mode='c'))  # (T, latent_dim, seq_h, seq_w)

        # Optional pre-computed retriever keys, sidecar `<clip><key_suffix>`.
        keys = None
        if self.key_suffix:
            key_path = path.with_name(path.stem + self.key_suffix)
            if key_path.is_file():
                keys = torch.from_numpy(np.load(key_path, mmap_mode='c'))

        usable = min(int(latents.shape[0]), len(traj_data))
        latents = latents[:usable]
        if keys is not None:
            keys = keys[:usable]
        pose = torch.zeros(usable, 5, dtype=torch.float32)
        curr_frame_idx = -1
        for i, entry in enumerate(traj_data[:usable]):
            pose[i, 0] = entry["x"]
            pose[i, 1] = entry["y"]
            pose[i, 2] = entry["z"]
            # pose[i, 3] = 0.0  pitch — not present in data, zero-imputed
            pose[i, 4] = entry["yaw"]
            if curr_frame_idx == -1 and entry.get("goal", {}).get("rangeSq") == 1:
                curr_frame_idx = i
        if self.standard_frame:
            pose = _standardize_loopnav_pose(pose)
        return latents, pose, curr_frame_idx, keys


class SoundSpacesBackend:
    """Backend for SoundSpaces episode dirs (SDXL latents + npz sidecars).

    Layout per episode dir (`sample["path"]`, produced by
    `scripts/data/build_latent_dataset_soundspaces.py`):
        latents.npy   (T, 4, 32, 32) float16, saved at SDXL tokenize() scale
        poses.npz     pos (T, 3) Habitat [x, height, z], heading (T,) rad
        meta.json     fps / legs (context vs eval phases) / sources / split
        audio.wav, actions.npz, closures.npz, legs.npz (not read here)

    Poses are ALWAYS remapped to the z-up standard frame: the Habitat pose
    convention matches raw LoopNav exactly — height in pos[:, 1], forward
    (-sin h, -cos h) in (x, z), verified mean cosine with walking motion
    +0.9996 — so `_standardize_loopnav_pose` applies verbatim and
    `global_to_local_pose` / worldmem / PluckerEmbedding are geometrically
    correct. No raw-convention checkpoints exist for this dataset, hence no
    `standard_frame` knob. pitch is zero-imputed (Habitat agents don't pitch).

    `curr_frame_idx` = the first frame of the first leg tagged
    `phase == "eval"` in meta.json (the start of the B→A return leg) — the
    SoundSpaces analog of LoopNav's first `rangeSq == 1` frame.
    """

    pose_fields = ["x", "y", "z", "pitch", "yaw"]

    def __init__(self, key_suffix: str | None = None,
                 audio_mode: str | None = None,
                 audio_window_hops: int = 48) -> None:
        # Sidecar filename suffix for pre-computed retriever keys inside the
        # episode dir: `latents.npy` → `latents<key_suffix>` (e.g.
        # ".keys_audio.npy" once the audio encoder lands). None → no keys.
        self.key_suffix = key_suffix
        # What (if anything) the per-frame AUDIO WINDOWS are returned as —
        # (T, 4, n_mels, audio_window_hops) causal windows from audio_feat.npy
        # (window ENDS at the frame's timestamp; log-mel planes normalized
        # with the dataset-global stats from audio_feat_params.json; head
        # padded at the normalized log-floor; 48 hops = 0.48 s, divisible by
        # patch_size 4 → 16x12 patch grid):
        #   None     — no audio (latents only);
        #   "frames" — windows REPLACE latents in the frames slot (the
        #              contrastive pretrain: QueryViT live-encodes the pool);
        #   "cue"    — frames stay SDXL latents and the windows ride along as
        #              a 5th "cues" element (zero-copy view; the formatter
        #              forwards the CURRENT frame's window as `query_cue` for
        #              pnp_cl's query_source="cue"; pool side via cached keys).
        assert audio_mode in (None, "frames", "cue"), audio_mode
        self.audio_mode = audio_mode
        self.audio_window_hops = audio_window_hops
        self._audio_norm = None      # (mean, std, log_eps) cached per backend

    def _audio_windows(self, d: Path, n_frames: int, fps: float) -> torch.Tensor:
        """(T, 4, n_mels, W) float32 windows, zero-copy strided over the clip."""
        if self._audio_norm is None:
            params = json.loads((d.parent / "audio_feat_params.json").read_text())
            self._audio_norm = (float(params["logmel_mean"]),
                                float(params["logmel_std"]),
                                float(params["log_eps"]),
                                int(round(params["sr"] / params["hop_length"])))
        mean, std, log_eps, hops_per_s = self._audio_norm
        hpf = int(round(hops_per_s / fps))                     # hops per frame (10)
        W = self.audio_window_hops

        feat = np.load(d / "audio_feat.npy", mmap_mode="c").astype(np.float32)
        feat[:2] = (feat[:2] - mean) / std                     # log-mel planes only
        floor = (np.log(log_eps) - mean) / std
        pad = np.empty((4, feat.shape[1], W), dtype=np.float32)
        pad[:2] = floor                                        # silence at the head
        pad[2:] = 0.0                                          # IPD of silence
        padded = np.ascontiguousarray(np.concatenate([pad, feat], axis=2))

        # Frame i's causal window ends at hop hpf*i: padded[..., hpf*i : hpf*i+W].
        # Uniform offsets → one strided VIEW; indexing by the sampler's frame
        # selection later materializes only the chosen windows.
        sc, sm, sh = padded.strides
        # writeable=True avoids torch.from_numpy's read-only warning; safe —
        # the base array is owned here and downstream only reads/indexes.
        windows = np.lib.stride_tricks.as_strided(
            padded, shape=(n_frames, 4, feat.shape[1], W),
            strides=(hpf * sh, sc, sm, sh), writeable=True)
        return torch.from_numpy(windows)

    def load_clip(self, sample: dict) -> tuple:
        """Returns (latents [T,4,32,32] fp32, pose [T,5], curr_frame_idx, keys).

        With audio_mode="frames", the first element is instead the per-frame
        audio windows (T, 4, n_mels, W) and latents.npy is never touched.
        """
        d = Path(sample["path"])

        keys = None
        if self.key_suffix:
            key_path = d / f"latents{self.key_suffix}"
            if key_path.is_file():
                keys = torch.from_numpy(np.load(key_path, mmap_mode="c"))

        with open(d / "meta.json") as f:
            meta = json.load(f)
        poses = np.load(d / "poses.npz")

        cues = None
        if self.audio_mode == "frames":
            frames = self._audio_windows(d, int(meta["n_frames"]), float(meta["fps"]))
        else:
            frames = torch.from_numpy(
                np.load(d / "latents.npy", mmap_mode="c").astype(np.float32))
            if self.audio_mode == "cue":
                cues = self._audio_windows(d, int(meta["n_frames"]), float(meta["fps"]))
        T = min(int(frames.shape[0]), len(poses["heading"]))
        frames = frames[:T]
        if keys is not None:
            keys = keys[:T]
        if cues is not None:
            cues = cues[:T]
        pose = torch.zeros(T, 5, dtype=torch.float32)
        pose[:, :3] = torch.from_numpy(np.asarray(poses["pos"][:T], dtype=np.float32))
        # pose[:, 3] = 0.0  pitch — Habitat agents don't pitch, zero-imputed
        pose[:, 4] = torch.from_numpy(np.asarray(poses["heading"][:T], dtype=np.float32))
        pose = _standardize_loopnav_pose(pose)

        curr_frame_idx = -1
        for leg in meta.get("legs", []):
            if leg.get("phase") == "eval":
                curr_frame_idx = min(int(leg["frames"][0]), T - 1)
                break
        if cues is not None:
            return frames, pose, curr_frame_idx, keys, cues
        return frames, pose, curr_frame_idx, keys


def _standardize_thor_pose(pose: torch.Tensor) -> torch.Tensor:
    """Map raw AI2-THOR (Unity) pose axes to the z-up standard frame.

    THOR/Unity is y-UP: `poses.npz["pos"]` is [x, height, z] and `heading` is
    the yaw about the up axis. The pose stack (`global_to_local_pose`,
    worldmem's FOV math, `PluckerEmbedding`) assumes z-up with viewing
    direction `(cos yaw, sin yaw)` in the (x, y) plane. Remap:

        [x_std, y_std, z_std] = [z_raw, x_raw, height_raw]

    a cyclic axis permutation (det +1, so handedness — and every cross product
    downstream — is preserved). Unlike LoopNav/Habitat, whose forward is
    `(-sin yaw, -cos yaw)` and which therefore needs the `[-z, -x, y]` map,
    THOR's heading needs NO offset or sign flip under this permutation:
    fitting `yaw_std = s*heading + c` against walking motion selects
    (s, c) = (+1, 0) with mean cosine +1.000. pitch/yaw columns pass through.
    """
    out = pose.clone()
    out[:, 0] = pose[:, 2]
    out[:, 1] = pose[:, 0]
    out[:, 2] = pose[:, 1]
    return out


def _thor_pitch_from_quat(quat: np.ndarray) -> np.ndarray:
    """Per-frame pitch (rad, + = looking up) from THOR's camera quaternion.

    `poses.npz` stores `pos`/`quat`/`heading` but NO pitch, while the Plücker
    ray pathway and worldmem's vertical FOV both need it. Unity forward is
    +z, i.e. column 2 of the rotation matrix, whose up-axis component is
    fy = 2(qy·qz - qw·qx); the standard-frame remap sends Unity's up (y) to
    the repo's z, so pitch = asin(fy) in the `d = [cos p cos y, cos p sin y,
    sin p]` convention. The stored layout is wxyz — verified by the encoder's
    pitch probe (yaw from this layout reproduces `heading` to +1.000, and the
    resulting range [-54.5, 0] deg matches the integrated `dpitch` action
    channel offset by THOR's default 30-deg-down camera horizon).
    """
    q = np.asarray(quat, dtype=np.float64)
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]          # wxyz
    n = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    n = np.where(n > 0, n, 1.0)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    fy = 2.0 * (qy * qz - qw * qx)
    return np.arcsin(np.clip(fy, -1.0, 1.0)).astype(np.float32)


def thor_interact_cumsum(interact: np.ndarray, codes=(1, 2, 3, 4)) -> torch.Tensor:
    """(T,) per-frame interaction codes -> (T, 4) cumulative per-code counts
    (one increment per frame carrying the code — the legacy label basis)."""
    onehot = np.stack([(interact == c) for c in codes], axis=1)
    return torch.from_numpy(np.cumsum(onehot, axis=0).astype(np.float32))


def thor_onset_cumsum(interact: np.ndarray, codes=(1, 2, 3, 4)) -> np.ndarray:
    """(T,) codes -> (T, 4) cumulative ONSET counts: one increment per
    interaction, at the first frame of its 3-6-frame span. Differencing two
    frames counts how many interactions BEGAN in the interval — it can never
    exceed the true count, so a window overlapping a span mid-way is not
    labelled a second time (the per-frame counts above are, which is what
    makes a stride-1/2 rollout repeat the same open label 3-6 steps running)."""
    onehot = np.stack([(interact == c) for c in codes], axis=1)
    onset = onehot & ~np.concatenate([np.zeros((1, len(codes)), dtype=bool), onehot[:-1]])
    return np.cumsum(onset, axis=0).astype(np.float32)


def thor_door_state(interact: np.ndarray, openness: np.ndarray) -> np.ndarray:
    """(T,) integrated door-state fraction of the currently-handled container.

    `actions.npz["openness"]` is ANIMATION-only — the door fraction on the
    3-6 swing frames (open 0.3..1.0, close 0.5..0.0) and 0 everywhere else.
    This integrates it into an absolute state: the swing values inside open
    (code 1) / close (code 2) spans, a plateau of 1.0 from an open's end to
    the next close's onset, 0 otherwise. The tour grammar handles containers
    serially and always closes before moving on, so a single global scalar is
    unambiguous (one container in flight at a time)."""
    T = len(interact)
    state = np.zeros(T, dtype=np.float32)
    plateau = 0.0
    for t in range(T):
        c = interact[t]
        if c == 1:
            state[t] = openness[t]
            plateau = 1.0
        elif c == 2:
            state[t] = openness[t]
            plateau = 0.0
        else:
            state[t] = plateau
    return state


class THORBackend:
    """Backend for AI2-THOR episode dirs (SDXL latents + npz sidecars).

    Layout per episode dir (`sample["path"]`, produced by
    `scripts/data/build_latent_dataset_ai2thor.py`):
        latents.npy   (T, 4, 32, 32) float16, saved at SDXL tokenize() scale
        poses.npz     pos (T, 3) Unity [x, height, z], quat (T, 4) wxyz,
                      heading (T,) rad
        meta.json     fps / legs (context vs eval phases) / stations / moves /
                      flips / inspections / split
        actions.npz, objvis.npz, closures.npz, legs.npz (not read here;
                      `actions.npz` carries the interaction channels used by
                      the eval batteries and any future action conditioning)

    Poses are ALWAYS remapped to the z-up standard frame (see
    `_standardize_thor_pose`) — this corpus has no raw-convention
    checkpoints, so there is no `standard_frame` knob. Unlike the other
    backends pitch is NOT zero-imputed: THOR agents pitch up to ~55 deg while
    inspecting stations, so it is recovered from `quat`.

    Every episode also yields per-frame CUMULATIVE interaction counts
    `(T, 4)` over the `actions.npz["interact"]` codes (1 open / 2 close /
    3 pickup / 4 put; 0 = none) as a 6th `load_clip` element. Cumulative, not
    per-frame, because the sampler selects individual frames while an
    interaction spans a median of 3 frames out of ~665: differencing the
    counts at two sampled frames recovers exactly the events in the interval
    between them, which per-frame labels cannot (measured: a goal-frame label
    misses 81% of the intervals that contain an interaction at ltp 64).
    The formatter turns that difference into `action_interact`.

    An 8th element `interact_state (T, 4)` carries the STATE-style label basis
    `[door_state, 0, onset_cum_pickup, onset_cum_put]` (see `thor_door_state`
    / `thor_onset_cumsum`): the formatter turns it into `action_state` — the
    absolute door fraction AT the goal plus once-per-interaction pickup/put
    flags, appended to the count basis for generators whose `interact_dim`
    exceeds 4 (`LDWMCore._generator_interact`; no runtime knob). Motivation: the per-frame counts label every swing frame, so short
    windows repeat an open label onto an already-open current frame — a
    combination training barely contains — and the generator learns the label
    as a state TOGGLE (observed as period-2 open/close flicker in stride-1
    rollouts). An absolute target state is idempotent and supervises mid-swing
    direction; column 1 is reserved (closing is the door fraction dropping).

    `curr_frame_idx` = meta.json's top-level `eval_start`, the generator's own
    declaration of where the evaluated phase begins. It is NOT the same as the
    first `phase == "eval"` leg: measured over 600 episodes it precedes that
    leg by a median 11 frames (99.5% of the time it falls inside the final
    `return` leg), so the rollout starts mid-approach and must predict both the
    arrival and the re-inspection rather than the reveal alone. The eval-leg
    scan is kept only as a fallback for episodes lacking the field.
    """

    pose_fields = ["x", "y", "z", "pitch", "yaw"]

    # Interaction codes in actions.npz["interact"] (0 = none, excluded).
    INTERACT_CODES = (1, 2, 3, 4)          # open / close / pickup / put

    def __init__(self, key_suffix: str | None = None,
                 mask_suffix: str | None = None) -> None:
        # Sidecar filename suffix for pre-computed retriever keys inside the
        # episode dir: `latents.npy` -> `latents<key_suffix>` (e.g.
        # ".keys_thor.npy"). None -> no cached keys (live encoding).
        self.key_suffix = key_suffix
        # Sidecar suffix for the per-frame CHANGE MASK (scripts/data/
        # build_masks_ai2thor.py): `latents<mask_suffix>`, bit-packed
        # (T, 128) uint8 -> (T, 32, 32) on the latent grid, 1 where the room
        # changed relative to the same viewpoint earlier in the episode.
        # Consumed as a per-token diffusion-loss weight; None -> uniform loss.
        self.mask_suffix = mask_suffix

    def _interact_cumsum(self, d: Path, T: int) -> torch.Tensor:
        """(T, 4) cumulative per-code interaction counts (see class docstring)."""
        return thor_interact_cumsum(np.load(d / "actions.npz")["interact"][:T])

    def load_clip(self, sample: dict) -> tuple:
        """Returns (latents [T,4,32,32] fp32, pose [T,5], curr_frame_idx, keys,
        cues=None, events [T,4], loss_mask, interact_state [T,4])."""
        d = Path(sample["path"])

        keys = None
        if self.key_suffix:
            key_path = d / f"latents{self.key_suffix}"
            if key_path.is_file():
                keys = torch.from_numpy(np.load(key_path, mmap_mode="c"))

        with open(d / "meta.json") as f:
            meta = json.load(f)
        poses = np.load(d / "poses.npz")

        frames = torch.from_numpy(
            np.load(d / "latents.npy", mmap_mode="c").astype(np.float32))
        T = min(int(frames.shape[0]), len(poses["heading"]))
        frames = frames[:T]
        if keys is not None:
            keys = keys[:T]

        pose = torch.zeros(T, 5, dtype=torch.float32)
        pose[:, :3] = torch.from_numpy(np.asarray(poses["pos"][:T], dtype=np.float32))
        pose[:, 3] = torch.from_numpy(_thor_pitch_from_quat(poses["quat"][:T]))
        pose[:, 4] = torch.from_numpy(np.asarray(poses["heading"][:T], dtype=np.float32))
        pose = _standardize_thor_pose(pose)

        curr_frame_idx = -1
        if meta.get("eval_start") is not None:
            curr_frame_idx = min(int(meta["eval_start"]), T - 1)
        else:                                    # fallback: first eval-phase leg
            for leg in meta.get("legs", []):
                if leg.get("phase") == "eval":
                    curr_frame_idx = min(int(leg["frames"][0]), T - 1)
                    break
        loss_mask = None
        if self.mask_suffix:
            mp = d / f"latents{self.mask_suffix}"
            if mp.is_file():
                packed = np.asarray(np.load(mp, mmap_mode="c")[:T])
                loss_mask = torch.from_numpy(
                    np.unpackbits(packed, axis=1).reshape(len(packed), 32, 32)
                    .astype(np.float32))
        # cues stays None (no audio here); events ride in the 6th slot, the
        # change mask in the 7th, the state-label basis in the 8th.
        interact_state = np.zeros((T, 4), dtype=np.float32)
        act_path = d / "actions.npz"
        if act_path.is_file():
            az = np.load(act_path)
            itr = np.asarray(az["interact"][:T])
            interact_state[:, 0] = thor_door_state(itr, np.asarray(az["openness"][:T]))
            interact_state[:, 2:] = thor_onset_cumsum(itr)[:, 2:]    # pickup / put onsets
            events = thor_interact_cumsum(itr)
        else:
            # No interaction log (e.g. the door-corridor / dynamic-agent corpus:
            # the ego never acts on the world) -> all-zero counts and state, so
            # the interaction channels are fed their null label everywhere.
            events = torch.zeros(T, 4, dtype=torch.float32)
        return (frames, pose, curr_frame_idx, keys, None,
                events, loss_mask, torch.from_numpy(interact_state))


class NavigationBackend:
    pose_fields = ["x", "y", "z", "pitch", "yaw"]

    def load_clip(self, sample: dict) -> tuple[torch.Tensor, torch.Tensor, None]:
        """Load all frames and poses for a clip.

        Returns:
            video:          Tensor of shape [T, C, H, W], dtype=torch.uint8
            pose:           Tensor of shape [T, 5] [x, y, z, pitch, yaw]; pitch zero-imputed
            curr_frame_idx: always None (NavigationBackend has no rangeSq concept)
        """
        path = Path(sample["path"])

        frame_files = sorted(
            (p for p in path.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}),
            key=lambda p: int(p.stem)
        )
        if len(frame_files) == 0:
            raise RuntimeError(f"No frames found in {path}")

        with open(path / "traj_data.pkl", "rb") as f:
            traj_data = pickle.load(f)
        position = traj_data["position"]
        yaw = traj_data["yaw"]

        T = len(frame_files)
        pose = torch.zeros(T, 5, dtype=torch.float32)
        frames = []
        for i in range(T):
            with Image.open(frame_files[i]) as img:
                frames.append(torch.from_numpy(np.array(img.convert("RGB"))).permute(2, 0, 1))
            pos_i = torch.tensor(position[i], dtype=torch.float32)
            pose[i, :pos_i.shape[0]] = pos_i   # fills x,y or x,y,z depending on dataset
            # pose[i, 3] = 0.0  pitch — not present in data, zero-imputed
            pose[i, 4] = torch.tensor(yaw[i], dtype=torch.float32).squeeze()

        return torch.stack(frames, dim=0), pose, None



class THORDynBackend(THORBackend):
    """THORBackend for the door-corridor / dynamic-agent corpus (`ai2thor_dyn`).
    Identical loading, plus REVEAL-ANCHORED SAMPLING support: the episode has
    no interactions (the ego never acts), so instead of interaction spans the
    backend hands the sampler two REVEAL pseudo-spans that both START at
    `eval_start + reveal_start_offset` (a fixed point early in go_door, see
    `reveal_spans`) and END at the look_A onset (code 0) or the look_B onset
    (code 1). With `ContextGoalSampler`'s `event_goal_anchor=end`,
    `event_phase=eval`, `event_pre=[0, 10]`, `event_post=[8, 30]` an anchored
    draw is then "current in the first go_door frames, BEFORE ANY end has been
    looked at, goal 0.8-3 s after end A (code 0) or end B (code 1) comes into
    view". Nothing in the current frame or in an earlier look settles the
    answer (looking at end A first and finding it empty would otherwise say
    'B'), and the two ends are drawn equally. The B pairs reach across look_A,
    so the anchored draw's horizon (`event_len_traj_pred`) must cover look_A's
    length plus the post-onset window. The interaction labels (`events` /
    `interact_state`) stay all-zero: the spans travel in a SEPARATE 9th return
    slot, so nothing leaks into `action_interact`. `reveal_anchor=False` ->
    plain THORBackend behaviour."""

    REVEAL_CODES = {0: "look_A", 1: "look_B"}      # code -> the leg whose onset ENDS the span

    def __init__(self, key_suffix: str | None = None, mask_suffix: str | None = None,
                 reveal_anchor: bool = True, reveal_start_offset: int = 12,
                 agent_state: bool = False,
                 agent_state_pair_s: float = 1.5, agent_state_px_min: int = 1,
                 agent_state_noise: float = 0.0) -> None:
        super().__init__(key_suffix=key_suffix, mask_suffix=mask_suffix)
        self.reveal_anchor = bool(reveal_anchor)
        # Span START = eval_start + reveal_start_offset (see `reveal_spans`);
        # must match the offset the reveal event index was built with.
        self.reveal_start_offset = int(reveal_start_offset)
        # AGENT-STATE OBSERVATION (`agent_state`): every frame also carries A2's
        # DETECTION in that frame's own egocentric coordinates, as a 10th
        # `load_clip` slot `(T, 6)` = `[vi, dx, dy, vi', dx', dy']`:
        #   vi      1 if A2 is in view (`agent2.npz` px >= `agent_state_px_min`,
        #           a detector-like threshold — the door-aperture glimpses sit at
        #           a few hundred px), else 0;
        #   dx, dy  A2's position relative to THIS frame's camera, metres, in the
        #           z-up standard frame's floor plane: dx forward (along the
        #           heading), dy lateral (left positive) — the yaw-only version
        #           of `global_to_local_pose`'s rotation, so a frame's state is
        #           frame-intrinsic and no consumer has to transform it; zero
        #           when vi = 0;
        #   the primed half = the same triple at the frame `agent_state_pair_s`
        #           seconds EARLIER (itself at the clip start), so one context
        #           slot carries velocity (~2 m at walking speed).
        # `agent_state_noise` > 0 perturbs (dx, dy) by that fraction of the
        # range (deterministic per episode) to mimic a detector. The formatter
        # divides the positions by `meters_per_waypoint`, like the poses.
        self.agent_state = bool(agent_state)
        self.agent_state_pair_s = float(agent_state_pair_s)
        self.agent_state_px_min = int(agent_state_px_min)
        self.agent_state_noise = float(agent_state_noise)

    def _agent_state(self, d: Path, pose: torch.Tensor, fps: float) -> torch.Tensor:
        """`(T, 6)` agent-state observation (see `__init__`) from `agent2.npz`
        and the standardized observer poses."""
        az = np.load(d / "agent2.npz")
        T = int(pose.shape[0])
        px = np.asarray(az["px"][:T])
        # A2's floor position in the standard frame: [x_std, y_std] = [z_raw, x_raw]
        # (the same cyclic remap as `_standardize_thor_pose`).
        a2 = np.stack([np.asarray(az["z"][:T], dtype=np.float64),
                       np.asarray(az["x"][:T], dtype=np.float64)], axis=1)
        obs = pose[:, :2].numpy().astype(np.float64)
        yaw = pose[:, 4].numpy().astype(np.float64)
        delta = a2 - obs
        cy, sy = np.cos(yaw), np.sin(yaw)
        fwd = cy * delta[:, 0] + sy * delta[:, 1]
        lat = -sy * delta[:, 0] + cy * delta[:, 1]
        if self.agent_state_noise > 0:
            rng = np.random.default_rng(abs(hash(d.name)) % (2 ** 32))
            rng_scale = self.agent_state_noise * np.hypot(fwd, lat)
            fwd = fwd + rng.normal(size=T) * rng_scale
            lat = lat + rng.normal(size=T) * rng_scale
        vi = (px >= self.agent_state_px_min).astype(np.float32)
        state3 = np.stack([vi, fwd * vi, lat * vi], axis=1).astype(np.float32)
        j = np.clip(np.arange(T) - int(round(self.agent_state_pair_s * fps)), 0, T - 1)
        return torch.from_numpy(np.concatenate([state3, state3[j]], axis=1))       # (T, 6)

    @staticmethod
    def reveal_spans(meta: dict, start_offset: int = 12) -> list[tuple[int, int, int]]:
        """`(code, start, end)` pseudo-spans in the sampler's span convention
        (shared with the event-index builder): end = the code's leg onset;
        start = `eval_start + start_offset` — a FIXED point early in go_door
        that depends on nothing A2 does. The sampler draws the anchored
        current `event_pre` frames BEFORE the start, so the current window is
        identical for every episode. (A start tied to A2 — e.g. the look_A
        onset pulled back to the first go_door frame with any A2 pixel — would
        leak the answer: which end is glimpsed through the aperture is fixed
        by the scene, so the pull-back, and with it the current's position
        along go_door, its distance/time to the goal and the pool's recency,
        encodes the final end and a predictor with every agent state zeroed
        still scores well on the 'memory-required' pairs. The first 12
        go_door frames are glimpse-free in ~95% of episodes.)"""
        legs = {L["name"]: (int(L["start"]), int(L["end"])) for L in meta.get("legs", [])}
        if "look_A" not in legs:
            return []
        a0 = int(meta.get("eval_start", legs["look_A"][0])) + int(start_offset)
        return [(c, a0, legs[n][0]) for c, n in THORDynBackend.REVEAL_CODES.items() if n in legs]

    def load_clip(self, sample: dict) -> tuple:
        """THORBackend's 8 slots + the reveal spans (9th; None when
        `reveal_anchor` is off) + the agent-state observation (10th; None when
        `agent_state` is off)."""
        out = super().load_clip(sample)
        d = Path(sample["path"]) if not Path(sample["path"]).is_file() else Path(sample["path"]).parent
        meta = json.loads((d / "meta.json").read_text())
        has_a2 = (d / "agent2.npz").is_file()
        spans = None
        if self.reveal_anchor:
            spans = self.reveal_spans(meta, self.reveal_start_offset)
        state = None
        if self.agent_state:
            if not has_a2:
                raise FileNotFoundError(f"THORDynBackend: agent_state=True but no agent2.npz in {d}")
            state = self._agent_state(d, out[1], float(meta.get("fps", 10.0)))
        return out + (spans, state)
