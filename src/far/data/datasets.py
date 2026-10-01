from __future__ import annotations

import json
import random
import warnings
import webdataset as wds
import numpy as np
import torch.distributed as dist

from pathlib import Path
from typing import Any, Iterator

from torch.utils.data import Dataset, IterableDataset


class WebVideoIterableDataset(IterableDataset):
    """
    IterableDataset over WebDataset shards.

    DDP-safe behavior:
    - shards are partitioned first across distributed ranks
    - then partitioned across dataloader workers within each rank
    - each worker/rank sees a disjoint subset of shard URLs

    Flow:
        raw_sample
        -> backend.decode_sample -> video_id, frames, pose
        -> sample dict {"video_id": ..., "num_frames": ...}
        -> sampler(sample) -> sampling_spec
        -> select frames, pose by frame_indices
        -> optional transform on frames
        -> yield dict matching VideoDataset.__getitem__ output
    """

    def __init__(
        self,
        manifest_path: str | Path,
        index_path: str | Path,
        backend,
        sampler=None,
        transform=None,
        cache_dir: str | None = None,
        cache_size: int = -1,
        shuffle_shards: bool = False,
        shuffle_samples: int | None = None,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.sampler = sampler
        self.transform = transform
        self.cache_dir = cache_dir
        self.cache_size = cache_size * 1024**3
        self.shuffle_shards = shuffle_shards
        self.shuffle_samples = shuffle_samples

        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        index = json.loads(Path(index_path).read_text(encoding="utf-8"))

        sample_by_id = {str(s["shard_id"]): s for s in manifest["samples"]}
        selected_ids = index["sample_ids"]

        missing = [sid for sid in selected_ids if sid not in sample_by_id]
        if missing:
            raise ValueError(
                f"{len(missing)} sample_ids from index not found in manifest. "
                f"First few: {missing[:5]}"
            )

        self.urls = [sample_by_id[sid]["path"] for sid in selected_ids]

    def _get_rank_world_size(self) -> tuple[int, int]:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return 0, 1

    def _partition_urls(self) -> list[str]:
        """Partition shard URLs across DDP ranks. Worker splitting is left to WebDataset."""
        rank, world_size = self._get_rank_world_size()
        return self.urls[rank::world_size]

    def _build_pipeline(self, urls: list[str]):
        kwargs = {}
        if self.cache_dir is not None:
            kwargs["cache_dir"] = self.cache_dir
            kwargs["cache_size"] = self.cache_size

        dataset = wds.WebDataset(
            urls,
            nodesplitter=lambda src: src,
            shardshuffle=self.shuffle_shards,
            handler=wds.warn_and_continue,
            **kwargs,
        )

        if self.shuffle_samples:
            dataset = dataset.shuffle(self.shuffle_samples)

        return dataset

    def __iter__(self) -> Iterator[dict[str, Any]]:
        local_urls = self._partition_urls()

        if len(local_urls) == 0:
            return

        for raw_sample in self._build_pipeline(local_urls):
            # Skip phantom samples that contain only WebDataset cache sidecar files
            # (.lock / .metadata) with no actual data keys.
            data_keys = {k for k in raw_sample if not k.startswith("__")
                         and not k.endswith(("lock", "metadata"))}
            if not data_keys:
                continue

            try:
                video_id, frames, pose = self.backend.decode_sample(raw_sample)
            except Exception as e:
                warnings.warn(f"[WebVideoIterableDataset] skipping sample '{raw_sample.get('__key__', '?')}': {e}")
                continue

            num_frames = int(frames.shape[0])
            sample = {"video_id": video_id, "num_frames": num_frames}

            if self.sampler is not None:
                sampling_spec = self.sampler(sample)
            else:
                frame_indices = list(range(num_frames))
                sampling_spec = {
                    "frame_indices": frame_indices,
                    "context_indices": None,
                    "goal_indices": None,
                }

            frame_indices = sampling_spec["frame_indices"]
            selected_frames = frames[frame_indices]
            selected_pose = pose[frame_indices]

            if self.transform is not None:
                selected_frames = self.transform(selected_frames)

            yield {
                "frames": selected_frames,
                "pose": selected_pose,
                "pose_fields": self.backend.pose_fields,
                "context_indices": sampling_spec["context_indices"],
                "goal_indices": sampling_spec["goal_indices"],
                "video_id": video_id,
            }


class FormattedIterableDataset(IterableDataset):
    def __init__(self, base_dataset, formatter):
        self.base_dataset = base_dataset
        self.formatter = formatter

    def __iter__(self):
        for item in self.base_dataset:
            yield self.formatter(item)


def _event_spans(events) -> list[tuple[int, int, int]] | None:
    """Interaction spans `(code, start, end)` from per-frame CUMULATIVE counts
    `(T, n_codes)`: a fire is a frame where a code's count increments, and
    consecutive fires of the same code (one 2-3-frame animation) form one span.
    None when the backend serves no events. Consumed by
    `ContextGoalSampler`'s event bias; cheap (numpy on a (T, 4) array)."""
    if events is None:
        return None
    ev = np.asarray(events)
    fires = np.diff(ev, axis=0, prepend=ev[:1]) > 0            # (T, n_codes) bool
    spans = []
    for c in range(fires.shape[1]):
        idx = np.flatnonzero(fires[:, c])
        if idx.size == 0:
            continue
        cuts = np.flatnonzero(np.diff(idx) > 1)
        for run in np.split(idx, cuts + 1):
            spans.append((c, int(run[0]), int(run[-1])))
    return sorted(spans, key=lambda t: t[1])


class VideoDataset(Dataset):
    """Map-style dataset over one manifest+index.

    EVENT-BIASED CLIP SUBSTITUTION (`event_frac` + `event_index_path`, train
    splits only): with probability `event_frac` the requested clip is REPLACED
    by one drawn uniformly from the event index — the clips on which the
    sampler's event-anchored draw is feasible (built offline by
    `scripts/data/build_event_index_ai2thor.py` with the SAME eligibility
    predicate the draw applies) — and the sampler is forced into the event
    draw (`force_event=True`); otherwise the requested clip is used with the
    uniform draw forced. This makes the marginal event-pair rate exactly
    `event_frac`, where the sampler's own per-clip Bernoulli tops out at
    p x (fraction of clips with an eligible span) — measured 6% at p=0.3 on
    ai2thor_v2, whose eval leg has interactions on only ~21% of clips.
    Substituted items no longer correspond to the requested index, so
    `dataset[i]` is NOT deterministic with this on — never enable it on an
    eval/test split. Draws use the stdlib `random` state, seeded per worker by
    the DataLoader (same discipline as the samplers)."""

    def __init__(
        self,
        manifest_path: str | Path,
        index_path: str | Path,
        backend,
        sampler=None,
        transform=None,
        event_index_path: str | Path | None = None,
        event_frac: float | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.index_path = Path(index_path)
        self.backend = backend
        self.sampler = sampler
        self.transform = transform
        self.event_frac = float(event_frac) if event_frac else None

        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        index = json.loads(self.index_path.read_text(encoding="utf-8"))

        manifest_samples = manifest["samples"]
        selected_ids = index["sample_ids"]

        sample_by_id = {}
        for sample in manifest_samples:
            video_id = str(sample["video_id"])
            if video_id in sample_by_id:
                raise ValueError(f"Duplicate video_id found in manifest: {video_id}")
            sample_by_id[video_id] = sample

        self.samples = []
        missing_ids = []
        for video_id in selected_ids:
            if video_id not in sample_by_id:
                missing_ids.append(video_id)
                continue
            self.samples.append(sample_by_id[video_id])

        if missing_ids:
            raise ValueError(
                f"{len(missing_ids)} sample_ids from index were not found in manifest. "
                f"First few missing ids: {missing_ids[:5]}"
            )

        self.event_positions = None
        if self.event_frac:
            assert event_index_path is not None, "event_frac needs event_index_path"
            event_index = json.loads(Path(event_index_path).read_text(encoding="utf-8"))
            pos_by_id = {str(s["video_id"]): i for i, s in enumerate(self.samples)}
            outside = [v for v in event_index["sample_ids"] if v not in pos_by_id]
            if outside:
                raise ValueError(
                    f"{len(outside)} event-index ids are not in this dataset's index — stale "
                    f"artifact? Rebuild it from {self.index_path}. First few: {outside[:5]}")
            self.event_positions = [pos_by_id[v] for v in event_index["sample_ids"]]
            assert self.event_positions, f"event index {event_index_path} is empty"
            # The artifact bakes in the eligibility params it was scanned with;
            # they must match the sampler that will consume the forced draws.
            recorded = event_index.get("event_params")
            if recorded is not None and self.sampler is not None:
                for k, v in recorded.items():
                    have = getattr(self.sampler, k, None)
                    have = sorted(have) if isinstance(have, set) else \
                        list(have) if isinstance(have, tuple) else have
                    if have != v:
                        raise ValueError(
                            f"event index was built with {k}={v} but the sampler has "
                            f"{k}={have} — rebuild {Path(event_index_path).name} with "
                            f"matching flags")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        force_event = None
        if self.event_frac:
            force_event = random.random() < self.event_frac
            if force_event:
                idx = random.choice(self.event_positions)
        sample = self.samples[idx]

        loaded = self.backend.load_clip(sample)
        frames, pose, curr_frame_idx = loaded[:3]
        keys = loaded[3] if len(loaded) > 3 else None   # optional pre-computed retriever keys
        cues = loaded[4] if len(loaded) > 4 else None   # optional per-frame retrieval-cue windows (e.g. audio)
        events = loaded[5] if len(loaded) > 5 else None # optional per-frame CUMULATIVE event counts (e.g. THOR interactions)
        loss_mask = loaded[6] if len(loaded) > 6 else None # optional per-frame (32,32) change mask -> diffusion-loss weight
        interact_state = loaded[7] if len(loaded) > 7 else None # optional per-frame [door_state, 0, onset_pickup, onset_put]
        sampler_spans = loaded[8] if len(loaded) > 8 else None  # optional backend-served (code, start, end) spans for the
                                                                # event-anchored draw (THORDynBackend reveal onsets); None ->
                                                                # spans are derived from the interaction counts below
        agent_state = loaded[9] if len(loaded) > 9 else None    # optional per-frame (T, 6) second-agent detection
                                                                # [vi, dx, dy, vi', dx', dy'] (THORDynBackend `agent_state`)

        num_frames = int(frames.shape[0])
        if self.sampler is not None:
            sampling_spec = self.sampler({"video_id": sample["video_id"], "num_frames": num_frames,
                                          "curr_frame_idx": curr_frame_idx,
                                          "event_spans": (sampler_spans if sampler_spans is not None
                                                          else _event_spans(events)),
                                          "force_event": force_event})
        else:
            sampling_spec = {
                "frame_indices": list(range(num_frames)),
                "context_indices": None,
                "context_mask": None,
                "goal_indices": None,
            }

        frame_indices = sampling_spec["frame_indices"]
        frames = frames[frame_indices]
        pose   = pose[frame_indices]
        if keys is not None:
            keys = keys[frame_indices]            # (len(frame_indices), key_dim), aligned with frames
        if cues is not None:
            cues = cues[frame_indices]            # (len(frame_indices), ...cue shape), aligned with frames
        if loss_mask is not None:
            loss_mask = loss_mask[frame_indices]  # (len(frame_indices), 32, 32), aligned with frames
        if events is not None:
            events = events[frame_indices]        # (len(frame_indices), n_codes) cumulative counts, aligned with frames
        if interact_state is not None:
            interact_state = interact_state[frame_indices]  # (len(frame_indices), 4), aligned with frames
        if agent_state is not None:
            agent_state = agent_state[frame_indices]        # (len(frame_indices), 6), aligned with frames

        if self.transform is not None:
            frames = self.transform(frames)

        return {
            "frames": frames,
            "pose": pose,
            "keys": keys,                         # None or (T, key_dim); split by the formatter
            "cues": cues,                         # None or (T, ...); formatter forwards the CURRENT frame's cue
            "events": events,                     # None or (T, n_codes) cumulative; formatter differences it per goal
            "interact_state": interact_state,     # None or (T, 4) state-label basis; formatter -> action_state per goal
            "loss_mask": loss_mask,               # None or (T, 32, 32) change mask; formatter keeps the GOAL rows
            "agent_state": agent_state,           # None or (T, 6) second-agent detection; formatter splits context / goal
            "event_anchored": bool(sampling_spec.get("event_anchored", False)),   # this draw came from the event-anchored branch
            "pose_fields": self.backend.pose_fields,
            "context_indices": sampling_spec["context_indices"],
            "context_mask": sampling_spec.get("context_mask"),
            "goal_indices": sampling_spec["goal_indices"],
            "video_id": sample["video_id"],
        }


class FormattedVideoDataset(Dataset):
    def __init__(self, base_dataset, formatter):
        self.base_dataset = base_dataset
        self.formatter = formatter

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        return self.formatter(item)