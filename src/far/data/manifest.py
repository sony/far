from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence
from typing import Any
import random

try:
    import av  # PyAV
except ImportError:
    av = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg"}


@dataclass
class ManifestSample:
    video_id: str
    source_type: str          # "avi" | "jpg_seq" | "latent"
    path: str
    dataset_name: str
    split: str


def build_manifest(
    roots: Sequence[Path],
    video_extensions: tuple[str, ...] = (".avi",),
    workers: int = 8,
    dataset_name: str = "default",
    split: str = "train",
    jpg_min_frames: int = 2,
) -> str:
    """
    Scans roots and returns a JSON manifest string.

    Design notes:
    - Encoded videos (.avi by default) are discovered by file extension.
    - JPG sequences are discovered as folders containing image files.
    - Precomputed latents (.npy + sidecar .json) are discovered as latent files.
    """
    roots = [Path(root).resolve() for root in roots]
    video_extensions = tuple(ext.lower() for ext in video_extensions)

    tasks: list[tuple[str, str, str, str]] = []

    for root in roots:
        if not root.exists():
            continue

        for video_path in _iter_video_files(root, video_extensions):
            tasks.append(("video", str(video_path), dataset_name, split))

        for seq_dir in _iter_jpg_sequence_dirs(root, jpg_min_frames):
            tasks.append(("jpg_seq", str(seq_dir), dataset_name, split))

        for latent_path in _iter_latent_files(root):
            tasks.append(("latent", str(latent_path), dataset_name, split))

    # Deduplicate paths in case nested roots overlap.
    deduped = {}
    for task in tasks:
        deduped[(task[0], task[1])] = task
    tasks = list(deduped.values())

    if workers <= 1:
        samples = [_probe_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            samples = list(ex.map(_probe_task, tasks, chunksize=32))

    valid_samples = [sample for sample in samples if sample is not None]
    valid_samples.sort(key=lambda x: (x.dataset_name, x.split, x.video_id, x.path))

    manifest = {
        "version": "v1",
        "num_samples": len(valid_samples),
        "samples": [asdict(sample) for sample in valid_samples],
    }
    return json.dumps(manifest, indent=2)


def _iter_video_files(root: Path, video_extensions: tuple[str, ...]) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in video_extensions:
            yield path


def _iter_latent_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.npy"):
        if path.is_file():
            yield path


def _iter_jpg_sequence_dirs(root: Path, jpg_min_frames: int) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        try:
            entries = list(path.iterdir())
        except OSError:
            continue
        num_images = sum(
            1 for p in entries if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if num_images >= jpg_min_frames:
            yield path


def _probe_task(task: tuple[str, str, str, str]) -> ManifestSample | None:
    source_kind, path_str, dataset_name, split = task
    path = Path(path_str)
    try:
        if source_kind == "video":
            return _probe_video_file(path, dataset_name, split)
        if source_kind == "jpg_seq":
            return _probe_jpg_sequence(path, dataset_name, split)
        if source_kind == "latent":
            return ManifestSample(
                video_id=_make_video_id(path),
                source_type="latent",
                path=str(path),
                dataset_name=dataset_name,
                split=split,
            )
    except Exception:
        return None
    return None


def _probe_video_file(path: Path, dataset_name: str, split: str) -> ManifestSample | None:
    if not path.is_file():
        return None
    return ManifestSample(
        video_id=_make_video_id(path),
        source_type=path.suffix.lower().lstrip("."),
        path=str(path),
        dataset_name=dataset_name,
        split=split,
    )


def _probe_jpg_sequence(path: Path, dataset_name: str, split: str) -> ManifestSample | None:
    frame_files = [
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not frame_files:
        return None
    return ManifestSample(
        video_id=_make_video_id(path),
        source_type="jpg_seq",
        path=str(path),
        dataset_name=dataset_name,
        split=split,
    )


def _make_video_id(path: Path) -> str:
    return path.as_posix()