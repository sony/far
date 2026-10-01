from __future__ import annotations

import json
import math
from pathlib import Path
import webdataset as wds


def chunked(xs, n):
    buf = []
    for x in xs:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


def collect_avi_trajectory_pairs(
    root_dir: str,
    avi_ext: str = ".avi",
    traj_ext: str = ".json",
    recursive: bool = True,
    skip_missing: bool = True,
) -> list[tuple[str, str]]:
    """
    Scan a directory and build a list of (avi_path, trajectory_path).

    Matching rule:
    - for each *.avi file, look for a trajectory file with the same stem
      in the same directory, e.g.:
        clip_001.avi <-> clip_001.json
    """
    root = Path(root_dir).resolve()

    if recursive:
        avi_files = sorted(root.rglob(f"*{avi_ext}"))
    else:
        avi_files = sorted(root.glob(f"*{avi_ext}"))

    pairs: list[tuple[str, str]] = []

    for avi_path in avi_files:
        if not avi_path.is_file():
            continue

        traj_path = avi_path.with_suffix(traj_ext)

        if traj_path.is_file():
            pairs.append((str(avi_path), str(traj_path)))
        elif not skip_missing:
            raise FileNotFoundError(
                f"Missing trajectory file for video: {avi_path} "
                f"(expected {traj_path})"
            )

    return pairs


def build_webdataset_shards(
    pairs: list[tuple[str, str]],
    out_dir: str,
    split: str = "train",
    samples_per_shard: int = 1000,
    traj_field_name: str = "json",
) -> None:
    """
    Write pairs into WebDataset tar shards.

    Each sample becomes:
      <key>.avi
      <key>.<traj_field_name>
      <key>.meta.json
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    nshards = math.ceil(len(pairs) / samples_per_shard) if pairs else 0
    print(f"Writing {len(pairs)} samples into {nshards} shard(s) under {out_path}")

    for shard_idx, shard_pairs in enumerate(chunked(pairs, samples_per_shard)):
        shard_name = f"{split}-{shard_idx:06d}.tar"
        shard_path = out_path / shard_name

        with wds.TarWriter(str(shard_path)) as sink:
            for sample_idx, (avi_path, traj_path) in enumerate(shard_pairs):
                avi_path = Path(avi_path)
                traj_path = Path(traj_path)

                key = f"{split}_{shard_idx:06d}_{sample_idx:06d}"

                sample = {
                    "__key__": key,
                    "avi": avi_path.read_bytes(),
                    traj_field_name: traj_path.read_bytes(),
                    "meta.json": json.dumps(
                        {
                            "orig_avi_path": str(avi_path),
                            "orig_traj_path": str(traj_path),
                            "split": split,
                        }
                    ).encode("utf-8"),
                }
                sink.write(sample)

        print(f"Wrote {shard_path}")


if __name__ == "__main__":
    root_dir = "datasets/loopnav/ABA"
    out_dir = "datasets/loopnav_shards"

    pairs = collect_avi_trajectory_pairs(
        root_dir=root_dir,
        avi_ext=".avi",
        traj_ext=".json",   # change to ".npy", ".npz", etc. if needed
        recursive=True,
        skip_missing=True,
    )

    print(f"Collected {len(pairs)} pairs")
    print("First 5 pairs:")
    for pair in pairs[:5]:
        print(pair)

    build_webdataset_shards(
        pairs=pairs,
        out_dir=out_dir,
        split="val",
        samples_per_shard=100,
        traj_field_name="json",   # should match traj_ext content type
    )