import json
import random
from pathlib import Path


VALID_SPLITS = {"train", "val", "test"}


def load_manifest(path):
    return json.loads(Path(path).read_text())


def save_manifest(manifest, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))


def create_splits(
    manifest,
    *,
    seed=42,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    split_by="video_id",
    dataset_name=None,
    preserve_existing=True,
):
    """
    Assign splits to samples in manifest.

    - preserves existing split if present
    - splits remaining samples by group (video_id or group_id)
    """

    samples = manifest["samples"]
    rng = random.Random(seed)

    # Separate samples
    groups = {}
    preserved = {}

    for s in samples:
        if dataset_name is not None and s.get("dataset_name") != dataset_name:
            continue

        vid = s["video_id"]

        if preserve_existing and s.get("split") in VALID_SPLITS:
            preserved[vid] = s["split"]
            continue

        key = s.get(split_by, vid)
        groups.setdefault(key, []).append(s)

    group_keys = list(groups.keys())
    rng.shuffle(group_keys)

    n = len(group_keys)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    assignments = dict(preserved)

    for i, key in enumerate(group_keys):
        if i < n_train:
            split = "train"
        elif i < n_train + n_val:
            split = "val"
        else:
            split = "test"

        for s in groups[key]:
            assignments[s["video_id"]] = split

    return assignments


def create_splits_from_list(
    manifest,
    *,
    train_list=None,
    val_list=None,
    test_list=None,
    dataset_name=None,
):
    """
    Assign splits to samples whose video_id contains a string from the given lists.

    Samples whose video_id doesn't match any list entry are left unassigned.
    """

    # Strip empty strings
    train_list = [s for s in (train_list or []) if s]
    val_list = [s for s in (val_list or []) if s]
    test_list = [s for s in (test_list or []) if s]

    assignments = {}
    for s in manifest["samples"]:
        if dataset_name is not None and s.get("dataset_name") != dataset_name:
            continue

        vid = s["video_id"]
        if any(name in vid for name in train_list):
            assignments[vid] = "train"
        elif any(name in vid for name in val_list):
            assignments[vid] = "val"
        elif any(name in vid for name in test_list):
            assignments[vid] = "test"

    return assignments


def apply_splits(manifest, assignments):
    for s in manifest["samples"]:
        vid = s["video_id"]
        if vid in assignments:
            s["split"] = assignments[vid]
    return manifest