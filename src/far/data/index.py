from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


def load_manifest(manifest_path: str | Path) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    if "samples" not in payload or not isinstance(payload["samples"], list):
        raise ValueError("Manifest must contain a list under key 'samples'.")

    return payload


def save_index(index_artifact: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(index_artifact, indent=2), encoding="utf-8")


def build_index(
    manifest: dict[str, Any],
    *,
    manifest_path: str | Path | None = None,
    split: list[str] | None = None,
    id_filter: Callable[[str], bool] | None = None,
    rule_description: str | None = None,
    id_field: str = "video_id",
    filter_by: str | None = None,
) -> dict[str, Any]:
    """
    Args:
        id_field:  Manifest field whose values are written to sample_ids.
        filter_by: Manifest field that --id-prefix / --id-substring filters are
                   applied to. Defaults to id_field when not specified.
    """
    effective_filter_by = filter_by if filter_by is not None else id_field
    selected_ids: list[str] = []

    for sample in manifest["samples"]:
        if id_field not in sample:
            raise KeyError(f"Field '{id_field}' not found in manifest sample: {sample}")
        sample_id = str(sample[id_field])

        if split is not None:
            sample_split = str(sample.get("split", ""))
            if sample_split not in split:
                continue

        if id_filter is not None:
            if effective_filter_by not in sample:
                raise KeyError(f"Field '{effective_filter_by}' not found in manifest sample: {sample}")
            if not id_filter(str(sample[effective_filter_by])):
                continue

        selected_ids.append(sample_id)

    return {
        "version": "v1",
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "num_selected_samples": len(selected_ids),
        "filters": {
            "split": split,
            "id_field": id_field,
            "filter_by": effective_filter_by,
            "rule_description": rule_description,
        },
        "sample_ids": selected_ids,
    }


def build_index_from_list(
    manifest: dict[str, Any],
    match_list: list[str],
    *,
    manifest_path: str | Path | None = None,
    split: list[str] | None = None,
    rule_description: str | None = None,
) -> dict[str, Any]:
    """Select samples whose video_id contains any string in match_list.

    One entry is added per matching (match_string, sample) pair, so a sample
    may appear multiple times if its video_id matches multiple strings.
    """
    match_list = [s for s in match_list if s]
    selected_ids: list[str] = []

    eligible = [
        str(s["video_id"]) for s in manifest["samples"]
        if split is None or str(s.get("split", "")) in split
    ]

    for match_str in match_list:
        for video_id in eligible:
            if match_str in video_id:
                selected_ids.append(video_id)

    return {
        "version": "v1",
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "num_selected_samples": len(selected_ids),
        "filters": {
            "split": split,
            "match_list": match_list,
            "rule_description": rule_description,
        },
        "sample_ids": selected_ids,
    }


def make_id_filter(
    *,
    prefix: str | None = None,
    substring: str | None = None,
) -> Callable[[str], bool] | None:
    if prefix is None and substring is None:
        return None

    def _filter(sample_id: str) -> bool:
        if prefix is not None and not sample_id.startswith(prefix):
            return False
        if substring is not None and substring not in sample_id:
            return False
        return True

    return _filter