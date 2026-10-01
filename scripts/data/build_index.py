import argparse
from pathlib import Path

from far.data.index import build_index, load_manifest, make_id_filter, save_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an index from a manifest.")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to manifest JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output index JSON.",
    )
    parser.add_argument(
        "--split",
        type=str,
        nargs="*",
        default=None,
        help="Allowed split values, e.g. --split train or --split train val.",
    )
    parser.add_argument(
        "--id-field",
        type=str,
        default="video_id",
        help="Manifest field whose values are written to sample_ids (default: video_id).",
    )
    parser.add_argument(
        "--filter-by",
        type=str,
        default=None,
        help="Manifest field that --id-prefix / --id-substring filters are applied to. "
             "Defaults to --id-field when not specified.",
    )
    parser.add_argument(
        "--filter-prefix",
        type=str,
        default=None,
        help="Keep only samples whose --filter-by field starts with this prefix.",
    )
    parser.add_argument(
        "--filter-substring",
        type=str,
        default=None,
        help="Keep only samples whose --filter-by field contains this substring.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    manifest = load_manifest(args.manifest)

    id_filter = make_id_filter(
        prefix=args.filter_prefix,
        substring=args.filter_substring,
    )

    effective_filter_by = args.filter_by or args.id_field
    rule_parts = []
    if args.filter_prefix is not None:
        rule_parts.append(f"{effective_filter_by}.startswith('{args.filter_prefix}')")
    if args.filter_substring is not None:
        rule_parts.append(f"'{args.filter_substring}' in {effective_filter_by}")
    rule_description = " and ".join(rule_parts) if rule_parts else None

    index_artifact = build_index(
        manifest=manifest,
        manifest_path=args.manifest,
        split=args.split,
        id_filter=id_filter,
        rule_description=rule_description,
        id_field=args.id_field,
        filter_by=args.filter_by,
    )

    save_index(index_artifact, args.output)
    print(f"Saved index to {args.output}")
    print(f"Selected {index_artifact['num_selected_samples']} samples")


if __name__ == "__main__":
    main()