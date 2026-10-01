import argparse
from pathlib import Path

from far.data.manifest import build_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a video/latent manifest.")
    parser.add_argument(
        "--roots",
        type=Path,
        nargs="+",
        required=True,
        help="One or more dataset roots to scan. Each root is searched for encoded "
             "videos (by --extensions), jpg-sequence directories, and .npy latent files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output manifest JSON.",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=[".avi"],
        help="Video file extensions to treat as encoded videos.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="default",
        help="Dataset name to store in the manifest.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split name to store in the manifest.",
    )
    parser.add_argument(
        "--jpg-min-frames",
        type=int,
        default=2,
        help="Minimum number of jpg frames for a folder to be considered a sequence.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    manifest = build_manifest(
        roots=args.roots,
        video_extensions=tuple(ext.lower() for ext in args.extensions),
        workers=args.workers,
        dataset_name=args.dataset_name,
        split=args.split,
        jpg_min_frames=args.jpg_min_frames,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(manifest, encoding="utf-8")
    print(f"Saved manifest to {args.output}")


if __name__ == "__main__":
    main()