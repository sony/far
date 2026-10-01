import argparse

from far.data.splits import (
    load_manifest,
    save_manifest,
    create_splits,
    apply_splits,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--split-by", default="video_id")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)

    parser.add_argument("--overwrite-existing", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    manifest = load_manifest(args.manifest)

    assignments = create_splits(
        manifest,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_by=args.split_by,
        dataset_name=args.dataset_name,
        preserve_existing=not args.overwrite_existing,
    )

    manifest = apply_splits(manifest, assignments)
    save_manifest(manifest, args.output)

    print("Saved:", args.output)


if __name__ == "__main__":
    main()