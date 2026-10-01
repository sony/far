#!/bin/bash
# Download the released checkpoints and datasets from Hugging Face into models/ and datasets/
# (a wrapper around scripts/download_release.py; everything is verified and resumable).
#   bash bash_scripts/download.sh checkpoints                 # every checkpoint bundle + tokenizer / detector (~13 GB)
#   bash bash_scripts/download.sh checkpoints ai2thor_v3      # one corpus's bundles
#   bash bash_scripts/download.sh demo ai2thor_v3             # a corpus's latents: demo | test | train (train includes test)
#   bash bash_scripts/download.sh test loopnav soundspaces_v2 # several corpora at once
# Corpora: loopnav soundspaces_v1 soundspaces_v2 ai2thor_v3 ai2thor_dyn. SoundSpaces is gated:
# accept the Matterport3D Terms of Use on its Hugging Face page and run `hf auth login` first.
# Extra flags after the corpora go to download_release.py (e.g. --verify, --remove-shards, --list).
set -euo pipefail
[ -f "$(dirname "${BASH_SOURCE[0]}")/env.local.sh" ] && . "$(dirname "${BASH_SOURCE[0]}")/env.local.sh"
cd "$(dirname "$(readlink -f "$0")")/.."
WHAT=${1:?usage: bash bash_scripts/download.sh checkpoints|demo|test|train [corpus ...] [--flags]}; shift
CORPORA=(); FLAGS=()
for x in "$@"; do if [[ $x == --* ]]; then FLAGS+=("$x"); else CORPORA+=(--corpus "$x"); fi; done
case "$WHAT" in
  checkpoints)     python scripts/download_release.py "${CORPORA[@]}" "${FLAGS[@]}" ;;
  demo|test|train) [ ${#CORPORA[@]} -gt 0 ] || { echo "name at least one corpus for the $WHAT tier"; exit 1; }
                   python scripts/download_release.py --no-models --data "$WHAT" "${CORPORA[@]}" "${FLAGS[@]}" ;;
  *) echo "unknown target '$WHAT': checkpoints | demo | test | train"; exit 1 ;;
esac
