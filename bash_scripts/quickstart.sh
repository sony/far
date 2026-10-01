#!/bin/bash
# One-command demo: FAR vs WorldMem on an AI2-THOR test clip -> results/quickstart/clip<k>/
# (downloads ~2.4 GB on the first run; needs a CUDA GPU with ~14 GB free). Options pass through
# to scripts/quickstart.py: --clip, --arms, --max-steps, --out, --device, --no-download.
#   bash bash_scripts/quickstart.sh
#   bash bash_scripts/quickstart.sh --arms far_multicue temporal worldmem --clip 341
set -euo pipefail
[ -f "$(dirname "${BASH_SOURCE[0]}")/env.local.sh" ] && . "$(dirname "${BASH_SOURCE[0]}")/env.local.sh"
cd "$(dirname "$(readlink -f "$0")")/.."
python scripts/quickstart.py "$@"
