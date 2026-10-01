# Per-machine overrides (python env, GPU set): bash_scripts/env.local.sh, see env.local.sh.example
[ -f "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh" ] && . "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh"
# AI2-THOR-dyn evaluation: the BOTH-ENDS painting probe for every arm, one shard per GPU
# (scripts/ai2thor_dyn/probe_paint_both_ends.py; shards split the test episodes, each writes its
# own rows file, so re-running with another shard count is safe). Needs the arms under
# models/ai2thor_dyn/<arm> (checkpoint step CKPT) and models/agent_cell_detector.pt
# (scripts/ai2thor_dyn/train_agent_cell_detector.py). Aggregate afterwards with
#   python scripts/ai2thor_dyn/aggregate_paint_both_ends.py $OUT $EPOCHS
#
#   bash bash_scripts/ai2thor_dyn/probe_both_ends_sharded.sh
#   EVAL_GPUS="0 1" ARMS="temporal B_rag" EPOCHS=2 bash bash_scripts/ai2thor_dyn/probe_both_ends_sharded.sh   # smoke
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
IFS=' ' read -r -a GPUS <<< "${EVAL_GPUS:-0 1 2 3}"
N=${#GPUS[@]}
ARMS=${ARMS:-"temporal worldmem A_nocue B_rag"}
OUT=${OUT:-results/paint_both_ends}
export EPOCHS=${EPOCHS:-10} CKPT=${CKPT:-0500000} EVAL_BS=${EVAL_BS:-8} MIN_CELLS=${MIN_CELLS:-10}
declare -A BUNDLE=([temporal]=temporal [worldmem]=worldmem [A_nocue]=far_meta [B_rag]=far_multicue)
mkdir -p "$OUT/logs"
for arm in $ARMS; do
    run="models/ai2thor_dyn/${BUNDLE[$arm]:-$arm}"
    PIDS=()
    for i in "${!GPUS[@]}"; do
        CUDA_VISIBLE_DEVICES=${GPUS[$i]} SHARD=$i NSHARDS=$N \
            python scripts/ai2thor_dyn/probe_paint_both_ends.py "$run" "$arm" "$OUT" > "$OUT/logs/${arm}_shard$i.log" 2>&1 &
        PIDS+=($!)
    done
    echo "$arm: $N shards on GPUs ${GPUS[*]} (logs $OUT/logs/${arm}_shard*.log)"
    for pid in "${PIDS[@]}"; do wait "$pid"; done
done
echo "done -> $OUT ; aggregate: python scripts/ai2thor_dyn/aggregate_paint_both_ends.py $OUT $EPOCHS"
