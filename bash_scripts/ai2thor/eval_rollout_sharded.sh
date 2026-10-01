# Per-machine overrides (python env, GPU set): bash_scripts/env.local.sh, see env.local.sh.example
[ -f "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh" ] && . "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh"
pip install -e .

# Sharded AI2-THOR rollout evaluation — one `scripts/world/eval_rollout_thor.py`
# process per GPU, zero inter-process communication (shards write disjoint
# (arm, clip) records into the config's output_dir; shard 0 builds the
# eligibility cache, the others wait for it). Existing records are SKIPPED, so
# re-running this script RESUMES an interrupted battery, and the shard count
# may change between launches. Arms + protocol live in
# configs/eval_rollout_thor.yaml; extra Hydra overrides pass through "$@":
#   bash bash_scripts/ai2thor/eval_rollout_sharded.sh
#   bash bash_scripts/ai2thor/eval_rollout_sharded.sh rollout_stride=4 num_clips=200
# The paired CONTROL battery goes into its OWN output_dir (the eligibility
# cache is protocol-specific; record dirs get an `_ctrl` suffix):
#   bash bash_scripts/ai2thor/eval_rollout_sharded.sh use_ctrl_set=true \
#       output_dir=results/rollout_eval/ai2thor_v3_stride2_ctrl
# NOTE: `ckpt: latest` arms change as their training runs advance — do not
# resume a battery after refreshing checkpoints without changing output_dir
# (pin numbered steps in the yaml for reproducible batteries).
# GPU list: edit GPUS below, or override per-launch without editing:
#   EVAL_GPUS="1 2" bash bash_scripts/ai2thor/eval_rollout_sharded.sh ...
# Monitor:  tail -f results/rollout_eval/logs/thor_shard*.log
#           ls <output_dir>/*/*.npz | wc -l   (expect #eligible x #arms)

IFS=' ' read -r -a GPUS <<< "${EVAL_GPUS:-0 1 2}"
N=${#GPUS[@]}
LOG_DIR=results/rollout_eval/logs
mkdir -p "$LOG_DIR"

PIDS=()
for i in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} \
    python -u scripts/world/eval_rollout_thor.py shard=$i num_shards=$N "$@" \
        > "$LOG_DIR/thor_shard$i.log" 2>&1 &
    PIDS+=($!)
    echo "shard $i/$N -> GPU ${GPUS[$i]} (pid ${PIDS[-1]}, log $LOG_DIR/thor_shard$i.log)"
done

FAIL=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "shard $i done"
    else
        echo "shard $i FAILED — see $LOG_DIR/thor_shard$i.log"
        FAIL=1
    fi
done
exit $FAIL
