# Per-machine overrides (python env, GPU set): bash_scripts/env.local.sh, see env.local.sh.example
[ -f "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh" ] && . "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh"
pip install -e .

# SoundSpaces v2: sharded per-frame rollout evaluation — one `scripts/world/eval_rollout.py`
# process per GPU, zero inter-process communication (shards write disjoint
# (arm, clip) records into the config's output_dir). Existing records are
# SKIPPED, so re-running this script RESUMES an interrupted battery, and the
# shard count may change between launches. Arms + protocol live in
# configs/eval_rollout_soundspaces_v2.yaml (shared knobs in configs/eval_rollout.yaml); extra Hydra overrides pass through "$@":
#   bash bash_scripts/soundspaces_v2/eval_rollout_sharded.sh
#   bash bash_scripts/soundspaces_v2/eval_rollout_sharded.sh rollout_stride=4 num_clips=200
# GPU list: edit GPUS below, or override per-launch without editing:
#   EVAL_GPUS="1 2" bash bash_scripts/soundspaces_v2/eval_rollout_sharded.sh ...
# Monitor:  tail -f results/rollout_eval/logs/eval_rollout_soundspaces_v2/shard*.log
#           ls <output_dir>/*/*.npz | wc -l   (expect num_clips x #arms)

IFS=' ' read -r -a GPUS <<< "${EVAL_GPUS:-0 1 2}"
N=${#GPUS[@]}
CONFIG=eval_rollout_soundspaces_v2
LOG_DIR=results/rollout_eval/logs/$CONFIG
mkdir -p "$LOG_DIR"

PIDS=()
for i in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} \
    python -u scripts/world/eval_rollout.py --config-name $CONFIG shard=$i num_shards=$N "$@" \
        > "$LOG_DIR/shard$i.log" 2>&1 &
    PIDS+=($!)
    echo "shard $i/$N -> GPU ${GPUS[$i]} (pid ${PIDS[-1]}, log $LOG_DIR/shard$i.log)"
done

FAIL=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "shard $i done"
    else
        echo "shard $i FAILED — see $LOG_DIR/shard$i.log"
        FAIL=1
    fi
done
exit $FAIL
