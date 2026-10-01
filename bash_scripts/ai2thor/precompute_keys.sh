# Pre-compute frozen-QueryViT content keys for every AI2-THOR latent episode:
# <episode>/latents<SUFFIX> (T, key_dim) fp32, aligned frame-for-frame with
# latents.npy, loaded at train time via <dataset>.shared.backend.key_suffix.
# fp32 eager, exactly the memory's live key path (keys and live queries MUST
# come from the same arithmetic — no autocast, no compile).
#
#   bash bash_scripts/ai2thor/precompute_keys.sh                      # defaults below
#   CKPT=<run>/checkpoints/0400000.pth.tar SUFFIX=.keys_thor_v2.npy \
#   LATENT_DIR=datasets/ai2thor_v2/latent GPUS=4,5,7 bash bash_scripts/ai2thor/precompute_keys.sh
#   PROCS_PER_GPU=2 ...   # two independent torchrun groups per GPU (disjoint shards)
#
# The query_vit overrides are REQUIRED (see pretrain_retriever.sh): keys pick
# up `null_interact_emb`, which trains away from zero, so the exporter's tower
# must be built with the SAME interact_dim as the checkpoint. Resume-safe
# (existing key files skipped), atomic writes.
# Per-machine overrides (python env, GPU set): bash_scripts/env.local.sh, see env.local.sh.example
[ -f "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh" ] && . "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh"
CKPT=${CKPT:-models/ai2thor_v3/retriever_jepa/checkpoints/0400000.pth.tar}
[ -f "$CKPT" ] || { echo "retriever checkpoint not found: $CKPT (run from the repo root; download it with: bash bash_scripts/download.sh checkpoints ai2thor_v3, or set CKPT=<path>)" >&2; exit 1; }
SUFFIX=${SUFFIX:-.keys_thor_v1.npy}
LATENT_DIR=${LATENT_DIR:-datasets/ai2thor_v3/latent}   # ai2thor_dyn: LATENT_DIR=datasets/ai2thor_dyn/latent
GPUS=${GPUS:-0,1,2}
PROCS_PER_GPU=${PROCS_PER_GPU:-1}
BATCH=${BATCH:-2048}
N=$(awk -F, '{print NF}' <<< "$GPUS")

run_group() {   # $1 = shard index, $2 = rdzv port
  CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$N --standalone --master_port=$2 \
      scripts/world/build_retriever_keys.py \
      --source latents --latent-dir $LATENT_DIR --latent-name latents.npy --ckpt $CKPT \
      --key-suffix $SUFFIX --batch-size $BATCH --prefetch 4 \
      --shard $1 --num-shards $PROCS_PER_GPU \
      --override model.memory.query_vit.img_height=32 \
      --override model.memory.query_vit.img_width=32 \
      --override model.memory.query_vit.patch_size=2 \
      --override model.memory.query_vit.in_chans=4 \
      --override model.memory.query_vit.interact_dim=4 "${@:3}"
}
if [ "$PROCS_PER_GPU" -gt 1 ]; then
  for s in $(seq 0 $((PROCS_PER_GPU - 1))); do run_group $s $((29600 + s)) "$@" & done; wait
else
  run_group 0 29600 "$@"
fi
