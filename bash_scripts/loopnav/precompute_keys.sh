# Pre-compute frozen-QueryViT content keys for every loopnav latent frame.
# Saves <clip>.keys.npy (T, key_dim) next to each <clip>.npy latent, aligned
# frame-for-frame. Loaded by LatentLoopNavBackend at train time so the
# retriever's per-frame key encodes become cached lookups. Re-run whenever the
# retriever checkpoint changes (keys are tied to specific QueryViT weights).
# Each rank processes a disjoint subset of latent files (files[rank::world_size]).
#   CKPT=models/loopnav/retriever_jepa/checkpoints/0400000.pth.tar bash bash_scripts/loopnav/precompute_keys.sh
CKPT=${CKPT:?set CKPT to the retriever (QueryViT) checkpoint the keys are tied to}
# Per-machine overrides (python env, GPU set): bash_scripts/env.local.sh, see env.local.sh.example
[ -f "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh" ] && . "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh"
CUDA_VISIBLE_DEVICES=${GPUS:-0} torchrun --nproc_per_node=$(awk -F, '{print NF}' <<< "${GPUS:-0}") --standalone scripts/world/build_retriever_keys.py \
    --latent-dir datasets/loopnav/latent \
    --ckpt $CKPT \
    --batch-size 1024
