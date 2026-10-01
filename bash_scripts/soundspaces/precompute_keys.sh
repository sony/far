# Per-machine overrides (python env, GPU set): bash_scripts/env.local.sh, see env.local.sh.example
[ -f "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh" ] && . "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh"
pip install -e .

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Cache per-frame AUDIO retriever keys for SoundSpaces — the prerequisite for
# the FAR audio arm (train_ldwm_pnp_cue_metaz.sh;
# cue mode NEVER live-encodes the pool; the backend reads
# `latents.keys_audio.npy` in each episode dir via key_suffix=".keys_audio.npy").
#   bash bash_scripts/soundspaces/precompute_keys.sh
#   GPUS=0,1 CKPT=<retriever ckpt> bash bash_scripts/soundspaces/precompute_keys.sh
#
# The 400k audio-jepa QueryViT (64x48 binaural windows, patch 4, 4 ch)
# null-action-encodes the SAME causal windows training sees — the script's
# `--source audio` mode goes through SoundSpacesBackend(audio_mode="frames"),
# so the slicing/normalization/padding contract (audio_feat_params.json) has
# a single source of truth and the cached keys live exactly in the query
# space. Output: `latents.keys_audio.npy` (T, 256) fp32 per episode dir.
# Resume-safe (existing key files are skipped) and atomic (tmp + rename),
# so preemption / re-runs are free.
CKPT=models/soundspaces_v1/retriever_audio/checkpoints/0400000.pth.tar

CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,3} torchrun --nproc_per_node=$(awk -F, '{print NF}' <<< "${GPUS:-0,1,2,3}") --standalone scripts/world/build_retriever_keys.py \
    --source audio \
    --latent-dir datasets/soundspaces/latent \
    --ckpt $CKPT \
    --key-suffix .keys_audio.npy \
    --batch-size 512 \
    --override model.memory.query_vit.img_height=64 \
    --override model.memory.query_vit.img_width=48 \
    --override model.memory.query_vit.patch_size=4 \
    --override model.memory.query_vit.in_chans=4
