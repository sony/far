# Per-machine overrides (python env, GPU set): bash_scripts/env.local.sh, see env.local.sh.example
[ -f "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh" ] && . "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh"
pip install -e .

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Cache per-frame LONGLIVE retriever keys for loopnav — the prerequisite for
# the LongLive world-model baseline arm. The 400k LongLive-pretrained QueryViT
# (recon + window-temporal-delta + smoothness objective, content mode)
# null-action-encodes every stored latent frame; suffix .keys_longlive.npy
# keeps the sidecars separate from the jepa keys (.keys.npy). pnp_cl consumes
# them via backend key_suffix=.keys_longlive.npy with scoring=visual — its
# F.normalize'd cosine ranking IS the paper's L2-normalized similarity
# retrieval (Alg. 1); temporal diversity comes from chunking (subsuming the
# saturated delta term's anti-duplicate role — see the pretrain launcher).
# Resume-safe (existing key files skipped).
CKPT=models/loopnav/retriever_longlive/checkpoints/0400000.pth.tar

CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,3,4,5,6,7} torchrun --nproc_per_node=$(awk -F, '{print NF}' <<< "${GPUS:-0,1,2,3,4,5,6,7}") --standalone scripts/world/build_retriever_keys.py \
    --source latents \
    --latent-dir datasets/loopnav/latent \
    --ckpt $CKPT \
    --key-suffix .keys_longlive.npy \
    --batch-size 512 \
    --override action_dim=5
