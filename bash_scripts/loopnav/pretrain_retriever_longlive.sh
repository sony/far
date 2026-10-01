# Per-machine overrides (python env, GPU set): bash_scripts/env.local.sh, see env.local.sh.example
[ -f "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh" ] && . "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh"
pip install -e .

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# ── LoopNav corpus (`loopnav_latent`): LongLive-RAG RETRIEVER PRE-TRAINING ──
# LongLive-RAG (arXiv:2606.02553) retrieval-embedding pre-training on the
# loopnav ViTVAE latents — the action-AGNOSTIC content-similarity BASELINE
# retriever for train_ldwm_longlive.sh (its keys: precompute_keys_longlive.sh).
#   bash bash_scripts/loopnav/pretrain_retriever_longlive.sh           # 3 GPUs, batch 3 x 32 = 96
#   GPUS=0,1,2,3 bash bash_scripts/loopnav/pretrain_retriever_longlive.sh
#
# Faithful port of the reference repo's `ae/` stage
# (github.com/qixinhu11/LongLive-RAG): latent autoencoder + Window Temporal
# Delta hinge (margin 0.85, window 3) + trajectory-smoothness penalty. The
# ENCODER is our QueryViT in content mode (key_dim embedding — architecture
# parity with the contrastive retriever; checkpoints drop into pnp_cl and
# build_retriever_keys unchanged); only the DECODER is the reference conv
# design. Reference hyperparams otherwise (AdamW 3e-4/1e-4, no warmup; their
# single-GPU batch 128 -> our 32 x 4). Data: CONTIGUOUS 8-frame clips at
# frame_stride=10: the temporal UNIT is 0.5 s at fps 20 — their unit is an AR
# BLOCK spanning seconds, not a 50 ms frame; stride-1 units are so redundant
# that the anti-collapse hinge loses to recon/smooth (observed); the delta
# window presumes adjacent-frame redundancy, and at fps 20 stride-1 frames sit
# 50 ms apart. action_dim=5 keeps the query_vit.* shapes matched to the
# world-model-time instantiation; key_suffix=null: sidecar retriever keys are
# irrelevant here, skip the IO.
#
# The decoder is discarded after training; the LongLive world-model arm uses
# the QueryViT + L2-normalised cosine retrieval with the CURRENT frame as the
# query (paper Alg. 1) — no action conditioning anywhere, by design.
CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2} torchrun --nproc_per_node=$(awk -F, '{print NF}' <<< "${GPUS:-0,1,2}") --standalone scripts/world/train_latent_retriever_longlive.py \
    run_name=retriever_longlive_pretrain \
    datasets=[loopnav_latent] \
    action_dim=5 \
    loopnav_latent.sampler.random_clip.clip_len=8 \
    loopnav_latent.sampler.random_clip.frame_stride=10 \
    loopnav_latent.shared.backend.key_suffix=null \
    +longlive.hidden_dims=[64,128,256] \
    +longlive.delta_weight=1.0 \
    +longlive.delta_margin=0.85 \
    +longlive.delta_window=3 \
    +longlive.smooth_weight=1.0 \
    lr=3e-4 \
    weight_decay=1e-4 \
    batch_size=${BATCH:-32} \
    iterations=400000 \
    warmup_steps=0 \
    log_every=100 \
    ckpt_every=25000 \
    eval_every=5000 \
    wandb.project=FAR-LoopNav \
    wandb.group=loopnav_latent \
    wandb.tags=[loopnav_latent,retriever_pretrain,longlive] \
