# Per-machine overrides (python env, GPU set): bash_scripts/env.local.sh, see env.local.sh.example
[ -f "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh" ] && . "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh"
pip install -e .

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# ── LoopNav corpus (`loopnav_latent`): FAR RETRIEVER PRE-TRAINING ──
# Contrastive (InfoNCE) pre-training of the FAR retriever's query encoder
# (QueryViT) only — builds ONLY the QueryViT (no generator / tokenizer /
# diffusion / compressor); the loss is computed by train_latent_retriever.py
# (ContrastivePretrainModule). The resulting checkpoint loads into the FAR
# world-model arms (train_ldwm_pnp*.sh) as the retriever init via
#   model.memory.ckpt_path=<this run>/checkpoints/latest.pth.tar
# (PnPRetrieval.load_weights loads the query_vit.* weights; the rest starts
# fresh), and its cached keys are exported FROM THE SAME checkpoint by
# precompute_keys.sh. Entry config: configs/world_pretrain.yaml (CL knobs under
# `contrastive.*`).
#   bash bash_scripts/loopnav/pretrain_retriever.sh                    # 2 GPUs, batch 2 x 32 = 64
#   GPUS=0,1,2,3 bash bash_scripts/loopnav/pretrain_retriever.sh
CUDA_VISIBLE_DEVICES=${GPUS:-0,1} torchrun --nproc_per_node=$(awk -F, '{print NF}' <<< "${GPUS:-0,1}") --standalone scripts/world/train_latent_retriever.py \
    run_name=retriever_jepa_pretrain_std \
    datasets=[loopnav_latent] \
    loopnav_latent.shared.backend.standard_frame=true \
    loopnav_latent.sampler.context_goal.context_pool_size=200 \
    goals_per_obs=1 \
    model/memory=pnp_cl \
    model.memory.chunk_size=10 \
    contrastive.positive_pair_type=[jepa] \
    contrastive.num_negatives=24 \
    contrastive.num_positives=8 \
    contrastive.local_pos_stats=\${loopnav_latent.spec.local_pos_stats} \
    image_height=360 \
    image_width=640 \
    action_dim=5 \
    pose_dim=5 \
    batch_size=${BATCH:-32} \
    lr=2e-4 \
    iterations=400000 \
    warmup_steps=5000 \
    log_every=100 \
    ckpt_every=50000 \
    eval_every=5000 \
    bfloat16=false \
    wandb.project=FAR-LoopNav \
    wandb.group=loopnav_latent \
    wandb.tags=[loopnav_latent,retriever_pretrain] \
