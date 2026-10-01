# Per-machine overrides (python env, GPU set): bash_scripts/env.local.sh, see env.local.sh.example
[ -f "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh" ] && . "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh"
CKPT=${CKPT:-models/loopnav/retriever_jepa/checkpoints/0400000.pth.tar}   # pre-trained retriever this arm starts from
[ -f "$CKPT" ] || { echo "retriever checkpoint not found: $CKPT (run from the repo root; download it with: bash bash_scripts/download.sh checkpoints loopnav, or set CKPT=<path>)" >&2; exit 1; }
pip install -e .

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# ── LoopNav corpus (`loopnav_latent`): FAR -- Multi-Cue, FIXED-FUSION ablation (no learned gate) ──
# Ablation of train_ldwm_pnp.sh: scoring=abl+visual+meta_z — the SAME
# per-query standardized channels as visual+meta_z, but fused with a FIXED
# neutral weight λ~ ≡ 0.5 (the learned gate's zero-init point) for the WHOLE
# run. No gate — this mode creates no fusion parameters, so the arm is
# "Multi-Cue where the gate never learns":
#   gap(Multi-Cue − this arm) = value of the LEARNED confidence weighting,
#                               standardization held fixed;
#   gap(this arm − Meta)      = value of the visual channel under neutral fusion.
# Everything else is IDENTICAL to train_ldwm_pnp.sh (same jepa tower, adapter
# encoder, keys cache, chunk/set layout, RAG settings, effective batch
# 4 x bs32 = 128; see it for the prerequisites) — only run_name and scoring differ.
#   bash bash_scripts/loopnav/train_ldwm_pnp_ablation.sh               # 4 GPUs, batch 4 x 32 = 128
#   GPUS=4,5,6,7 bash bash_scripts/loopnav/train_ldwm_pnp_ablation.sh
# Resume: append from_checkpoint=<run_dir>/checkpoints/latest.pth.tar.
CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,3} torchrun --nproc_per_node=$(awk -F, '{print NF}' <<< "${GPUS:-0,1,2,3}") --standalone scripts/world/train_latent.py \
    run_name=nwm_cdit_b_pnp_ablz_adapter_posefa_l1 \
    datasets=[loopnav_latent] \
    loopnav_latent.shared.backend.standard_frame=true \
    loopnav_latent.shared.backend.key_suffix=.keys_jepa.npy \
    model/generator=cdit_b_2_mod_pose_fa \
    model.generator.context_size=4 \
    'model.generator.local_pos_stats=${loopnav_latent.spec.local_pos_stats}' \
    model/tokenizer=vit_vae \
    model/memory=pnp_cl \
    model.memory.ckpt_path=$CKPT \
    model.memory.scoring=abl+visual+meta_z \
    model.memory.encoder_mode=adapter \
    'model.memory.local_pos_stats=${loopnav_latent.spec.local_pos_stats}' \
    model.memory.num_sets=5 \
    model.memory.chunk_size=10 \
    model.retriever_loss_weight=1.0 \
    rag_every=20 \
    rag_noise_levels=4 \
    local_rag=true \
    local_rag_weight=1.0 \
    model.condition_on_context_time=true \
    model.condition_on_context_pose=true \
    image_height=360 \
    image_width=640 \
    action_dim=5 \
    pose_dim=5 \
    batch_size=${BATCH:-32} \
    eval_batch_size=${EVAL_BATCH:-64} \
    lr=1e-4 \
    lr_mem=1e-4 \
    bfloat16=false \
    torch_compile=true \
    wandb.project=FAR-LoopNav \
    wandb.group=loopnav_latent \
    wandb.tags=[loopnav_latent] \
