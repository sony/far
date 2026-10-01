# Per-machine overrides (python env, GPU set): bash_scripts/env.local.sh, see env.local.sh.example
[ -f "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh" ] && . "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh"
CKPT=${CKPT:-models/loopnav/retriever_jepa/checkpoints/0400000.pth.tar}   # pre-trained retriever this arm starts from
[ -f "$CKPT" ] || { echo "retriever checkpoint not found: $CKPT (run from the repo root; download it with: bash bash_scripts/download.sh checkpoints loopnav, or set CKPT=<path>)" >&2; exit 1; }
pip install -e .

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# ── LoopNav corpus (`loopnav_latent`): FAR -- Meta ──
# Trains the FAR retriever with METADATA-ONLY scoring (scoring=meta): chunks
# are ranked by the closed-form relative-geometry cross-scorer logit ALONE (no
# visual term, no visual_scale parameter), and the retriever loss (EMDR^2)
# trains ONLY the geometry head. The scoring-channel factorial:
#   FAR -- Visual     scoring=visual         encoder_mode=adapter   train_ldwm_pnp_visual_only.sh
#   FAR -- Meta       scoring=meta           encoder_mode=frozen    <- THIS RUN
#   FAR -- Multi-Cue  scoring=visual+meta_z  encoder_mode=adapter   train_ldwm_pnp.sh
# Tower init (the jepa pretrain, step 400k), cached keys, chunk/set layout,
# RAG settings and batch mirror train_ldwm_pnp.sh (see it for the
# prerequisites) so the arms stay samples-matched; keys/tower never enter the
# meta ranking and are kept for provenance parity and telemetry.
#   bash bash_scripts/loopnav/train_ldwm_pnp_meta_only.sh              # 4 GPUs, batch 4 x 32 = 128
#   GPUS=4,5,6,7 bash bash_scripts/loopnav/train_ldwm_pnp_meta_only.sh
# Resume: append from_checkpoint=<run_dir>/checkpoints/latest.pth.tar.
CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,3} torchrun --nproc_per_node=$(awk -F, '{print NF}' <<< "${GPUS:-0,1,2,3}") --standalone scripts/world/train_latent.py \
    run_name=nwm_cdit_b_pnp_meta_only_posefa_l1 \
    datasets=[loopnav_latent] \
    loopnav_latent.shared.backend.standard_frame=true \
    loopnav_latent.shared.backend.key_suffix=.keys_jepa.npy \
    model/generator=cdit_b_2_mod_pose_fa \
    model.generator.context_size=4 \
    'model.generator.local_pos_stats=${loopnav_latent.spec.local_pos_stats}' \
    model/tokenizer=vit_vae \
    model/memory=pnp_cl \
    model.memory.ckpt_path=$CKPT \
    model.memory.scoring=meta \
    model.memory.encoder_mode=frozen \
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
