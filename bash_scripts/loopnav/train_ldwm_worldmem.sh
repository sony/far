# Per-machine overrides (python env, GPU set): bash_scripts/env.local.sh, see env.local.sh.example
[ -f "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh" ] && . "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh"
pip install -e .

# ── LoopNav corpus (`loopnav_latent`): WORLDMEM baseline ("WorldMem") ──
# Trains the FOV-overlap / pose-heuristic retrieval arm: identical to
# train_ldwm_temporal.sh except the memory block (model/memory=worldmem with
# the config's loopnav camera / radius defaults), so the arms stay matched in
# generator / batch / horizon / pool.
#   bash bash_scripts/loopnav/train_ldwm_worldmem.sh                   # 4 GPUs, batch 4 x 32 = 128
#   GPUS=4,5,6,7 bash bash_scripts/loopnav/train_ldwm_worldmem.sh
# Resume: append from_checkpoint=<run_dir>/checkpoints/latest.pth.tar.

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,3} torchrun --nproc_per_node=$(awk -F, '{print NF}' <<< "${GPUS:-0,1,2,3}") --standalone scripts/world/train_latent.py \
    run_name=nwm_cdit_b_worldmem_posefa_ddim20 \
    datasets=[loopnav_latent] \
    loopnav_latent.shared.backend.standard_frame=true \
    model/generator=cdit_b_2_mod_pose_fa \
    model.generator.context_size=4 \
    'model.generator.local_pos_stats=${loopnav_latent.spec.local_pos_stats}' \
    model/tokenizer=vit_vae \
    model/memory=worldmem \
    'model.memory.local_pos_stats=${loopnav_latent.spec.local_pos_stats}' \
    model.condition_on_context_time=true \
    model.condition_on_context_pose=true \
    image_height=360 \
    image_width=640 \
    action_dim=5 \
    pose_dim=5 \
    batch_size=${BATCH:-32} \
    eval_batch_size=${EVAL_BATCH:-64} \
    lr=1e-4 \
    bfloat16=false \
    torch_compile=true \
    wandb.project=FAR-LoopNav \
    wandb.group=loopnav_latent \
    wandb.tags=[loopnav_latent] \
