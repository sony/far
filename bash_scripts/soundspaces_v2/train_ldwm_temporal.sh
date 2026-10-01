# Per-machine overrides (python env, GPU set): bash_scripts/env.local.sh, see env.local.sh.example
[ -f "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh" ] && . "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh"
pip install -e .

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# Growable allocator segments — prevents the fragmentation OOM at the 2nd+
# eval (training re-carves the arena between evals; fixed segments then
# cannot serve the large eval-shaped blocks again).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── V2 CORPUS ─ soundspaces_v2_latent: periodic 360° scans (scan_A +
# region-change scan_mid_K) close v1's observability gaps (ABA coverage
# 0.43→0.90, unsupported runs 9.1s→0.9s median). Same episode format, same
# audio contract (v1 normalization stats), same 400k retriever + v2 keys.
# Trajectory-PAIRED with v1 (same seeds/waypoints) — cross-corpus
# comparisons at matched clips are valid.

# ── SoundSpaces v2 corpus (`soundspaces_v2_latent`): TEMPORAL baseline ("Temporal") ──
# Trains a pose-conditioned CDiT-B world model on SDXL latents whose context
# is the most recent frames of the retrieval pool (model/memory=temporal — no
# learned memory). Siblings, matched in generator / batch / horizon / pool:
# train_ldwm_worldmem.sh (WorldMem) and the FAR arms (train_ldwm_pnp_*.sh).
#   bash bash_scripts/soundspaces_v2/train_ldwm_temporal.sh          # 4 GPUs, batch 4 x 16 = 64
#   GPUS=4,5,6,7 bash bash_scripts/soundspaces_v2/train_ldwm_temporal.sh
# Resume: append from_checkpoint=<run_dir>/checkpoints/latest.pth.tar.
# Conventions:
#   - tokenizer sdxl_vae (scale 0.13025; latents pre-computed at this scale),
#     latent grid 32x32x4 -> generator in_channels/input_size auto-derived;
#   - fps 10: len_traj_pred 128 = a 12.8 s goal window; max_frame_offset 10
#     in the dataset spec keeps stride_to_seconds = 1.0;
#   - fov 90x90 deg (Habitat square RGB sensor) -> half-FOV 45/45, overriding
#     the loopnav-camera defaults in the generator config;
#   - poses standardized to the z-up frame by the backend (always on).
CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,3} torchrun --nproc_per_node=$(awk -F, '{print NF}' <<< "${GPUS:-0,1,2,3}") --standalone scripts/world/train_latent.py \
    run_name=ssv2_cdit_b_temporal_posefa_ps2 \
    datasets=[soundspaces_v2_latent] \
    `# retrieval pool (frames of history the memory selects from; 200 = 20 s at`\
    `# fps 10). Both samplers matched so in-training eval sees training conditions.`\
    soundspaces_v2_latent.sampler.context_goal.context_pool_size=200 \
    soundspaces_v2_latent.sampler.context_goal_test.context_pool_size=200 \
    model/generator=cdit_b_2_mod_pose_fa \
    model.generator.context_size=4 \
    model.generator.patch_size=2 \
    model.generator.fov_half_h=45.0 \
    model.generator.fov_half_v=45.0 \
    'model.generator.local_pos_stats=${soundspaces_v2_latent.spec.local_pos_stats}' \
    model.generator.stride_to_seconds=1.0 \
    model/tokenizer=sdxl_vae \
    model/memory=temporal \
    model.condition_on_context_time=true \
    model.condition_on_context_pose=true \
    image_height=256 \
    image_width=256 \
    len_traj_pred=128 \
    action_dim=5 \
    pose_dim=5 \
    batch_size=${BATCH:-16} \
    eval_batch_size=${EVAL_BATCH:-32} \
    lr=1e-4 \
    bfloat16=false \
    torch_compile=true \
    wandb.project=FAR-SoundSpaces \
    wandb.group=soundspaces_v2_latent \
    wandb.tags=[soundspaces_v2_latent] \
