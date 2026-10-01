# Per-machine overrides (python env, GPU set): bash_scripts/env.local.sh, see env.local.sh.example
[ -f "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh" ] && . "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh"
pip install -e .

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# Growable allocator segments — prevents the fragmentation OOM at the 2nd+
# eval (training re-carves the arena between evals; fixed segments then
# cannot serve the large eval-shaped blocks again).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── SoundSpaces corpus (`soundspaces_latent`): WORLDMEM baseline ("WorldMem") ──
# Trains the FOV-overlap / pose-heuristic retrieval arm: identical to
# train_ldwm_temporal.sh except the memory block — the arms must stay matched
# in generator / batch / horizon for the comparison to be a clean factorial.
#   bash bash_scripts/soundspaces/train_ldwm_worldmem.sh          # 4 GPUs, batch 4 x 16 = 64
#   GPUS=4,5,6,7 bash bash_scripts/soundspaces/train_ldwm_worldmem.sh
# Resume: append from_checkpoint=<run_dir>/checkpoints/latest.pth.tar.
#   - fov_half 45/45 = the Habitat 90 deg square RGB sensor (the loopnav
#     defaults 52.5/37.5 describe a different camera);
#   - radius is the visibility-sphere around the GOAL, in METRES here (loopnav
#     used 30 Minecraft units). MP3D rooms are ~5-10 m, so 5.0 keeps the
#     overlap score GRADED: too small a sphere is either fully seen or fully
#     missed (binary, uninformative), too large and every candidate covers it
#     (flat). Tune this first if worldmem looks degenerate — the retrieval demo
#     showed scores saturating at ~1.00 with radius 3.0 on small scenes.
CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,3} torchrun --nproc_per_node=$(awk -F, '{print NF}' <<< "${GPUS:-0,1,2,3}") --standalone scripts/world/train_latent.py \
    run_name=ss_cdit_b_worldmem_posefa_ps2 \
    datasets=[soundspaces_latent] \
    model/generator=cdit_b_2_mod_pose_fa \
    model.generator.context_size=4 \
    model.generator.patch_size=2 \
    model.generator.fov_half_h=45.0 \
    model.generator.fov_half_v=45.0 \
    'model.generator.local_pos_stats=${soundspaces_latent.spec.local_pos_stats}' \
    model.generator.stride_to_seconds=1.0 \
    model/tokenizer=sdxl_vae \
    model/memory=worldmem \
    'model.memory.local_pos_stats=${soundspaces_latent.spec.local_pos_stats}' \
    model.memory.radius=5.0 \
    model.memory.fov_half_h=45.0 \
    model.memory.fov_half_v=45.0 \
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
    wandb.group=soundspaces_latent \
    wandb.tags=[soundspaces_latent] \
