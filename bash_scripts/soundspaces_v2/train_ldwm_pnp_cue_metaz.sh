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

# ── SoundSpaces v2 corpus (`soundspaces_v2_latent`): FAR -- Multi-Cue (headline arm) ──
# Trains the FAR retriever (model/memory=pnp_cl) with AUDIO-CUE queries fused
# with metadata through the z-gate (scoring=visual+meta_z, query_source=cue).
# Siblings, matched in generator / batch / horizon / pool: train_ldwm_temporal.sh
# (Temporal), train_ldwm_worldmem.sh (WorldMem) and train_ldwm_pnp_meta_only.sh
# (FAR -- Meta).
#   - the query encoder reads the CURRENT STEP AUDIO WINDOW (4, 64, 48) instead
#     of a latent frame — the memory stays modality-agnostic ("cue");
#   - the pool side is scored against CACHED audio keys (key_suffix), never
#     live-encoded: cue mode asserts context_keys is present;
#   - the "visual" channel of the z-gate is therefore the AUDIO cosine (the
#     name is historical); the gate learns per-query audio-vs-geometry trust,
#     which on soundspaces has real in-distribution variation to learn from
#     (source audibility changes with room/occlusion) unlike loopnav.
#
# PREREQUISITES:
#   1. audio retriever checkpoint: the jepa pretrain at step 400k (pinned
#      below; bash_scripts/soundspaces/pretrain_audio_retriever.sh trains one);
#   2. cached per-frame audio keys `latents.keys_audio.npy` in every episode
#      dir, exported FROM THE SAME checkpoint:
#        bash bash_scripts/soundspaces/precompute_keys.sh
#      Keys and ckpt_path must stay in lockstep — mismatched key/query spaces
#      degrade retrieval silently.
#   bash bash_scripts/soundspaces_v2/train_ldwm_pnp_cue_metaz.sh          # 4 GPUs, batch 4 x 16 = 64
#   GPUS=4,5,6,7 bash bash_scripts/soundspaces_v2/train_ldwm_pnp_cue_metaz.sh
# Resume: append from_checkpoint=<run_dir>/checkpoints/latest.pth.tar.
CKPT=${CKPT:-models/soundspaces_v1/retriever_audio/checkpoints/0400000.pth.tar}
[ -f "$CKPT" ] || { echo "retriever checkpoint not found: $CKPT (run from the repo root; download it with: bash bash_scripts/download.sh checkpoints soundspaces_v1, or set CKPT=<path>)" >&2; exit 1; }
CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,3} torchrun --nproc_per_node=$(awk -F, '{print NF}' <<< "${GPUS:-0,1,2,3}") --standalone scripts/world/train_latent.py \
    run_name=ssv2_cdit_b_pnp_amz_posefa_ps2 \
    datasets=[soundspaces_v2_latent] \
    `# retrieval pool (frames of history the memory selects from; 200 = 20 s at`\
    `# fps 10) -> floor((pool-1)/chunk) = 19 candidate chunks vs num_sets*num_memories`\
    `# = 15 selected; raise the pool, not the chunk granularity, for more ranking`\
    `# pressure. Both samplers matched so in-training eval sees training conditions.`\
    soundspaces_v2_latent.sampler.context_goal.context_pool_size=200 \
    soundspaces_v2_latent.sampler.context_goal_test.context_pool_size=200 \
    soundspaces_v2_latent.shared.backend.audio_mode=cue \
    soundspaces_v2_latent.shared.backend.key_suffix=.keys_audio.npy \
    model/generator=cdit_b_2_mod_pose_fa \
    model.generator.context_size=4 \
    model.generator.patch_size=2 \
    model.generator.fov_half_h=45.0 \
    model.generator.fov_half_v=45.0 \
    'model.generator.local_pos_stats=${soundspaces_v2_latent.spec.local_pos_stats}' \
    model.generator.stride_to_seconds=1.0 \
    model/tokenizer=sdxl_vae \
    model/memory=pnp_cl \
    model.memory.ckpt_path=$CKPT \
    model.memory.scoring=visual+meta_z \
    model.memory.query_source=cue \
    model.memory.encoder_mode=adapter \
    model.memory.query_vit.img_height=64 \
    model.memory.query_vit.img_width=48 \
    model.memory.query_vit.patch_size=4 \
    model.memory.query_vit.in_chans=4 \
    'model.memory.local_pos_stats=${soundspaces_v2_latent.spec.local_pos_stats}' \
    model.memory.stride_to_seconds=1.0 \
    model.memory.num_sets=5 \
    model.memory.chunk_size=10 \
    model.retriever_loss_weight=1.0 \
    rag_every=20 \
    rag_noise_levels=4 \
    local_rag=true \
    local_rag_weight=1.0 \
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
    lr_mem=1e-4 \
    bfloat16=false \
    torch_compile=true \
    wandb.project=FAR-SoundSpaces \
    wandb.group=soundspaces_v2_latent \
    wandb.tags=[soundspaces_v2_latent] \
