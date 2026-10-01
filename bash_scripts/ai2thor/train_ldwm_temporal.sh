# Per-machine overrides (python env, GPU set): bash_scripts/env.local.sh, see env.local.sh.example
[ -f "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh" ] && . "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh"
pip install -e .

# ── AI2-THOR v3 corpus (`ai2thor_v3_latent`): TEMPORAL baseline ("Temporal") ──
# Trains a pose-conditioned CDiT-B world model on SDXL latents whose context is
# the most recent frames of the retrieval pool (model/memory=temporal — no
# learned memory). Sibling launchers, matched in generator / batch / horizon /
# pool / loss mask so the comparison is a clean factorial:
# train_ldwm_worldmem.sh (WorldMem) and train_ldwm_pnp_vmz.sh (FAR -- Multi-Cue).
#   bash bash_scripts/ai2thor/train_ldwm_temporal.sh               # 4 GPUs, batch 4 x 16 = 64
#   GPUS=4,5,6,7 bash bash_scripts/ai2thor/train_ldwm_temporal.sh
# Resume: append from_checkpoint=<run_dir>/checkpoints/latest.pth.tar.
#
# CORPUS: single-room TOUR episodes with k<=5 interaction chains (every act
# episode fires >= 1), plus seed-matched `_ctrl` static siblings held out for
# paired eval. 8.1k episodes / ~9.2M frames (median 1088 per episode), 119
# scenes, scene-disjoint split (95 train / 24 test). Generator rules:
# hidden-inventory carry, ~1.9 s post-interaction dwell (post-change evidence
# exists in the pool), canonical 1.0 m / 25 deg inspection pose (stale/fresh
# views at the SAME pose), deterministic retrace. See
# configs/dataset/ai2thor_v3_latent.yaml.
#
# CONVENTIONS:
#   - tokenizer sdxl_vae (scale 0.13025; latents pre-computed at this scale by
#     build_latent_dataset_ai2thor.py), latent grid 4x32x32 -> generator
#     in_channels/input_size auto-derived by build_ldwm;
#   - fps 10, max_frame_offset 10 in the dataset spec -> stride_to_seconds 1.0;
#     len_traj_pred 32 = a 3.2 s goal window (the agent steps 0.05 m);
#   - fov 90x90 deg (THOR's default fieldOfView with square 256x256 frames)
#     -> half-FOV 45/45, overriding the loopnav-camera defaults;
#   - poses standardized to the z-up frame by THORBackend (always on), with
#     pitch recovered from the stored quaternion (THOR agents pitch up to ~55
#     deg while inspecting stations — it is NOT zero-imputed here);
#   - interact_dim 5: the per-goal multi-hot of open/close/pickup/put occurring
#     in (current, goal] PLUS the absolute door-fraction at the goal (the
#     state label; see LDWMCore._generator_interact), injected through adaLN.
#     The count channels are REQUIRED for this corpus — a pose delta cannot
#     express "the container opens", so without it the eval-leg rollout never
#     renders the reveal and the memory probe (what is inside?) never fires.
#     Zero-init, so it costs nothing when the label is absent or all-zero.
#
# PREREQUISITES: the AI2-THOR v3 train tier (bash bash_scripts/download.sh train ai2thor_v3),
#   which ships the latents, the union loss masks (.change_interact.npy), the manifest,
#   the indices and the event index.

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# Growable allocator segments — prevents the fragmentation OOM at the 2nd+
# eval (training re-carves the arena between evals; fixed segments then
# cannot serve the large eval-shaped blocks again).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,3} torchrun --nproc_per_node=$(awk -F, '{print NF}' <<< "${GPUS:-0,1,2,3}") --standalone scripts/world/train_latent.py \
    run_name=thor3_cdit_b_temporal_posefa \
    datasets=[ai2thor_v3_latent] \
    `# MASK-WEIGHTED DIFFUSION LOSS: the per-goal union change mask (tokens that differ`\
    `# vs the same viewpoint earlier OR vs the pre-swing anchor during interactions)`\
    `# reweights the per-token loss so the changed region carries ~lambda/(1+lambda)`\
    `# ~= 23% of it. Mean-1 weight: loss scale / LR unchanged; lambda=0 = no weighting.`\
    ai2thor_v3_latent.shared.backend.mask_suffix=.change_interact.npy \
    model.mask_loss_lambda=0.3 \
    `# masks above this token fraction are gated off (fridge/cabinet door swings);`\
    `# 0.15 keeps the union's small-container swing rows (11-13% of tokens).`\
    model.mask_max_frac=0.15 \
    `# retrieval pool (frames of history the memory selects from; 400 = 40 s at`\
    `# fps 10). Both samplers matched so in-training eval sees training conditions.`\
    ai2thor_v3_latent.sampler.context_goal.context_pool_size=400 \
    ai2thor_v3_latent.sampler.context_goal_test.context_pool_size=400 \
    `# EVENT-ANCHORED (current, goal) pairs: with prob 0.3 the drawn clip is swapped`\
    `# for one from the event index and the pair is anchored on an EVAL-LEG open/close`\
    `# span (current 3 frames into .. 5 before the animation, goal 0-8 frames after its`\
    `# start). Identical on every AI2-THOR arm; the event_* knobs are checked against`\
    `# the index at startup.`\
    ai2thor_v3_latent.train.base_dataset.event_frac=0.3 \
    ai2thor_v3_latent.sampler.context_goal.event_phase=eval \
    ai2thor_v3_latent.sampler.context_goal.event_codes=null \
    'ai2thor_v3_latent.sampler.context_goal.event_pre=[-3,5]' \
    'ai2thor_v3_latent.sampler.context_goal.event_post=[0,8]' \
    ai2thor_v3_latent.sampler.context_goal.event_goal_anchor=start \
    `# context pool = exploration-phase frames only (before eval_start), for`\
    `# both the train and the in-training-eval samplers — the rollout protocol.`\
    ai2thor_v3_latent.sampler.context_goal.pool_phase=pre_eval \
    ai2thor_v3_latent.sampler.context_goal_test.pool_phase=pre_eval \
    model/generator=cdit_b_2_mod_pose_fa \
    model.generator.context_size=4 \
    model.generator.patch_size=2 \
    model.generator.fov_half_h=45.0 \
    model.generator.fov_half_v=45.0 \
    'model.generator.local_pos_stats=${ai2thor_v3_latent.spec.local_pos_stats}' \
    model.generator.stride_to_seconds=1.0 \
    `# 5 = the 4 interaction-count channels + the absolute door-fraction state label`\
    `# at the goal (LDWMCore._generator_interact); the retriever still gets the 4 counts.`\
    model.generator.interact_dim=5 \
    model/tokenizer=sdxl_vae \
    model/memory=temporal \
    model.condition_on_context_time=true \
    model.condition_on_context_pose=true \
    image_height=256 \
    image_width=256 \
    len_traj_pred=32 \
    action_dim=5 \
    pose_dim=5 \
    batch_size=${BATCH:-16} \
    eval_batch_size=${EVAL_BATCH:-32} \
    lr=1e-4 \
    latest_every=2500 \
    bfloat16=false \
    torch_compile=true \
    wandb.project=FAR-AI2THOR \
    wandb.group=ai2thor_v3_latent \
    wandb.tags=[ai2thor_v3_latent] \
