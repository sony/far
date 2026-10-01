# Per-machine overrides (python env, GPU set): bash_scripts/env.local.sh, see env.local.sh.example
[ -f "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh" ] && . "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh"
pip install -e .

# ── AI2-THOR dynamic-agent (door-corridor) corpus (`ai2thor_dyn_latent`): WORLDMEM baseline ("WorldMem") ──
# Trains the FOV-overlap / pose-heuristic retrieval arm: identical to
# train_ldwm_temporal.sh (see it for the corpus, conventions, prerequisites
# and batch sizing) except the memory block, so the arms stay matched in
# generator / batch / horizon / pool / loss mask. Mirrors
# bash_scripts/ai2thor/train_ldwm_worldmem.sh (the AI2-THOR v3 arm).
#   bash bash_scripts/ai2thor_dyn/train_ldwm_worldmem.sh             # 4 GPUs, batch 4 x 16 = 64
#   GPUS=4,5,6,7 bash bash_scripts/ai2thor_dyn/train_ldwm_worldmem.sh
# Resume: append from_checkpoint=<run_dir>/checkpoints/latest.pth.tar.
#
# WorldMem knobs:
#   - fov_half 45/45 = THOR's default 90 deg fieldOfView with square frames;
#   - radius is the visibility sphere around the GOAL, in METRES. 2.0 (the
#     AI2-THOR v3 value) keeps the sphere local to the goal's vicinity so the
#     overlap stays GRADED; it is the first knob to tune if worldmem looks
#     degenerate — too small is binary (fully seen / fully missed), too large
#     encloses the room and goes flat.
#   - What this arm is expected to do here: the reveal query (A1 outside,
#     looking down the corridor at end A/B) pose-matches the PROLOGUE frames
#     pro_look_A/B EXACTLY — which show that end EMPTY. FOV overlap therefore
#     hands the generator the right appearance and the wrong state, and has
#     no channel to the doorway-crossing frames (inside the room, facing the
#     aperture) that carry A2's direction. That is the comparison this arm
#     exists to lose: same viewpoint, different state.

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# Growable allocator segments — prevents the fragmentation OOM at the 2nd+
# eval (training re-carves the arena between evals; fixed segments then
# cannot serve the large eval-shaped blocks again).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# AGENT-STATE knobs: the detection threshold (px; corpus visible-frame
# percentiles 5/10/25/50 = 24/82/369/722 -> 100 keeps 89%, glimpses through the
# door aperture become misses), the predictor's loss weight in the training
# loss, and its weight inside every EMDR2 candidate reward.
AGENT_PX_MIN=${AGENT_PX_MIN:-100}
AGENT_LOSS_W=${AGENT_LOSS_W:-1.0}
AGENT_REWARD_W=${AGENT_REWARD_W:-1.0}

CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,3} torchrun --nproc_per_node=$(awk -F, '{print NF}' <<< "${GPUS:-0,1,2,3}") --standalone scripts/world/train_latent.py \
    `# _rev_agent = REVEAL-ANCHORED sampling (35%) + the AGENT-STATE observation`\
    run_name=thordyn_cdit_b_worldmem_posefa_rev_agent \
    datasets=[ai2thor_dyn_latent] \
    `# MASK-WEIGHTED DIFFUSION LOSS on A2's silhouette — see the temporal`\
    `# launcher. Same lambda on every arm so the factorial stays matched.`\
    ai2thor_dyn_latent.shared.backend.mask_suffix=.agent.npy \
    `# AGENT-STATE OBSERVATION: every frame carries A2's detection [vi, dx, dy] in its`\
    `# own egocentric floor frame plus the same triple 1.5 s earlier (velocity); vi =`\
    `# px >= AGENT_PX_MIN. The generator's predictor turns the retrieved slots' states`\
    `# into the goal's state (BCE + position loss, in the training loss and in every`\
    `# EMDR2 candidate reward) and the DiT is conditioned on it. Same on every arm.`\
    ai2thor_dyn_latent.shared.backend.agent_state=true \
    ai2thor_dyn_latent.shared.backend.agent_state_pair_s=1.5 \
    ai2thor_dyn_latent.shared.backend.agent_state_px_min=$AGENT_PX_MIN \
    model.agent_loss_weight=$AGENT_LOSS_W \
    model.agent_reward_weight=$AGENT_REWARD_W \
    `# REVEAL-ANCHORED PAIRS (THORDynBackend serves the two reveal onsets as spans):`\
    `# 35% of train draws are 'current 0.5-2 s BEFORE an end comes into view, goal`\
    `# 0.8-3 s after', ~50/50 across the ends — the pair that must be answered from`\
    `# memory. Exact rate via the reveal index; the in-training eval draws the same`\
    `# pairs at event_bias_p so eval/mask_* measure reveals. 0 / null = plain uniform.`\
    ai2thor_dyn_latent.train.base_dataset.event_frac=0.35 \
    ai2thor_dyn_latent.sampler.context_goal_test.event_bias_p=0.35 \
    `# anchored-draw horizon (both samplers): the B-end pairs span look_A and the`\
    `# current sits before the door-aperture glimpse; the uniform draw keeps len_traj_pred.`\
    ai2thor_dyn_latent.sampler.context_goal.event_len_traj_pred=135 \
    ai2thor_dyn_latent.sampler.context_goal_test.event_len_traj_pred=135 \
    `# LAMBDA 5.0 = the FAR arms'. w = (1+lam*m/frac)/(1+lam), so A2's silhouette`\
    `# carries (frac+lam)/(1+lam) of the loss: 84% at 5.0 (A2 is ~3.4% of the grid).`\
    model.mask_loss_lambda=5.0 \
    model.mask_max_frac=0.15 \
    `# retrieval pool — matched to the temporal arm; see that script. The pool`\
    `# SUBSAMPLES the pre-eval span, so its size sets the density, not the reach.`\
    ai2thor_dyn_latent.sampler.context_goal.context_pool_size=400 \
    ai2thor_dyn_latent.sampler.context_goal_test.context_pool_size=400 \
    `# context pool = observation history only (before eval_start), for both`\
    `# the train and the in-training-eval samplers — the rollout protocol.`\
    ai2thor_dyn_latent.sampler.context_goal.pool_phase=pre_eval \
    ai2thor_dyn_latent.sampler.context_goal_test.pool_phase=pre_eval \
    `# the pose_fa DiT + the agent-state predictor (AgentStateCDiT; flat DiT fields,`\
    `# so the model.generator.* overrides below apply unchanged)`\
    model/generator=cdit_b_2_agent_fa \
    `# CONTEXT 4 (3 memories) = the FAR arms', so the retrieval claim is about WHICH`\
    `# 3 frames each rule picks from the same pool.`\
    model.generator.context_size=4 \
    model.generator.patch_size=2 \
    model.generator.fov_half_h=45.0 \
    model.generator.fov_half_v=45.0 \
    'model.generator.local_pos_stats=${ai2thor_dyn_latent.spec.local_pos_stats}' \
    model.generator.stride_to_seconds=1.0 \
    `# 5 = the 4 count channels + the door-fraction state label, as on the AI2-THOR`\
    `# v3 arms — all-zero on this corpus (see the temporal launcher).`\
    model.generator.interact_dim=5 \
    model/tokenizer=sdxl_vae \
    model/memory=worldmem \
    'model.memory.local_pos_stats=${ai2thor_dyn_latent.spec.local_pos_stats}' \
    model.memory.radius=2.0 \
    model.memory.fov_half_h=45.0 \
    model.memory.fov_half_v=45.0 \
    model.condition_on_context_time=true \
    model.condition_on_context_pose=true \
    image_height=256 \
    image_width=256 \
    `# HORIZON 32 frames (3.2 s) for the UNIFORM draw = the short, rollout-like`\
    `# pairs; the reveal-anchored draw has its own horizon (event_len_traj_pred above).`\
    len_traj_pred=32 \
    action_dim=5 \
    pose_dim=5 \
    `# 16 per rank = the FAR arms', so the arms are matched on samples/step (global`\
    `# 64) as well as on LR and warmup.`\
    batch_size=${BATCH:-16} \
    eval_batch_size=${EVAL_BATCH:-32} \
    lr=1e-4 \
    latest_every=2500 \
    bfloat16=false \
    torch_compile=true \
    wandb.project=FAR-AI2THOR \
    wandb.group=ai2thor_dyn_latent \
    wandb.tags=[ai2thor_dyn_latent] \
