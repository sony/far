# Per-machine overrides (python env, GPU set): bash_scripts/env.local.sh, see env.local.sh.example
[ -f "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh" ] && . "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh"
pip install -e .

# ── AI2-THOR v3 corpus (`ai2thor_v3_latent`): WORLDMEM baseline ("WorldMem") ──
# Trains the FOV-overlap / pose-heuristic retrieval arm: identical to
# train_ldwm_temporal.sh (see it for the corpus, conventions and
# prerequisites) except the memory block, so the arms stay matched in
# generator / batch / horizon / pool / loss mask.
#   bash bash_scripts/ai2thor/train_ldwm_worldmem.sh               # 4 GPUs, batch 4 x 16 = 64
#   GPUS=4,5,6,7 bash bash_scripts/ai2thor/train_ldwm_worldmem.sh
# Resume: append from_checkpoint=<run_dir>/checkpoints/latest.pth.tar.
#
# WorldMem knobs:
#   - fov_half 45/45 = THOR's default 90 deg fieldOfView with square frames
#     (the loopnav defaults 52.5/37.5 describe a different camera);
#   - radius is the visibility sphere around the GOAL, in METRES. Room scale
#     here is SMALL: the median episode walks a 3.1 m diagonal (p90 extent
#     2.8 x 4.0 m), vs 5-10 m MP3D rooms where soundspaces uses 5.0. A 5 m
#     sphere would enclose an entire iTHOR room, so every candidate frame
#     would "cover" the goal and the score would go flat; 2.0 keeps it local
#     to the goal's vicinity and the overlap GRADED. This is the first knob to
#     tune if worldmem looks degenerate — too small is binary (fully seen /
#     fully missed), too large is uninformative.
#   - single-room episodes make pose proximity a STRONG baseline here (much
#     more so than in the multi-room soundspaces corpus): the discriminating
#     cases are same-pose-different-state, i.e. the container the agent opens
#     on the return leg. That is the comparison this arm exists to lose.

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# Growable allocator segments — prevents the fragmentation OOM at the 2nd+
# eval (training re-carves the arena between evals; fixed segments then
# cannot serve the large eval-shaped blocks again).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,3} torchrun --nproc_per_node=$(awk -F, '{print NF}' <<< "${GPUS:-0,1,2,3}") --standalone scripts/world/train_latent.py \
    run_name=thor3_cdit_b_worldmem_posefa \
    datasets=[ai2thor_v3_latent] \
    `# MASK-WEIGHTED DIFFUSION LOSS: the per-goal union change mask reweights the`\
    `# per-token loss so the changed region carries ~lambda/(1+lambda) ~= 23% of it.`\
    `# Same lambda on every AI2-THOR arm; see train_ldwm_temporal.sh.`\
    ai2thor_v3_latent.shared.backend.mask_suffix=.change_interact.npy \
    model.mask_loss_lambda=0.3 \
    `# masks above this token fraction are gated off (fridge/cabinet door swings);`\
    `# 0.15 keeps the union's small-container swing rows (11-13% of tokens).`\
    model.mask_max_frac=0.15 \
    `# retrieval pool — matched to the temporal arm; see that script.`\
    ai2thor_v3_latent.sampler.context_goal.context_pool_size=400 \
    ai2thor_v3_latent.sampler.context_goal_test.context_pool_size=400 \
    `# EVENT-ANCHORED (current, goal) pairs on eval-leg interaction spans, drawn at`\
    `# an exact 0.3 rate through the event index — identical on every AI2-THOR arm;`\
    `# see train_ldwm_temporal.sh.`\
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
    model/memory=worldmem \
    'model.memory.local_pos_stats=${ai2thor_v3_latent.spec.local_pos_stats}' \
    model.memory.radius=2.0 \
    model.memory.fov_half_h=45.0 \
    model.memory.fov_half_v=45.0 \
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
