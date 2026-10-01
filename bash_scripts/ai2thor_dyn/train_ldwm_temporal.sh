# Per-machine overrides (python env, GPU set): bash_scripts/env.local.sh, see env.local.sh.example
[ -f "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh" ] && . "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh"
pip install -e .

# ── AI2-THOR dynamic-agent (door-corridor) corpus (`ai2thor_dyn_latent`): TEMPORAL baseline ("Temporal") ──
# Trains the recency-retrieval arm: a pose-conditioned CDiT-B world model on
# SDXL latents with the agent-state predictor (cdit_b_2_agent_fa), whose
# context is the most recent frames of the retrieval pool (model/memory=
# temporal — no learned memory). Sibling launchers, matched in generator /
# batch / horizon / pool / loss mask so the comparison is a clean factorial:
# train_ldwm_worldmem.sh (WorldMem), train_ldwm_pnp_meta_B_rag.sh (FAR -- Meta,
# the headline arm) and train_ldwm_pnp_meta_A_nocue.sh (its no-cue ablation).
#   bash bash_scripts/ai2thor_dyn/train_ldwm_temporal.sh             # 4 GPUs, batch 4 x 16 = 64
#   GPUS=4,5,6,7 bash bash_scripts/ai2thor_dyn/train_ldwm_temporal.sh
# Resume: append from_checkpoint=<run_dir>/checkpoints/latest.pth.tar.
#
# CORPUS: the observer (A1) watches a corridor through a doorway from inside a
# room; a second agent (A2) crosses the doorway K times (alternating ends,
# EVERY crossing observed), stops at its final end, then A1 walks out
# (`eval_start` = the go_door leg) and looks at both corridor ends — the model
# must render A2 at the end it ARRIVED at, composed onto the EMPTY end views it
# saw in the prologue (pro_look_A/B are pose-identical to the reveal legs
# look_A/B). 1,002 `_act` episodes (median 1,488 frames, eval_start at a
# median of 1,388) over 7 scenes, seed carve: 800 train / 202 test, every scene
# in both splits. See configs/dataset/ai2thor_dyn_latent.yaml.
#
# Mirrors bash_scripts/ai2thor/train_ldwm_temporal.sh (the AI2-THOR v3 arm) in
# generator / horizon / conventions so the two corpora's arms stay comparable.
# What differs, and why:
#   - mask_suffix .agent.npy: A2's exact silhouette on the latent grid (built
#     by scripts/data/build_agent_masks_dyn.py from masks/%06d.png; all-zero
#     when A2 is off screen) replaces the TOUR
#     change/union masks. ~3.4% of tokens when present -> lambda 5.0 puts ~84%
#     of the loss on the agent instead of its area share, so the small
#     doorway-crossing / corridor-end agent cannot wash out under plain MSE.
#     mask_max_frac 0.15 is kept for parity but never gates here (A2 at its
#     closest is ~10% of the grid).
#   - event anchoring is REVEAL-anchored rather than interaction-anchored:
#     this corpus has no interactions (actions.npz absent -> THORBackend
#     zero-fills the counts / state labels), so THORDynBackend serves the
#     look_A / look_B reveal onsets as the event spans instead (see the
#     REVEAL-ANCHORED PAIRS note in the command).
#   - interact_dim 5 is KEPT although every label is zero here: the channel is
#     zero-init and costs nothing when the label is absent, and it keeps the
#     generator structurally identical to the AI2-THOR v3 checkpoints
#     (cross-corpus fine-tuning / shared eval code stay drop-in).
#   - the AGENT-STATE observation + predictor (below) exist only on this corpus.
#
# CONVENTIONS (unchanged from the AI2-THOR v3 arms):
#   - tokenizer sdxl_vae (scale 0.13025; latents pre-computed at this scale by
#     build_latent_dataset_ai2thor.py), latent grid 4x32x32;
#   - fps 10, max_frame_offset 10 in the dataset spec -> stride_to_seconds 1.0;
#   - fov 90x90 deg (THOR's default fieldOfView with square 256x256 frames)
#     -> half-FOV 45/45;
#   - poses standardized to the z-up frame by THORBackend (pitch recovered
#     from the corpus's yaw_pitch_quat).
# What recency can and cannot do here: the pool is the pre-eval span, whose
# tail is the end of the watch leg — so this arm sees the LAST crossing only
# when A2's final pulse sits within the retrieved recency window (short
# `post` / dwell); long post-crossing dwells push it out of reach.
#
# PREREQUISITES:
#   python scripts/data/build_latent_dataset_ai2thor.py \
#       --dataset-root /path/to/door_corridor_raw_episodes \
#       --output-dir datasets/ai2thor_dyn/latent \
#       --sidecars poses.npz,agent2.npz,meta.json          # latents
#   python scripts/data/build_agent_masks_dyn.py             # latents.agent.npy
#   python scripts/data/build_manifest_ai2thor.py \
#       --latent-root datasets/ai2thor_dyn/latent \
#       --manifest-out results/manifests/ai2thor_dyn/ai2thor_dyn_latent.json \
#       --index-dir results/indices/ai2thor_dyn --dataset-name ai2thor_dyn \
#       --index-prefix ai2thor_dyn_latent --kind act \
#       --split-field split_seed --allow-scene-overlap
#   python scripts/data/build_event_index_ai2thor.py --spans reveal \
#       --event-goal-anchor end --event-pre 0 10 --event-post 8 30 \
#       --reveal-start-offset 12 ...                        # reveal index (train split)
# BATCH SIZE: 16 per rank (the run name's bs is per rank), global 4 x 16 = 64
# — matched to the FAR arms (context 4 / pool 400 / batch 16 / mask lambda 5.0)
# so the baseline-vs-FAR factorial differs only in the memory module. Peak
# VRAM per rank at context 4 / pool 400 is ~1.3 GB + 3.4 GB x batch_size
# (fp32, torch.compile; ~56 GB at 16, plus ~1 GB of DDP buckets); the
# in-training eval (eval_batch_size 32 x 4 goals, bf16 no_grad on the EMA,
# incl. decode + DreamSim/LPIPS) peaks at ~13 GB at pool 400.

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
    run_name=thordyn_cdit_b_temporal_posefa_rev_agent \
    datasets=[ai2thor_dyn_latent] \
    `# MASK-WEIGHTED DIFFUSION LOSS on A2's silhouette (see header). Mean-1 weight,`\
    `# so the loss scale and LR are unchanged; same lambda on every arm.`\
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
    `# retrieval pool (frames of history the memory selects from; 400 = 40 s at fps`\
    `# 10, drawn uniformly from the pre-eval span, which it SUBSAMPLES: the size sets`\
    `# the density, not the reach; A2 is on screen for ~16% of it). Matched to the`\
    `# FAR arms; both samplers matched so in-training eval sees training conditions.`\
    ai2thor_dyn_latent.sampler.context_goal.context_pool_size=400 \
    ai2thor_dyn_latent.sampler.context_goal_test.context_pool_size=400 \
    `# context pool = OBSERVATION history only (before eval_start = go_door), for`\
    `# both samplers — a reveal must be answered from memory of the crossings, never`\
    `# from a fresher eval-leg look at the corridor ends. The rollout protocol.`\
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
    `# v3 arms — all-zero on this corpus (see header).`\
    model.generator.interact_dim=5 \
    model/tokenizer=sdxl_vae \
    model/memory=temporal \
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
