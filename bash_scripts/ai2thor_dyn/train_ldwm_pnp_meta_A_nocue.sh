# Per-machine overrides (python env, GPU set): bash_scripts/env.local.sh, see env.local.sh.example
[ -f "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh" ] && . "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh"
pip install -e .

# ── AI2-THOR dynamic-agent (door-corridor) corpus (`ai2thor_dyn_latent`): FAR -- Meta, NO-CUE ablation (arm A) ──
# Identical to train_ldwm_pnp_meta_B_rag.sh (the headline FAR -- Meta arm:
# metadata-only scoring, chunk 30, pool 400, K 5, context 4, batch 16, RAG on
# the anchored draws only at prior temperature 0.1) except
# `model.memory.metadata_agent_state=false`: the cross-scorer ranks by geometry
# + recency alone and never sees the second-agent detections (the generator's
# predictor still does). A vs B therefore isolates whether a geometry + time
# scorer can find the last crossing from the reward alone, or whether it needs
# the cue. See train_ldwm_temporal.sh for the corpus and prerequisites.
#   bash bash_scripts/ai2thor_dyn/train_ldwm_pnp_meta_A_nocue.sh       # 4 GPUs, batch 4 x 16 = 64
#   GPUS=4,5,6,7 bash bash_scripts/ai2thor_dyn/train_ldwm_pnp_meta_A_nocue.sh
# Resume: append from_checkpoint=<run_dir>/checkpoints/latest.pth.tar.
#
# `meta` = the closed-form cross-scorer alone. Per (frame, goal) it reads the
# goal's position in the frame's egocentric coords, relative yaw/pitch,
# distance, Δt-to-goal, Δt-to-current (Fourier-featurised, 3-layer MLP), plus
# the AGENT-STATE CUE (`metadata_agent_state`): the frame's second-agent
# detection [vi, dx, dy, vi', dx', dy'], its pair velocity rotated into the
# current frame, and the current frame's own detection — a zero-init block, so
# the ranking starts at the geometry / recency prior and learns "an agent was
# seen here, heading that way" from the EMDR2 reward (diffusion loss + the
# predictor's loss, the term that separates the last crossing from the one
# before). No visual channel, no z-gate, no query adapter (encoder_mode
# frozen): the vision tower (QueryViT) still encodes the per-goal query — an
# unused cosine — and its cached pool keys are kept ONLY so the pool is never
# live-encoded (400 x QueryViT per step). It ranks nothing. Watch `rank_corr`
# (prior vs reward over the K sets), `local_rank_corr` and
# eval/agent_auc_memreq.
#
# PREREQUISITES (compute-only under scoring=meta — see above):
#   1. retriever checkpoint: the object-centric fine-tune at step 12.5k (pinned
#      below; bash_scripts/ai2thor_dyn/pretrain_retriever_hard.sh trains one);
#   2. cached per-frame keys `latents$KEYS` (default .keys_obj_v1.npy) in every
#      episode dir, exported FROM THAT SAME checkpoint:
#        python scripts/world/build_retriever_keys.py --source latents \
#            --latent-dir datasets/ai2thor_dyn/latent --ckpt $CKPT \
#            --key-suffix .keys_obj_v1.npy --latent-name latents.npy \
#            --override model.memory.query_vit.img_height=32 [... as precompute_keys.sh]
#      (`--latent-name latents.npy` is REQUIRED here: the agent-mask sidecar
#      `latents.agent.npy` sits next to the latents and the default *.npy
#      discovery would sweep it in.) Keys and ckpt_path must stay in
#      lockstep — mismatched key/query spaces degrade retrieval silently.
#   3. interact_dim=4 on the query_vit is REQUIRED, not optional: the tower
#      (and its keys) carry `null_interact_emb` on the key path. Building the
#      memory at the config default 0 drops those params on load and puts
#      live queries in a DIFFERENT space from the cached keys — silently.
# Retriever init + cached pool keys (MUST come from the same tower):
#   default  = the object-centric fine-tune (models/ai2thor_dyn/retriever_object,
#              step 12.5k) + `.keys_obj_v1.npy`;
#   jepa baseline (agent-blind): CKPT=models/ai2thor_v3/retriever_jepa/checkpoints/0400000.pth.tar KEYS=.keys_thor_v1.npy bash <this>

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# Growable allocator segments — prevents the fragmentation OOM at the 2nd+
# eval (training re-carves the arena between evals; fixed segments then
# cannot serve the large eval-shaped blocks again).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT=${CKPT:-models/ai2thor_dyn/retriever_object/checkpoints/0012500.pth.tar}
[ -f "$CKPT" ] || { echo "retriever checkpoint not found: $CKPT (run from the repo root; download it with: bash bash_scripts/download.sh checkpoints ai2thor_dyn, or set CKPT=<path>)" >&2; exit 1; }
KEYS=${KEYS:-.keys_obj_v1.npy}

# AGENT-STATE knobs: the detection threshold (px; corpus visible-frame
# percentiles 5/10/25/50 = 24/82/369/722 -> 100 keeps 89%, glimpses through the
# door aperture become misses), the predictor's loss weight in the training
# loss, and its weight inside every EMDR2 candidate reward.
AGENT_PX_MIN=${AGENT_PX_MIN:-100}
AGENT_LOSS_W=${AGENT_LOSS_W:-1.0}
AGENT_REWARD_W=${AGENT_REWARD_W:-1.0}

CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,3} torchrun --nproc_per_node=$(awk -F, '{print NF}' <<< "${GPUS:-0,1,2,3}") --standalone scripts/world/train_latent.py \
    `# _rev_agent = REVEAL-ANCHORED sampling (35%) + the AGENT-STATE observation`\
    run_name=thordyn_cdit_b_pnp_meta_posefa_A_nocue_ctx4_rev_agent \
    datasets=[ai2thor_dyn_latent] \
    ai2thor_dyn_latent.shared.backend.key_suffix=$KEYS \
    `# MASK-WEIGHTED DIFFUSION LOSS on A2's silhouette — see the temporal launcher.`\
    `# LAMBDA 5.0: w = (1 + lam*m/frac)/(1+lam), so A2's region carries (frac+lam)/`\
    `# (1+lam) = 84% of the loss (A2 is ~3.4% of the grid); goals with no A2 on screen`\
    `# are unweighted. Mean-1 weight, so the loss scale and LR are unchanged.`\
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
    model.mask_loss_lambda=5.0 \
    model.mask_max_frac=0.15 \
    `# RAG REWARD weighting (the per-set and within-chunk rewards that rank the`\
    `# candidate chunks), set explicitly to the same 5.0: candidate sets are ranked by`\
    `# how well they render A2's region, not by overall scene match.`\
    model.rag_mask_loss_lambda=5.0 \
    `# retrieval pool — matched to the temporal / worldmem arms.`\
    ai2thor_dyn_latent.sampler.context_goal.context_pool_size=400 \
    ai2thor_dyn_latent.sampler.context_goal_test.context_pool_size=400 \
    `# context pool = observation history only (before eval_start), for both`\
    `# the train and the in-training-eval samplers — the rollout protocol.`\
    ai2thor_dyn_latent.sampler.context_goal.pool_phase=pre_eval \
    ai2thor_dyn_latent.sampler.context_goal_test.pool_phase=pre_eval \
    `# the pose_fa DiT + the agent-state predictor (AgentStateCDiT; flat DiT fields,`\
    `# so the model.generator.* overrides below apply unchanged)`\
    model/generator=cdit_b_2_agent_fa \
    `# CONTEXT 4 (3 memory slots) = the baseline arms'; 3 slots leave no room for`\
    `# coverage by luck, so the retrieval claim is strict. The RAG candidates' reach`\
    `# (K 5 x chunk 30 ~ 48 s) does not depend on it.`\
    model.generator.context_size=4 \
    model.generator.patch_size=2 \
    model.generator.fov_half_h=45.0 \
    model.generator.fov_half_v=45.0 \
    'model.generator.local_pos_stats=${ai2thor_dyn_latent.spec.local_pos_stats}' \
    model.generator.stride_to_seconds=1.0 \
    `# 5 = the 4 count channels + the door-fraction state label, as on the AI2-THOR`\
    `# v3 arms — all-zero on this corpus (see the temporal launcher). The RETRIEVER`\
    `# still receives the plain 4-dim counts it was pretrained on.`\
    model.generator.interact_dim=5 \
    model/tokenizer=sdxl_vae \
    model/memory=pnp_cl \
    model.memory.ckpt_path=$CKPT \
    model.memory.scoring=meta \
    `# ARM A: the scorer WITHOUT the agent-state cue — geometry + recency only`\
    `# (the generator's predictor still gets the observation; only the RETRIEVER`\
    `# is blind to it). A vs B = what the cue buys the retriever.`\
    model.memory.metadata_agent_state=false \
    `# no query adapter: the visual channel is unused under scoring=meta`\
    model.memory.encoder_mode=frozen \
    model.memory.query_vit.img_height=32 \
    model.memory.query_vit.img_width=32 \
    model.memory.query_vit.patch_size=2 \
    model.memory.query_vit.in_chans=4 \
    model.memory.query_vit.interact_dim=4 \
    'model.memory.local_pos_stats=${ai2thor_dyn_latent.spec.local_pos_stats}' \
    model.memory.stride_to_seconds=1.0 \
    `# 3-layer metadata cross-scorer MLP (default 2): one extra hidden layer for`\
    `# sharper geometry pair functions; zero-init last layer keeps the step-0 no-op.`\
    model.memory.metadata_layers=3 \
    `# PRIOR temperature 0.1: the tanh-bounded meta logit lives in [-1,1]; at tau 1 the`\
    `# distilled prior can only match a near-one-hot posterior by driving the pre-`\
    `# activations into saturation (cue block untrainable), at 0.1 the same logit gap`\
    `# gives e^20, so the prior sharpens inside the tanh's linear range.`\
    model.memory.softmax_temperature=0.1 \
    model.memory.num_sets=5 \
    model.memory.chunk_size=30 \
    model.retriever_loss_weight=1.0 \
    `# RAG every 7 steps on the ANCHORED draws only: the uniform draws' reward (the`\
    `# predictor's loss with the answer already in the current frame) teaches the`\
    `# scorer to avoid sightings; scoring only the ~35% anchored rows costs ~1/3 per`\
    `# step, hence the 3x cadence at the same budget.`\
    rag_every=7 \
    rag_anchored_only=true \
    rag_noise_levels=4 \
    local_rag=true \
    local_rag_weight=1.0 \
    model.condition_on_context_time=true \
    model.condition_on_context_pose=true \
    image_height=256 \
    image_width=256 \
    `# HORIZON 32 frames (3.2 s) for the UNIFORM draw = the short, rollout-like`\
    `# pairs; the reveal-anchored draw has its own horizon (event_len_traj_pred above).`\
    len_traj_pred=32 \
    action_dim=5 \
    pose_dim=5 \
    batch_size=${BATCH:-16} \
    eval_batch_size=${EVAL_BATCH:-32} \
    lr=1e-4 \
    lr_mem=1e-4 \
    latest_every=2500 \
    bfloat16=false \
    torch_compile=true \
    wandb.project=FAR-AI2THOR \
    wandb.group=ai2thor_dyn_latent \
    wandb.tags=[ai2thor_dyn_latent] \
