# Per-machine overrides (python env, GPU set): bash_scripts/env.local.sh, see env.local.sh.example
[ -f "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh" ] && . "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh"
pip install -e .

# ── AI2-THOR v3 corpus (`ai2thor_v3_latent`): FAR -- Multi-Cue (headline arm) ──
# Trains the FAR retriever (model/memory=pnp_cl) scoring on BOTH channels
# through the z-gate (scoring=visual+meta_z) — the fused sibling of
# train_ldwm_temporal.sh (Temporal) / train_ldwm_worldmem.sh (WorldMem), matched
# to them in generator / batch / horizon / pool / loss mask so the three form a
# clean factorial. See train_ldwm_temporal.sh for the corpus and conventions.
#   bash bash_scripts/ai2thor/train_ldwm_pnp_vmz.sh                # 4 GPUs, batch 4 x 16 = 64
#   GPUS=4,5,6,7 bash bash_scripts/ai2thor/train_ldwm_pnp_vmz.sh
# Resume: append from_checkpoint=<run_dir>/checkpoints/latest.pth.tar.
#
# `vmz` = visual + meta_z, as in the loopnav sibling
# (bash_scripts/loopnav/train_ldwm_pnp.sh, run_name ..._pnp_vmz_...). This arm
# has NO cue channel: the soundspaces fused arm queries with an AUDIO CUE
# window (query_source=cue), but AI2-THOR has no cue modality, so the query
# here is the current FRAME latent and `query_source` stays at its default
# `frame`. The two fused channels are:
#   - VISION  — QueryViT key/query cosine over SDXL latents, the encoder
#     pre-trained by bash_scripts/ai2thor/pretrain_retriever.sh;
#   - METADATA — the closed-form relative-geometry cross-scorer (goal position
#     in each frame's egocentric coords, relative yaw/pitch, distance, Δt).
# The z-gate standardizes each channel PER QUERY over the pool's valid frames
# and mixes them as (1−λ~)·z_meta + λ~·z_visual, λ from one shared linear gate
# on DETACHED features, squashed to λ~ = ε + (1−2ε)λ (gate_eps 0.05) so neither
# channel is ever fully switched off. Zero-init gate ⇒ λ~ = 0.5 at step 0, i.e.
# an equal-weight ensemble of the two rankings.
#
# Why this corpus should reward the gate: single-room episodes make pose
# proximity a STRONG baseline, so the metadata channel is genuinely informative
# — but the discriminating cases are same-pose-different-state (the container
# the agent opened on the outbound leg), where geometry alone cannot tell the
# stale view from the fresh one and the visual channel must carry the decision.
# That is in-distribution variation for the gate to actually learn from.
#
# PREREQUISITES:
#   1. retriever checkpoint: the jepa pretrain at step 400k (pinned below;
#      bash_scripts/ai2thor/pretrain_retriever.sh trains one);
#   2. cached per-frame keys `latents$KEYS` (default .keys_thor_v1.npy) in every
#      episode dir, exported FROM THE SAME checkpoint by
#      bash_scripts/ai2thor/precompute_keys.sh. Keys and ckpt_path must stay in
#      lockstep — mismatched key/query spaces degrade retrieval silently.
#   3. interact_dim=4 on the query_vit is REQUIRED, not optional: the keys were
#      exported from an interact_dim=4 tower, and `null_interact_emb` (which
#      sits on the key path) trained away from zero. Building the memory at the
#      config default 0 drops those params on load and puts live queries in a
#      DIFFERENT space from the cached keys — silently, with no error.
# Retriever init + cached pool keys (MUST come from the same tower):
#   default  = the released jepa tower (models/ai2thor_v3/retriever_jepa, step
#              400k) + its keys (`latents.keys_thor_v1.npy`);
#   own tower: CKPT=<pretrain run>/checkpoints/<step>.pth.tar KEYS=.keys_thor_v2.npy bash <this>

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# Growable allocator segments — prevents the fragmentation OOM at the 2nd+
# eval (training re-carves the arena between evals; fixed segments then
# cannot serve the large eval-shaped blocks again).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT=${CKPT:-models/ai2thor_v3/retriever_jepa/checkpoints/0400000.pth.tar}
[ -f "$CKPT" ] || { echo "retriever checkpoint not found: $CKPT (run from the repo root; download it with: bash bash_scripts/download.sh checkpoints ai2thor_v3, or set CKPT=<path>)" >&2; exit 1; }
KEYS=${KEYS:-.keys_thor_v1.npy}
CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,3} torchrun --nproc_per_node=$(awk -F, '{print NF}' <<< "${GPUS:-0,1,2,3}") --standalone scripts/world/train_latent.py \
    run_name=thor3_cdit_b_pnp_vmz_posefa \
    datasets=[ai2thor_v3_latent] \
    ai2thor_v3_latent.shared.backend.key_suffix=$KEYS \
    `# MASK-WEIGHTED DIFFUSION LOSS: the per-goal union change mask reweights the`\
    `# per-token loss so the changed region carries ~lambda/(1+lambda) ~= 23% of it.`\
    `# Same lambda on every AI2-THOR arm; see train_ldwm_temporal.sh.`\
    ai2thor_v3_latent.shared.backend.mask_suffix=.change_interact.npy \
    model.mask_loss_lambda=0.3 \
    `# masks above this token fraction are gated off (fridge/cabinet door swings);`\
    `# 0.15 keeps the union's small-container swing rows (11-13% of tokens).`\
    model.mask_max_frac=0.15 \
    `# rag_mask_loss_lambda inherits the lambda above, so the RAG per-set and`\
    `# within-chunk REWARDS rank candidate sets by how well they predict what`\
    `# CHANGED; set model.rag_mask_loss_lambda=0 to weight only the generator's loss.`\
    `# retrieval pool — matched to the temporal / worldmem arms.`\
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
    model/memory=pnp_cl \
    model.memory.ckpt_path=$CKPT \
    model.memory.scoring=visual+meta_z \
    model.memory.encoder_mode=adapter \
    model.memory.query_vit.img_height=32 \
    model.memory.query_vit.img_width=32 \
    model.memory.query_vit.patch_size=2 \
    model.memory.query_vit.in_chans=4 \
    model.memory.query_vit.interact_dim=4 \
    'model.memory.local_pos_stats=${ai2thor_v3_latent.spec.local_pos_stats}' \
    model.memory.stride_to_seconds=1.0 \
    `# 3-layer metadata cross-scorer MLP (default 2): one extra hidden layer for`\
    `# sharper geometry pair functions; zero-init last layer keeps the step-0 no-op.`\
    model.memory.metadata_layers=3 \
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
    wandb.group=ai2thor_v3_latent \
    wandb.tags=[ai2thor_v3_latent] \
