# Per-machine overrides (python env, GPU set): bash_scripts/env.local.sh, see env.local.sh.example
[ -f "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh" ] && . "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh"
pip install -e .

# ── AI2-THOR: FAR VISUAL RETRIEVER PRE-TRAINING (corpus = $DATASET, default ai2thor_v3_latent) ──
# Contrastive (InfoNCE) pre-training of the FAR retriever's query encoder
# (QueryViT) on the AI2-THOR v2 latent corpus — the same script/module as the
# loopnav jepa pretrain and the soundspaces audio pretrain
# (train_latent_retriever / ContrastivePretrainModule). The checkpoint is the
# `CKPT` init of the FAR world-model arms (train_ldwm_pnp_*.sh) and the tower
# that exports their cached keys (precompute_keys.sh).
#   bash bash_scripts/ai2thor/pretrain_retriever.sh                # 4 GPUs, batch 4 x 16 = 64
#   GPUS=4,5,6,7 bash bash_scripts/ai2thor/pretrain_retriever.sh
#
# QueryViT is instantiated over the SDXL latents this corpus ships: (4, 32, 32)
# at patch_size 2 -> a 16x16 = 256-token grid.
#
# Positives: `jepa` — the goal frame's own content key, i.e. the encoder is
# trained so that the action-conditioned query at the current frame predicts
# the FUTURE observation's embedding. Negatives come from the same episode's
# pool (hard negatives: same kitchen, different station/heading), which
# prevents the episode-identity shortcut.
#
# SAMPLER — deliberately DIFFERENT from the world-model launchers in one knob:
#   * event bias ON, identical windows to train_ldwm_*.sh: with p=0.3 the
#     (current, goal) pair sits on an open/close/pickup/put span, so the
#     jepa target is the state change itself (the fresh view of the container
#     / surface), not the average over a 6.4 s window;
#   * pool_phase=past (NOT pre_eval): the negatives — and the pose-positive
#     candidates — are drawn from the ENTIRE past up to the current frame,
#     including the agent's own eval-leg frames. The world-model arms restrict
#     their pool to the exploration phase (the rollout protocol); the retriever
#     pretrain must not, because the eval-leg past is where the same-pose
#     FRESH views of a re-observed station live — exactly the hard negatives
#     (stale vs fresh at one pose) the encoder has to learn to separate.
#     Same for the in-training-eval sampler, so recall is measured on the
#     same pool the encoder trains against.
#
# PREREQUISITES: the AI2-THOR v2 latent corpus (`datasets/ai2thor_v2/latent` with its manifest and
#   indices), the corpus this tower was pre-trained on. It is not part of the public release;
#   the AI2-THOR v3 train tier (bash bash_scripts/download.sh train ai2thor_v3) is the
#   alternative corpus to pre-train on (override datasets= and the ai2thor_v2_latent.* keys).
#
# The checkpoint loads into world-model training as the retriever init via
#   CKPT=<this run>/checkpoints/<step>.pth.tar   (train_ldwm_pnp_*.sh read $CKPT)
# and its cached keys are exported with the SAME tower by
#   CKPT=<this run>/checkpoints/<step>.pth.tar SUFFIX=.keys_thor_v2.npy \
#   LATENT_DIR=datasets/ai2thor_v2/latent bash bash_scripts/ai2thor/precompute_keys.sh
# (then launch the FAR arms with KEYS=.keys_thor_v2.npy; the released tower's
# keys on this corpus are `.keys_thor_v1.npy`, the launchers' default).
#
# The interact_dim override is REQUIRED, not cosmetic: keys are encoded with
# no action/stride/interaction, so they pick up `null_interact_emb` — which is
# zero at init but TRAINS. Building the exporter's QueryViT at the config
# default interact_dim=0 silently drops the interact params (strict=False ->
# "unexpected") and yields keys that differ from what the live encoder
# computes. Cached keys and live queries must come from the SAME architecture.

# DATASET: the corpus to pre-train on. Default ai2thor_v3_latent (released); the released
# ai2thor_v3/retriever_jepa tower was pre-trained on ai2thor_v2_latent, which is not released,
# so a v3 pre-training will not reproduce it exactly.
DATASET=${DATASET:-ai2thor_v3_latent}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,3} torchrun --nproc_per_node=$(awk -F, '{print NF}' <<< "${GPUS:-0,1,2,3}") --standalone scripts/world/train_latent_retriever.py \
    run_name=retriever_${DATASET%_latent}_jepa_pretrain_ltp64 \
    datasets=[${DATASET}] \
    ${DATASET}.sampler.context_goal.context_pool_size=200 \
    `# EVENT-ANCHORED pairs — same windows as the world-model launchers.`\
    ${DATASET}.sampler.context_goal.event_bias_p=0.3 \
    ${DATASET}.sampler.context_goal.event_phase=any \
    ${DATASET}.sampler.context_goal.event_codes=null \
    "${DATASET}.sampler.context_goal.event_pre=[-3,5]" \
    "${DATASET}.sampler.context_goal.event_post=[0,8]" \
    ${DATASET}.sampler.context_goal.event_goal_anchor=start \
    `# negatives from the ENTIRE past (exploration + eval-leg-so-far), see header.`\
    ${DATASET}.sampler.context_goal.pool_phase=past \
    ${DATASET}.sampler.context_goal_test.pool_phase=past \
    goals_per_obs=1 \
    model/memory=pnp_cl \
    model.memory.chunk_size=10 \
    model.memory.query_vit.img_height=32 \
    model.memory.query_vit.img_width=32 \
    model.memory.query_vit.patch_size=2 \
    model.memory.query_vit.in_chans=4 \
    model.memory.query_vit.interact_dim=4 \
    contrastive.positive_pair_type=[jepa] \
    contrastive.num_positives=8 \
    contrastive.num_negatives=24 \
    contrastive.cl_pose_radius=0.5 \
    contrastive.cl_pose_fov_half_h=180.0 \
    contrastive.cl_pose_fov_half_v=180.0 \
    contrastive.local_pos_stats=\${${DATASET}.spec.local_pos_stats} \
    image_height=256 \
    image_width=256 \
    len_traj_pred=64 \
    action_dim=5 \
    pose_dim=5 \
    batch_size=${BATCH:-16} \
    lr=2e-4 \
    iterations=400000 \
    warmup_steps=5000 \
    log_every=100 \
    ckpt_every=25000 \
    eval_every=5000 \
    bfloat16=false \
    wandb.project=FAR-AI2THOR \
    wandb.group=${DATASET} \
    wandb.tags=[${DATASET},retriever_pretrain,visual] \
