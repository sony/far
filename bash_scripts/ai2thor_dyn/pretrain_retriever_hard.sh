# Per-machine overrides (python env, GPU set): bash_scripts/env.local.sh, see env.local.sh.example
[ -f "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh" ] && . "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh"
pip install -e .

# ── AI2-THOR dynamic-agent (door-corridor) corpus (`ai2thor_dyn_latent`): FAR RETRIEVER FINE-TUNE ──
# Fine-tunes the FAR retriever's query encoder (QueryViT) with the
# OBJECT-CENTRIC term of scripts/world/train_latent_retriever_hard.py, starting
# from the AI2-THOR v3 jepa tower. The checkpoint is the `CKPT` init of the
# dynamic-agent FAR arms (train_ldwm_pnp_meta*.sh) and the tower that exports
# their cached keys (`.keys_obj_v1.npy`).
#   bash bash_scripts/ai2thor_dyn/pretrain_retriever_hard.sh            # 4 GPUs, batch 4 x 16 = 64
#   GPUS=4,5,6,7 INIT=<ckpt> bash bash_scripts/ai2thor_dyn/pretrain_retriever_hard.sh
#
# Objective:
#   query     = the real deployment query (eval-leg current + reveal pose);
#   positives = the pool frames of the LATEST sighting of the second agent
#               (presence = the .agent.npy silhouette area per frame, served
#               through the backend's mask_suffix), weighted by visible area;
#   negatives = agent-free pool frames (the pose-identical empty prologue view
#               of each end and its neighbours are all in here, unmined);
#   neutral   = earlier crossings (opposite heading — never negatives).
# Episode-level rule (every eval-leg query of an agent episode gets the same
# positives), so the mapping is consistent and needs no memorisation; the
# jepa term is kept for view structure with presence frames excluded from its
# negatives. Long-horizon draws and pose-matched hard negatives are OFF (both
# hurt retrieval of the crossing on this corpus).
# Init = the AI2-THOR v3 jepa tower (models/ai2thor_v3/retriever_jepa, step
# 400k) that exported this corpus's `.keys_thor_v1.npy` keys (weights only).
# NOTE: the fine-tuned tower defines a NEW key space — re-export keys
# (build_retriever_keys.py --latent-name latents.npy, a new --key-suffix)
# before wiring it into a world-model run.
# In the log, `obj_margin` (best positive minus best negative, cosine) must go
# positive on the VAL line, not only on train.

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

INIT=${INIT:-models/ai2thor_v3/retriever_jepa/checkpoints/0400000.pth.tar}
CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,3} torchrun --nproc_per_node=$(awk -F, '{print NF}' <<< "${GPUS:-0,1,2,3}") --standalone scripts/world/train_latent_retriever_hard.py \
    run_name=retriever_thordyn_object_ft \
    datasets=[ai2thor_dyn_latent] \
    `# the agent silhouette mask = per-frame presence for the object term`\
    ai2thor_dyn_latent.shared.backend.mask_suffix=.agent.npy \
    ai2thor_dyn_latent.sampler.context_goal.context_pool_size=400 \
    ai2thor_dyn_latent.sampler.context_goal_test.context_pool_size=400 \
    `# pool = the PRE-EVAL history only (prologue + watch leg), exactly the deployment`\
    `# pool. NOT pool_phase=past: eval-leg look_A/look_B frames showing the agent at its`\
    `# FINAL end would become 'latest sighting' positives — the answer would leak in.`\
    ai2thor_dyn_latent.sampler.context_goal.pool_phase=pre_eval \
    ai2thor_dyn_latent.sampler.context_goal_test.pool_phase=pre_eval \
    ai2thor_dyn_latent.sampler.context_goal.long_horizon_p=0.0 \
    ai2thor_dyn_latent.sampler.context_goal_test.long_horizon_p=0.0 \
    goals_per_obs=1 \
    model/memory=pnp_cl \
    model.memory.chunk_size=10 \
    model.memory.query_vit.img_height=32 \
    model.memory.query_vit.img_width=32 \
    model.memory.query_vit.patch_size=2 \
    model.memory.query_vit.in_chans=4 \
    model.memory.query_vit.interact_dim=4 \
    contrastive.init_from=$INIT \
    contrastive.positive_pair_type=[jepa] \
    contrastive.num_positives=8 \
    contrastive.num_negatives=24 \
    contrastive.obj_weight=1.0 \
    contrastive.obj_num_positives=8 \
    contrastive.obj_pulse_gap_s=2.0 \
    contrastive.obj_min_frac=0.0 \
    contrastive.obj_last_only=true \
    contrastive.num_hard_negatives=0 \
    contrastive.stride_clip=null \
    contrastive.max_frame_offset=10.0 \
    contrastive.fps=10.0 \
    contrastive.cl_pose_radius=0.5 \
    contrastive.cl_pose_fov_half_h=180.0 \
    contrastive.cl_pose_fov_half_v=180.0 \
    contrastive.local_pos_stats=\${ai2thor_dyn_latent.spec.local_pos_stats} \
    image_height=256 \
    image_width=256 \
    len_traj_pred=64 \
    action_dim=5 \
    pose_dim=5 \
    batch_size=${BATCH:-16} \
    lr=5e-5 \
    iterations=50000 \
    warmup_steps=2000 \
    log_every=100 \
    ckpt_every=2500 \
    eval_every=2500 \
    bfloat16=false \
    wandb.project=FAR-AI2THOR \
    wandb.group=ai2thor_dyn_latent \
    wandb.tags=[ai2thor_dyn_latent,retriever_pretrain,object_centric] \
