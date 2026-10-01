# Per-machine overrides (python env, GPU set): bash_scripts/env.local.sh, see env.local.sh.example
[ -f "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh" ] && . "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh"
pip install -e .

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# ── SoundSpaces corpus (`soundspaces_latent`): FAR AUDIO RETRIEVER PRE-TRAINING ──
# Contrastive (InfoNCE) pre-training of the FAR retriever's AUDIO query encoder
# for SoundSpaces — the SAME script/module as the visual jepa pretrain (train_latent_retriever /
# ContrastivePretrainModule): QueryViT is a generic 2D patch encoder, here
# instantiated over the (4, 64, 48) binaural feature windows (logmel L/R +
# cos/sin IPD, 0.48 s causal, AST-style freq x time patches -> 16x12 grid).
# The backend serves the windows IN THE FRAMES SLOT (audio_mode="frames"),
# so the whole sampler/formatter stack is reused untouched.
#
# Positives: `pose` type with fov_half = 180 deg -> the FOV cone is the full
# sphere, i.e. pure RADIUS (0.5 m) same-place positives with NO heading
# constraint — return-leg queries must match reversed-heading outbound
# memories (the loop-closure use case), so the encoder is trained for heading
# invariance. Negatives come from the same episode's pool (hard negatives:
# same room layout, different position) — prevents the episode-identity
# shortcut. closures.npz is NOT needed for training; keep it for the
# validation-gate notebook (recall@k on held-out scenes).
#
# The checkpoint loads into the FAR world-model arms (train_ldwm_pnp_*.sh)
# as the retriever init via
#   model.memory.ckpt_path=<this run>/checkpoints/latest.pth.tar
# and its cached keys are exported FROM THE SAME checkpoint by precompute_keys.sh.
#   bash bash_scripts/soundspaces/pretrain_audio_retriever.sh          # 4 GPUs, batch 4 x 16 = 64
#   GPUS=4,5,6,7 bash bash_scripts/soundspaces/pretrain_audio_retriever.sh
CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,3} torchrun --nproc_per_node=$(awk -F, '{print NF}' <<< "${GPUS:-0,1,2,3}") --standalone scripts/world/train_latent_retriever.py \
    run_name=retriever_audio_jepa_pretrain_ltp64 \
    datasets=[soundspaces_latent] \
    soundspaces_latent.shared.backend.audio_mode=frames \
    soundspaces_latent.sampler.context_goal.context_pool_size=200 \
    goals_per_obs=1 \
    model/memory=pnp_cl \
    model.memory.chunk_size=10 \
    model.memory.query_vit.img_height=64 \
    model.memory.query_vit.img_width=48 \
    model.memory.query_vit.patch_size=4 \
    model.memory.query_vit.in_chans=4 \
    contrastive.positive_pair_type=[jepa] \
    contrastive.num_positives=8 \
    contrastive.num_negatives=24 \
    contrastive.cl_pose_radius=0.5 \
    contrastive.cl_pose_fov_half_h=180.0 \
    contrastive.cl_pose_fov_half_v=180.0 \
    contrastive.local_pos_stats=\${soundspaces_latent.spec.local_pos_stats} \
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
    wandb.project=FAR-SoundSpaces \
    wandb.group=soundspaces_latent \
    wandb.tags=[soundspaces_latent,retriever_pretrain,audio] \
