# Per-machine overrides (python env, GPU set): bash_scripts/env.local.sh, see env.local.sh.example
[ -f "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh" ] && . "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh"
CKPT=${CKPT:-models/loopnav/retriever_jepa/checkpoints/0400000.pth.tar}   # pre-trained retriever this arm starts from
[ -f "$CKPT" ] || { echo "retriever checkpoint not found: $CKPT (run from the repo root; download it with: bash bash_scripts/download.sh checkpoints loopnav, or set CKPT=<path>)" >&2; exit 1; }
pip install -e .

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# ── LoopNav corpus (`loopnav_latent`): FAR -- Multi-Cue (headline arm) ──
# Trains the FAR retriever (model/memory=pnp_cl) scoring on BOTH channels
# through the z-gate (scoring=visual+meta_z): the VISUAL channel is the
# QueryViT key/query cosine over ViT-VAE latents, the METADATA channel the
# closed-form relative-geometry cross-scorer. Sibling launchers, matched in
# generator / batch / horizon / pool / RAG settings so the arms stay
# samples-matched: train_ldwm_temporal.sh (Temporal), train_ldwm_worldmem.sh
# (WorldMem), train_ldwm_longlive.sh (LongLive-RAG), train_ldwm_pnp_meta_only.sh
# (FAR -- Meta), train_ldwm_pnp_visual_only.sh (FAR -- Visual) and
# train_ldwm_pnp_ablation.sh (the fixed-fusion ablation of this arm).
#   bash bash_scripts/loopnav/train_ldwm_pnp.sh                        # 4 GPUs, batch 4 x 32 = 128
#   GPUS=4,5,6,7 bash bash_scripts/loopnav/train_ldwm_pnp.sh
# Resume: append from_checkpoint=<run_dir>/checkpoints/latest.pth.tar.
#
# Z-GATE: each channel is STANDARDIZED per query over the pool's valid frames
# and mixed as (1−λ~)·z_meta + λ~·z_visual. λ comes from ONE shared linear
# gate scoring each channel from the same DETACHED features — [raw top1, mean,
# std] + a fixed one-hot source tag + Δt-to-goal via per-channel dt·tag slots
# — via a softmax over the channel logits, then squashed to λ~ = ε + (1−2ε)λ
# (ε = gate_eps 0.05) so both channels always keep ≥ ε weight. The gate cannot
# train either scorer to manipulate its own confidence (features are
# detached). Zero-init gate ⇒ λ~ = 0.5 (an equal-weight ensemble of the two
# rankings) at step 0; no visual_scale parameter exists in this mode, so there
# is no runaway-scale degree of freedom. encoder_mode=adapter trains a
# zero-init query adapter over the frozen tower.
#
# PREREQUISITES: the FAR retriever checkpoint (models/loopnav/retriever_jepa,
# step 400k; bash_scripts/loopnav/pretrain_retriever.sh trains one) and the
# cached per-clip keys `<clip>.keys_jepa.npy` exported FROM THAT SAME tower
# (scripts/world/build_retriever_keys.py --key-suffix .keys_jepa.npy, i.e.
# bash_scripts/loopnav/precompute_keys.sh at a non-default suffix; point
# `key_suffix` below at whichever name was written). Keys and ckpt_path must
# stay in lockstep — mismatched key/query spaces degrade retrieval silently.
CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,3} torchrun --nproc_per_node=$(awk -F, '{print NF}' <<< "${GPUS:-0,1,2,3}") --standalone scripts/world/train_latent.py \
    run_name=nwm_cdit_b_pnp_vmz_adapter_posefa_l1 \
    datasets=[loopnav_latent] \
    loopnav_latent.shared.backend.standard_frame=true \
    loopnav_latent.shared.backend.key_suffix=.keys_jepa.npy \
    model/generator=cdit_b_2_mod_pose_fa \
    model.generator.context_size=4 \
    'model.generator.local_pos_stats=${loopnav_latent.spec.local_pos_stats}' \
    model/tokenizer=vit_vae \
    model/memory=pnp_cl \
    model.memory.ckpt_path=$CKPT \
    model.memory.scoring=visual+meta_z \
    model.memory.encoder_mode=adapter \
    'model.memory.local_pos_stats=${loopnav_latent.spec.local_pos_stats}' \
    model.memory.num_sets=5 \
    model.memory.chunk_size=10 \
    model.retriever_loss_weight=1.0 \
    rag_every=20 \
    rag_noise_levels=4 \
    local_rag=true \
    local_rag_weight=1.0 \
    model.condition_on_context_time=true \
    model.condition_on_context_pose=true \
    image_height=360 \
    image_width=640 \
    action_dim=5 \
    pose_dim=5 \
    batch_size=${BATCH:-32} \
    eval_batch_size=${EVAL_BATCH:-64} \
    lr=1e-4 \
    lr_mem=1e-4 \
    bfloat16=false \
    torch_compile=true \
    wandb.project=FAR-LoopNav \
    wandb.group=loopnav_latent \
    wandb.tags=[loopnav_latent] \
