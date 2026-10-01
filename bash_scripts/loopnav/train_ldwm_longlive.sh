# Per-machine overrides (python env, GPU set): bash_scripts/env.local.sh, see env.local.sh.example
[ -f "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh" ] && . "$(dirname "${BASH_SOURCE[0]}")/../env.local.sh"
pip install -e .

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# Growable allocator segments — prevents the fragmentation OOM at the 2nd+
# eval (training re-carves the arena between evals; fixed segments then
# cannot serve the large eval-shaped blocks again).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── LoopNav corpus (`loopnav_latent`): LongLive-RAG baseline ("LongLive-RAG") ──
# Trains the LongLive-RAG (arXiv:2606.02553) retrieval arm — the
# action-AGNOSTIC content-similarity baseline — on the same pose-conditioned
# CDiT-B world model as the other loopnav arms (matched in generator / batch /
# horizon / pool). Faithful mapping of their design onto the FAR retriever
# machinery (model/memory=pnp_cl):
#   - query_conditioning=none  — queries are CONTENT embeddings of the current
#     frame (null action/stride, the same encode as keys): the paper's Alg. 1
#     "embedding of the most recent completed latent". NOTE: in rollouts this
#     queries with the GENERATED current frame — exposure-bias inheritance is
#     an intended, documented property of the method on this task;
#   - encoder_mode=frozen + retriever_loss_weight=0 + local_rag=false — the
#     encoder is OFFLINE-trained and never updated during generation training
#     ("does not update the base generator or add trainable layers");
#   - scoring=visual — L2-normalized cosine over cached keys IS their
#     similarity retrieval; chunking provides the anti-duplicate role their
#     (domain-saturated) temporal-delta term was meant to play, and the
#     retriever's current-frame exclusion mirrors their recency guard R;
#   - encoder = QueryViT trained by THEIR objective (recon + window temporal
#     delta + smoothness; see pretrain_retriever_longlive.sh) — architecture
#     parity with the contrastive arms, so the comparison isolates query
#     semantics + pretraining objective.
# Keys are convention-free (content encode never reads poses), so the raw-
# convention pretrain pairs fine with this standard-frame posefa run.
#
# PREREQUISITES: the LongLive-pretrained QueryViT (models/loopnav/
# retriever_longlive, step 400k; bash_scripts/loopnav/pretrain_retriever_longlive.sh
# trains one) and the cached keys `<clip>.keys_longlive.npy` from the SAME ckpt:
#   bash bash_scripts/loopnav/precompute_keys_longlive.sh
#   bash bash_scripts/loopnav/train_ldwm_longlive.sh                   # 4 GPUs, batch 4 x 32 = 128
#   GPUS=4,5,6,7 bash bash_scripts/loopnav/train_ldwm_longlive.sh
# Resume: append from_checkpoint=<run_dir>/checkpoints/latest.pth.tar.
CKPT=${CKPT:-models/loopnav/retriever_longlive/checkpoints/0400000.pth.tar}
[ -f "$CKPT" ] || { echo "retriever checkpoint not found: $CKPT (run from the repo root; download it with: bash bash_scripts/download.sh checkpoints loopnav, or set CKPT=<path>)" >&2; exit 1; }
CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,3} torchrun --nproc_per_node=$(awk -F, '{print NF}' <<< "${GPUS:-0,1,2,3}") --standalone scripts/world/train_latent.py \
    run_name=nwm_cdit_b_pnp_longlive_posefa_l0 \
    datasets=[loopnav_latent] \
    loopnav_latent.shared.backend.standard_frame=true \
    loopnav_latent.shared.backend.key_suffix=.keys_longlive.npy \
    model/generator=cdit_b_2_mod_pose_fa \
    model.generator.context_size=4 \
    'model.generator.local_pos_stats=${loopnav_latent.spec.local_pos_stats}' \
    model/tokenizer=vit_vae \
    model/memory=pnp_cl \
    model.memory.ckpt_path=$CKPT \
    model.memory.scoring=visual \
    model.memory.query_conditioning=none \
    model.memory.encoder_mode=frozen \
    'model.memory.local_pos_stats=${loopnav_latent.spec.local_pos_stats}' \
    model.memory.num_sets=5 \
    model.memory.chunk_size=10 \
    model.retriever_loss_weight=0.0 \
    local_rag=false \
    model.condition_on_context_time=true \
    model.condition_on_context_pose=true \
    image_height=360 \
    image_width=640 \
    action_dim=5 \
    pose_dim=5 \
    batch_size=${BATCH:-32} \
    eval_batch_size=${EVAL_BATCH:-64} \
    lr=1e-4 \
    bfloat16=false \
    torch_compile=true \
    wandb.project=FAR-LoopNav \
    wandb.group=loopnav_latent \
    wandb.tags=[loopnav_latent] \
