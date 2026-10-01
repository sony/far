import torch
import torch.nn as nn


class LDWMCore(nn.Module):
    """
    Latent Diffusion World Model — latent-space operations only.

    Owns the denoising generator and the diffusion process. All inputs and
    outputs are in the tokenizer's latent space; pixel-space conversion is
    left to PixelLDWM.

    Memory modules follow a unified API
    (see `far.memories.{baselines, worldmem, pnp_cl}`):

        forward(context_frames, context_poses, context_times,
                action, stride, context_mask=None, context_keys=None)
            -> (sel_frames, sel_poses, sel_times, sel_scores, sel_chunk_scores)

    `sel_frames` is either `(B, num_goals, num_memories, C, H, W)` for
    single-set memories (treated as `K = 1` here) or
    `(B, num_goals, K, num_memories, C, H, W)` for set-aware memories
    (`pnp_cl`'s `num_sets` axis). The generator is trained on the top set in
    `compute_loss_latent` (hard-EM). pnp_cl's retriever (QueryViT) is warm-started
    by a SEPARATE contrastive pre-training stage (`train_latent_retriever`); that
    loss is not produced by the memory here, so no memory returns an aux loss.
    """

    def __init__(
        self,
        generator: nn.Module,
        diffusion,
        memory: nn.Module = None,
        condition_on_context_time: bool = True,
        condition_on_context_pose: bool = True,
        retriever_loss_weight: float = 1.0,
        mask_loss_lambda: float = 0.0,
        mask_max_frac: float = 0.1,
        rag_mask_loss_lambda: float = None,
        agent_loss_weight: float = 1.0,
        agent_reward_weight: float = 1.0,
        agent_cond_binary: bool = True,
    ):
        super().__init__()
        self.generator = generator
        # AGENT-STATE PROTOCOL — active only when the generator exposes
        # `predict_agent` (AgentStateCDiT) AND the batch carries
        # `context_agent_state`; every other generator / dataset is untouched.
        # Per (sample, goal, set) the core gathers the selected slots'
        # second-agent detections by nearest pool time, the generator's
        # predictor turns them into the goal's `[logit vi, dx, dy]`, the DiT is
        # conditioned on the BINARISED state `[±1, dx, dy]` (`agent_cond_binary`,
        # see `_agent_cond`: the ground-truth state in training, the sign of the
        # prediction when sampling; `false` = the raw soft prediction, under
        # which the DiT only ever sees hedged logits and learns to paint the
        # agent at low probabilities), and:
        #   `agent_loss_weight`   x the predictor's own loss is added to the
        #                         training loss (compute_loss_latent);
        #   `agent_reward_weight` x the same loss is added to every candidate
        #                         set's / frame's EMDR2 reward (rag_step,
        #                         rag_step_ddim) — the term that separates the
        #                         last crossing from the one before it.
        self.agent_loss_weight = float(agent_loss_weight)
        self.agent_reward_weight = float(agent_reward_weight)
        self.agent_cond_binary = bool(agent_cond_binary)
        self.diffusion = diffusion
        self.memory = memory
        # Default number of retrieved frames per set (= context_size - 1). Cached
        # while `generator` is still the raw module — at train time it gets wrapped
        # in DDP/torch.compile, after which `self.generator.context_size` is no
        # longer reachable. `prepare_context` passes this to the memory's forward
        # unless a caller overrides it (rag_step uses num_memories=1 → a ctx=2
        # single-document probe; the generator now accepts a variable frame count).
        self.num_memories = generator.context_size - 1
        self.condition_on_context_time = condition_on_context_time
        self.condition_on_context_pose = condition_on_context_pose
        # Coefficient on the RAG retriever posterior-distillation loss
        # `L_ret = -Σ_j q_ij log p_η(z_ij|Q_i)` computed lazily in `rag_step`.
        # Trains the retriever (pnp_cl's QueryViT) to match the EM posterior
        # over the K context sets. 0.0 disables the EM retriever update.
        self.retriever_loss_weight = retriever_loss_weight
        # ── Mask-weighted diffusion loss ──────────────────────────────────
        # The plain latent MSE is dominated by the ~98% of tokens a pose-matched
        # context frame already predicts, so "copy the best-matching context
        # frame" is near-optimal and the object that actually changed
        # contributes almost nothing. When the batch carries a per-goal change
        # mask (`goal_loss_mask`, from scripts/data/build_masks_ai2thor.py)
        # the per-token loss is reweighted so the changed region carries a fixed
        # SHARE of the loss:
        #     w = (1 + lambda * m / mean(m)) / (1 + lambda)
        # `w` has mean 1 per sample, so the loss scale — and with it the LR and
        # the loss curve — is unchanged; only the spatial allocation shifts.
        # For a binary mask over a fraction f of tokens the in-mask share is
        # (f + lambda)/(1 + lambda) ~= lambda/(1 + lambda) for small f, so
        # lambda 0.25 / 0.43 / 1.0 -> ~20 / 30 / 50%. 0.0 disables it entirely
        # (bit-identical to an unweighted run).
        self.mask_loss_lambda = float(mask_loss_lambda)
        # Masks covering more than this fraction of tokens are rejected as
        # viewpoint smear rather than a localized change (measured: genuine
        # change regions ~1.8% of tokens, viewpoint-misaligned ones ~76%);
        # those goals fall back to the uniform loss.
        self.mask_max_frac = float(mask_max_frac)
        # Same reweighting applied to the RAG per-set / per-frame REWARD in
        # `rag_step`, so the retriever's EM posterior ranks candidate sets by how
        # well they predict WHAT CHANGED rather than by overall scene match (the
        # measured retrieval bias: pose-matched but stale frames win). None =
        # inherit `mask_loss_lambda`; set explicitly (e.g. 0.0) to ablate the two
        # independently — they answer different questions. Applies to BOTH RAG
        # rewards: the multi-noise diffusion loss (`rag_step`) and the sampled
        # x0 MSE (`rag_step_ddim`).
        self.rag_mask_loss_lambda = (None if rag_mask_loss_lambda is None
                                     else float(rag_mask_loss_lambda))
        # Telemetry (non-persistent): fraction of goals that actually received a
        # weight, and the realized in-mask share of the loss (tracks
        # lambda/(1+lambda) where masks are present).
        self.register_buffer('last_mask_weighted_frac', torch.zeros(1), persistent=False)
        self.register_buffer('last_mask_loss_share', torch.zeros(1), persistent=False)
        # Actual prediction ERROR inside vs outside the change region (the two
        # above only describe how the WEIGHT is allocated, which is fixed by
        # lambda and the mask area). `in/out` is the training-time analogue of
        # the offline change-region diagnostic: a model that copies a
        # pose-matched context frame is near-perfect outside the region and
        # wrong inside it, so in/out >> 1; a blurry model sits at in/out <~ 1.
        # NOTE: measured on the model's target-space residual at a UNIFORMLY
        # sampled t, so high-t steps (target ~ pure noise) dilute the ratio
        # toward 1 — read the trend, not the absolute value.
        self.register_buffer('last_mask_in_mse', torch.zeros(1), persistent=False)
        self.register_buffer('last_mask_out_mse', torch.zeros(1), persistent=False)
        # Agent-state telemetry: the predictor's loss on the last training step
        # (`train/agent_loss`) and its last prediction from `sample_latent`,
        # `(B*num_goals, 3)`, which the eval reads against `goal_agent_state`.
        self.register_buffer('last_agent_loss', torch.zeros(1), persistent=False)
        # RETRIEVAL TELEMETRY (agent-state protocol, set by compute_loss_latent
        # from the top set's retrieved slots, current frame excluded): the
        # fraction of slots whose detection has vi=1 — the retriever bootstrap
        # made visible (climbs from ~0 once the scorer seeks sightings) — and
        # the slots' mean time offset from the current frame in seconds (~5 s
        # = recency, ~40 s = the last crossing).
        self.register_buffer('last_ret_sighting_frac', torch.zeros(1), persistent=False)
        self.register_buffer('last_ret_dt', torch.zeros(1), persistent=False)
        self.last_agent_pred = None
        # Running EMA of `2 * std(per_set_loss across K)` used as the EM
        # temperature τ in `rag_step` (over the de-noised multi-noise spread).
        # Initialised to 0.1; decay = 0.99 → effective window of ~100 updates.
        self.register_buffer('per_set_loss_tau_ema', torch.tensor(0.1))
        # τ EMA for rag_step_ddim's SAMPLED-outcome reward (mask-weighted x0 MSE
        # on DDIM samples) — a different scale from the diffusion-loss spread,
        # so it gets its own EMA. persistent=False: pre-existing checkpoints
        # load unchanged and a resume merely re-warms it.
        self.register_buffer('rag_ddim_tau_ema', torch.tensor(0.1), persistent=False)
        # rag_step_ddim telemetry (non-persistent): cross-set std over K of the
        # sampled x0 error, split by region. `last_rag_mse_spread` = the
        # UNWEIGHTED whole-frame MSE (the reward before mask weighting);
        # `last_rag_in_mask_spread` = the MSE INSIDE the change region, on goals
        # that carry a usable mask. in_mask >> mse means the candidate sets
        # differ where the state changed and barely elsewhere — the retriever
        # is being ranked on state, which is the point of the sampled reward.
        self.register_buffer('last_rag_mse_spread', torch.zeros(1), persistent=False)
        self.register_buffer('last_rag_in_mask_spread', torch.zeros(1), persistent=False)
        self.per_set_loss_tau_ema_decay = 0.99
        # Telemetry: last `rag_step`'s retriever loss `L_ret`. Non-persistent;
        # `train_latent.py` reads it by name (`last_em_ret_loss` → `train/ret_loss`).
        self.register_buffer('last_em_ret_loss', torch.zeros(1), persistent=False)
        # Telemetry: last `rag_step`'s RE-RANK RATE — the fraction of (sample, goal)
        # for which the EM posterior's best set differs from the retriever prior's
        # best set, i.e. `argmax_K(q) != argmax_K(p_η)`. ~0 ⇒ the EM only confirms
        # the prior (no re-ranking → the retriever gets ~no gradient); > 0 ⇒ the EM
        # is actually re-ordering sets. Read as `last_em_rerank_rate → train/rerank_rate`.
        self.register_buffer('last_em_rerank_rate', torch.zeros(1), persistent=False)
        # Telemetry: last `rag_step`'s Spearman ρ between the retriever prior and the
        # reward ranking over the K sets — ρ(rank p_η, rank −per_set_loss), averaged
        # over (sample, goal). +1 ⇒ the prior ranks sets the way the reward does
        # (good); 0 ⇒ unrelated; <0 ⇒ anti-aligned. Rank-based, so unlike CE/rerank
        # it is invariant to the prior's temperature/peakedness and uses the FULL
        # ordering. Read as `last_em_rank_corr → train/rank_corr`.
        self.register_buffer('last_em_rank_corr', torch.zeros(1), persistent=False)
        # LOCAL (within-chunk) RAG (rag_step's local block): `per_frame_loss_tau_ema`
        # is the EM temperature over the within-chunk per-FRAME loss spread — the
        # analog of `per_set_loss_tau_ema` at a smaller scale, so it gets its own
        # EMA. `last_em_local_ret_loss` is the last local CE (non-persistent
        # telemetry), read as `last_em_local_ret_loss → train/local_ret_loss`.
        self.register_buffer('per_frame_loss_tau_ema', torch.tensor(0.1))
        self.register_buffer('last_em_local_ret_loss', torch.zeros(1), persistent=False)
        # Telemetry: Spearman ρ(local prior, −per-frame reward) over the S
        # within-chunk frames — the local analog of `last_em_rank_corr` (over the
        # K sets). Read as `last_em_local_rank_corr → train/local_rank_corr`.
        self.register_buffer('last_em_local_rank_corr', torch.zeros(1), persistent=False)

    def forward(self, x_latent, t, action, stride, context_latent,
                context_time=None, context_pose=None, context_chunk_score=None,
                context_mask=None, action_interact=None, agent_state=None):
        """Denoising forward pass — thin delegation to the generator.

        Args:
            x_latent:            (N, C, H, W) noisy latent at diffusion timestep t.
            t:                   (N,) diffusion timestep indices.
            action:              (N, action_dim) pose delta in local coords.
            stride:              (N,) relative time to the target frame.
            context_latent:      (N, context_size, C, H, W) conditioning frame latents.
            context_time:        (N, context_size) per-frame timestamps in seconds.
            context_pose:        (N, context_size, pose_dim) per-frame poses.
            context_chunk_score: (N, context_size - 1) per-past-frame relevance
                                 from a memory module's per-chunk score output
                                 (worldmem's confidence or pnp_cl's chunk
                                 scores). Forwarded to generators that accept
                                 the kwarg (cdit_mod_score_{xa,fa}); silently
                                 omitted when None so other generators stay
                                 source-compatible.

        Returns:
            (N, out_channels, H, W) predicted noise (or noise + variance).

        Note: per-*set* scores (pnp_cl's `selected_scores`) are not forwarded;
        they weight the per-set diffusion loss in `compute_loss_latent` instead.
        """
        kwargs = dict(
            context_time=context_time if self.condition_on_context_time else None,
            context_pose=context_pose if self.condition_on_context_pose else None,
        )
        if context_chunk_score is not None:
            kwargs['context_chunk_score'] = context_chunk_score
        # Frame-validity mask for generators that consume the RAW pool
        # (a generator that attends over the whole pool, run with memory=null). Only forwarded when
        # non-None — callers gate on `accepts_context_mask`, so generators
        # without the kwarg never receive it.
        if context_mask is not None:
            kwargs['context_mask'] = context_mask
        # Interaction channel — only for generators built with it (interact_dim
        # > 0), so pose-only generators never see the kwarg.
        if action_interact is not None:
            kwargs['action_interact'] = action_interact
        # Predicted second-agent state — only AgentStateCDiT takes the kwarg, and
        # the core only produces it when that generator is in use.
        if agent_state is not None:
            kwargs['agent_state'] = agent_state
        return self.generator(x_latent, t, action, context_latent, stride, **kwargs)

    # ── Agent-state protocol helpers ──────────────────────────────────────────
    def _agent_module(self):
        """The generator with DDP / torch.compile wrappers peeled off, or None
        when it has no `predict_agent` (every non-agent-state generator)."""
        g = self.generator
        for _ in range(4):
            if hasattr(g, 'module'):
                g = g.module
            elif hasattr(g, '_orig_mod'):
                g = g._orig_mod
            else:
                break
        return g if hasattr(g, 'predict_agent') else None

    @staticmethod
    def _slot_states(context_agent_state, context_times, sel_times):
        """Gather the per-slot agent detections of selected frames by NEAREST
        pool time (the memories return times, not indices; pool times are
        unique per frame, and left-padding copies frame 0's state anyway).
        context_agent_state (B, T, S), context_times (B, T), sel_times (B, *dims)
        -> (B, *dims, S)."""
        B, T = context_times.shape
        flat = sel_times.reshape(B, -1)                                          # (B, n)
        idx = (context_times[:, None, :] - flat[:, :, None]).abs().argmin(-1)   # (B, n)
        out = torch.gather(context_agent_state, 1,
                           idx[..., None].expand(-1, -1, context_agent_state.shape[-1]))
        return out.reshape(*sel_times.shape, context_agent_state.shape[-1])

    def _agent_predict(self, mod, slot_state, slot_pose, slot_time, action, stride, slot_mask=None):
        """Run the generator's predictor on flattened (N, T_eff, ...) slots; the
        current frame is the tail slot, so `cur_time` = its time. Returns the
        (N, 3) prediction with grad (the caller decides what to charge it to).

        NOT routed through the DDP wrapper — DDP's autograd traversal only
        needs the predictor's parameters to be REACHABLE from the forward's
        output, which `_agent_cond` guarantees; a second DDP forward per step
        would instead reset that traversal and flag them unused."""
        return mod.predict_agent(slot_state, slot_pose, slot_time, slot_mask,
                                 action, stride, slot_time[:, -1])

    def _agent_cond(self, pred, target=None):
        """The DiT's conditioning input `(N, 3)`, never carrying gradient into
        the predictor (it is trained only by its own loss / reward), plus a
        zero-weight link to the live prediction so the predictor's parameters
        stay reachable from the DDP-wrapped forward's output
        (find_unused_parameters=True would otherwise mark them unused and error
        when the aux loss's backward reaches them; the link contributes exactly
        zero gradient).

        `agent_cond_binary`: the state is BINARISED to `[±1, dx, dy]` — in
        training (`target` = the goal's ground-truth `[vi, dx, dy]`) the DiT is
        teacher-forced on the true state, so it learns both a confident
        'present' and a confident 'absent' arm; when sampling (`target` None)
        it gets the predictor's decision, sign(logit), with the position zeroed
        when absent. `False` = the raw soft prediction, detached."""
        link = 0.0 * pred
        if not self.agent_cond_binary:
            return pred.detach() + link
        if target is not None:
            vi = (target[:, :1].float() > 0.5).float()
            cond = torch.cat([2.0 * vi - 1.0, target[:, 1:3].float() * vi], dim=1)
        else:
            vi = (pred[:, :1].detach().float() > 0).float()
            cond = torch.cat([2.0 * vi - 1.0, pred[:, 1:3].detach().float() * vi], dim=1)
        return cond.to(pred.dtype) + link

    def prepare_context(
        self,
        context_latent,
        context_poses,
        context_times,
        action,
        stride,
        context_mask=None,
        context_keys=None,
        query_cue=None,
        action_interact=None,
        num_memories: int = None,
        return_local: bool = False,
        context_agent_state=None,
    ):
        """Apply memory-based, per-goal context selection with a set (K) axis.

        Memory modules follow the unified API documented on the class:

            forward(ctx_frames, ctx_poses, ctx_times, action, stride,
                    context_mask=None, context_keys=None)
                -> 5-tuple (sel_frames, sel_poses, sel_times,
                            sel_scores, sel_chunk_scores)

        `context_mask` and (when available) `context_keys` are forwarded to the
        memory here. The goal pose for pose-based retrieval is carried by
        `action` (= the formatter's `goal_poses_nl`); memories that need it in
        raw `context_poses` units invert the normalisation themselves via a
        `local_pos_stats` constructor arg (see `worldmem.WorldMemRetrieval` and
        `pnp_cl.PnPRetrieval`).

        The current frame is appended at the tail of every set, and all returned
        tensors carry a leading `(B, num_goals, K)`.

        Returns:
            context_latent: (B, num_goals, K, T_eff, C, H, W).
            context_poses:  (B, num_goals, K, T_eff, pose_dim) or None.
            context_times:  (B, num_goals, K, T_eff) or None.
            context_score:  (B, num_goals, K) per-set score summing to 1 along K,
                            or None when the memory produces no scores.
            chunk_scores:   (B, num_goals, K, num_memories) per-chunk relevance
                            (one entry per *past* frame in the set; the current
                            frame is excluded), or None. WorldMem and pnp_cl
                            populate this; baselines do not. Forwarded to
                            generators that accept `context_chunk_score`
                            (cdit_mod_score_{xa,fa}).
        """
        B         = context_latent.shape[0]
        num_goals = action.shape[1]
        # Optional within-chunk candidates for local RAG — populated only when
        # `return_local` and the memory (pnp_cl) emits a `LocalCandidates` bundle.
        local_bundle = None

        # Retrieval requires per-frame poses + times. Without them, we just
        # broadcast the original context across goals — memories that don't
        # need them (e.g. AR rollout's inner sample_latent calls) become no-ops.
        can_retrieve = (
            self.memory is not None
            and context_poses is not None
            and context_times is not None
        )

        if can_retrieve:
            # Pre-computed retriever keys — forwarded only when present, so they
            # reach pnp_cl (which consumes them as cached retrieval/scoring keys)
            # without changing the call for memories that don't.
            mem_kwargs = {}
            if context_keys is not None:
                mem_kwargs['context_keys'] = context_keys
            # Retrieval-cue window (e.g. the current frame's binaural audio):
            # forwarded ONLY to memories configured for cue queries, so
            # baseline memories never see an unexpected kwarg even when the
            # dataset supplies cues. The attribute is probed on the BASE
            # memory: unwrap DDP (`.module` — DDP does not delegate attribute
            # access) and the eval-time wrapper convention (`.memory` —
            # RecordRetrieval / SubsampledPool / TeacherForcedQuery), which
            # would otherwise silently hide `query_source` and default to
            # 'frame'.
            mem_inner = self.memory
            for _ in range(8):                     # bounded unwrap walk
                if hasattr(mem_inner, 'module'):
                    mem_inner = mem_inner.module
                elif isinstance(getattr(mem_inner, 'memory', None), torch.nn.Module):
                    mem_inner = mem_inner.memory
                else:
                    break
            if query_cue is not None and getattr(mem_inner, 'query_source', 'frame') == 'cue':
                mem_kwargs['query_cue'] = query_cue
            # Interaction labels reach the retriever ONLY when its encoder was
            # built with the pathway (interact_dim > 0), so datasets that emit
            # `action_interact` stay compatible with pose-only retrievers.
            if action_interact is not None and getattr(
                    getattr(mem_inner, 'query_vit', None), 'interact_dim', 0) > 0:
                mem_kwargs['action_interact'] = action_interact
            # Second-agent detections as a RETRIEVAL cue — only for a pnp_cl whose
            # metadata scorer was built with the block (`metadata_agent_state`);
            # every other memory never sees the kwarg.
            if context_agent_state is not None and getattr(mem_inner, 'uses_agent_state', False):
                mem_kwargs['context_agent_state'] = context_agent_state
            if return_local:
                mem_kwargs['return_local'] = True
            # `num_memories` (frames per set) is supplied per-call: defaults to
            # context_size-1, overridden by rag_step to 1 (ctx=2 single-doc probe).
            nm = num_memories if num_memories is not None else self.num_memories
            mem_out = self.memory(
                context_latent, context_poses, context_times,
                action, stride, context_mask, num_memories=nm, **mem_kwargs,
            )
            # Memories return the 5-tuple; pnp_cl additionally returns a
            # `LocalCandidates` bundle as a 6th element when `return_local` is set
            # (other memories ignore the flag and stay at 5).
            if return_local and len(mem_out) == 6:
                (sel_latent, sel_poses, sel_times,
                 sel_scores, sel_chunk_scores, local_bundle) = mem_out
            else:
                (sel_latent, sel_poses, sel_times,
                 sel_scores, sel_chunk_scores) = mem_out

            # Normalise to a per-set axis at dim 2. Single-set memories return
            # `(B, num_goals, num_memories, C, H, W)` — 6 dims — and we insert
            # `K=1`. Set-aware memories (pnp_cl) already carry the
            # `(B, num_goals, K, ...)` axis (7 dims). The dim check is unchanged
            # but the comment is now in the unified vocabulary.
            if sel_latent.dim() == 6:
                sel_latent = sel_latent.unsqueeze(2)
                sel_poses  = sel_poses.unsqueeze(2)
                sel_times  = sel_times.unsqueeze(2)
                # Single-set memory: drop the set-level score (trivial — one set).
                sel_scores = None
                # Per-chunk scores (e.g. WorldMem's per-frame confidence) stay
                # meaningful and are preserved: insert the K=1 axis to match the
                # (B, ng, K, M) shape that downstream callers expect.
                if sel_chunk_scores is not None:
                    sel_chunk_scores = sel_chunk_scores.unsqueeze(2)
            K = sel_latent.shape[2]

            # Append the current frame at the tail of each set's context.
            cur_l = context_latent[:, -1]                                       # (B, C, H, W)
            cur_p = context_poses[:, -1]                                        # (B, pose_dim)
            cur_t = context_times[:, -1]                                        # (B,)
            cur_l = cur_l[:, None, None, None].expand(B, num_goals, K, 1, *cur_l.shape[1:])
            cur_p = cur_p[:, None, None, None].expand(B, num_goals, K, 1, cur_p.shape[-1])
            cur_t = cur_t[:, None, None, None].expand(B, num_goals, K, 1)
            context_latent = torch.cat([sel_latent, cur_l], dim=3)             # (B, ng, K, T_eff, C, H, W)
            context_poses  = torch.cat([sel_poses,  cur_p], dim=3)
            context_times  = torch.cat([sel_times,  cur_t], dim=3)
            context_score  = sel_scores                                         # (B, ng, K) or None
            chunk_scores   = sel_chunk_scores                                   # (B, ng, K, num_memories) or None
        else:
            # No memory (or no poses/times): broadcast original context as one set.
            context_latent = context_latent[:, None, None].expand(
                B, num_goals, 1, *context_latent.shape[1:]
            )
            if context_poses is not None:
                context_poses = context_poses[:, None, None].expand(
                    B, num_goals, 1, *context_poses.shape[1:]
                )
            if context_times is not None:
                context_times = context_times[:, None, None].expand(
                    B, num_goals, 1, *context_times.shape[1:]
                )
            context_score = None
            chunk_scores  = None
        if return_local:
            return (context_latent, context_poses, context_times,
                    context_score, chunk_scores, local_bundle)
        return context_latent, context_poses, context_times, context_score, chunk_scores

    def _mask_loss_weight(self, goal_loss_mask, lam, repeat: int = 1):
        """Mean-1 per-token loss weight from a per-goal change mask.

        `w = (1 + lam * m / mean(m)) / (1 + lam)` on goals whose mask is
        non-empty and smaller than `mask_max_frac` (larger ones are viewpoint
        smear, not a localized change), and exactly 1 elsewhere — so the loss
        SCALE is untouched and only its spatial allocation moves. `repeat`
        tiles the weight over a trailing per-goal axis with `repeat_interleave`
        semantics, matching how `rag_step` lays out its K candidate sets (and
        its local block's S within-chunk frames) in the batch dim.

        Returns `(w, valid, m)`, all None when disabled or no mask is supplied.
        """
        if goal_loss_mask is None or lam is None or lam <= 0:
            return None, None, None
        m = goal_loss_mask.reshape(-1, 1, *goal_loss_mask.shape[-2:]).float()
        frac = m.mean(dim=(1, 2, 3), keepdim=True)
        valid = (frac > 0) & (frac < self.mask_max_frac)
        w = torch.where(valid,
                        (1.0 + lam * m / frac.clamp(min=1e-6)) / (1.0 + lam),
                        torch.ones_like(m))
        if repeat > 1:
            w = w.repeat_interleave(repeat, dim=0)
        return w, valid, m

    def _generator_interact(self, action_interact, action_state):
        """The interaction tensor the GENERATOR is conditioned on, keyed by its
        declared `interact_dim` — no runtime knob:

          interact_dim == counts width (4): the legacy per-frame-count
              multi-hot, bit-identical for every existing checkpoint.
          interact_dim == 4 + k (k <= 4): the counts with the first k columns
              of the STATE basis `[door_state, 0, pickup_onset, put_onset]`
              appended (k=1 -> door fraction at the goal). The absolute door
              state is idempotent — a swing's repeated per-frame labels cannot
              re-toggle a door the state column pins — and supervises
              mid-swing direction; see `thor_door_state`.

        The MEMORY always receives the plain counts (`action_interact`) it was
        pretrained on. Datasets without a state basis (everything but THOR)
        pass action_state=None and require interact_dim <= 4."""
        # Probe through DDP's `.module` — DDP does not delegate attribute
        # access, so probing the wrapper directly reads 0 during multi-GPU
        # training and silently severs the generator's interaction channel
        # (torch.compile's OptimizedModule DOES delegate, so it needs no step).
        _gen = getattr(self.generator, 'module', self.generator)
        dim = getattr(_gen, 'interact_dim', 0)
        if dim == 0 or action_interact is None:
            return None
        extra = dim - action_interact.shape[-1]
        if extra <= 0:
            return action_interact
        assert action_state is not None, (
            f"generator interact_dim={dim} expects {extra} state column(s) but the "
            f"dataset served no interact_state (action_state is None)")
        return torch.cat([action_interact, action_state[..., :extra]], dim=-1)

    def compute_loss_latent(
        self,
        context_latent: torch.Tensor,
        goal_latent: torch.Tensor,
        action: torch.Tensor,
        stride: torch.Tensor,
        context_poses: torch.Tensor = None,
        context_times: torch.Tensor = None,
        context_mask: torch.Tensor = None,
        context_keys: torch.Tensor = None,
        query_cue: torch.Tensor = None,
        action_interact: torch.Tensor = None,
        action_state: torch.Tensor = None,
        goal_loss_mask: torch.Tensor = None,
        context_agent_state: torch.Tensor = None,
        goal_agent_state: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Diffusion training loss over the top retrieved set.

        The goal pose for `pnp_cl`'s `'pose'` positives is recovered inside the
        memory by inverting the formatter's `_nl` normalisation on `action`
        (which IS `goal_poses_nl`) — same pattern as
        `worldmem.WorldMemRetrieval`.

        **Hard-EM generator step.** The generator is trained on the SINGLE top
        set (set 0 — the highest-scored set, which `sample_latent` deploys), at
        one noise level and full batch. The retriever's multi-set RAG signal is
        computed separately and lazily in `rag_step` (forward-only); it does not
        enter `opt_loss` here. The retriever's contrastive loss is trained in a
        separate pre-training stage (`train_latent_retriever`), NOT here.
        Returns **two scalars**:

        - `opt_loss`: the loss to backward through = `gen_loss` (top-set
          denoising loss). No memory auxiliary loss is added.
        - `gen_loss`: identical to `opt_loss` (kept as a separate name for the
          clean diffusion-quality curve / forward-compat).

        Args:
            context_latent: (B, context_size, C, H, W) conditioning latents.
            goal_latent:    (B, num_goals, C, H, W) target latents.
            action:         (B, num_goals, action_dim) actions for each goal.
            stride:         (B, num_goals) relative times for each goal.
            context_poses:  (B, context_size, pose_dim) per-frame poses.
            context_times:  (B, context_size) per-frame timestamps in seconds.

        Returns:
            `(opt_loss, gen_loss)` — `opt_loss == gen_loss`.
        """
        # The retriever's contrastive loss is trained separately (pre-training,
        # `train_latent_retriever`).
        (
            context_latent, sel_poses, sel_times,
            context_score, chunk_scores,
        ) = self.prepare_context(
            context_latent, context_poses, context_times,
            action=action, stride=stride, context_mask=context_mask,
            context_keys=context_keys, query_cue=query_cue,
            action_interact=action_interact,
            context_agent_state=context_agent_state,
        )
        # context_latent: (B, num_goals, K, T_eff, C, H, W).
        B, num_goals, K = context_latent.shape[:3]
        BG = B * num_goals

        # Hard-EM: train the generator on the SINGLE top set (set 0 — the
        # highest-scored set, the one `sample_latent` deploys), at one noise
        # level and full batch (K=1 cost, single clean context). The retriever's
        # multi-set RAG signal is computed separately + lazily in `rag_step`. No
        # contrastive loss is computed here (it lives on the pre-training path).
        top_l, top_p, top_t = self._select_top_set(context_latent, sel_poses, sel_times)
        top_cs = chunk_scores[:, :, 0] if chunk_scores is not None else None
        x_start = goal_latent.reshape(BG, *goal_latent.shape[2:])
        context = top_l.reshape(BG, *top_l.shape[2:])
        act     = action.reshape(BG, -1)
        strd    = stride.reshape(BG)
        ctx_time = top_t.reshape(BG, *top_t.shape[2:]) if top_t is not None else None
        ctx_pose = top_p.reshape(BG, *top_p.shape[2:]) if top_p is not None else None
        # `.detach()` keeps the generator's chunk-score injection from sending a
        # second gradient back into the memory's queries/keys (the retriever
        # trains via `rag_step`, not this path).
        ctx_chunk_score = top_cs.reshape(BG, top_cs.shape[-1]).detach() if top_cs is not None else None
        t = torch.randint(0, self.diffusion.num_timesteps, (BG,), device=x_start.device)
        noise = torch.randn_like(x_start)
        model_kwargs = dict(action=act, context_latent=context, stride=strd,
                            context_time=ctx_time, context_pose=ctx_pose,
                            context_chunk_score=ctx_chunk_score)
        # Raw-pool generators (accepts_context_mask, run with memory=null) get
        # the frame-validity mask so left-padding never enters their implicit
        # retrieval. Probed through DDP's `.module` — DDP does not delegate
        # attribute access (same trap as the memory's query_source).
        _gen = getattr(self.generator, 'module', self.generator)
        if context_mask is not None and getattr(_gen, 'accepts_context_mask', False):
            model_kwargs['context_mask'] = (
                context_mask[:, None, :].expand(B, num_goals, context_mask.shape[-1]).reshape(BG, -1))
        # Per-goal interaction multi-hot, flattened into the batch dim like
        # `action`/`stride`. Pose deltas cannot express "the container opens",
        # so a THOR rollout without this never renders the eval-leg reveal.
        gen_interact = self._generator_interact(action_interact, action_state)
        if gen_interact is not None and getattr(_gen, 'interact_dim', 0) > 0:
            model_kwargs['action_interact'] = gen_interact.reshape(BG, -1)
        # Agent-state protocol: the top set's slot detections -> the predictor's
        # `[logit vi, dx, dy]` for the goal; the DiT is conditioned on it
        # (detached) and the predictor's own loss is added to `opt_loss` below.
        agent_aux = None
        mod = self._agent_module() if context_agent_state is not None else None
        if mod is not None and top_t is not None:
            slot_state = self._slot_states(context_agent_state, context_times, top_t).reshape(BG, top_t.shape[-1], -1)
            with torch.no_grad():                                                   # telemetry, retrieved slots only
                if slot_state.shape[1] > 1:
                    self.last_ret_sighting_frac.copy_((slot_state[:, :-1, 0] > 0).float().mean().reshape(1))
                    self.last_ret_dt.copy_((ctx_time[:, -1:] - ctx_time[:, :-1]).float().mean().reshape(1))
            pred = self._agent_predict(mod, slot_state, ctx_pose, ctx_time, act, strd)
            tgt = goal_agent_state.reshape(BG, -1) if goal_agent_state is not None else None
            model_kwargs['agent_state'] = self._agent_cond(pred, tgt)
            if tgt is not None:
                agent_aux = mod.agent_loss(pred, tgt)
        # Mask-weighted loss (see `mask_loss_lambda`): normalise the mask into a
        # mean-1 per-token weight, rejecting masks that are empty (nothing to
        # emphasise) or implausibly large (viewpoint smear). `w` broadcasts over
        # the channel axis, so mean_flat becomes a weighted mean.
        w, valid, m = self._mask_loss_weight(goal_loss_mask, self.mask_loss_lambda)
        if w is not None:
            with torch.no_grad():
                self.last_mask_weighted_frac.copy_(valid.float().mean().reshape(1))
                share = (w * m).sum(dim=(1, 2, 3)) / w.sum(dim=(1, 2, 3)).clamp(min=1e-6)
                sel = valid.reshape(-1)
                self.last_mask_loss_share.copy_(
                    (share[sel].mean() if bool(sel.any()) else share.new_zeros(())).reshape(1))
        _terms = self.diffusion.training_losses(self, x_start, t, model_kwargs,
                                                noise=noise, loss_weight=w,
                                                return_per_token=w is not None)
        loss = _terms["loss"]
        if w is not None and "per_token" in _terms:
            with torch.no_grad():
                pt = _terms["per_token"]                                  # (BG, H, W)
                mm, vv = m[:, 0], valid[:, 0, 0, 0]                       # (BG, H, W), (BG,)
                den_i = mm.sum(dim=(1, 2)).clamp(min=1e-6)
                den_o = (1 - mm).sum(dim=(1, 2)).clamp(min=1e-6)
                e_in = (pt * mm).sum(dim=(1, 2)) / den_i
                e_out = (pt * (1 - mm)).sum(dim=(1, 2)) / den_o
                if bool(vv.any()):
                    self.last_mask_in_mse.copy_(e_in[vv].mean().reshape(1))
                    self.last_mask_out_mse.copy_(e_out[vv].mean().reshape(1))
        gen_loss = loss.mean()
        opt_loss = gen_loss
        if agent_aux is not None:
            # `gen_loss` stays the diffusion-only curve; the predictor term is
            # optimised alongside it and logged on its own.
            opt_loss = opt_loss + self.agent_loss_weight * agent_aux.mean()
            self.last_agent_loss.copy_(agent_aux.detach().mean().reshape(1))
        return opt_loss, gen_loss

    def rag_step(
        self,
        context_latent,
        goal_latent,
        action,
        stride,
        context_poses=None,
        context_times=None,
        context_mask=None,
        context_keys=None,
        query_cue=None,
        action_interact=None,
        action_state=None,
        goal_loss_mask=None,
        num_noise_levels: int = 4,
        local_rag: bool = False,
        local_rag_weight: float = 1.0,
        context_agent_state=None,
        goal_agent_state=None,
        select=None,
    ):
        """Lazy RAG update (the every-N step paired with `compute_loss_latent`'s
        per-step hard-EM generator training).

        Retrieves the K candidate sets, evaluates each set's diffusion loss
        averaged over `num_noise_levels` STRATIFIED timesteps **under no_grad**
        (cheap — no generator gradient, no retained activations); τ is an EMA of
        `2·std` over that de-noised spread.

        Returns **only** the EMDR² RETRIEVER loss (the reader term
        `log p(a | Z_top-K)` is the generator, trained by `compute_loss_latent`):

            L_ret = − log Σ_k SG(p(a | z_k)) · p(z_k | q)

        with the reader surrogate `p(a | z_k) = exp(−ℓ_k / τ)` (SG — `ℓ_k` is the
        detached per-set diffusion loss). Implemented stably as
        `−logsumexp_k(−ℓ_k/τ + log p_k)`; the gradient flows only through the
        retriever prior `p_k = context_score` (→ QueryViT / adapters), and equals
        the earlier EM posterior-distillation gradient `−Σ_k γ_k ∇ log p_k`. The
        local (within-chunk) term has the same form over the S frames and REUSES
        the same τ (`per_set_loss_tau_ema`). Returns
        `retriever_loss_weight · (L_ret + local_rag_weight · L_ret^local)`, or
        `None` when there is no RAG signal (K ≤ 1 or no scores).

        Call this every `rag_every` steps and `.backward()` the result; the
        retriever gradient accumulates into `.grad` alongside the per-step
        generator (reader) loss before the optimizer step.
        """
        # Retrieval only — the memory returns its 5-tuple and skips the InfoNCE
        # encode. `context_keys` still feeds cached retrieval + per-frame scoring.
        # num_memories=1 → each set is ONE retrieved frame + the current frame
        # (context_size=2): the discriminative single-document setting for the
        # retriever reward. The generator (compute_loss_latent) keeps training on
        # the full context_size-1 set; the generator now handles a variable count.
        # `select` (B,) bool — restrict the whole RAG step (candidates, reward forwards, τ EMA, retriever
        # loss, telemetry) to these rows: `rag_anchored_only` scores only the reveal-anchored draws, the
        # ones where memory is the answer (the uniform draws' reward is the predictor's loss on a context
        # the current frame already answers, which teaches the scorer to AVOID sightings).
        # A rank with no selected row keeps the full batch and returns the loss x 0, so DDP's collective
        # in the memory's backward still fires on every rank (returning None would deadlock the others).
        _rag_w = 1.0
        if select is not None:
            sel = select.bool()
            if bool(sel.any()):
                _n = sel.shape[0]
                _take = lambda x: x[sel] if (torch.is_tensor(x) and x.dim() > 0 and x.shape[0] == _n) else x
                (context_latent, goal_latent, action, stride, context_poses, context_times, context_mask, context_keys,
                 query_cue, action_interact, action_state, goal_loss_mask, context_agent_state, goal_agent_state) = map(
                    _take, (context_latent, goal_latent, action, stride, context_poses, context_times, context_mask, context_keys,
                            query_cue, action_interact, action_state, goal_loss_mask, context_agent_state, goal_agent_state))
            else:
                _rag_w = 0.0
        pc = self.prepare_context(
            context_latent, context_poses, context_times,
            action=action, stride=stride, context_mask=context_mask,
            context_keys=context_keys, query_cue=query_cue,
            action_interact=action_interact,
            num_memories=1, return_local=local_rag,
            context_agent_state=context_agent_state,
        )
        if local_rag:
            (context_latent, sel_poses, sel_times,
             context_score, chunk_scores, local_bundle) = pc
        else:
            (context_latent, sel_poses, sel_times,
             context_score, chunk_scores) = pc
            local_bundle = None
        B, num_goals, K = context_latent.shape[:3]
        if context_score is None or K <= 1:
            return None                                     # no RAG signal to train on
        BG, BGK = B * num_goals, B * num_goals * K

        x_start = goal_latent.unsqueeze(2).expand(B, num_goals, K, *goal_latent.shape[2:]).reshape(BGK, *goal_latent.shape[2:])
        context = context_latent.reshape(BGK, *context_latent.shape[3:])
        act     = action.unsqueeze(2).expand(B, num_goals, K, action.shape[-1]).reshape(BGK, action.shape[-1])
        strd    = stride.unsqueeze(2).expand(B, num_goals, K).reshape(BGK)
        ctx_time = sel_times.reshape(BGK, *sel_times.shape[3:]) if sel_times is not None else None
        ctx_pose = sel_poses.reshape(BGK, *sel_poses.shape[3:]) if sel_poses is not None else None
        ctx_chunk_score = chunk_scores.reshape(BGK, chunk_scores.shape[-1]) if chunk_scores is not None else None
        base_kwargs = dict(action=act, context_latent=context, stride=strd,
                           context_time=ctx_time, context_pose=ctx_pose,
                           context_chunk_score=ctx_chunk_score)
        # The reward must be scored under the same conditioning the reader
        # (compute_loss_latent) trains with: without the interaction flag a
        # reveal goal is unpredictable from ANY context and the per-set losses
        # flatten exactly on the discriminating samples.
        _gen = getattr(self.generator, 'module', self.generator)
        gen_interact = self._generator_interact(action_interact, action_state)
        if gen_interact is not None and getattr(_gen, 'interact_dim', 0) > 0:
            base_kwargs['action_interact'] = (
                gen_interact.reshape(BG, -1).repeat_interleave(K, dim=0))

        # Agent-state protocol: each candidate set's slots -> its own prediction,
        # which conditions that set's reward forward; the predictor's loss on the
        # set joins the reward (`agent_reward_weight`). Forward-only, like the
        # diffusion reward — the predictor is trained in compute_loss_latent.
        agent_set = None
        mod = self._agent_module() if context_agent_state is not None else None
        if mod is not None and sel_times is not None:
            with torch.no_grad():
                slot_state = self._slot_states(context_agent_state, context_times, sel_times).reshape(BGK, sel_times.shape[-1], -1)
                pred = self._agent_predict(mod, slot_state, ctx_pose, ctx_time, act, strd)
                tgt = (goal_agent_state.unsqueeze(2).expand(B, num_goals, K, goal_agent_state.shape[-1]).reshape(BGK, -1)
                       if goal_agent_state is not None else None)
                base_kwargs['agent_state'] = self._agent_cond(pred, tgt)
                if tgt is not None:
                    agent_set = (self.agent_reward_weight * mod.agent_loss(pred, tgt)).reshape(B, num_goals, K)

        # Change-region weighting of the REWARD (see `rag_mask_loss_lambda`): the
        # candidate sets are then ranked by how well each predicts the region
        # that actually changed, instead of by overall scene match. τ is an EMA
        # of the per-set spread, so it re-normalises to the weighted scale on
        # its own. `repeat=K`: the K sets are laid out with repeat_interleave.
        _rag_lam = (self.mask_loss_lambda if self.rag_mask_loss_lambda is None
                    else self.rag_mask_loss_lambda)
        w_set, _, _ = self._mask_loss_weight(goal_loss_mask, _rag_lam, repeat=K)

        # ── Multi-noise, stratified per-set loss (forward-only) ────────────────
        T = self.diffusion.num_timesteps
        M = max(1, int(num_noise_levels))
        with torch.no_grad():
            acc = None
            for m in range(M):
                lo, hi = (m * T) // M, ((m + 1) * T) // M
                hi = max(hi, lo + 1)                          # guard tiny strata
                t_goal = torch.randint(lo, hi, (BG,), device=x_start.device)   # one t per goal, stratum m
                t      = t_goal.repeat_interleave(K)          # shared across the K sets
                noise  = torch.randn(BG, *goal_latent.shape[2:], device=x_start.device).repeat_interleave(K, dim=0)
                loss_m = self.diffusion.training_losses(
                    self, x_start, t, base_kwargs, noise=noise, loss_weight=w_set)["loss"]
                acc = loss_m if acc is None else acc + loss_m
            per_set_loss = (acc / M).reshape(B, num_goals, K)          # (B, ng, K) — detached
            # Telemetry (non-persistent attr, analysis only): the reward's two terms per candidate set
            # and the sets' times, so a probe can see which term prefers which candidate.
            self.last_rag_terms = dict(diffusion=per_set_loss.detach().clone(),
                                       agent=(agent_set.detach().clone() if agent_set is not None else None),
                                       sel_times=(sel_times.detach().clone() if sel_times is not None else None))
            if agent_set is not None:
                per_set_loss = per_set_loss + agent_set                  # + the predictor's loss on this set

            # τ — EMA of 2·std over the de-noised multi-noise per-set spread.
            spread = (per_set_loss.std(dim=2) * 2.0).mean()
            d = self.per_set_loss_tau_ema_decay
            if _rag_w > 0: self.per_set_loss_tau_ema.mul_(d).add_(spread, alpha=(1.0 - d))
            tau = self.per_set_loss_tau_ema.clamp(min=1e-8)

        log_p = context_score.clamp(min=1e-12).log()                  # (B, ng, K) = log p_η — grad → retriever
        # EMDR² retriever loss: −log Σ_k SG(p(a|z_k)) p(z_k|q), reader surrogate
        # p(a|z_k) = exp(−ℓ_k/τ) (SG — ℓ_k is the detached per-set diffusion loss,
        # τ = per_set_loss_tau_ema). = −logsumexp_k(−ℓ_k/τ + log p_k); the gradient
        # flows only through log p_k (the retriever prior). The reward is centred by
        # its per-(sample,goal) min — a detached constant that keeps the loss value
        # O(1) without changing the gradient (logsumexp is shift-equivariant).
        adv = per_set_loss - per_set_loss.amin(dim=2, keepdim=True)   # (B, ng, K) ≥ 0, detached
        ret_loss = -torch.logsumexp(-adv / tau + log_p, dim=2).mean()
        self.last_em_ret_loss.copy_(ret_loss.detach().reshape(1))
        with torch.no_grad():                                         # telemetry only (RAG posterior + rank agreement)
            q = torch.softmax(log_p - per_set_loss / tau, dim=2)      # RAG posterior γ_k
            # Re-rank rate: fraction of (sample, goal) where the posterior's best
            # set differs from the prior's best set. ~0 ⇒ the reward only confirms
            # the prior (no re-ranking signal for the retriever).
            rerank = (q.argmax(dim=2) != log_p.argmax(dim=2)).float().mean()
            self.last_em_rerank_rate.copy_(rerank.reshape(1))
            # Spearman ρ(prior, −reward) over the K sets, per (sample, goal): rank
            # the sets by the prior (log_p) and by −per_set_loss, then correlate the
            # ranks. Rank-based ⇒ τ/peakedness-invariant; measures whether the
            # retriever ranks sets the way the reward does.
            rp = log_p.argsort(dim=2).argsort(dim=2).float()              # ranks of the prior
            rr = (-per_set_loss).argsort(dim=2).argsort(dim=2).float()    # ranks of −reward
            rp = rp - rp.mean(dim=2, keepdim=True)
            rr = rr - rr.mean(dim=2, keepdim=True)
            denom = (rp.pow(2).sum(dim=2).sqrt() * rr.pow(2).sum(dim=2).sqrt()).clamp(min=1e-12)
            rank_corr = ((rp * rr).sum(dim=2) / denom).mean()            # mean Spearman ρ over (sample, goal)
            self.last_em_rank_corr.copy_(rank_corr.reshape(1))

        # ── Local (within-chunk) RAG ──────────────────────────────────────────
        # The global block above ranks CHUNKS (each via its max-frame). This
        # ranks the S FRAMES *inside* one chunk by their single-frame diffusion
        # reward and distils that into the retriever's per-frame prior — the
        # signal the chunk-max collapse otherwise discards (only the argmax frame
        # gets gradient). The chunk is drawn at RANDOM from the K top-scoring sets,
        # per (sample, goal): the top chunk's frames are the most uniformly
        # relevant (flattest local reward) and top-only never visits an
        # under-ranked chunk to correct it, so exploration over the top-K beats
        # always refining set 0. Reward is forward-only with shared t+noise across
        # the S frames (common random numbers) so the spread isolates the frame.
        local_ret_loss = None
        if local_rag and local_bundle is not None:
            dev = context_latent.device
            S = local_bundle.chunks.shape[4]
            with torch.no_grad():
                rk = torch.randint(0, K, (B, num_goals), device=dev)             # random chunk per (sample, goal)
            bb = torch.arange(B,         device=dev)[:, None]
            gg = torch.arange(num_goals, device=dev)[None, :]
            cand   = local_bundle.chunks     [bb, gg, rk, 0]                      # (B, ng, S, C, H, W)
            cand_p = local_bundle.poses      [bb, gg, rk, 0]                      # (B, ng, S, pose_dim)
            cand_t = local_bundle.times      [bb, gg, rk, 0]                      # (B, ng, S)
            fprior = local_bundle.frame_prior[bb, gg, rk, 0]                      # (B, ng, S) — grad, sums to 1 over S
            # Current frame (shared across sets) appended at each candidate's
            # tail, mirroring prepare_context's [retrieved, current] → ctx=2.
            cur_l = context_latent[:, :, 0, -1]                                   # (B, ng, C, H, W)
            cur_p = sel_poses[:, :, 0, -1] if sel_poses is not None else None     # (B, ng, pose_dim)
            cur_t = sel_times[:, :, 0, -1] if sel_times is not None else None     # (B, ng)
            BGS = B * num_goals * S
            loc_ctx = torch.stack(
                [cand, cur_l[:, :, None].expand(B, num_goals, S, *cur_l.shape[2:])], dim=3,
            )                                                                     # (B, ng, S, 2, C, H, W)
            loc_x    = goal_latent.unsqueeze(2).expand(B, num_goals, S, *goal_latent.shape[2:]).reshape(BGS, *goal_latent.shape[2:])
            loc_c    = loc_ctx.reshape(BGS, 2, *cand.shape[3:])
            loc_act  = action.unsqueeze(2).expand(B, num_goals, S, action.shape[-1]).reshape(BGS, action.shape[-1])
            loc_strd = stride.unsqueeze(2).expand(B, num_goals, S).reshape(BGS)
            loc_time = (torch.stack([cand_t, cur_t[:, :, None].expand(B, num_goals, S)], dim=3).reshape(BGS, 2)
                        if cand_t is not None else None)
            loc_pose = (torch.stack([cand_p, cur_p[:, :, None].expand(B, num_goals, S, cand_p.shape[-1])], dim=3).reshape(BGS, 2, cand_p.shape[-1])
                        if cand_p is not None else None)
            loc_cs   = fprior.reshape(BGS, 1).detach()                            # per-frame relevance injection (1 past frame)
            loc_kwargs = dict(action=loc_act, context_latent=loc_c, stride=loc_strd,
                              context_time=loc_time, context_pose=loc_pose,
                              context_chunk_score=loc_cs)
            if 'action_interact' in base_kwargs:
                loc_kwargs['action_interact'] = (
                    gen_interact.reshape(BG, -1).repeat_interleave(S, dim=0))
            # Agent-state protocol on the single-frame candidates: [frame, current].
            agent_frame = None
            if mod is not None and loc_time is not None and loc_pose is not None:
                with torch.no_grad():
                    loc_state = self._slot_states(context_agent_state, context_times,
                                                  loc_time.reshape(B, num_goals, S, 2)).reshape(BGS, 2, -1)
                    pred_l = self._agent_predict(mod, loc_state, loc_pose, loc_time, loc_act, loc_strd)
                    tgt_l = (goal_agent_state.unsqueeze(2).expand(B, num_goals, S, goal_agent_state.shape[-1]).reshape(BGS, -1)
                             if goal_agent_state is not None else None)
                    loc_kwargs['agent_state'] = self._agent_cond(pred_l, tgt_l)
                    if tgt_l is not None:
                        agent_frame = (self.agent_reward_weight * mod.agent_loss(pred_l, tgt_l)).reshape(B, num_goals, S)
            # Same weighting for the within-chunk reward; S frames per goal.
            w_loc, _, _ = self._mask_loss_weight(goal_loss_mask, _rag_lam, repeat=S)
            with torch.no_grad():
                acc_l = None
                for m in range(M):
                    lo, hi = (m * T) // M, ((m + 1) * T) // M
                    hi = max(hi, lo + 1)
                    t_g = torch.randint(lo, hi, (BG,), device=dev)               # one t per goal, shared across the S frames
                    t_l = t_g.repeat_interleave(S)
                    noise_l = torch.randn(BG, *goal_latent.shape[2:], device=dev).repeat_interleave(S, dim=0)
                    loss_l = self.diffusion.training_losses(
                        self, loc_x, t_l, loc_kwargs, noise=noise_l, loss_weight=w_loc)["loss"]
                    acc_l = loss_l if acc_l is None else acc_l + loss_l
                per_frame_loss = (acc_l / M).reshape(B, num_goals, S)            # (B, ng, S) — detached
                if agent_frame is not None:
                    per_frame_loss = per_frame_loss + agent_frame
                # `per_frame_loss_tau_ema` is kept as a DIAGNOSTIC of the within-chunk
                # spread (train/local_tau_ema); it is NOT the EM temperature. The local
                # EMDR² loss REUSES the global τ (`tau` = per_set_loss_tau_ema) so the
                # finer within-chunk decision sits on the same loss scale as the chunk
                # decision → a proportionally softer posterior (no over-sharpening).
                d = self.per_set_loss_tau_ema_decay
                if _rag_w > 0: self.per_frame_loss_tau_ema.mul_(d).add_((per_frame_loss.std(dim=2) * 2.0).mean(), alpha=(1.0 - d))
            log_p_l = fprior.clamp(min=1e-12).log()                             # (B, ng, S) — grad → retriever (per-frame keys)
            # EMDR² retriever loss over the S within-chunk frames (same form as the
            # global term; reward exp(−ℓ_frame/τ), reusing the global `tau`).
            adv_l = per_frame_loss - per_frame_loss.amin(dim=2, keepdim=True)   # (B, ng, S) ≥ 0, detached
            local_ret_loss = -torch.logsumexp(-adv_l / tau + log_p_l, dim=2).mean()
            self.last_em_local_ret_loss.copy_(local_ret_loss.detach().reshape(1))
            with torch.no_grad():                                              # local rank-agreement telemetry
                rp_l = log_p_l.argsort(dim=2).argsort(dim=2).float()
                rr_l = (-per_frame_loss).argsort(dim=2).argsort(dim=2).float()
                rp_l = rp_l - rp_l.mean(dim=2, keepdim=True)
                rr_l = rr_l - rr_l.mean(dim=2, keepdim=True)
                denom_l = (rp_l.pow(2).sum(dim=2).sqrt() * rr_l.pow(2).sum(dim=2).sqrt()).clamp(min=1e-12)
                self.last_em_local_rank_corr.copy_(
                    ((rp_l * rr_l).sum(dim=2) / denom_l).mean().reshape(1))

        # The reader term of EMDR² (log p(a | Z_top-K)) is the generator, trained by
        # the main forward (`compute_loss_latent`); rag_step returns ONLY the
        # retriever term. No generator forward here ⇒ nothing for torch.compile to
        # corrupt across the no_grad reward forwards, and the generator is not
        # backward'd twice per rag step.
        ret_total = ret_loss if local_ret_loss is None else ret_loss + local_rag_weight * local_ret_loss
        return self.retriever_loss_weight * ret_total * _rag_w          # x0 = no selected row on this rank (see `select`)

    def rag_step_ddim(
        self,
        context_latent,
        goal_latent,
        action,
        stride,
        context_poses=None,
        context_times=None,
        context_mask=None,
        context_keys=None,
        query_cue=None,
        action_interact=None,
        action_state=None,
        goal_loss_mask=None,
        ddim_steps: int = 8,
        mse_weight: float = 1.0,
        local_rag: bool = False,
        local_rag_weight: float = 1.0,
        context_agent_state=None,
        goal_agent_state=None,
        select=None,
    ):
        """`rag_step` with a SAMPLED-outcome reward instead of the diffusion loss.

        Motivation: single-step per-set predictions do not differ in OBJECT
        STATE across candidate contexts — the diffusion loss has nothing
        state-like to rank — while DDIM samples of >= 8 steps separate stale
        from fresh context (measured). So each of the K
        sets gets one `ddim_steps`-step sample from pure noise (the SAME z
        across sets — the paired comparison), and the per-set reward is the
        MASK-WEIGHTED x0 error

            ℓ_k = mse_weight · mean_tokens( w · (x0_k − goal)² )

        with `w` the same mean-1 per-token weight `compute_loss_latent` /
        `rag_step` apply (`_mask_loss_weight`; λ = `rag_mask_loss_lambda`, None
        → `mask_loss_lambda`; masks failing the `mask_max_frac` gate → w ≡ 1):
        the change region carries ~λ/(1+λ) of the reward instead of its ~2%
        area share, so a set is ranked by whether its sample renders WHAT
        CHANGED rather than by how well it reproduces the unchanged background
        (which every pose-matched set does). With no mask or λ = 0 this is the
        plain sample MSE. The EMDR² retriever term, τ EMA (its own buffer —
        different reward scale), telemetry, and the optional LOCAL
        (within-chunk) term all mirror `rag_step`, with the local per-frame
        reward built by the same weighted sample-MSE construction (so sharing
        the global τ is exact here, not a scale compromise). Sampling is no_grad; gradient flows only through the
        retriever prior / per-frame prior. Unlike `rag_step`, `action_interact`
        also conditions the GENERATOR (a reveal sample cannot be rendered,
        hence not rewarded, without it). Cost: ddim_steps · (K + S·local_rag)
        forwards per goal — use a larger `rag_every` than with `rag_step`.
        """
        # `select` (B,) bool — restrict the whole RAG step (candidates, reward forwards, τ EMA, retriever
        # loss, telemetry) to these rows: `rag_anchored_only` scores only the reveal-anchored draws, the
        # ones where memory is the answer (the uniform draws' reward is the predictor's loss on a context
        # the current frame already answers, which teaches the scorer to AVOID sightings).
        # A rank with no selected row keeps the full batch and returns the loss x 0, so DDP's collective
        # in the memory's backward still fires on every rank (returning None would deadlock the others).
        _rag_w = 1.0
        if select is not None:
            sel = select.bool()
            if bool(sel.any()):
                _n = sel.shape[0]
                _take = lambda x: x[sel] if (torch.is_tensor(x) and x.dim() > 0 and x.shape[0] == _n) else x
                (context_latent, goal_latent, action, stride, context_poses, context_times, context_mask, context_keys,
                 query_cue, action_interact, action_state, goal_loss_mask, context_agent_state, goal_agent_state) = map(
                    _take, (context_latent, goal_latent, action, stride, context_poses, context_times, context_mask, context_keys,
                            query_cue, action_interact, action_state, goal_loss_mask, context_agent_state, goal_agent_state))
            else:
                _rag_w = 0.0
        pc = self.prepare_context(
            context_latent, context_poses, context_times,
            action=action, stride=stride, context_mask=context_mask,
            context_keys=context_keys, query_cue=query_cue,
            action_interact=action_interact, num_memories=1,
            return_local=local_rag,
            context_agent_state=context_agent_state,
        )
        if local_rag:
            (context_latent, sel_poses, sel_times,
             context_score, chunk_scores, local_bundle) = pc
        else:
            (context_latent, sel_poses, sel_times,
             context_score, chunk_scores) = pc
            local_bundle = None
        B, num_goals, K = context_latent.shape[:3]
        if context_score is None or K <= 1:
            return None
        BG, BGK = B * num_goals, B * num_goals * K

        x_start = goal_latent.unsqueeze(2).expand(B, num_goals, K, *goal_latent.shape[2:]).reshape(BGK, *goal_latent.shape[2:])
        context = context_latent.reshape(BGK, *context_latent.shape[3:])
        act     = action.unsqueeze(2).expand(B, num_goals, K, action.shape[-1]).reshape(BGK, action.shape[-1])
        strd    = stride.unsqueeze(2).expand(B, num_goals, K).reshape(BGK)
        ctx_time = sel_times.reshape(BGK, *sel_times.shape[3:]) if sel_times is not None else None
        ctx_pose = sel_poses.reshape(BGK, *sel_poses.shape[3:]) if sel_poses is not None else None
        ctx_chunk_score = chunk_scores.reshape(BGK, chunk_scores.shape[-1]) if chunk_scores is not None else None
        base_kwargs = dict(action=act, context_latent=context, stride=strd,
                           context_time=ctx_time, context_pose=ctx_pose,
                           context_chunk_score=ctx_chunk_score)
        _gen = getattr(self.generator, 'module', self.generator)
        gen_interact = self._generator_interact(action_interact, action_state)
        if gen_interact is not None and getattr(_gen, 'interact_dim', 0) > 0:
            base_kwargs['action_interact'] = (
                gen_interact.reshape(BG, -1).repeat_interleave(K, dim=0))

        # Agent-state protocol (see rag_step): per-set prediction conditions the
        # sampled forward; the predictor's loss on the set joins the reward.
        agent_set = None
        mod = self._agent_module() if context_agent_state is not None else None
        if mod is not None and sel_times is not None:
            with torch.no_grad():
                slot_state = self._slot_states(context_agent_state, context_times, sel_times).reshape(BGK, sel_times.shape[-1], -1)
                pred = self._agent_predict(mod, slot_state, ctx_pose, ctx_time, act, strd)
                tgt = (goal_agent_state.unsqueeze(2).expand(B, num_goals, K, goal_agent_state.shape[-1]).reshape(BGK, -1)
                       if goal_agent_state is not None else None)
                base_kwargs['agent_state'] = self._agent_cond(pred, tgt)
                if tgt is not None:
                    agent_set = (self.agent_reward_weight * mod.agent_loss(pred, tgt)).reshape(B, num_goals, K)

        # Lazily build/re-use the ddim-respaced twin of the training diffusion
        # (create_diffusion defaults == the training config: linear, 1000 steps,
        # learn_sigma).
        if getattr(self, '_rag_ddim_steps', None) != int(ddim_steps):
            from ..generators.diffusion import create_diffusion
            self._rag_ddim = create_diffusion(f"ddim{int(ddim_steps)}")
            self._rag_ddim_steps = int(ddim_steps)

        # Same mean-1 per-token weight as compute_loss_latent / rag_step; None
        # when no mask is served or λ = 0 (→ plain sample MSE).
        _rag_lam = (self.mask_loss_lambda if self.rag_mask_loss_lambda is None
                    else self.rag_mask_loss_lambda)
        w_set, valid_set, m_set = self._mask_loss_weight(goal_loss_mask, _rag_lam, repeat=K)

        def _sampled_reward(kwargs, x_tgt, n_rep, w):
            """Mask-weighted MSE on a shared-noise DDIM sample per candidate
            (z drawn per goal, repeated over the candidate axis — common random
            numbers, so the reward spread isolates the candidate). `w` is the
            (N, 1, H, W) per-token weight or None. Returns
            (weighted reward, unweighted per-sample MSE, per-token sq error)."""
            z = torch.randn(BG, *goal_latent.shape[2:],
                            device=x_tgt.device).repeat_interleave(n_rep, dim=0)
            x0 = self._rag_ddim.ddim_sample_loop(
                self.forward, z.shape, z, clip_denoised=False,
                model_kwargs=kwargs, device=z.device)
            sq = (x0 - x_tgt) ** 2                                              # (N, C, H, W)
            mse_c = sq.flatten(1).mean(1)
            r = mse_c if w is None else (sq * w).flatten(1).mean(1)             # w broadcasts over C
            return mse_weight * r, mse_c, sq

        with torch.no_grad():
            per_set, mse_c, sq_c = _sampled_reward(base_kwargs, x_start, K, w_set)
            per_set = per_set.reshape(B, num_goals, K)
            if agent_set is not None:
                per_set = per_set + agent_set
            spread = (per_set.std(dim=2) * 2.0).mean()
            d = self.per_set_loss_tau_ema_decay
            if _rag_w > 0: self.rag_ddim_tau_ema.mul_(d).add_(spread, alpha=(1.0 - d))
            tau = self.rag_ddim_tau_ema.clamp(min=1e-8)

        log_p = context_score.clamp(min=1e-12).log()
        adv = per_set - per_set.amin(dim=2, keepdim=True)
        ret_loss = -torch.logsumexp(-adv / tau + log_p, dim=2).mean()
        self.last_em_ret_loss.copy_(ret_loss.detach().reshape(1))
        with torch.no_grad():
            q = torch.softmax(log_p - per_set / tau, dim=2)
            self.last_em_rerank_rate.copy_(
                (q.argmax(dim=2) != log_p.argmax(dim=2)).float().mean().reshape(1))
            # Combined-reward rank agreement — SAME semantics as rag_step's
            # rank_corr (Spearman ρ(prior, −reward) over K), so the field stays
            # comparable across rag modes.
            rp_c = log_p.argsort(dim=2).argsort(dim=2).float()
            rr_c = (-per_set).argsort(dim=2).argsort(dim=2).float()
            rp_c = rp_c - rp_c.mean(dim=2, keepdim=True)
            rr_c = rr_c - rr_c.mean(dim=2, keepdim=True)
            dn_c = (rp_c.pow(2).sum(dim=2).sqrt() * rr_c.pow(2).sum(dim=2).sqrt()).clamp(min=1e-12)
            self.last_em_rank_corr.copy_(((rp_c * rr_c).sum(dim=2) / dn_c).mean().reshape(1))
            # Region-split telemetry (see the buffer comments in __init__):
            # cross-set spread of the UNWEIGHTED whole-frame MSE vs of the MSE
            # INSIDE the change region, the latter over goals with a usable mask.
            self.last_rag_mse_spread.copy_(mse_c.reshape(B, num_goals, K).std(dim=2).mean().reshape(1))
            if m_set is not None and bool(valid_set.any()):
                mm = m_set.reshape(-1, *m_set.shape[-2:])                         # (BG, H, W) — per goal, shared over K
                if mm.shape[0] == BGK:
                    mm = mm[::K]
                err = sq_c.mean(1).reshape(BG, K, *sq_c.shape[-2:])               # (BG, K, H, W)
                e_in = (err * mm[:, None]).sum(dim=(2, 3)) / mm.sum(dim=(1, 2)).clamp(min=1e-6)[:, None]
                vv = valid_set.reshape(-1)
                vv = vv[::K] if vv.shape[0] == BGK else vv
                self.last_rag_in_mask_spread.copy_(e_in.std(dim=1)[vv].mean().reshape(1))

        # ── Local (within-chunk) term — rag_step's local block with the sampled
        # reward. Same candidate packing (random top-K chunk per (sample, goal),
        # [frame, current] context, fprior as the per-frame relevance/prior);
        # the SAME τ as the global term — here both rewards are literally the
        # same weighted sample-MSE construction, so the shared scale is exact,
        # not a design compromise. Cost: adds S · ddim_steps forwards per goal.
        local_ret_loss = None
        if local_rag and local_bundle is not None:
            dev = context_latent.device
            S = local_bundle.chunks.shape[4]
            with torch.no_grad():
                rk = torch.randint(0, K, (B, num_goals), device=dev)
            bb = torch.arange(B,         device=dev)[:, None]
            gg = torch.arange(num_goals, device=dev)[None, :]
            cand   = local_bundle.chunks     [bb, gg, rk, 0]
            cand_p = local_bundle.poses      [bb, gg, rk, 0]
            cand_t = local_bundle.times      [bb, gg, rk, 0]
            fprior = local_bundle.frame_prior[bb, gg, rk, 0]
            cur_l = context_latent[:, :, 0, -1]
            cur_p = sel_poses[:, :, 0, -1] if sel_poses is not None else None
            cur_t = sel_times[:, :, 0, -1] if sel_times is not None else None
            BGS = B * num_goals * S
            loc_ctx = torch.stack(
                [cand, cur_l[:, :, None].expand(B, num_goals, S, *cur_l.shape[2:])], dim=3)
            loc_x    = goal_latent.unsqueeze(2).expand(B, num_goals, S, *goal_latent.shape[2:]).reshape(BGS, *goal_latent.shape[2:])
            loc_kwargs = dict(
                action=action.unsqueeze(2).expand(B, num_goals, S, action.shape[-1]).reshape(BGS, action.shape[-1]),
                context_latent=loc_ctx.reshape(BGS, 2, *cand.shape[3:]),
                stride=stride.unsqueeze(2).expand(B, num_goals, S).reshape(BGS),
                context_time=(torch.stack([cand_t, cur_t[:, :, None].expand(B, num_goals, S)], dim=3).reshape(BGS, 2)
                              if cand_t is not None else None),
                context_pose=(torch.stack([cand_p, cur_p[:, :, None].expand(B, num_goals, S, cand_p.shape[-1])], dim=3).reshape(BGS, 2, cand_p.shape[-1])
                              if cand_p is not None else None),
                context_chunk_score=fprior.reshape(BGS, 1).detach())
            if 'action_interact' in base_kwargs:
                loc_kwargs['action_interact'] = (
                    gen_interact.reshape(BG, -1).repeat_interleave(S, dim=0))
            agent_frame = None
            if mod is not None and loc_kwargs['context_time'] is not None and loc_kwargs['context_pose'] is not None:
                with torch.no_grad():
                    lt = loc_kwargs['context_time']
                    loc_state = self._slot_states(context_agent_state, context_times,
                                                  lt.reshape(B, num_goals, S, 2)).reshape(BGS, 2, -1)
                    pred_l = self._agent_predict(mod, loc_state, loc_kwargs['context_pose'], lt,
                                                 loc_kwargs['action'], loc_kwargs['stride'])
                    tgt_l = (goal_agent_state.unsqueeze(2).expand(B, num_goals, S, goal_agent_state.shape[-1]).reshape(BGS, -1)
                             if goal_agent_state is not None else None)
                    loc_kwargs['agent_state'] = self._agent_cond(pred_l, tgt_l)
                    if tgt_l is not None:
                        agent_frame = (self.agent_reward_weight * mod.agent_loss(pred_l, tgt_l)).reshape(B, num_goals, S)
            w_loc, _, _ = self._mask_loss_weight(goal_loss_mask, _rag_lam, repeat=S)
            with torch.no_grad():
                per_frame = _sampled_reward(loc_kwargs, loc_x, S, w_loc)[0].reshape(B, num_goals, S)
                if agent_frame is not None:
                    per_frame = per_frame + agent_frame
                d = self.per_set_loss_tau_ema_decay
                if _rag_w > 0: self.per_frame_loss_tau_ema.mul_(d).add_(
                    (per_frame.std(dim=2) * 2.0).mean(), alpha=(1.0 - d))
            log_p_l = fprior.clamp(min=1e-12).log()
            adv_l = per_frame - per_frame.amin(dim=2, keepdim=True)
            local_ret_loss = -torch.logsumexp(-adv_l / tau + log_p_l, dim=2).mean()
            self.last_em_local_ret_loss.copy_(local_ret_loss.detach().reshape(1))
            with torch.no_grad():
                rp_l = log_p_l.argsort(dim=2).argsort(dim=2).float()
                rr_l = (-per_frame).argsort(dim=2).argsort(dim=2).float()
                rp_l = rp_l - rp_l.mean(dim=2, keepdim=True)
                rr_l = rr_l - rr_l.mean(dim=2, keepdim=True)
                denom_l = (rp_l.pow(2).sum(dim=2).sqrt() * rr_l.pow(2).sum(dim=2).sqrt()).clamp(min=1e-12)
                self.last_em_local_rank_corr.copy_(
                    ((rp_l * rr_l).sum(dim=2) / denom_l).mean().reshape(1))

        ret_total = ret_loss if local_ret_loss is None else ret_loss + local_rag_weight * local_ret_loss
        return self.retriever_loss_weight * ret_total * _rag_w          # x0 = no selected row on this rank (see `select`)

    def _select_top_set(self, context_latent, sel_poses, sel_times):
        """Collapse the K-set axis to the single best set for generation.

        The beam search returns sets sorted by score (set 0 = highest), so for
        sampling we use set 0. Returns tensors with the K axis removed.
        """
        context_latent = context_latent[:, :, 0]                               # (B, num_goals, T_eff, ...)
        sel_poses = sel_poses[:, :, 0] if sel_poses is not None else None
        sel_times = sel_times[:, :, 0] if sel_times is not None else None
        return context_latent, sel_poses, sel_times

    def sample_latent(
        self,
        context_latent: torch.Tensor,
        action: torch.Tensor,
        stride: torch.Tensor,
        num_goals: int = 1,
        progress: bool = False,
        context_poses: torch.Tensor = None,
        context_times: torch.Tensor = None,
        context_mask: torch.Tensor = None,
        context_keys: torch.Tensor = None,
        query_cue: torch.Tensor = None,
        action_interact: torch.Tensor = None,
        action_state: torch.Tensor = None,
        use_ddim: bool = False,
        context_agent_state: torch.Tensor = None,
    ) -> torch.Tensor:
        """Run reverse diffusion conditioned on latent context frames.

        When the memory returns K candidate sets, the top-scored set (set 0) is
        used for generation.

        Args:
            context_latent: (B, context_size, C, H, W) conditioning latents.
            action:         (B * num_goals, action_dim) actions toward each goal.
            stride:         (B * num_goals,) relative times for each goal.
            num_goals:      number of future frames to generate per batch element.
            progress:       show tqdm progress bar during sampling.
            use_ddim:       sample with `ddim_sample_loop` (deterministic, η=0)
                            instead of ancestral DDPM. Pair with a `"ddimN"`
                            `timestep_respacing` on the diffusion object.
            context_poses:  (B, context_size, pose_dim) per-frame poses.
            context_times:  (B, context_size) per-frame timestamps in seconds.
            context_keys:   (B, context_size, key_dim) pre-computed retriever
                            keys. When supplied, the memory uses them as cached
                            retrieval/scoring keys instead of encoding the
                            context frames with QueryViT — keeping the retrieval
                            key space identical to training. Only valid for REAL
                            context frames (the keys exist for them); `None`
                            falls back to live encoding. AR rollout leaves this
                            `None` (its context contains generated frames).

        Returns:
            (B * num_goals, C, H, W) sampled latents.
        """
        # action/stride arrive flattened across goals; unflatten for per-goal
        # memory retrieval, then collapse the K-set axis and re-flatten.
        B_in = context_latent.shape[0]
        action_per_goal = action.reshape(B_in, num_goals, -1)
        stride_per_goal = stride.reshape(B_in, num_goals)
        context_latent, sel_poses, sel_times, _, chunk_scores = self.prepare_context(
            context_latent, context_poses, context_times,
            action=action_per_goal, stride=stride_per_goal, context_mask=context_mask,
            context_keys=context_keys, query_cue=query_cue,
            action_interact=action_interact,
            context_agent_state=context_agent_state,
        )
        # context_latent: (B, num_goals, K, T_eff, ...) → take the top set.
        context_latent, sel_poses, sel_times = self._select_top_set(
            context_latent, sel_poses, sel_times,
        )
        # chunk_scores: (B, num_goals, K, num_memories) → top set → (B, ng, M).
        if chunk_scores is not None:
            chunk_scores = chunk_scores[:, :, 0]
        device   = context_latent.device
        expanded = context_latent.flatten(0, 1)                              # (B*num_goals, T_eff, C, H, W)
        # Size the noise `z` from the generator's `in_channels` and the context's
        # spatial dims.
        out_c = self.generator.in_channels
        z = torch.randn(B_in * num_goals, out_c, *expanded.shape[3:], device=device)

        ctx_time = sel_times.flatten(0, 1) if sel_times is not None else None
        ctx_pose = sel_poses.flatten(0, 1) if sel_poses is not None else None
        # `.detach()` for consistency with `compute_loss_latent` — sampling
        # is no-grad anyway, but keeping the convention identical avoids any
        # surprise if someone later wraps sample_latent in a grad-enabled path.
        ctx_chunk_score = chunk_scores.flatten(0, 1).detach() if chunk_scores is not None else None

        model_kwargs = dict(action=action, context_latent=expanded, stride=stride,
                            context_time=ctx_time, context_pose=ctx_pose,
                            context_chunk_score=ctx_chunk_score)
        # Same raw-pool mask forwarding as compute_loss_latent (see there).
        _gen = getattr(self.generator, 'module', self.generator)
        if context_mask is not None and getattr(_gen, 'accepts_context_mask', False):
            model_kwargs['context_mask'] = (
                context_mask[:, None, :].expand(B_in, num_goals, context_mask.shape[-1])
                .reshape(B_in * num_goals, -1))
        gen_interact = self._generator_interact(action_interact, action_state)
        if gen_interact is not None and getattr(_gen, 'interact_dim', 0) > 0:
            model_kwargs['action_interact'] = gen_interact.reshape(B_in * num_goals, -1)
        # Agent-state protocol: predict ONCE from the top set's slots and condition
        # every denoising step on it; the prediction is kept for the eval
        # (`last_agent_pred`, (B*num_goals, 3)).
        mod = self._agent_module() if context_agent_state is not None else None
        if mod is not None and sel_times is not None:
            with torch.no_grad():
                slot_state = self._slot_states(context_agent_state, context_times, sel_times).flatten(0, 1)
                pred = self._agent_predict(mod, slot_state, ctx_pose, ctx_time, action, stride)
                model_kwargs['agent_state'] = self._agent_cond(pred)     # the predictor's DECISION (binarised)
            self.last_agent_pred = pred.detach()
        sample_loop = self.diffusion.ddim_sample_loop if use_ddim else self.diffusion.p_sample_loop
        return sample_loop(
            self.forward, z.shape, z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=progress,
            device=device,
        )
