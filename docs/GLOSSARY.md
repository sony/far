# Glossary

Names that appear in configs, launchers, run names and analysis code, and what
they mean in the paper's terms.

## Method and arms

| Name in code | Paper name | Meaning |
|---|---|---|
| `pnp`, `pnp_cl`, `PnPRetrieval` | FAR retriever | the learned, action-conditioned retrieval memory (`src/far/memories/pnp_cl.py`); "cl" = trained with a contrastive pre-training stage |
| `meta`, `pnp_meta`, `meta_only` | FAR -- Meta | the retriever scoring past frames with the closed-form metadata cross-scorer only (relative pose, time offsets) |
| `visual`, `pnp_visual` | FAR -- Visual | the retriever scoring with the visual (QueryViT) channel only |
| `vmz`, `visual+meta_z` | FAR -- Multi-Cue (LoopNav, AI2-THOR) | visual + metadata channels, each standardised per query and fused by a learned gate |
| `amz`, `cue_metaz`, `audio+meta_z` | FAR -- Multi-Cue (SoundSpaces) | the same fusion with the binaural-audio channel in place of the visual one |
| `ablz`, `pnp_ablation` | FAR -- Multi-Cue, fixed fusion | the visual and metadata channels mixed with a fixed weight of 0.5 instead of the learned gate |
| `jepa`, `frozen_encoder`, `encoder_mode=frozen` | FAR (frozen encoder) | the retriever's visual encoder frozen at its contrastive pre-training weights |
| `l0` / `l1` in run names | -- | retriever loss weight 0 / 1 (`model.retriever_loss_weight`) |
| `temporal` | Temporal | the most-recent-frames baseline (`TemporalRetrieval`, K frames spaced `interval` seconds apart) |
| `worldmem` | WorldMem | the FOV-overlap greedy retrieval baseline re-implemented from WorldMem |
| `longlive` | LongLive-RAG | the content-query retrieval baseline (retriever trained without the action) |
| `agent_fa`, `rev_agent`, `A_nocue` / `B_rag` | dynamic-agent arms | AI2-THOR-dyn arms whose generator also predicts the second agent's state; A = metadata scorer without the agent-state cue, B = with the cue |

## Generators

| Name | Meaning |
|---|---|
| `cdit` | Context-conditioned DiT (from Navigation World Models) |
| `cdit_b_2_mod_pose_fa` (`posefa`) | the main generator: full-attention CDiT with per-token Plücker-ray pose and trajectory-time conditioning; `b` = base size, `2` = patch size |
| `cdit_b_2_agent_fa` | `pose_fa` + the agent-state predictor (dynamic-agent corpus) |
| `ddim20`, `ps2` in run names | DDIM with 20 sampling steps; "ps2" = the SoundSpaces pool/sampler setting used for all SoundSpaces arms |

## Corpora

| Name | Corpus |
|---|---|
| `loopnav`, `loopnav_latent` | LoopNav (Minecraft loop navigation); `_latent` = pre-computed ViT-VAE latents |
| `ss`, `soundspaces_latent` | SoundSpaces v1: exploration with endpoint scans only |
| `ssv2`, `soundspaces_v2_latent` | SoundSpaces v2: exploration with periodic 360-degree scans (the headline SoundSpaces corpus) |
| `thor3`, `ai2thor_v3_latent` | AI2-THOR v3: object-interaction tours with a retrace (the headline AI2-THOR corpus) |
| `thor2`, `ai2thor_v2_latent` | the earlier AI2-THOR corpus (retriever pre-training data) |
| `thordyn`, `dyn`, `ai2thor_dyn_latent` | AI2-THOR-dyn: the door-corridor corpus with a second, moving agent |
| `_ctrl` | the static control sibling of each AI2-THOR v3 test episode (same stations, nothing moved) |

## Evaluation vocabulary

| Term | Meaning |
|---|---|
| battery | one evaluation sweep: every arm rolled out on every test clip, one `.npz` record per (arm, clip) under `results/rollout_eval/<battery>/<arm>/` |
| arm | one trained model (a memory variant) in a battery |
| replace-tail rollout | autoregressive rollout in which the generated frame replaces only the tail slot of the context; the retrievable history stays real |
| context size / pool | the number of frames the generator conditions on (`generator.context_size`) versus the number of past frames the memory may choose from (`context_pool_size`) |
| RAG step, EMDR² | the retriever's training signal: the diffusion loss of the generator under each candidate retrieval set, turned into a posterior the retriever is distilled towards |
| stale frame, stale-margin success | the exploration-phase view of the same viewpoint before an interaction; a step succeeds when the generation is closer to the ground truth than that stale view is |
| chunk score, `context_chunk_score` | the retriever's per-chunk relevance score, injected into the generator's context tokens |
