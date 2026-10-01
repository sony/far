"""Per-frame rollout evaluation over the test set — data collection only.

For every (arm, clip): frame-by-frame AR rollout along the return leg
(rollout_meta.ipynb protocol — replace-tail pool of REAL frames, per-step
re-centering, per-step retrieval), then per-step PSNR/LPIPS/DreamSim vs GT,
per-step covariates for alignment-free analysis, and per-step retrieval
diagnostics. One .npz record per (arm, clip); existing records are skipped, so
the script is resumable and shards trivially across GPUs:

    CUDA_VISIBLE_DEVICES=0 python scripts/world/eval_rollout.py shard=0 num_shards=4 &
    CUDA_VISIBLE_DEVICES=1 python scripts/world/eval_rollout.py shard=1 num_shards=4 &
    ...

Presentation (phase windows, end-aligned / fractional / distance-to-closure
curves, paired stats) is a separate analysis step over the saved records.
"""

import json
import os
import random
import time

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from far.constants import CONFIG_ROOT, ROOT
from far.data.transforms import global_to_local_pose, normalize_data


# ─────────────────────────────────────────────────────────────────────────────
# Model building — mirrors rollout_meta.ipynb's build_model (incl. the config
# schema migrations for pre-refactor runs).
# ─────────────────────────────────────────────────────────────────────────────

def migrate_memory_cfg(cfg):
    if cfg.model.get("memory") is None:
        return
    mem = cfg.model.memory
    for stale in ("beam_size", "lambda_div", "key_adapter_hidden",
                  "random_replace_prob", "random_replace_warmup_prob",
                  "random_replace_warmup_steps"):
        mem.pop(stale, None)
    query_vit_cfg = mem.get("query_vit")
    if query_vit_cfg is not None:
        query_vit_cfg.pop("key_adapter_hidden", None)
        if "query_adapter_hidden" in query_vit_cfg:
            mem["query_adapter_hidden"] = query_vit_cfg.pop("query_adapter_hidden")
    if "metadata_scorer" in mem:
        mem["scoring"] = "visual+meta" if mem.pop("metadata_scorer") else "visual"
    if "query_adapter_hidden" in mem or "freeze_encoder" in mem:
        qah = mem.pop("query_adapter_hidden", None)
        fe = mem.pop("freeze_encoder", False)
        if qah is not None:
            mem["encoder_mode"], mem["adapter_hidden"] = "adapter", qah
        else:
            mem["encoder_mode"] = "frozen" if fe else "finetune"

def _run_label(run_dir):
    """Short display name of a checkpoint bundle: the run_name segment of a timestamped run dir
    (`<stamp>__<script>__<run_name>__...`), else the bundle path itself (`models/<corpus>/<arm>`)."""
    parts = os.path.basename(os.path.normpath(run_dir)).split("__")
    return parts[2] if len(parts) > 2 else os.path.normpath(run_dir)


def build_model(run_dir, ckpt_step, device):
    cfg = OmegaConf.load(f"{run_dir}/config.yaml")
    OmegaConf.set_struct(cfg, False)
    migrate_memory_cfg(cfg)
    # Configs saved on other machines may reference a pre-downloaded local VAE
    # folder; fall back to the HF id (resolved from the local cache) when the
    # folder is absent here.
    tok_ckpt = cfg.model.tokenizer.get("ckpt_path")
    if tok_ckpt and "sdxl-vae-fp16-fix" in str(tok_ckpt) and not os.path.isdir(str(tok_ckpt)):
        cfg.model.tokenizer.ckpt_path = "madebyollin/sdxl-vae-fp16-fix"
    ckpt = torch.load(f"{run_dir}/checkpoints/{ckpt_step}.pth.tar",
                      map_location="cpu", weights_only=False)
    if "memory" in ckpt and cfg.model.get("memory") is not None:
        cfg.model.memory.ckpt_path = None
    model = instantiate(cfg.model).to(device).eval()
    model.core.generator.load_state_dict(ckpt["ema"], strict=True)
    if "memory" in ckpt:
        mem_sd = {}
        for k, v in ckpt["memory"].items():
            k = k.replace("_orig_mod.", "")
            for old, new in (("query_vit.query_proj.", "query_proj."),
                             ("query_vit.key_proj.", "key_proj.")):
                if k.startswith(old):
                    k = new + k[len(old):]
            mem_sd[k] = v
        model.core.memory.load_state_dict(mem_sd, strict=True)
    print(f"[build] {_run_label(run_dir):40s} mem={type(model.core.memory).__name__:18s} "
          f"steps={ckpt['train_steps']}")
    return model, cfg


class RecordRetrieval(torch.nn.Module):
    """Transparent pass-through stashing the memory's most recent 5-tuple."""

    def __init__(self, memory):
        super().__init__()
        self.memory, self.last = memory, None

    def forward(self, *args, **kw):
        out = self.memory(*args, **kw)
        self.last = out
        return out


class SubsampledPool(torch.nn.Module):
    """Restrict the inner memory's retrieval pool to fixed PAST positions
    (worldmem train-matched pool cap; the current frame is always forwarded)."""

    def __init__(self, memory, pos):
        super().__init__()
        self.memory = memory
        self.pos = pos

    def forward(self, frames, poses, times, action, stride, context_mask=None, **kw):
        T = frames.shape[1]
        idx = torch.cat([self.pos[self.pos < T - 1],
                         torch.tensor([T - 1], dtype=torch.long)]).to(frames.device)
        cm = None if context_mask is None else context_mask[:, idx]
        if kw.get("context_keys") is not None:      # keep keys frame-aligned
            kw["context_keys"] = kw["context_keys"][:, idx]
        return self.memory(frames[:, idx], poses[:, idx], times[:, idx],
                           action, stride, cm, **kw)


# ─────────────────────────────────────────────────────────────────────────────
# Rollout (rollout_meta.ipynb protocol) with per-step capture.
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def rollout_clip(model, ep, cfg, spec, stats_np, device):
    """AR-roll one model along one episode; returns (pred latents, retrieval per-step)."""
    mem_orig = model.core.memory
    mem = mem_orig
    if (cfg.worldmem_pool_size is not None and mem is not None
            and type(mem).__name__ == "WorldMemRetrieval"):
        valid_pos = np.flatnonzero(ep["ctx_mask"])
        if cfg.worldmem_pool_size < len(valid_pos):
            sub = np.sort(np.random.default_rng(ep["seed"]).choice(
                valid_pos, size=cfg.worldmem_pool_size, replace=False))
            mem = SubsampledPool(mem, torch.as_tensor(sub, dtype=torch.long)).to(device)
    recorder = None
    if mem is not None:
        mem = recorder = RecordRetrieval(mem).to(device)
    model.core.memory = mem

    pool_lat = [f for f in ep["ctx_lat"]]
    pool_pose_g = ep["ctx_pose_g"].copy()
    pool_idx = list(ep["ctx_idx"])
    pool_mask = list(ep["ctx_mask"])
    n_real = len(pool_lat)
    origin_idx = pool_idx[0]
    # Audio streams (soundspaces cue arms). POSITION-tied real measurements:
    # the query cue and the tail slot's key follow the GT position the agent
    # reached — only the visual latent in the tail is generated. Baselines
    # accept-and-ignore context_keys; query_cue is forwarded by LDWMCore only
    # to memories with query_source="cue".
    has_audio = ep.get("ctx_keys") is not None
    pool_keys = [k for k in ep["ctx_keys"]] if has_audio else None
    cue_tail = ep["cue_curr"] if has_audio else None

    preds, ret_dt, ret_dist = [], [], []
    try:
        for tpos in ep["targets"]:
            tgt_idx, tgt_pose_g = int(ep["goal_idx"][tpos]), ep["goal_pose_g"][tpos]
            curr_pose_g = pool_pose_g[-1]
            local = global_to_local_pose(
                np.concatenate([pool_pose_g, tgt_pose_g[None]]), curr_pose_g)
            local[:, :3] /= spec.meters_per_waypoint
            ctx_poses = torch.as_tensor(local[:-1], dtype=torch.float32)
            act = local[-1].copy()
            act[:3] = normalize_data(act[None, :3], stats_np)[0]
            ctx_times = torch.as_tensor(
                (np.asarray(pool_idx) - origin_idx) / spec.fps, dtype=torch.float32)
            stride_nl = (tgt_idx - origin_idx) / spec.max_frame_offset

            audio_kw = {}
            if has_audio:
                audio_kw = dict(
                    context_keys=torch.stack(pool_keys)[None].to(device),
                    query_cue=cue_tail[None].to(device),
                )
            pred = model.core.sample_latent(
                torch.stack(pool_lat)[None].to(device),
                torch.as_tensor(act, dtype=torch.float32)[None].to(device),
                torch.tensor([stride_nl], dtype=torch.float32).to(device),
                num_goals=1,
                context_poses=ctx_poses[None].to(device),
                context_times=ctx_times[None].to(device),
                context_mask=torch.as_tensor(np.asarray(pool_mask))[None].to(device),
                use_ddim=(cfg.sampler == "ddim"),
                **audio_kw,
            )[0].float().cpu()
            preds.append(pred)

            # Retrieval diagnostics: picked frames' Δt to current and pose
            # distance to the CURRENT GT pose (top set for set-aware memories).
            if recorder is not None and recorder.last is not None:
                sel_t = recorder.last[2]
                sel_p = recorder.last[1]
                if sel_t.dim() == 4:                       # (1,1,K,M) set-aware
                    sel_t, sel_p = sel_t[0, 0, 0], sel_p[0, 0, 0]
                else:
                    sel_t, sel_p = sel_t[0, 0], sel_p[0, 0]
                cur_t = float(ctx_times[-1])
                ret_dt.append((sel_t.detach().cpu().numpy() - cur_t))
                # pose distance in the LOCAL frame (current at origin): the
                # selected poses are already re-centred to the current pose.
                d = np.linalg.norm(sel_p.detach().cpu().numpy()[:, :3], axis=1) * spec.meters_per_waypoint
                ret_dist.append(d)

            if len(pool_lat) == n_real:
                pool_lat.append(pred)
                if has_audio:
                    pool_keys.append(ep["goal_keys"][tpos])
                pool_pose_g = np.concatenate([pool_pose_g, tgt_pose_g[None]])
                pool_idx.append(tgt_idx)
                pool_mask.append(True)
            else:
                pool_lat[-1] = pred
                if has_audio:
                    pool_keys[-1] = ep["goal_keys"][tpos]
                pool_pose_g[-1] = tgt_pose_g
                pool_idx[-1] = tgt_idx
            if has_audio:
                cue_tail = ep["goal_cues"][tpos]
    finally:
        model.core.memory = mem_orig
    ret = None
    if ret_dt:
        ret = (np.stack(ret_dt), np.stack(ret_dist))       # (steps, M) each
    return torch.stack(preds), ret


def build_episode(item, cfg, spec, clip_seed):
    """Deterministic episode: full return leg (minus start_offset) at stride."""
    n = len(item["context_indices"])
    ctx_lat = item["frames"][:n].clone()
    goal_lat = item["frames"][n:]
    pose_g = item["pose"].numpy().astype(np.float64)
    ctx_pose_g, goal_pose_g = pose_g[:n], pose_g[n:]
    ctx_idx = np.asarray(item["context_indices"])
    goal_idx = np.asarray(item["goal_indices"])
    ctx_mask = np.asarray(item["context_mask"], dtype=bool)
    # Audio streams (backends with key_suffix / audio_mode="cue" set).
    keys_all = item.get("keys")
    cues_all = item.get("cues")
    has_audio = keys_all is not None and cues_all is not None
    if has_audio:
        keys_all = torch.as_tensor(np.asarray(keys_all), dtype=torch.float32)
        cues_all = torch.as_tensor(np.asarray(cues_all), dtype=torch.float32)
        ctx_keys, goal_keys = keys_all[:n].clone(), keys_all[n:]
        goal_cues = cues_all[n:]
        cue_curr = cues_all[n - 1]

    so = int(cfg.start_offset)
    if so > 0:
        if so >= len(goal_idx):
            return None
        ctx_lat = torch.cat([ctx_lat, goal_lat[:so]])
        ctx_pose_g = np.concatenate([ctx_pose_g, goal_pose_g[:so]])
        ctx_idx = np.concatenate([ctx_idx, goal_idx[:so]])
        ctx_mask = np.concatenate([ctx_mask, np.ones(so, dtype=bool)])
        goal_lat, goal_pose_g, goal_idx = goal_lat[so:], goal_pose_g[so:], goal_idx[so:]
        if has_audio:
            ctx_keys = torch.cat([ctx_keys, goal_keys[:so]])
            cue_curr = goal_cues[so - 1]
            goal_keys, goal_cues = goal_keys[so:], goal_cues[so:]

    # Real trajectory length (the sampler repeat-pads short clips).
    eff_len = int(np.searchsorted(goal_idx, goal_idx[-1])) + 1
    stride = int(cfg.rollout_stride)
    steps = eff_len // stride
    if cfg.max_steps is not None:
        steps = min(steps, int(cfg.max_steps))
    # Floor the requirement at ONE step regardless of `min_steps`: a return leg
    # shorter than the stride would otherwise pass eligibility with an empty
    # target list and crash the rollout (`torch.stack([])`).
    if steps < max(1, int(cfg.min_steps)):
        return None
    targets = list(range(stride - 1, steps * stride, stride))
    ep = dict(ctx_lat=ctx_lat, goal_lat=goal_lat, ctx_pose_g=ctx_pose_g,
              goal_pose_g=goal_pose_g, ctx_idx=ctx_idx, goal_idx=goal_idx,
              ctx_mask=ctx_mask, targets=targets, eff_len=eff_len,
              seed=clip_seed, video_id=item["video_id"])
    if has_audio:
        ep.update(ctx_keys=ctx_keys, goal_keys=goal_keys,
                  goal_cues=goal_cues, cue_curr=cue_curr)
    return ep


@hydra.main(config_path=CONFIG_ROOT, config_name="eval_rollout", version_base=None)
def main(cfg):
    os.chdir(ROOT)
    os.environ.setdefault("FAR_MODELS_ROOT", os.path.join(ROOT, "models"))
    os.environ.setdefault("FAR_RESULT_ROOT", os.path.join(ROOT, "results"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_root = cfg.output_dir
    os.makedirs(out_root, exist_ok=True)

    # ── Models ───────────────────────────────────────────────────────────────
    models, data_cfg = {}, None
    for label, arm in cfg.arms.items():
        models[label], c = build_model(arm.run_dir, str(arm.ckpt), device)
        data_cfg = c
    dn = str(cfg.dataset)
    # Arms trained before this dataset group existed (e.g. v1-trained arms
    # evaluated on the v2 corpus) lack it in their saved configs — graft the
    # group from CONFIG_ROOT; its ${...} interpolations resolve against
    # data_cfg's root (len_traj_pred etc.), same as at training time.
    if dn not in data_cfg:
        data_cfg[dn] = OmegaConf.load(os.path.join(CONFIG_ROOT, "dataset", f"{dn}.yaml"))
        print(f"[data] grafted dataset group '{dn}' from CONFIG_ROOT")
    def _conv(run_dir):
        c = OmegaConf.load(f"{run_dir}/config.yaml")
        blk = c[dn] if dn in c else OmegaConf.load(os.path.join(CONFIG_ROOT, "dataset", f"{dn}.yaml"))
        return bool(blk.shared.backend.get("standard_frame", False))
    frames_conv = {label: _conv(arm.run_dir) for label, arm in cfg.arms.items()}
    assert len(set(frames_conv.values())) == 1, f"mixed pose conventions: {frames_conv}"

    # Eval diffusion + runtime retrieval knobs (rollout_meta.ipynb semantics).
    respace = f"ddim{cfg.sampling_steps}" if cfg.sampler == "ddim" else str(cfg.sampling_steps)
    eval_diffusion = instantiate(data_cfg.model.diffusion, timestep_respacing=respace)
    for m in models.values():
        m.core.diffusion = eval_diffusion
        if cfg.context_size is not None:
            m.core.num_memories = int(cfg.context_size) - 1
        if cfg.pnp_chunk_size is not None and hasattr(m.core.memory, "chunk_size"):
            m.core.memory.chunk_size = int(cfg.pnp_chunk_size)

    # ── Dataset (context_traj sampler; global poses kept) ────────────────────
    data_cfg[dn].test.base_dataset.sampler = data_cfg[dn].sampler.context_traj
    data_cfg[dn].test.base_dataset.sampler.context_pool_size = int(cfg.context_pool_size)
    data_cfg.len_traj_pred = 1024
    # Cue-querying arms (soundspaces pnp_audio / pnp_amz) need the dataset to
    # serve binaural windows + cached audio keys regardless of WHICH arm's
    # config the dataset was built from — force the backend fields when any
    # loaded memory queries by cue.
    if any(getattr(m.core.memory, "query_source", "frame") == "cue" for m in models.values()):
        data_cfg[dn].shared.backend.audio_mode = "cue"
        data_cfg[dn].shared.backend.key_suffix = ".keys_audio.npy"
        print("[data] cue arm detected -> backend audio_mode=cue, key_suffix=.keys_audio.npy")
    dataset = instantiate(data_cfg[dn].test.base_dataset)
    spec = instantiate(data_cfg[dn].spec)
    stats_np = {k: v.unsqueeze(0).numpy() for k, v in spec.local_pos_stats.items()}
    tok = next(iter(models.values())).tokenizer

    # ── Metric models ────────────────────────────────────────────────────────
    import lpips as lpips_pkg
    lpips_fn = lpips_pkg.LPIPS(net="alex", verbose=False).to(device).eval()
    try:
        import peft.import_utils as _piu
        import peft.tuners.lora.torchao as _plt
        _piu.is_torchao_available = lambda *a, **k: False
        _plt.is_torchao_available = lambda *a, **k: False
    except Exception:
        pass
    from dreamsim import dreamsim as _dreamsim
    from far.constants import hub_cache_dir
    ds_model, _ = _dreamsim(pretrained=True, device=str(device), cache_dir=hub_cache_dir())

    @torch.no_grad()
    def detok(lat, bs=16):
        out = [tok.detokenize(lat[i:i + bs].to(device)).clamp(-1, 1)
               for i in range(0, lat.shape[0], bs)]
        return torch.cat(out) * 0.5 + 0.5                    # (N, 3, H, W) on device, [0, 1]

    @torch.no_grad()
    def frame_metrics(pred_lat, gt_lat, bs=8):
        psnr, lp, dsim = [], [], []
        for i in range(0, pred_lat.shape[0], bs):
            p, g = detok(pred_lat[i:i + bs]), detok(gt_lat[i:i + bs])
            mse = ((p - g) ** 2).flatten(1).mean(1)
            psnr.append((-10 * torch.log10(mse)).cpu())
            lp.append(lpips_fn(p * 2 - 1, g * 2 - 1).flatten().cpu())
            dsim.append(ds_model(g, p).flatten().cpu())
        return (torch.cat(psnr).numpy(), torch.cat(lp).numpy(), torch.cat(dsim).numpy())

    # ── Eligible clips (deterministic), sharded ──────────────────────────────
    protocol = OmegaConf.to_container(cfg, resolve=True)
    protocol.pop("arms")
    with open(os.path.join(out_root, "protocol.json"), "w") as f:
        json.dump(protocol, f, indent=2, default=str)

    # ── Eligible clips: seeded-permutation scan, METADATA ONLY ───────────────
    # The scan must NOT hold episode tensors: each episode pins ~0.1 GB
    # (cloned pool latents + a goal-latent VIEW that keeps the whole clip
    # array alive), so retaining thousands of them exhausted host RAM
    # (6 shards x ~330 GB at num_clips=3000 froze the server). Only
    # (idx, n_steps, video_id) is kept; episodes are rebuilt deterministically
    # per clip inside the record loop, one at a time.
    # Shard 0 scans and writes `eligible_clips.json` (atomic, with the
    # protocol params that define eligibility); other shards WAIT for it
    # instead of repeating the scan I/O six-fold. A cache whose params
    # mismatch or whose list is too short is rescanned.
    elig_params = {k: (None if cfg[k] is None else int(cfg[k]) if k != "sampler" else cfg[k])
                   for k in ("num_clips", "start_offset", "rollout_stride",
                             "min_steps", "max_steps", "context_pool_size", "seed")}
    elig_params["dataset"] = str(cfg.dataset)
    elig_params["min_coverage"] = None if cfg.min_coverage is None else float(cfg.min_coverage)
    # Observability filter: per-episode outbound->return view-coverage scores
    # (scripts/data/build_observability_soundspaces.py). Clips below the floor
    # — or unscored — are ineligible.
    obs_cov = None
    if cfg.min_coverage is not None:
        with open(str(cfg.observability_path)) as f:
            obs_cov = {e["video_id"]: e["coverage"] for e in json.load(f)["episodes"]}
        print(f"[obs] coverage filter >= {float(cfg.min_coverage)} "
              f"({len(obs_cov)} episodes scored)")
    # Store the EFFECTIVE floor (>= 1 step) so caches written before the
    # zero-step-episode fix mismatch and are rebuilt automatically.
    elig_params["min_steps"] = max(1, elig_params["min_steps"] or 0)
    elig_path = os.path.join(out_root, "eligible_clips.json")

    def load_elig():
        # Cache validity: params must match, and the list must actually cover
        # the request — either >= num_clips entries, or (num_clips=null =
        # "every eligible clip") a scan that provably walked the WHOLE
        # dataset (`scanned` == len(dataset); absent in pre-knob caches, so
        # those are conservatively rebuilt).
        try:
            with open(elig_path) as f:
                d = json.load(f)
            if d.get("params") != elig_params:
                return None
            if cfg.num_clips is None:
                if d.get("scanned") == len(dataset):
                    return d["clips"]
            elif len(d["clips"]) >= int(cfg.num_clips):
                return d["clips"][: int(cfg.num_clips)]
        except (OSError, ValueError, KeyError):
            pass
        return None

    eligible = load_elig()
    if eligible is None and int(cfg.shard) != 0:
        print(f"[shard {int(cfg.shard)}] waiting for {elig_path} from shard 0 ...")
        t_wait = time.time()
        while eligible is None and time.time() - t_wait < 3600:
            time.sleep(10)
            eligible = load_elig()
    if eligible is None:
        order = np.random.default_rng(int(cfg.seed)).permutation(len(dataset))
        eligible, scanned = [], 0
        for idx in order:
            if cfg.num_clips is not None and len(eligible) >= int(cfg.num_clips):
                break
            idx = int(idx)
            random.seed(int(cfg.seed) + idx)
            item = dataset[idx]
            scanned += 1
            ep = build_episode(item, cfg, spec, clip_seed=int(cfg.seed) + idx)
            if ep is not None and (obs_cov is None or
                                   obs_cov.get(ep["video_id"], -1.0) >= float(cfg.min_coverage)):
                eligible.append([idx, len(ep["targets"]), ep["video_id"]])
            del item, ep
        n_aba = sum(("/ABA/" in v or "_ABA_" in v) for _, _, v in eligible)
        print(f"[clips] {len(eligible)} eligible episodes (scanned {scanned} of "
              f"{len(dataset)}; ABA {n_aba} / ABCA {len(eligible) - n_aba})")
        tmp = f"{elig_path}.tmp{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump({"params": elig_params, "clips": eligible,
                       "scanned": scanned}, f)
        os.replace(tmp, elig_path)
    else:
        print(f"[clips] loaded {len(eligible)} eligible episodes from cache")

    my = [e for i, e in enumerate(eligible) if i % int(cfg.num_shards) == int(cfg.shard)]
    for label in models:
        os.makedirs(os.path.join(out_root, label), exist_ok=True)
    # Pending work for THIS shard, CLIP-major (each episode is rebuilt once and
    # rolled by every arm that still misses its record, then freed). The
    # step-weighted ETA uses the cached per-clip step counts.
    todo = []
    for clip_idx, n_steps, _vid in my:
        missing = [l for l in models
                   if not os.path.exists(os.path.join(out_root, l, f"{clip_idx:05d}.npz"))]
        if missing:
            todo.append((int(clip_idx), int(n_steps), missing))
    n_pending = sum(len(ls) for _, _, ls in todo)
    total_steps = sum(n * len(ls) for _, n, ls in todo)
    print(f"[shard {int(cfg.shard)}] {n_pending} records pending ({total_steps} steps)")
    done_recs, done_steps, t0 = 0, 0, time.time()
    for clip_idx, n_steps, labels in todo:
        random.seed(int(cfg.seed) + clip_idx)
        item = dataset[clip_idx]
        ep = build_episode(item, cfg, spec, clip_seed=int(cfg.seed) + clip_idx)
        assert ep is not None and len(ep["targets"]) == n_steps, \
            f"clip {clip_idx}: episode rebuild diverged from the eligibility scan"
        for label in labels:
            rec_path = os.path.join(out_root, label, f"{clip_idx:05d}.npz")
            if os.path.exists(rec_path):
                continue
            torch.manual_seed(ep["seed"])
            preds, ret = rollout_clip(models[label], ep, cfg, spec, stats_np, device)
            gt_lat = ep["goal_lat"][ep["targets"]]
            psnr, lp, dsim = frame_metrics(preds, gt_lat)

            tg = np.asarray(ep["targets"])
            gidx = ep["goal_idx"][tg].astype(np.int64)
            cur0 = int(ep["ctx_idx"][-1])
            end_pose = ep["goal_pose_g"][ep["eff_len"] - 1]
            cov = dict(
                frame_idx=gidx,
                sec_from_start=(gidx - cur0) / spec.fps,
                sec_to_end=(int(ep["goal_idx"][ep["eff_len"] - 1]) - gidx) / spec.fps,
                frac=np.arange(1, len(tg) + 1) / len(tg),
                dist_to_closure=np.linalg.norm(
                    ep["goal_pose_g"][tg][:, :3] - end_pose[None, :3], axis=1),
            )
            payload = dict(psnr=psnr, lpips=lp, dreamsim=dsim,
                           video_id=np.array(ep["video_id"]),
                           n_steps=np.array(len(tg)), **cov)
            if ret is not None:
                payload["ret_dt"], payload["ret_dist"] = ret
            np.savez_compressed(rec_path, **payload)
            done_recs += 1
            done_steps += len(tg)
            el = time.time() - t0
            eta = (total_steps - done_steps) * el / max(done_steps, 1)
            print(f"[{label}] clip {clip_idx:5d} ({ep['video_id'].split('/')[-1]}): "
                  f"{len(tg)} steps  lpips {lp.mean():.3f}  dreamsim {dsim.mean():.3f}  | "
                  f"shard {done_recs}/{n_pending} recs  "
                  f"ETA {eta/60:.0f} min ({time.strftime('%H:%M', time.localtime(time.time() + eta))})")
        del item, ep
    print("done")


if __name__ == "__main__":
    main()
