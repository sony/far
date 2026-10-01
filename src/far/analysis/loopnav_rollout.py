"""LoopNav rollout: the machinery behind `notebooks/rollout_loopnav.ipynb` (knobs +
calls), the LoopNav analog of `ss_rollout` — same episode dict / rollout protocol /
figures, minus the audio streams. Every arm is its own training run (identical
cdit_b posefa generator, different memory), rolled along the SAME GT trajectory:
replace-tail protocol (the pool's history stays 100% real; one generated tail slot
is overwritten every step) unless `insert_generated`; poses re-centred on the new
current frame every step exactly as the training formatter does.

POSE CONVENTION: the runs decide it (`loopnav_latent.shared.backend.standard_frame`
in each run config — all arms in RUNS must agree). The rollout math uses the run's
convention verbatim; every top-down FIGURE is drawn in the z-up standard frame,
so `plot_ep` swaps standardized copies of the poses in for plotting only.
The metric / figure / video helpers are shared with `ss_rollout`.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from hydra.utils import instantiate
from omegaconf import OmegaConf

from far.data.transforms import global_to_local_pose, normalize_data
from far.analysis import ss_rollout as S
from far.analysis.ss_rollout import (SubsampledPool, TeacherForcedQuery, make_recorder, make_picks_recorder,  # noqa: F401
                                      compute_metrics, plot_compare, write_rollout_video, plot_progression,
                                      export_frames, export_records, write_retrieval_geography, write_context_video,
                                      install_eval_knobs)
from far.analysis.thor_rollout import Decoder  # noqa: F401  (shared SDXL/ViTVAE decode helper)

DISPLAY_NAME = {
    "GT":                "Ground truth",
    "B_posefa_temporal": "Temporal",
    "B_posefa_worldmem": "WorldMem",
    "pnp_longlive":      "LongLive-RAG",
    "pnp_learned_vmz":   "FAR -- Multi-Cue",       # the panel video adds "(ours)" under the OURS card
    "pnp_meta_only":     "FAR -- Meta",
    "pnp_visual_only":   "FAR -- Visual",
    "pnp_jepa_frozen":   "FAR -- Frozen Retriever",
}


# ── models ────────────────────────────────────────────────────────────────────
def _migrate_config(cfg):
    """Schema migrations for older run configs so they instantiate today: the
    pnp_cl query-adapter move (QueryViT -> PnPRetrieval), the removed beam /
    diversity / key-adapter knobs, and the knob consolidation (metadata_scorer
    -> scoring; freeze_encoder / query_adapter_hidden -> encoder_mode /
    adapter_hidden)."""
    if cfg.model.get("memory") is None:
        return
    mem = cfg.model.memory
    for stale in ("beam_size", "lambda_div", "key_adapter_hidden", "random_replace_prob",
                  "random_replace_warmup_prob", "random_replace_warmup_steps"):
        mem.pop(stale, None)
    qv = mem.get("query_vit")
    if qv is not None:
        qv.pop("key_adapter_hidden", None)
        if "query_adapter_hidden" in qv:
            mem["query_adapter_hidden"] = qv.pop("query_adapter_hidden")
    if "metadata_scorer" in mem:
        mem["scoring"] = "visual+meta" if mem.pop("metadata_scorer") else "visual"
    if "query_adapter_hidden" in mem or "freeze_encoder" in mem:
        qah, fe = mem.pop("query_adapter_hidden", None), mem.pop("freeze_encoder", False)
        if qah is not None:
            mem["encoder_mode"], mem["adapter_hidden"] = "adapter", qah
        else:
            mem["encoder_mode"] = "frozen" if fe else "finetune"


def build_model(run_dir: str, ckpt_step: str, device):
    """Instantiate a run's PixelLDWM from its saved (migrated) config and load the
    EMA generator + the jointly-trained memory state (`ckpt["memory"]`, loaded on
    top of — and instead of — the pretrain init its config's `ckpt_path` names,
    whose file may not exist on this machine). Memory keys are renamed from the
    torch.compile'd / pre-refactor layout (`_orig_mod.`, `query_vit.query_proj.`)."""
    cfg = OmegaConf.load(f"{run_dir}/config.yaml")
    OmegaConf.set_struct(cfg, False)
    _migrate_config(cfg)
    ckpt = torch.load(f"{run_dir}/checkpoints/{ckpt_step}.pth.tar", map_location="cpu", weights_only=False)
    if "memory" in ckpt and cfg.model.get("memory") is not None:
        cfg.model.memory.ckpt_path = None
    model = instantiate(cfg.model).to(device).eval()
    model.core.generator.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ckpt["ema"].items()}, strict=True)
    if "memory" in ckpt:
        sd = {}
        for k, v in ckpt["memory"].items():
            k = k.replace("_orig_mod.", "")
            for old, new in (("query_vit.query_proj.", "query_proj."), ("query_vit.key_proj.", "key_proj.")):
                if k.startswith(old):
                    k = new + k[len(old):]
            sd[k] = v
        model.core.memory.load_state_dict(sd, strict=True)
    gen = model.core.generator
    print(f"{_run_label(run_dir):38s} gen={type(gen).__module__.split('.')[-1]}"
          f"({sum(p.numel() for p in gen.parameters()) / 1e6:.0f}M) mem={type(model.core.memory).__name__:18s} "
          f"steps={ckpt['train_steps']} mem_in_ckpt={'memory' in ckpt}")
    return model, cfg

def _run_label(run_dir):
    """Short display name of a checkpoint bundle: the run_name segment of a timestamped run dir
    (`<stamp>__<script>__<run_name>__...`), else the bundle path itself (`models/<corpus>/<arm>`)."""
    parts = os.path.basename(os.path.normpath(run_dir)).split("__")
    return parts[2] if len(parts) > 2 else os.path.normpath(run_dir)


def build_models(runs: dict, device):
    """{label: (run_dir, step)} -> ({label: model}, data_cfg of the LAST run). Asserts
    one pose convention across the runs (a raw-convention checkpoint fed standardized
    poses, or vice versa, degrades silently) and prints each pnp arm's scoring recipe."""
    models, cfg = {}, None
    for label, (rd, st) in runs.items():
        models[label], cfg = build_model(rd, st, device)
    frames = {l: bool(OmegaConf.load(f"{rd}/config.yaml").loopnav_latent.shared.backend.get("standard_frame", False))
              for l, (rd, _) in runs.items()}
    assert len(set(frames.values())) == 1, f"mixed pose conventions in RUNS: {frames}"
    print("pose convention (standard_frame):", next(iter(frames.values())))
    for l, m in models.items():
        if hasattr(m.core.memory, "scoring"):
            vs = f"  visual_scale={float(m.core.memory.visual_scale):.3f}" if hasattr(m.core.memory, "visual_scale") else ""
            print(f"{l}: scoring={m.core.memory.scoring}  adapter={m.core.memory.query_proj is not None}{vs}")
    return models, cfg


# ── dataset / episode ─────────────────────────────────────────────────────────
def build_dataset(data_cfg, group: str, pool_size: int):
    """RAW (unformatted) test dataset with ContextTrajectorySampler: latents come
    pre-computed from the LatentLoopNavBackend (item["frames"] is (T, 16, 18, 32))
    and the GLOBAL poses the formatter would discard are kept for the per-step
    re-centring. Returns (dataset, spec, stats_np, standard_frame)."""
    from far.constants import CONFIG_ROOT
    if group not in data_cfg:
        data_cfg[group] = OmegaConf.load(os.path.join(CONFIG_ROOT, "dataset", f"{group}.yaml"))
        print(f"grafted dataset group '{group}' from CONFIG_ROOT")
    dg = data_cfg[group]
    dg.test.base_dataset.sampler = dg.sampler.context_traj
    dg.test.base_dataset.sampler.context_pool_size = pool_size
    data_cfg.len_traj_pred = pool_size
    dataset = instantiate(dg.test.base_dataset)
    spec = instantiate(dg.spec)
    stats_np = {k: v.unsqueeze(0).numpy() for k, v in spec.local_pos_stats.items()}
    std = bool(dg.shared.backend.get("standard_frame", False))
    print("test clips:", len(dataset), "| fps:", spec.fps, "| max_frame_offset:", spec.max_frame_offset,
          "| meters_per_waypoint:", spec.meters_per_waypoint, "| standard_frame:", std)
    return dataset, spec, stats_np, std


def standardize(pose_g: np.ndarray) -> np.ndarray:
    """Raw Minecraft poses [x, height, z, pitch, yaw] (forward = (-sin yaw, -cos yaw) in
    (x, z)) -> the z-up standard frame [-z, -x, height, pitch, yaw] (forward =
    (cos yaw, sin yaw) in (x, y)) — the `standard_frame: true` backend remap."""
    out = np.asarray(pose_g, dtype=np.float64).copy()
    out[:, 0], out[:, 1], out[:, 2] = -pose_g[:, 2], -pose_g[:, 0], pose_g[:, 1]
    return out


def load_episode(dataset, clip_idx: int, seed: int = 0, standard_frame: bool = False) -> dict:
    """Sample the clip and unpack it into an episode dict: latents, global poses,
    indices, pool mask. `ep["std"]` records the pose convention for `plot_ep`."""
    import random
    random.seed(seed)
    item = dataset[clip_idx]
    n = len(item["context_indices"])
    lat_all = item["frames"]; pose_g = item["pose"].numpy().astype(np.float64)
    ep_path = next(s["path"] for s in dataset.samples if s["video_id"] == item["video_id"])
    return dict(
        clip_idx=clip_idx, video_id=item["video_id"], ep_path=ep_path, n_pool=n, std=standard_frame,
        ctx_lat=lat_all[:n].clone(), goal_lat=lat_all[n:],
        ctx_pose_g=pose_g[:n], goal_pose_g=pose_g[n:],
        ctx_idx=np.asarray(item["context_indices"]), goal_idx=np.asarray(item["goal_indices"]),
        ctx_mask=np.asarray(item["context_mask"], dtype=bool), src_xy=np.zeros((0, 2)))


def apply_window(ep: dict, spec, start_offset: int, stride: int, max_steps=None,
                 pool_phase: str = "traversed") -> list:
    """Shift the rollout start `start_offset` frames into the return path and build
    the step targets — the rollout runs from there to the END of the real trajectory
    (repeat-padding excluded), or for `max_steps` steps, so START_OFFSET + MAX_STEPS
    select an INTERVAL.

    POOL_PHASE decides what the walked-past frames do, as in `thor_rollout`:
      "traversed"   — they JOIN the retrievable pool (the default): the agent may
          answer from the view it passed seconds ago.
      "exploration" — they do not; only the start frame is appended as the CURRENT
          observation, so every RETRIEVABLE frame predates the return path and the
          arms must answer from the outbound leg alone.
    Mutates `ep`; stores/returns ep["targets"]."""
    assert pool_phase in ("traversed", "exploration"), pool_phase
    ep.setdefault("eval_start", int(ep["goal_idx"][0]))   # the return path's first frame, before any slicing
    if start_offset > 0:
        assert start_offset < len(ep["goal_idx"]), "START_OFFSET exceeds the sampled trajectory"
        absorb = slice(0, start_offset) if pool_phase == "traversed" else slice(start_offset - 1, start_offset)
        n_abs = start_offset if pool_phase == "traversed" else 1
        ep["ctx_lat"] = torch.cat([ep["ctx_lat"], ep["goal_lat"][absorb]])
        ep["ctx_pose_g"] = np.concatenate([ep["ctx_pose_g"], ep["goal_pose_g"][absorb]])
        ep["ctx_idx"] = np.concatenate([ep["ctx_idx"], ep["goal_idx"][absorb]])
        ep["ctx_mask"] = np.concatenate([ep["ctx_mask"], np.ones(n_abs, dtype=bool)])
        for k in ("goal_lat", "goal_pose_g", "goal_idx"):
            ep[k] = ep[k][start_offset:]
    eff = int(np.searchsorted(ep["goal_idx"], ep["goal_idx"][-1])) + 1
    steps = eff // stride
    assert steps >= 1, "trajectory too short for this stride/start_offset"
    if max_steps is not None:
        steps = min(steps, int(max_steps))
    ep["targets"], ep["stride"] = list(range(stride - 1, steps * stride, stride)), int(stride)
    retr = np.asarray(ep["ctx_idx"])[np.asarray(ep["ctx_mask"], dtype=bool)][:-1]   # the memory treats the tail as "now"
    print(f"clip {os.path.basename(ep['video_id'])}: pool={len(ep['ctx_idx'])} ({pool_phase}: "
          f"{int((retr >= ep['eval_start']).sum())} return-path frames retrievable) traj={len(ep['goal_idx'])} "
          f"curr_idx={ep['ctx_idx'][-1]} rollout frames [{ep['goal_idx'][ep['targets'][0]]}..{ep['goal_idx'][ep['targets'][-1]]}]")
    return ep["targets"]


def plot_ep(ep: dict) -> dict:
    """The episode with its poses in the z-up standard frame — for the top-down
    figures only (the `ss_rollout` map helpers assume that frame). A no-op copy when
    the runs already use `standard_frame: true`."""
    if ep.get("std", False):
        return ep
    out = dict(ep); out["ctx_pose_g"], out["goal_pose_g"] = standardize(ep["ctx_pose_g"]), standardize(ep["goal_pose_g"])
    return out


def inspect_trajectory(ep: dict, spec, decode):
    """Top-down map + decoded GT frames along the rollout window, before generation."""
    return S.inspect_trajectory(plot_ep(ep), spec, decode)


def plot_summary(ep: dict, spec, psnr, lpips_steps, dreamsim_steps, save_path=None):
    return S.plot_summary(plot_ep(ep), spec, psnr, lpips_steps, dreamsim_steps, save_path=save_path)


def write_geography(ep: dict, spec, pick_times, pick_scores, gt_img, pred_img, psnr, decode, save_path, **kw):
    """`ss_rollout.write_retrieval_geography` on the standardized poses."""
    return write_retrieval_geography(plot_ep(ep), spec, pick_times, pick_scores, gt_img, pred_img, psnr, decode, save_path, **kw)


def thor_episode(ep: dict, stride: int) -> dict:
    """Adapt a LoopNav episode for `thor_rollout.write_rollout_video` (the panel-style
    video): the OUTBOUND leg becomes the history prelude (every frame before the first
    return-leg frame, read from the clip's latent `.npy`), the return path the
    evaluation phase. No interaction / agent sidecars, so captions and verdicts stay off."""
    lat = np.load(ep["ep_path"], mmap_mode="c")
    es = int(ep["goal_idx"][0]); out = dict(ep)
    out.update(eval_start=es, stride=int(stride), eff_len=len(ep["goal_idx"]), meta={}, targets=ep["targets"],
               ctx_idx=np.arange(0, es), ctx_lat=torch.from_numpy(np.asarray(lat[:es], dtype=np.float32)), ctx_mask=np.ones(es, dtype=bool),
               ctx_fire=np.zeros((es, 4)))
    return out


# ── rollout ───────────────────────────────────────────────────────────────────
def rollout(model, label, ep, spec, stats_np, device, sampler="ddim", seed=0,
            insert_generated=False, teacher_forced=None, worldmem_pool_size=None):
    """AR-roll one model along ep["targets"] (replace-tail unless insert_generated).
    `teacher_forced`: None | "query" (memory queries with the GT frame of the current
    position; the generator still conditions on the generated tail) | "full" (the
    pool tail IS the GT frame — the no-exposure-bias upper bound). pnp_cl
    live-encodes its retrieval keys (no cached `context_keys`)."""
    targets = ep["targets"]
    mem_orig = model.core.memory
    while hasattr(mem_orig, "memory"):       # shed wrappers left by a crashed run
        mem_orig = mem_orig.memory
    mem = mem_orig
    if worldmem_pool_size is not None and mem is not None and type(mem).__name__ == "WorldMemRetrieval":
        valid_pos = np.flatnonzero(np.asarray(ep["ctx_mask"]))
        if worldmem_pool_size < len(valid_pos):
            sub = np.sort(np.random.default_rng(seed).choice(valid_pos, size=worldmem_pool_size, replace=False))
            mem = SubsampledPool(mem, torch.as_tensor(sub, dtype=torch.long), n_base=len(ep["ctx_mask"])).to(device)
            print(f"[{label}] worldmem pool capped: {len(valid_pos)} valid -> {worldmem_pool_size}")
    tf_query = None
    if teacher_forced == "query" and mem is not None:
        mem = tf_query = TeacherForcedQuery(mem).to(device)
    model.core.memory = mem
    pool_lat = [f for f in ep["ctx_lat"]]; pool_pose_g = ep["ctx_pose_g"].copy()
    pool_idx = list(ep["ctx_idx"]); pool_mask = list(ep["ctx_mask"])
    n_real, origin_idx, gt_tail = len(pool_lat), pool_idx[0], pool_lat[-1]
    preds = []
    from tqdm.auto import tqdm
    pbar = tqdm(targets, desc=label, unit="frame", leave=False)
    try:
        for tpos in pbar:
            tgt_idx, tgt_pose_g = int(ep["goal_idx"][tpos]), ep["goal_pose_g"][tpos]
            if tf_query is not None:
                tf_query.gt = gt_tail
            local = global_to_local_pose(np.concatenate([pool_pose_g, tgt_pose_g[None]]), pool_pose_g[-1])
            local[:, :3] /= spec.meters_per_waypoint
            ctx_poses = torch.as_tensor(local[:-1], dtype=torch.float32)
            action = local[-1].copy(); action[:3] = normalize_data(action[None, :3], stats_np)[0]
            ctx_times = torch.as_tensor((np.asarray(pool_idx) - origin_idx) / spec.fps, dtype=torch.float32)
            stride_nl = (tgt_idx - origin_idx) / spec.max_frame_offset
            with torch.no_grad():
                pred = model.core.sample_latent(
                    torch.stack(pool_lat)[None].to(device),
                    torch.as_tensor(action, dtype=torch.float32)[None].to(device),
                    torch.tensor([stride_nl], dtype=torch.float32).to(device),
                    num_goals=1,
                    context_poses=ctx_poses[None].to(device),
                    context_times=ctx_times[None].to(device),
                    context_mask=torch.as_tensor(np.asarray(pool_mask))[None].to(device),
                    use_ddim=(sampler == "ddim"),
                )[0].float().cpu()
            preds.append(pred)
            tail_new = ep["goal_lat"][tpos].clone() if teacher_forced == "full" else pred
            if insert_generated or len(pool_lat) == n_real:
                pool_lat.append(tail_new); pool_pose_g = np.concatenate([pool_pose_g, tgt_pose_g[None]])
                pool_idx.append(tgt_idx); pool_mask.append(True)
            else:
                pool_lat[-1], pool_pose_g[-1], pool_idx[-1] = tail_new, tgt_pose_g, tgt_idx
            gt_tail = ep["goal_lat"][tpos]
            pbar.set_postfix(target_frame=tgt_idx, pool=len(pool_lat))
    finally:
        model.core.memory = mem_orig
    return torch.stack(preds)


def run_rollouts(models, ep, spec, stats_np, device, **kw):
    """Roll every arm on the SAME clip with the same noise seed; hooks on each arm's
    base memory record its picks (+ scores) and, for fused pnp arms, the channel
    attribution. Returns (preds, score_stats, attribution, pick_times, pick_scores)."""
    preds, score_stats, attribution, pick_times, pick_scores = {}, {}, {}, {}, {}
    for label, model in models.items():
        mem = model.core.memory
        while hasattr(mem, "memory"):
            mem = mem.memory
        handle, prows, pscore_rows, srows, arows = None, [], [], [], None
        if hasattr(mem, "chunk_size"):
            arows = [] if getattr(mem, "scoring", "").startswith("visual+meta") else None
            handle = mem.register_forward_hook(make_recorder(srows, arows, prows, pscore_rows), with_kwargs=True)
        elif mem is not None:
            handle = mem.register_forward_hook(make_picks_recorder(prows, pscore_rows), with_kwargs=True)
        torch.manual_seed(kw.get("seed", 0))
        preds[label] = rollout(model, label, ep, spec, stats_np, device, **kw)
        if handle is not None:
            handle.remove()
            pick_times[label], pick_scores[label] = prows, pscore_rows
            if hasattr(mem, "chunk_size"):
                score_stats[label] = np.asarray(srows)
                if arows is not None:
                    attribution[label] = np.asarray(arows)
    return preds, score_stats, attribution, pick_times, pick_scores
