"""SoundSpaces rollout machinery backing `notebooks/rollout_soundspaces.ipynb`
(the soundspaces sibling of `thor_rollout` — same episode-dict + knob-function
pattern).

AUDIO IS OBSERVED, NOT GENERATED: the binaural cue at the current position and
the cached audio keys of pool frames are REAL measurements tied to POSITIONS
(like poses/actions), so at every rollout step the query cue and the tail
slot's key follow the GT position the agent has reached — only the VISUAL
latent in the tail is generated. Under insert_generated the appended history
likewise carries GT-position audio keys (the audio there was really heard)."""

import json
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from hydra.utils import instantiate
from omegaconf import OmegaConf

from far.data.transforms import global_to_local_pose, normalize_data, tensor_to_image
from far.analysis.thor_rollout import Decoder, _lpips, _dreamsim, export_records  # noqa: F401  (shared helpers)

HXY = [0, 1]                     # standardized z-up: horizontal plane = [x, y]
DISPLAY_NAME = {
    "GT":             "Ground truth",
    "temporal":       "Temporal",
    "worldmem":       "WorldMem",
    "pnp_amz":        "FAR -- Multi-Cue",
    "pnp_meta_only":  "FAR -- Meta",
}


def _heading(yw):
    return (np.cos(yw), np.sin(yw))


# ── models ────────────────────────────────────────────────────────────────────
def build_model(run_dir: str, ckpt_step: str, device):
    """Instantiate a run's PixelLDWM from its saved config and load the EMA
    generator + jointly-trained memory. Runs from other machines reference a
    pre-downloaded local VAE folder; fall back to the HF id when absent."""
    cfg = OmegaConf.load(f"{run_dir}/config.yaml")
    OmegaConf.set_struct(cfg, False)
    tok_ckpt = cfg.model.tokenizer.get("ckpt_path")
    if tok_ckpt and "sdxl-vae-fp16-fix" in str(tok_ckpt) and not os.path.isdir(str(tok_ckpt)):
        cfg.model.tokenizer.ckpt_path = "madebyollin/sdxl-vae-fp16-fix"
    ckpt = torch.load(f"{run_dir}/checkpoints/{ckpt_step}.pth.tar", map_location="cpu", weights_only=False)
    if "memory" in ckpt and cfg.model.get("memory") is not None:
        cfg.model.memory.ckpt_path = None      # trained memory state supersedes the pretrain init
    model = instantiate(cfg.model).to(device).eval()
    model.core.generator.load_state_dict(
        {k.removeprefix("_orig_mod."): v for k, v in ckpt["ema"].items()}, strict=True)
    if "memory" in ckpt:
        model.core.memory.load_state_dict(
            {k.removeprefix("_orig_mod."): v for k, v in ckpt["memory"].items()}, strict=True)
    gen = model.core.generator
    print(f"{_run_label(run_dir):34s} gen={type(gen).__module__.split('.')[-1]}"
          f"({sum(p.numel() for p in gen.parameters()) / 1e6:.0f}M) "
          f"mem={type(model.core.memory).__name__:22s} steps={ckpt['train_steps']}")
    return model, cfg

def _run_label(run_dir):
    """Short display name of a checkpoint bundle: the run_name segment of a timestamped run dir
    (`<stamp>__<script>__<run_name>__...`), else the bundle path itself (`models/<corpus>/<arm>`)."""
    parts = os.path.basename(os.path.normpath(run_dir)).split("__")
    return parts[2] if len(parts) > 2 else os.path.normpath(run_dir)


def build_models(runs: dict, device):
    """{label: (run_dir, step)} -> ({label: model}, data_cfg of the LAST run).
    Prints each pnp arm's scoring recipe for the record."""
    models, cfg = {}, None
    for label, (rd, st) in runs.items():
        models[label], cfg = build_model(rd, st, device)
    for l, m in models.items():
        if hasattr(m.core.memory, "scoring"):
            print(f"{l}: scoring={m.core.memory.scoring}  query_source={m.core.memory.query_source}  "
                  f"adapter={m.core.memory.query_proj is not None}  gate={hasattr(m.core.memory, 'gate')}")
    return models, cfg


def install_eval_knobs(models, data_cfg, sampler="ddim", sampling_steps=20,
                      pnp_chunk_size=None, context_size=None):
    """(Re)install the eval diffusion at the requested respacing and the runtime
    retrieval knobs on every model. None = each run's trained value."""
    respace = f"ddim{sampling_steps}" if sampler == "ddim" else str(sampling_steps)
    eval_diffusion = instantiate(data_cfg.model.diffusion, timestep_respacing=respace)
    for m in models.values():
        m.core.diffusion = eval_diffusion
        if pnp_chunk_size is not None and hasattr(m.core.memory, "chunk_size"):
            m.core.memory.chunk_size = pnp_chunk_size
        if context_size is not None:
            m.core.num_memories = context_size - 1


# ── dataset / episode ─────────────────────────────────────────────────────────
def build_dataset(data_cfg, group: str, pool_size: int):
    """RAW (unformatted) test dataset with ContextTrajectorySampler in CUE mode:
    latents in the frames slot, binaural windows as item["cues"], cached audio
    keys as item["keys"] — baseline arms simply ignore the extra streams."""
    from far.constants import CONFIG_ROOT
    if group not in data_cfg:
        data_cfg[group] = OmegaConf.load(os.path.join(CONFIG_ROOT, "dataset", f"{group}.yaml"))
        print(f"grafted dataset group '{group}' from CONFIG_ROOT")
    dg = data_cfg[group]
    dg.test.base_dataset.sampler = dg.sampler.context_traj
    dg.test.base_dataset.sampler.context_pool_size = pool_size
    data_cfg.len_traj_pred = pool_size
    dg.shared.backend.audio_mode = "cue"
    dg.shared.backend.key_suffix = ".keys_audio.npy"
    dataset = instantiate(dg.test.base_dataset)
    spec = instantiate(dg.spec)
    stats_np = {k: v.unsqueeze(0).numpy() for k, v in spec.local_pos_stats.items()}
    print("test clips:", len(dataset), "| fps:", spec.fps,
          "| max_frame_offset:", spec.max_frame_offset)
    return dataset, spec, stats_np


def load_episode(dataset, clip_idx: int, seed: int = 0) -> dict:
    """Sample the clip and unpack it into an episode dict: latents, audio keys +
    cue windows, global poses, indices, pool mask, sound-source positions."""
    import random
    random.seed(seed)
    item = dataset[clip_idx]
    n = len(item["context_indices"])
    lat_all = item["frames"]
    keys_all = torch.as_tensor(np.asarray(item["keys"]), dtype=torch.float32)
    cues_all = torch.as_tensor(np.asarray(item["cues"]), dtype=torch.float32)
    pose_g = item["pose"].numpy().astype(np.float64)
    ep_path = next(s["path"] for s in dataset.samples if s["video_id"] == item["video_id"])
    meta = json.loads(open(os.path.join(ep_path, "meta.json")).read())
    # habitat [x, height, z] -> z-up standard frame [-z, -x, height]
    src_xy = np.array([[-s["pos"][2], -s["pos"][0]] for s in meta.get("sources", [])])
    return dict(
        clip_idx=clip_idx, video_id=item["video_id"], ep_path=ep_path, n_pool=n,
        ctx_lat=lat_all[:n].clone(), goal_lat=lat_all[n:],
        ctx_keys=keys_all[:n].clone(), goal_keys=keys_all[n:],
        goal_cues=cues_all[n:], cue_curr=cues_all[n - 1], ctx_cues=cues_all[:n].clone(),
        ctx_pose_g=pose_g[:n], goal_pose_g=pose_g[n:],
        ctx_idx=np.asarray(item["context_indices"]), goal_idx=np.asarray(item["goal_indices"]),
        ctx_mask=np.asarray(item["context_mask"], dtype=bool), src_xy=src_xy)


def apply_window(ep: dict, spec, start_offset: int, stride: int, max_steps=None,
                 pool_phase: str = "traversed") -> list:
    """Shift the rollout start `start_offset` frames into the return path
    (frames, poses, indices, AND the audio key/cue streams move together) and
    build the step targets; `max_steps` caps them, so START_OFFSET + MAX_STEPS
    select an INTERVAL of the return path (None = run to its end).

    POOL_PHASE decides what the walked-past frames do, as in `thor_rollout`:
      "traversed"   — the frames walked before the start JOIN the retrievable
          pool (the default): the agent may answer from the view it passed
          seconds ago.
      "exploration" — they do not; only the start frame is appended as the
          CURRENT observation, so every RETRIEVABLE frame predates the return
          path and the arms must answer from the outbound leg alone.
    Mutates `ep`; stores/returns ep["targets"]."""
    assert pool_phase in ("traversed", "exploration"), pool_phase
    ep.setdefault("eval_start", int(ep["goal_idx"][0]))   # the return path's first frame, before any slicing
    if start_offset > 0:
        assert start_offset < len(ep["goal_idx"]), "START_OFFSET exceeds the sampled trajectory"
        absorb = slice(0, start_offset) if pool_phase == "traversed" else slice(start_offset - 1, start_offset)
        n_abs = start_offset if pool_phase == "traversed" else 1
        ep["ctx_lat"] = torch.cat([ep["ctx_lat"], ep["goal_lat"][absorb]])
        ep["ctx_keys"] = torch.cat([ep["ctx_keys"], ep["goal_keys"][absorb]])
        ep["ctx_pose_g"] = np.concatenate([ep["ctx_pose_g"], ep["goal_pose_g"][absorb]])
        ep["ctx_idx"] = np.concatenate([ep["ctx_idx"], ep["goal_idx"][absorb]])
        ep["ctx_mask"] = np.concatenate([ep["ctx_mask"], np.ones(n_abs, dtype=bool)])
        ep["cue_curr"] = ep["goal_cues"][start_offset - 1]
        ep["ctx_cues"] = torch.cat([ep["ctx_cues"], ep["goal_cues"][absorb]])
        for k in ("goal_lat", "goal_keys", "goal_cues", "goal_pose_g", "goal_idx"):
            ep[k] = ep[k][start_offset:]
    eff = int(np.searchsorted(ep["goal_idx"], ep["goal_idx"][-1])) + 1
    steps = eff // stride
    assert steps >= 1, "trajectory too short for this stride/start_offset"
    if max_steps is not None:
        steps = min(steps, int(max_steps))
    ep["targets"], ep["stride"] = list(range(stride - 1, steps * stride, stride)), int(stride)
    retr = np.asarray(ep["ctx_idx"])[np.asarray(ep["ctx_mask"], dtype=bool)][:-1]   # the memory treats the tail as "now"
    print(f"clip {ep['video_id']}: pool={len(ep['ctx_idx'])} ({pool_phase}: {int((retr >= ep['eval_start']).sum())} "
          f"return-path frames retrievable) traj={len(ep['goal_idx'])} "
          f"curr_idx={ep['ctx_idx'][-1]} rollout frames "
          f"[{ep['goal_idx'][ep['targets'][0]]}..{ep['goal_idx'][ep['targets'][-1]]}]")
    return ep["targets"]


def _draw_topdown(ax, ep, targets, window=True):
    valid_xy = ep["ctx_pose_g"][ep["ctx_mask"]][:, HXY]
    traj_xy = ep["goal_pose_g"][:, HXY]
    ax.plot(valid_xy[:, 0], valid_xy[:, 1], ":", color="dimgray", lw=1.2, label="context path")
    ax.plot(traj_xy[:, 0], traj_xy[:, 1], "-",
            color="lightcoral" if window else "tab:red", lw=1.2 if window else 1.5, label="GT trajectory")
    if window:
        win_xy = ep["goal_pose_g"][targets][:, HXY]
        ax.plot(win_xy[:, 0], win_xy[:, 1], "-", color="tab:red", lw=2.0, label="rollout window")
    else:
        step_xy = ep["goal_pose_g"][targets][:, HXY]
        ax.scatter(step_xy[:, 0], step_xy[:, 1], s=18, color="tab:blue", zorder=3, label="rollout steps")
    ax.plot(*ep["ctx_pose_g"][-1, HXY], marker="*", ms=13, color="black", label="current")
    if len(ep["src_xy"]):
        ax.scatter(ep["src_xy"][:, 0], ep["src_xy"][:, 1], marker="*", s=160, color="tab:orange",
                   edgecolors="black", zorder=4, label="sound source")
    hx, hy = _heading(ep["ctx_pose_g"][-1, 4])
    L = 0.06 * max(np.ptp(traj_xy[:, 0]), np.ptp(traj_xy[:, 1]), 1.0)
    ax.arrow(*ep["ctx_pose_g"][-1, HXY], L * hx, L * hy, head_width=0.25 * L, color="black")
    all_xy = np.concatenate([valid_xy, traj_xy, ep["ctx_pose_g"][-1:, HXY]])
    lo, hi = all_xy.min(0), all_xy.max(0)
    cx, cy = (lo + hi) / 2
    half = 0.55 * max(hi[0] - lo[0], hi[1] - lo[1], 4.0)
    ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="best")


def inspect_trajectory(ep: dict, spec, decode):
    """Top-down map + current frame + evenly spaced GT frames (no generation)."""
    targets = ep["targets"]
    n_show = min(8, len(targets))
    show = np.linspace(0, len(targets) - 1, n_show).round().astype(int)
    img = decode(torch.cat([ep["ctx_lat"][-1:], ep["goal_lat"][[targets[s] for s in show]]]))
    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(2, 7, wspace=0.05, hspace=0.15)
    ax = fig.add_subplot(gs[:, :2])
    _draw_topdown(ax, ep, targets, window=True)
    ax.set_title("top-down", fontsize=12)
    cells = [(r, col) for r in range(2) for col in range(2, 7)]
    axc = fig.add_subplot(gs[cells[0]])
    axc.imshow(tensor_to_image(img[0]))
    axc.set_title(f"current (idx {ep['ctx_idx'][-1]})", fontsize=10); axc.axis("off")
    for k, s in enumerate(show):
        axc = fig.add_subplot(gs[cells[k + 1]])
        axc.imshow(tensor_to_image(img[k + 1]))
        dt = (ep["goal_idx"][targets[s]] - ep["ctx_idx"][-1]) / spec.fps
        axc.set_title(f"GT step {s + 1} (+{dt:.2f}s)", fontsize=10); axc.axis("off")
    plt.suptitle(f"{ep['video_id']} — trajectory inspection (no generation yet)", fontsize=13)
    plt.show()


def write_context_video(ep: dict, spec, decode, path: str):
    """The agent's REAL walked history — the valid pool frames in temporal order
    at capture speed. SDXL decode of a 1024 pool takes ~30-60 s."""
    import av
    from PIL import Image, ImageDraw
    os.makedirs(os.path.dirname(path), exist_ok=True)
    valid = np.flatnonzero(np.asarray(ep["ctx_mask"]))
    container = av.open(path, "w")
    stream = container.add_stream("h264", rate=int(spec.fps))
    stream.height, stream.width = 256, 256
    stream.pix_fmt = "yuv420p"; stream.options = {"crf": "20"}
    for s in range(0, len(valid), 16):
        imgs = decode(ep["ctx_lat"][valid[s:s + 16]])
        for k, im in zip(valid[s:s + 16], imgs):
            fi = int(ep["ctx_idx"][k])
            u8 = (im.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            pim = Image.fromarray(u8); d = ImageDraw.Draw(pim)
            text = f"idx {fi}  t={(fi - int(ep['ctx_idx'][0])) / spec.fps:.1f}s"
            d.rectangle([0, 0, 10 + 7 * len(text), 22], fill=(0, 0, 0))
            d.text((5, 5), text, fill=(255, 255, 255))
            for pkt in stream.encode(av.VideoFrame.from_ndarray(
                    np.ascontiguousarray(np.asarray(pim)), format="rgb24")):
                container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()
    print(f"context video: {path}  ({len(valid)} frames @ {int(spec.fps)} fps)")


# ── rollout ───────────────────────────────────────────────────────────────────
class SubsampledPool(torch.nn.Module):
    """Restricts the inner memory's retrieval pool to fixed PAST positions
    (worldmem cap); slices the aligned context_keys stream identically."""

    def __init__(self, memory, pos, n_base):
        super().__init__()
        self.memory, self.pos, self.n_base = memory, pos, n_base

    def forward(self, frames, poses, times, action, stride, context_mask=None, **kw):
        T = frames.shape[1]
        idx = torch.cat([self.pos[self.pos < T - 1],
                         torch.arange(self.n_base, max(self.n_base, T - 1), dtype=torch.long),
                         torch.tensor([T - 1], dtype=torch.long)]).to(frames.device)
        cm = None if context_mask is None else context_mask[:, idx]
        if kw.get("context_keys") is not None:
            kw["context_keys"] = kw["context_keys"][:, idx]
        return self.memory(frames[:, idx], poses[:, idx], times[:, idx], action, stride, cm, **kw)


class TeacherForcedQuery(torch.nn.Module):
    """Clean-query wrapper. On soundspaces every pnp arm queries via the
    (always real) audio cue and pnp_meta scores geometry, so this is a no-op
    for ALL arms here — kept for protocol parity with the loopnav notebook."""

    def __init__(self, memory):
        super().__init__()
        self.memory, self.gt = memory, None

    def forward(self, frames, *args, **kw):
        if self.gt is not None:
            frames = frames.clone()
            frames[:, -1] = self.gt.to(frames.device, frames.dtype)
        return self.memory(frames, *args, **kw)


def rollout(model, label, ep, spec, stats_np, device, sampler="ddim", seed=0,
            insert_generated=False, teacher_forced=None, worldmem_pool_size=None):
    """AR-roll one model along ep["targets"]. Replace-tail protocol (history
    stays 100% real) unless insert_generated; audio streams threaded per step:
    the pool's cached keys are POSITION-tied, the query cue is the real
    binaural window at the CURRENT (GT) position."""
    targets = ep["targets"]
    mem_orig = model.core.memory
    while hasattr(mem_orig, "memory"):       # shed wrappers left by a crashed run
        mem_orig = mem_orig.memory
    mem = mem_orig
    if (worldmem_pool_size is not None and mem is not None
            and type(mem).__name__ == "WorldMemRetrieval"):
        valid_pos = np.flatnonzero(np.asarray(ep["ctx_mask"]))
        if worldmem_pool_size < len(valid_pos):
            sub = np.sort(np.random.default_rng(seed).choice(valid_pos, size=worldmem_pool_size, replace=False))
            mem = SubsampledPool(mem, torch.as_tensor(sub, dtype=torch.long), n_base=len(ep["ctx_mask"])).to(device)
            print(f"[{label}] worldmem pool capped: {len(valid_pos)} valid -> {worldmem_pool_size}")
    tf_query = None
    if teacher_forced == "query" and mem is not None:
        mem = tf_query = TeacherForcedQuery(mem).to(device)
    model.core.memory = mem
    pool_lat = [f for f in ep["ctx_lat"]]
    pool_keys = [k for k in ep["ctx_keys"]]
    pool_pose_g = ep["ctx_pose_g"].copy()
    pool_idx = list(ep["ctx_idx"])
    pool_mask = list(ep["ctx_mask"])
    n_real, origin_idx = len(pool_lat), pool_idx[0]
    gt_tail, cue_tail = pool_lat[-1], ep["cue_curr"]
    preds = []
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
                    context_keys=torch.stack(pool_keys)[None].to(device),
                    query_cue=cue_tail[None].to(device),
                    use_ddim=(sampler == "ddim"),
                )[0].float().cpu()
            preds.append(pred)
            tail_new = ep["goal_lat"][tpos].clone() if teacher_forced == "full" else pred
            if insert_generated or len(pool_lat) == n_real:
                pool_lat.append(tail_new); pool_keys.append(ep["goal_keys"][tpos])
                pool_pose_g = np.concatenate([pool_pose_g, tgt_pose_g[None]])
                pool_idx.append(tgt_idx); pool_mask.append(True)
            else:
                pool_lat[-1], pool_keys[-1] = tail_new, ep["goal_keys"][tpos]
                pool_pose_g[-1], pool_idx[-1] = tgt_pose_g, tgt_idx
            gt_tail, cue_tail = ep["goal_lat"][tpos], ep["goal_cues"][tpos]
            pbar.set_postfix(target_frame=tgt_idx, pool=len(pool_lat))
    finally:
        model.core.memory = mem_orig
    return torch.stack(preds)


# ── per-step retrieval recording (hooked on the BASE memory) ─────────────────
def _chunk_ids(out, times, S):
    t = out[2][0, 0, 0]
    pos = torch.tensor([torch.isclose(times[0], v).nonzero()[0, 0] for v in t])
    drop = (times.shape[1] - 1) % S
    return set(((pos - drop) // S).tolist())


def make_recorder(srows, arows, prows, pscores):
    busy = {"on": False}

    def hook(mod, args, kwargs, out):
        if busy["on"]:
            return
        t = out[2][0, 0]
        if t.dim() > 1:                        # pnp sel_times (num_sets, K) -> set 0
            t = t[0]
        prows.append(t.detach().float().cpu().numpy())
        v = out[4][0, 0, 0].detach().float().cpu()
        pscores.append(v.numpy().copy())
        t2 = torch.topk(v, min(2, v.numel())).values
        srows.append((t2[0].item(), (t2[0] - t2[-1]).item()))
        if arows is None:
            return
        busy["on"] = True
        try:
            with torch.no_grad():
                ms, mod.meta_scorer = mod.meta_scorer, None
                vis = mod(*args, **kwargs)     # pure-audio ranking, same state
                mod.meta_scorer = ms
                sc, mod.scoring = mod.scoring, "meta"
                met = mod(*args, **kwargs)     # pure-meta ranking, same state
                mod.scoring = sc
            times = args[2]
            f, vv, m = (_chunk_ids(o, times, mod.chunk_size) for o in (out, vis, met))
            arows.append((len(f & vv & m), len((f & vv) - m), len((f & m) - vv), len(f - vv - m)))
        finally:
            busy["on"] = False
    return hook


def make_picks_recorder(prows, pscores):
    def hook(mod, args, kwargs, out):
        prows.append(out[2][0, 0].detach().float().cpu().numpy())
        sc = out[4]
        pscores.append(None if sc is None else sc[0, 0].detach().float().cpu().numpy())
    return hook


def run_rollouts(models, ep, spec, stats_np, device, **kw):
    """Roll every arm on the SAME clip with the same noise seed. Returns
    (preds, score_stats, attribution, pick_times, pick_scores)."""
    preds, score_stats, attribution, pick_times, pick_scores = {}, {}, {}, {}, {}
    for label, model in models.items():
        mem = model.core.memory
        while hasattr(mem, "memory"):
            mem = mem.memory
        handle, prows, pscore_rows = None, [], []
        srows, arows = [], None
        if hasattr(mem, "chunk_size"):         # pnp arms
            arows = [] if getattr(mem, "scoring", "").startswith("visual+meta") else None
            handle = mem.register_forward_hook(make_recorder(srows, arows, prows, pscore_rows), with_kwargs=True)
        elif mem is not None:                  # temporal / worldmem
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


# ── metrics + figures ─────────────────────────────────────────────────────────
def compute_metrics(pred_img, gt_img, device):
    """Per-step PSNR / LPIPS / DreamSim dicts vs GT."""
    psnr = {l: (-10 * torch.log10(((img - gt_img) ** 2).flatten(1).mean(1))).numpy()
            for l, img in pred_img.items()}
    lp = _lpips(device)

    @torch.no_grad()
    def lpips_curve(img, bs=8):
        return torch.cat([lp(img[i:i + bs].to(device) * 2 - 1, gt_img[i:i + bs].to(device) * 2 - 1
                             ).flatten().cpu() for i in range(0, img.shape[0], bs)]).numpy()
    ds = _dreamsim(device)

    @torch.no_grad()
    def ds_curve(img, bs=8):
        return torch.cat([ds(gt_img[i:i + bs].to(device), img[i:i + bs].to(device)
                             ).flatten().cpu() for i in range(0, img.shape[0], bs)]).numpy()
    return psnr, {l: lpips_curve(i) for l, i in pred_img.items()}, {l: ds_curve(i) for l, i in pred_img.items()}


def plot_compare(ep, spec, cur_img, gt_img, pred_img, psnr, title_extra="", save_path=None):
    """One row of GT plus one row per model, PSNR-stamped."""
    targets = ep["targets"]
    rows = [("GT", gt_img)] + list(pred_img.items())
    n_show = min(8, len(targets))
    show = np.linspace(0, len(targets) - 1, n_show).round().astype(int)
    fig, axes = plt.subplots(len(rows), n_show + 1, figsize=(3.2 * (n_show + 1), 3.3 * len(rows)))
    for r, (name, imgs) in enumerate(rows):
        axes[r, 0].imshow(tensor_to_image(cur_img))
        axes[r, 0].set_title("current" if r == 0 else "")
        axes[r, 0].set_ylabel(name, fontsize=12)
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        for c, s in enumerate(show):
            ax = axes[r, c + 1]
            ax.imshow(tensor_to_image(imgs[s]))
            dt = (ep["goal_idx"][targets[s]] - ep["ctx_idx"][-1]) / spec.fps
            ax.set_title(f"step {s + 1} (+{dt:.2f}s)" if r == 0 else f"PSNR {psnr[name][s]:.1f}", fontsize=9)
            ax.axis("off")
    plt.suptitle(f"{ep['video_id']} — {len(targets)} steps{title_extra}", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_summary(ep, spec, psnr, lpips_steps, dreamsim_steps, save_path=None):
    """Top-down map (left) + per-step metric curves (right)."""
    targets = ep["targets"]
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.2], hspace=0.3, wspace=0.16)
    ax = fig.add_subplot(gs[:, 0])
    _draw_topdown(ax, ep, targets, window=False)
    ax.set_title("top-down: context path, GT trajectory, rollout poses")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    panels = [("PSNR (dB)", psnr, ".1f", "higher is better"),
              ("LPIPS", lpips_steps, ".3f", "lower is better"),
              ("DreamSim", dreamsim_steps, ".3f", "lower is better")]
    ax0 = None
    for i, (name, data, fmt, note) in enumerate(panels):
        axm = fig.add_subplot(gs[i, 1], sharex=ax0)
        ax0 = ax0 or axm
        for label, v in data.items():
            axm.plot(np.arange(1, len(v) + 1), v, marker="o", ms=3,
                     label=f"{label} (mean {format(v.mean(), fmt)})")
        axm.set_ylabel(name); axm.grid(alpha=0.3); axm.legend(fontsize=8)
        axm.set_title(f"per-step {name.split()[0]} vs GT ({note})", fontsize=10)
        if i < len(panels) - 1:
            axm.tick_params(labelbottom=False)
    axm.set_xlabel("rollout step")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ── rollout video (GT vs arms + evolving LPIPS strip) ────────────────────────
def _to_u8(img, scale):
    from PIL import Image
    u8 = (img.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    if scale > 1:
        im = Image.fromarray(u8)
        u8 = np.asarray(im.resize((im.width * scale, im.height * scale), Image.LANCZOS))
    return u8


def _err_overlay(u8, img_t, gt_t, scale, thresh, patch, alpha):
    """Non-overlapping patch x patch squares (NATIVE 256-px units); a square is
    tinted red iff its mean |pred - GT| exceeds the ABSOLUTE thresh."""
    diff = (img_t - gt_t).abs().mean(0).numpy()
    P = patch
    gh, gw = diff.shape[0] // P, diff.shape[1] // P
    grid = diff[:gh * P, :gw * P].reshape(gh, P, gw, P).mean((1, 3))
    PS = P * scale
    mask = np.zeros(u8.shape[:2], bool)
    mask[:gh * PS, :gw * PS] = np.kron(grid > thresh, np.ones((PS, PS), bool))
    out = u8.copy()
    out[mask] = (out[mask] * (1 - alpha) + np.array([255, 0, 0]) * alpha).round().astype(np.uint8)
    return out


def write_rollout_video(ep, spec, cur_img, gt_img, pred_img, lpips_steps, save_path,
                        rollout_stride, display_name=None, scale=2,
                        error_overlay=True, error_thresh=0.1, error_patch=16, error_alpha=0.2,
                        chart_h=300, chart_w_frac=1.0):
    """GT vs each arm side by side (up to 4 panels/row) with an evolving
    per-step LPIPS strip along the bottom. SCALE=2 lifts the composite past
    1080 lines so hosted transcodes keep a 1080p rung."""
    import av
    import matplotlib as mpl
    from PIL import Image, ImageDraw, ImageFont
    display_name = display_name or DISPLAY_NAME
    targets = ep["targets"]
    label_px = 20 * scale
    font = ImageFont.truetype(os.path.join(os.path.dirname(mpl.__file__),
                                           "mpl-data", "fonts", "ttf", "DejaVuSans-Bold.ttf"), label_px)

    def label2(frame_u8, name, tag):
        im = Image.fromarray(frame_u8)
        d = ImageDraw.Draw(im, "RGBA")
        tw = max(d.textlength(name, font=font), d.textlength(tag, font=font))
        d.rectangle([0, 0, tw + 14, 2 * label_px + 14], fill=(0, 0, 0, 185))
        d.text((7, 4), name, fill=(255, 255, 255), font=font)
        d.text((7, label_px + 8), tag, fill=(210, 210, 210), font=font)
        return np.asarray(im.convert("RGB"))

    panels = [("GT", gt_img)] + list(pred_img.items())
    n_p, K = len(panels), gt_img.shape[0]
    ncols = min(n_p, 4)
    nrows = int(np.ceil(n_p / ncols))
    video_fps = max(1, round(spec.fps / rollout_stride))
    hold = video_fps // 2 + 1
    cur_u8 = _to_u8(cur_img, scale)

    t_sec = (ep["goal_idx"][targets] - ep["ctx_idx"][-1]) / spec.fps
    grid_w = ncols * cur_u8.shape[1]
    chart_hs = chart_h * scale
    chart_w = int(grid_w * chart_w_frac) // 2 * 2
    dpi = 100 * scale
    figc, axc = plt.subplots(figsize=(chart_w / dpi, chart_hs / dpi), dpi=dpi)
    ymax = 1.05 * max(v.max() for v in lpips_steps.values())
    solid = {}
    for i, (nm, v) in enumerate(lpips_steps.items()):
        col = plt.cm.tab10(i)
        axc.plot(t_sec, v, color=col, lw=1.0, alpha=0.25)
        solid[nm] = axc.plot([], [], color=col, lw=2.2, label=display_name.get(nm, nm))[0]
    cursor = axc.axvline(float(t_sec[0]), color="black", lw=1.0, alpha=0.6)
    axc.set_xlim(float(t_sec[0]), float(t_sec[-1])); axc.set_ylim(0, float(ymax))
    axc.set_xlabel("seconds ahead", fontsize=14)
    axc.set_ylabel("LPIPS  (lower = better)", fontsize=14)
    axc.tick_params(labelsize=11); axc.grid(alpha=0.3)
    axc.legend(loc="upper left", fontsize=11, ncol=min(len(solid), 3),
               frameon=True, fancybox=False, edgecolor="black")
    figc.tight_layout(pad=0.6)

    def chart(k):
        kk = max(k + 1, 0)
        for nm, ln in solid.items():
            ln.set_data(t_sec[:kk], lpips_steps[nm][:kk])
        cursor.set_xdata([float(t_sec[min(max(k, 0), K - 1)])])
        figc.canvas.draw()
        buf = np.asarray(figc.canvas.buffer_rgba())[..., :3]
        out = np.full((chart_hs, grid_w, 3), 255, np.uint8)
        h, w = min(chart_hs, buf.shape[0]), min(chart_w, buf.shape[1])
        x0 = (grid_w - w) // 2
        out[:h, x0:x0 + w] = buf[:h, :w]
        return out

    frames = []
    for k in range(-hold, K):
        tiles = []
        for name, imgs in panels:
            tile = cur_u8.copy() if k < 0 else _to_u8(imgs[k], scale)
            if error_overlay and k >= 0 and name != "GT":
                tile = _err_overlay(tile, imgs[k], gt_img[k], scale, error_thresh, error_patch, error_alpha)
            tag = "current" if k < 0 else \
                f"step {k + 1} (+{(ep['goal_idx'][targets[k]] - ep['ctx_idx'][-1]) / spec.fps:.1f}s)"
            tiles.append(label2(tile, display_name.get(name, name), tag))
        tiles += [np.zeros_like(tiles[0])] * (nrows * ncols - n_p)
        rows_ = [np.concatenate(tiles[r * ncols:(r + 1) * ncols], axis=1) for r in range(nrows)]
        frames.append(np.concatenate([np.concatenate(rows_, axis=0), chart(k)], axis=0))
    plt.close(figc)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    container = av.open(save_path, "w")
    stream = container.add_stream("h264", rate=video_fps)
    stream.height, stream.width = frames[0].shape[0], frames[0].shape[1]
    stream.pix_fmt = "yuv420p"; stream.options = {"crf": "20"}
    for f in frames:
        for pkt in stream.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()
    print(f"saved {save_path}  ({len(frames)} frames @ {video_fps} fps, "
          f"{frames[0].shape[1]}x{frames[0].shape[0]})")
    return save_path


def thor_episode(ep: dict, stride: int) -> dict:
    """Adapt a SoundSpaces episode for `thor_rollout.write_rollout_video` (the panel-style
    video): the OUTBOUND leg becomes the history prelude — every frame before the first
    rollout target, read from the episode's `latents.npy` (not the sampled pool, so the
    prelude does not skip and every retrieved pick decodes) — and the return path the
    evaluation phase (`eval_start` = the first goal frame). No interaction / agent
    sidecars exist here, so captions, verdicts and sightings stay off."""
    lat = np.load(os.path.join(ep["ep_path"], "latents.npy"), mmap_mode="c")
    es = int(ep["goal_idx"][0]); out = dict(ep)
    out.update(eval_start=es, stride=int(stride), eff_len=len(ep["goal_idx"]), meta={}, targets=ep["targets"],
               ctx_idx=np.arange(0, es), ctx_lat=torch.from_numpy(np.asarray(lat[:es], dtype=np.float32)), ctx_mask=np.ones(es, dtype=bool),
               ctx_fire=np.zeros((es, 4)))
    return out


def plot_progression(ep, spec, cur_img, gt_img, pred_img, lpips_steps, display_name=None,
                     n_frames=6, scale=2, error_overlay=True, error_thresh=0.1,
                     error_patch=16, error_alpha=0.2, save_path=None):
    """Static frame progression with the video cell's conventions (DISPLAY_NAME
    row labels + the patch-wise error overlay), evenly spaced current -> end."""
    display_name = display_name or DISPLAY_NAME
    targets = ep["targets"]
    show = np.linspace(0, len(targets) - 1, min(n_frames, len(targets))).round().astype(int)
    rows = [("GT", gt_img)] + list(pred_img.items())
    fig, axes = plt.subplots(len(rows), len(show) + 1,
                             figsize=(2.6 * (len(show) + 1), 2.7 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, (name, imgs) in enumerate(rows):
        axes[r, 0].imshow(tensor_to_image(cur_img))
        axes[r, 0].set_title("current" if r == 0 else "", fontsize=15)
        axes[r, 0].set_ylabel(display_name.get(name, name), fontsize=17)
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        for c, s in enumerate(show):
            ax = axes[r, c + 1]
            u8 = _to_u8(imgs[s], scale)
            if error_overlay and name != "GT":
                u8 = _err_overlay(u8, imgs[s], gt_img[s], scale, error_thresh, error_patch, error_alpha)
            ax.imshow(u8); ax.axis("off")
            if r == 0:
                dt = (ep["goal_idx"][targets[s]] - ep["ctx_idx"][-1]) / spec.fps
                ax.set_title(f"+{dt:.1f}s", fontsize=15)
            else:
                ax.set_title(f"LPIPS {lpips_steps[name][s]:.3f}", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


# ── retrieval-geography video ────────────────────────────────────────────────
def export_frames(ep, spec, decode, gt_img, pred_img, cur_img, pick_times, save_dir, all_gt=False):
    """Dump the rollout's frames as stand-alone PNGs under `<save_dir>/frames/`:
      current.png                          the current frame the rollout starts from
      gt_targets/step<k>_f<idx>.png        the GT frame at each rollout step (what the arms
                                           are compared against; stride-aligned)
      <arm>/step<k>_f<idx>.png             each arm's generated frame per step
      <arm>_top1/step<k>_f<idx>_ctx<fi>.png the TOP-1 retrieved context frame the arm read
                                           for that step (`fi` = its episode frame index),
                                           decoded from the pool; arms without picks skip it
      gt/f<idx>.png                        every GT frame of the return path (`all_gt`)
    `<idx>` is the episode frame index (the axis of `ep["goal_idx"]`), `<k>` the rollout
    step. Picks are `pick_times` (seconds from the pool origin, rank order), resolved to
    frames exactly as `write_retrieval_geography` does. Returns the frames root."""
    from far.analysis.figure_io import save_frames, slug
    root = os.path.join(save_dir, "frames")
    targets = np.asarray(ep["targets"])
    goal_idx = np.asarray(ep["goal_idx"]); tgt_idx = goal_idx[targets]
    save_frames(root, {"current": cur_img})
    save_frames(os.path.join(root, "gt_targets"),
                {f"step{k:03d}_f{int(fi):06d}": gt_img[k] for k, fi in enumerate(tgt_idx)})
    for label, imgs in pred_img.items():
        save_frames(os.path.join(root, slug(label)),
                    {f"step{k:03d}_f{int(fi):06d}": imgs[k] for k, fi in enumerate(tgt_idx)})
    # top-1 context frame per step: pool latent nearest the pick time (+ inserted / GT goal frames)
    idx2lat = {int(i): l for i, l, m in zip(ep["ctx_idx"], ep["ctx_lat"], ep["ctx_mask"]) if m}
    idx2lat.update({int(i): l for i, l in zip(ep["goal_idx"], ep["goal_lat"])})
    fi_of = lambda t: int(round(float(t) * spec.fps)) + int(ep["ctx_idx"][0])
    top1 = {}
    for label in pred_img:
        rows = pick_times.get(label) or []
        if not len(rows):
            continue
        fis = []
        for si in range(len(targets)):
            picks = np.asarray(rows[si])
            if picks.ndim > 1:
                picks = picks[0]
            fis.append(fi_of(np.atleast_1d(picks)[0]))
        top1[label] = fis
    uniq = sorted({fi for fis in top1.values() for fi in fis if fi in idx2lat})
    dec = decode(torch.stack([idx2lat[fi] for fi in uniq])) if uniq else torch.zeros(0)
    img_of = {fi: dec[k] for k, fi in enumerate(uniq)}
    for label, fis in top1.items():
        save_frames(os.path.join(root, slug(label) + "_top1"),
                    {f"step{k:03d}_f{int(tgt_idx[k]):06d}_ctx{int(fi):06d}": img_of[fi]
                     for k, fi in enumerate(fis) if fi in img_of})
    if all_gt:
        out = {}
        for s0 in range(0, len(goal_idx), 16):
            dec = decode(ep["goal_lat"][s0:s0 + 16])
            out.update({f"f{int(goal_idx[s0 + j]):06d}": dec[j] for j in range(dec.shape[0])})
        save_frames(os.path.join(root, "gt"), out)
    n_top = sum(len(v) for v in top1.values())
    print(f"frames -> {root}: current, {len(tgt_idx)} GT targets, {len(pred_img)} arms x {len(tgt_idx)} generations, "
          f"{n_top} top-1 context frames" + (f", {len(goal_idx)} GT return-path frames" if all_gt else ""))
    return root


def topdown_mesh(ep_path, cache_dir="results/figures/topdown_maps", ss_python=None, gpu=0, force=False,
                 margin=2.0, fit_sources=False, cut_height=1.8, below=0.6, px_per_m=200.0, fade=0.12, desat=0.15):
    """The scene MESH rendered top-down for this episode, as `(img, extent)` ready for
    `ax.imshow(img, extent=extent)` in the z-up standard frame the episode poses use
    ([u, v] = [-z, -x] of habitat's (x, z); `_standardize_loopnav_pose`).

    Rendering needs habitat-sim, which lives in another conda env: the render is cached
    under `cache_dir/<traj_id>.{png,json}` by `scripts/soundspaces/render_topdown.py` (an
    orthographic camera clipped `cut_height` m above the floor, geometry-less pixels white,
    mesh faded / desaturated for print), and this call shells out to `ss_python` to build
    the cache when it is missing (`force` re-renders). `margin` = metres around the
    trajectory bbox; the figure crops with its own limits, so keep it generous."""
    import json as _json, subprocess, sys
    from PIL import Image
    from far.constants import ROOT
    meta = _json.load(open(os.path.join(ep_path, "meta.json")))
    traj_id = meta.get("traj_id", os.path.basename(ep_path))
    cache_dir = cache_dir if os.path.isabs(cache_dir) else os.path.join(ROOT, cache_dir)
    png, js = os.path.join(cache_dir, f"{traj_id}.png"), os.path.join(cache_dir, f"{traj_id}.json")
    if force or not (os.path.isfile(png) and os.path.isfile(js)):
        script = os.path.join(ROOT, "scripts", "soundspaces", "render_topdown.py")
        if not os.path.isfile(script):
            raise FileNotFoundError(f"{script} not found: the top-down mesh renderer is figure tooling that is not part of "
                                    "the public release; pass a cached render or drop the mesh map")
        ss_python = ss_python or os.environ.get("SS_PYTHON")
        if not ss_python:
            raise RuntimeError("set SS_PYTHON to the python of an environment with habitat-sim and sound-spaces "
                               "installed (scripts/soundspaces/render_topdown.py runs there)")
        cmd = [ss_python, script, "--clip-dir", ep_path, "--out", cache_dir,
               "--margin", str(margin), "--cut-height", str(cut_height), "--below", str(below), "--px-per-m", str(px_per_m),
               "--fade", str(fade), "--desat", str(desat), "--gpu", str(gpu)] + (["--fit-sources"] if fit_sources else []) \
              + (["--force"] if force else [])
        env = {**os.environ, "MAGNUM_LOG": "quiet", "HABITAT_SIM_LOG": "quiet"}
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if r.returncode != 0 or not os.path.isfile(png):
            raise RuntimeError(f"mesh render failed ({r.returncode}):\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
        print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else f"[rendered] {png}", file=sys.stderr)
    img = np.asarray(Image.open(png).convert("RGB"))
    x_lo, x_hi, z_lo, z_hi = _json.load(open(js))["extent_xz"]
    # habitat (x, z) -> standard (u, v) = (-z, -x): image rows become columns (reversed)
    return img.transpose(1, 0, 2)[:, ::-1], [-z_hi, -z_lo, -x_hi, -x_lo]


def write_retrieval_geography(ep, spec, pick_times, pick_scores, gt_img, pred_img, psnr,
                              decode, save_path, pick_top_n=3, fov_half_deg=45,
                              video_fps=4, n_pick_thumbs=4):
    """WHERE each arm's memory reads from, one video frame per rollout step:
    row 1 top-down pick maps (score-colored, viridis, per panel-step min-max),
    row 2 top-N picked frames as a 2x2 tile (rank order TL,TR,BL,BR), row 3 the
    generated frame, PSNR-stamped (leading column: the GT target)."""
    import av
    from matplotlib.patches import Wedge
    targets = ep["targets"]
    idx2pose = {int(i): p for i, p, m in zip(ep["ctx_idx"], ep["ctx_pose_g"], ep["ctx_mask"]) if m}
    idx2pose.update({int(i): p for i, p in zip(ep["goal_idx"], ep["goal_pose_g"])})
    idx2lat = {int(i): l for i, l, m in zip(ep["ctx_idx"], ep["ctx_lat"], ep["ctx_mask"]) if m}
    idx2lat.update({int(i): l for i, l in zip(ep["goal_idx"], ep["goal_lat"])})
    arms_p = [l for l in pred_img if len(pick_times.get(l, []))]
    cmap = plt.get_cmap("viridis")
    arm_col = dict(zip(arms_p, ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]))
    valid_xy = ep["ctx_pose_g"][ep["ctx_mask"]][:, HXY]
    traj_xy = ep["goal_pose_g"][:, HXY]
    all_xy = np.concatenate([valid_xy, traj_xy])
    arc_r = 0.045 * float(np.ptp(all_xy, axis=0).max())
    pad = 0.06 * float(np.ptp(all_xy, axis=0).max())
    xlim = (all_xy[:, 0].min() - pad, all_xy[:, 0].max() + pad)
    ylim = (all_xy[:, 1].min() - pad, all_xy[:, 1].max() + pad)

    def fov_arc(ax, pose, color, alpha=0.30):
        yaw = np.degrees(float(pose[4]))
        ax.add_patch(Wedge(tuple(pose[HXY]), arc_r, yaw - fov_half_deg, yaw + fov_half_deg,
                           facecolor=color, edgecolor="none", alpha=alpha, zorder=4))

    def step_picks(label, si):
        picks = np.asarray(pick_times[label][si])
        if picks.ndim > 1:
            picks = picks[0]
        return np.atleast_1d(picks)

    def fi_of(t):
        return int(round(float(t) * spec.fps)) + int(ep["ctx_idx"][0])

    top_fis = {l: [[fi_of(t) for t in step_picks(l, si)[:n_pick_thumbs]]
                   for si in range(len(targets))] for l in arms_p}
    uniq = sorted({fi for v in top_fis.values() for fis in v for fi in fis if fi in idx2lat})
    dec = decode(torch.stack([idx2lat[fi] for fi in uniq])) if uniq else torch.zeros(0)
    pick_img = {fi: dec[k] for k, fi in enumerate(uniq)}

    ncol = len(arms_p) + 1
    fig, axes = plt.subplots(3, ncol, figsize=(3.6 * ncol, 10.6),
                             gridspec_kw=dict(height_ratios=[1.15, 1.0, 1.0]), squeeze=False)

    def draw_map(ax, si, label=None):
        ax.clear()
        ax.plot(valid_xy[:, 0], valid_xy[:, 1], ":", color="dimgray", lw=0.9, alpha=0.8)
        ax.plot(traj_xy[:, 0], traj_xy[:, 1], "-", color="tab:red", lw=1.1, alpha=0.35)
        cur = ep["goal_pose_g"][targets[si]][HXY]
        if label is not None:
            picks = step_picks(label, si)
            scs = (pick_scores.get(label) or [None] * len(pick_times[label]))[si]
            scs = None if scs is None else np.atleast_1d(np.asarray(scs, dtype=float))
            if pick_top_n is not None:
                picks = picks[:pick_top_n]
                scs = None if scs is None else scs[:pick_top_n]
            if scs is not None:
                rng = float(scs.max() - scs.min())
                norm = (scs - scs.min()) / rng if rng > 0 else np.full(len(scs), 0.5)
            for k, t in enumerate(picks):
                fi = fi_of(t)
                if fi not in idx2pose:
                    continue
                col = cmap(float(norm[k])) if scs is not None else arm_col[label]
                pk = idx2pose[fi]
                ax.plot([cur[0], pk[HXY][0]], [cur[1], pk[HXY][1]], "-", color=col, lw=1.1, alpha=0.8)
                ax.plot(*pk[HXY], "o", color=col, ms=4.5, mec="black", mew=0.3)
                fov_arc(ax, pk, col)
        ax.plot(*cur, "o", color="black", ms=8, zorder=5)
        fov_arc(ax, ep["goal_pose_g"][targets[si]], "black", alpha=0.22)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("trajectory" if label is None else label, fontsize=10)

    geo_frames = []
    axes[0, 0].set_visible(False)
    axes[1, 0].set_visible(False)
    for si in range(len(targets)):
        axes[2, 0].clear()
        axes[2, 0].imshow(tensor_to_image(gt_img[si]))
        axes[2, 0].set_title("GT target", fontsize=10); axes[2, 0].axis("off")
        for j, label in enumerate(arms_p):
            draw_map(axes[0, j + 1], si, label)
            ax = axes[1, j + 1]; ax.clear()
            tiles, tags = [], []
            for fi in top_fis[label][si]:
                if fi in pick_img:
                    tiles.append(np.asarray(tensor_to_image(pick_img[fi])))
                    tags.append(f"{(fi - int(ep['goal_idx'][targets[si]])) / spec.fps:+.1f}s")
                else:
                    tiles.append(None); tags.append("n/a")
            while len(tiles) < n_pick_thumbs:
                tiles.append(None); tags.append("")
            proto = next((t for t in tiles if t is not None), None)
            if proto is not None:
                tiles = [t if t is not None else np.zeros_like(proto) for t in tiles]
                gap, (th, tw) = 4, proto.shape[:2]
                grid = np.zeros((2 * th + gap, 2 * tw + gap, proto.shape[2]), dtype=proto.dtype)
                quad = [(0, 0), (0, 1), (1, 0), (1, 1)]
                for t, (r, cc) in zip(tiles, quad):
                    grid[r * (th + gap):r * (th + gap) + th, cc * (tw + gap):cc * (tw + gap) + tw] = t
                ax.imshow(grid)
                for tag, (r, cc) in zip(tags, quad):
                    if tag:
                        ax.text(cc * (tw + gap) + 10, r * (th + gap) + 14, tag, fontsize=6,
                                color="white", va="top", bbox=dict(fc="black", alpha=0.6, pad=1))
            ax.set_title(f"top-{n_pick_thumbs} picks (rank TL,TR,BL,BR)", fontsize=9)
            ax.axis("off")
            ax = axes[2, j + 1]; ax.clear()
            ax.imshow(tensor_to_image(pred_img[label][si]))
            ax.set_title(f"PSNR {psnr[label][si]:.1f}", fontsize=10); ax.axis("off")
        dt = (ep["goal_idx"][targets[si]] - ep["ctx_idx"][-1]) / spec.fps
        fig.suptitle(f"{ep['video_id']} — step {si + 1}/{len(targets)} (+{dt:.1f}s)", fontsize=12)
        fig.canvas.draw()
        geo_frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)

    h, w = geo_frames[0].shape[:2]
    h -= h % 2; w -= w % 2
    geo_frames = [f[:h, :w] for f in geo_frames]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    container = av.open(save_path, "w")
    stream = container.add_stream("h264", rate=video_fps)
    stream.height, stream.width = h, w
    stream.pix_fmt = "yuv420p"; stream.options = {"crf": "20"}
    for f in geo_frames:
        for pkt in stream.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()
    print(f"saved {save_path}  ({len(geo_frames)} frames @ {video_fps} fps, {w}x{h})")
    return save_path
