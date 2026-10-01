"""AI2-THOR rollout analysis — the machinery behind `notebooks/rollout_ai2thor.ipynb`.

The notebook is knobs + calls; the machinery lives here so the notebook stays
short. The design notes for each step are the docstrings of the functions that
implement them.

Pipeline (one call per notebook cell):
    build_model / build_dataset      -> models, data_cfg, dataset, spec
    install_eval_knobs               -> eval diffusion, chunk size, context size
    load_episode + print_revisits    -> `ep` dict + the revisit menu
    resolve_window + apply_window    -> START_OFFSET / MAX_STEPS / targets
    inspect_trajectory / write_episode_video
    run_rollouts                     -> preds, pick_times, pick_scores, step_interact
    decode_all + metric_curves       -> images + per-step metric per arm
    plot_compare / plot_metric_curve / change_region_diagnostic /
    write_rollout_video / plot_retrieval_geography

Corpus conventions (ai2thor_v2): TOUR episodes fire up to 5 interaction chains
on an EXPLORATION leg, then RETRACE deterministically on an EVAL leg where the
agent revisits each station. `THORBackend` standardizes poses to the z-up frame
[x, y, z, pitch, yaw] (forward = (cos yaw, sin yaw)) and serves per-frame
CUMULATIVE interaction counts `events` (T, 4) over {open, close, pickup, put};
the per-step `action_interact` is their difference over (current, target],
exactly as the training formatter builds it.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from far.data.transforms import global_to_local_pose, normalize_data, tensor_to_image

INTERACT_CODES = ["open", "close", "pickup", "put"]   # THORBackend.INTERACT_CODES order
THOR_ACTIONS = {"PickupObject": "pickup", "PutObject": "put", "OpenObject": "open", "CloseObject": "close"}


# ── models / dataset ──────────────────────────────────────────────────────────
def build_model(run_dir: str, ckpt_step: str, device):
    """Instantiate a run's PixelLDWM from its saved config and load the EMA
    generator + jointly-trained memory from `checkpoints/<ckpt_step>.pth.tar`.
    Runs from other machines reference a pre-downloaded local VAE folder; when
    it is absent here the HF id is used (resolved from the local HF cache)."""
    cfg = OmegaConf.load(f"{run_dir}/config.yaml")
    OmegaConf.set_struct(cfg, False)
    tok_ckpt = cfg.model.tokenizer.get("ckpt_path")
    if tok_ckpt and "sdxl-vae-fp16-fix" in str(tok_ckpt) and not os.path.isdir(str(tok_ckpt)):
        cfg.model.tokenizer.ckpt_path = "madebyollin/sdxl-vae-fp16-fix"
    ckpt = torch.load(f"{run_dir}/checkpoints/{ckpt_step}.pth.tar", map_location="cpu", weights_only=False)
    if "memory" in ckpt and cfg.model.get("memory") is not None:
        cfg.model.memory.ckpt_path = None          # the trained memory state supersedes the pretrain init
    model = instantiate(cfg.model).to(device).eval()
    model.core.generator.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ckpt["ema"].items()}, strict=True)
    if "memory" in ckpt:
        model.core.memory.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ckpt["memory"].items()}, strict=True)
    gen = model.core.generator
    model.train_steps = int(ckpt["train_steps"])   # provenance for eval records ("latest" goes stale)
    print(f"{_run_label(run_dir):36s} gen={type(gen).__module__.split('.')[-1]}"
          f"({sum(p.numel() for p in gen.parameters()) / 1e6:.0f}M) mem={type(model.core.memory).__name__:18s} "
          f"steps={ckpt['train_steps']} interact_dim={getattr(gen, 'interact_dim', 0)}")
    return model, cfg

def _run_label(run_dir):
    """Short display name of a checkpoint bundle: the run_name segment of a timestamped run dir
    (`<stamp>__<script>__<run_name>__...`), else the bundle path itself (`models/<corpus>/<arm>`)."""
    parts = os.path.basename(os.path.normpath(run_dir)).split("__")
    return parts[2] if len(parts) > 2 else os.path.normpath(run_dir)


def build_models(runs: dict, device):
    """{label: (run_dir, step)} -> ({label: model}, data_cfg of the LAST run).
    Every arm must carry interact_dim=4: an arm without it silently ignores
    `action_interact` and can never render a reveal."""
    models, cfg = {}, None
    for label, (rd, st) in runs.items():
        models[label], cfg = build_model(rd, st, device)
    for label, m in models.items():
        assert getattr(m.core.generator, "interact_dim", 0) >= len(INTERACT_CODES), \
            f"{label}: generator interact_dim < {len(INTERACT_CODES)} — cannot render interactions"
        # interact_dim > 4 (the AI2-THOR v3 corpus): the count basis plus leading columns of the
        # STATE basis [door fraction at the goal, 0, pickup/put onsets] —
        # `LDWMCore._generator_interact` appends them from `action_state`,
        # which the rollout below builds per step from `interact_state`.
    return models, cfg


def build_dataset(data_cfg, group: str, pool_size: int):
    """RAW (unformatted) test dataset with ContextTrajectorySampler — keeps the
    GLOBAL poses and the cumulative `events` the formatter would drop.
    Returns (dataset, spec, stats_np, trained_pool_size)."""
    from far.constants import CONFIG_ROOT
    if group not in data_cfg:
        data_cfg[group] = OmegaConf.load(os.path.join(CONFIG_ROOT, "dataset", f"{group}.yaml"))
        print(f"grafted dataset group '{group}' from CONFIG_ROOT")
    dg = data_cfg[group]
    dg.test.base_dataset.sampler = dg.sampler.context_traj
    dg.test.base_dataset.sampler.context_pool_size = pool_size
    data_cfg.len_traj_pred = 1024
    dataset = instantiate(dg.test.base_dataset)
    spec = instantiate(dg.spec)
    stats_np = {k: v.unsqueeze(0).numpy() for k, v in spec.local_pos_stats.items()}
    trained_pool = int(dg.sampler.context_goal.context_pool_size)
    print(f"test clips: {len(dataset)} | fps {spec.fps} | trained context_pool_size {trained_pool}")
    return dataset, spec, stats_np, trained_pool


def use_ctrl_index(data_cfg, group: str):
    """Repoint the group's TEST index at the seed-matched `_ctrl` siblings
    (`<...>_test_ctrl.json`, positionally aligned with the act index — the same
    CLIP_IDX addresses the paired control episode: same scene, same route, NO
    state changes). Call BEFORE `build_dataset`. Grafts the group from
    CONFIG_ROOT first when the loaded run config lacks it (cross-corpus RUNS),
    mirroring `build_dataset`. Ctrl episodes still fire open/close events at
    the same stations (observe-only visits, `moves: []`), so revisit-anchored
    windows work — but every reveal shows an UNCHANGED interior, which is the
    paired-control point."""
    from far.constants import CONFIG_ROOT
    if group not in data_cfg:
        data_cfg[group] = OmegaConf.load(os.path.join(CONFIG_ROOT, "dataset", f"{group}.yaml"))
        print(f"grafted dataset group '{group}' from CONFIG_ROOT")
    bd = data_cfg[group].test.base_dataset
    ip = str(bd.index_path).replace("_test.json", "_test_ctrl.json")
    assert ip != str(bd.index_path), f"unexpected test index_path: {bd.index_path}"
    if not os.path.isfile(ip):
        raise FileNotFoundError(
            f"{ip} — build it: python scripts/data/build_index.py --manifest <group manifest> "
            f"--output {ip} --split test --filter-substring _ctrl")
    bd.index_path = ip
    print(f"test index -> CTRL siblings ({os.path.basename(ip)}; same CLIP_IDX ordering as the act set)")


def install_eval_knobs(models, data_cfg, sampler="ddim", sampling_steps=20, pnp_chunk_size=None, context_size=None):
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


# ── episode ───────────────────────────────────────────────────────────────────
def station_xyz(obj_id: str) -> np.ndarray:
    """'Type|-00.33|+00.76|+02.98' -> standard-frame [z, x, height]. Compound ids
    ('Sink|..|..|..|SinkBasin') keep the coordinates in fields 1-3."""
    q = obj_id.split("|")
    return np.array([float(q[3]), float(q[1]), float(q[2])])


def load_episode(dataset, clip_idx: int, seed: int = 0) -> dict:
    """Sample the clip and unpack it into an episode dict: latents, global
    poses, absolute frame indices, pool mask, cumulative events, meta.json."""
    import random
    random.seed(seed)
    item = dataset[clip_idx]
    n = len(item["context_indices"])
    ev_all = np.asarray(item["events"], dtype=np.float32)
    st_all = item.get("interact_state")
    st_all = None if st_all is None else np.asarray(st_all, dtype=np.float32)
    ag_all = item.get("agent_state")                  # (T, 6) [vi, dx, dy, vi', dx', dy'] (THORDynBackend agent_state)
    ag_all = None if ag_all is None else np.asarray(ag_all, dtype=np.float32)
    pose_g = item["pose"].numpy().astype(np.float64)
    ep_path = next(s_["path"] for s_ in dataset.samples if s_["video_id"] == item["video_id"])
    meta = json.loads(open(os.path.join(ep_path, "meta.json")).read())
    ep = dict(
        clip_idx=clip_idx, video_id=item["video_id"], ep_path=ep_path, meta=meta,
        eval_start=int(meta["eval_start"]), n_pool=n,
        ctx_lat=item["frames"][:n].clone(), goal_lat=item["frames"][n:],
        ctx_pose_g=pose_g[:n], goal_pose_g=pose_g[n:], pose_all=pose_g,
        ctx_idx=np.asarray(item["context_indices"]), goal_idx=np.asarray(item["goal_indices"]),
        ctx_mask=np.asarray(item["context_mask"], dtype=bool),
        ctx_ev=ev_all[:n], goal_ev=ev_all[n:],
        ctx_state=None if st_all is None else st_all[:n],
        goal_state=None if st_all is None else st_all[n:],
        # second-agent detection per pool / goal frame, and the pair spacing the
        # backend used for the primed half (needed to build the generated
        # frames' states in `rollout`)
        ctx_agent=None if ag_all is None else ag_all[:n],
        goal_agent=None if ag_all is None else ag_all[n:],
        agent_pair_s=float(getattr(dataset.backend, "agent_state_pair_s", 0.0)),
    )
    ep["idx_all"] = np.concatenate([ep["ctx_idx"], ep["goal_idx"]])
    return ep


def nice_type(obj_id: str, stations: dict = None) -> str:
    """'DiningTable|+02.42|...' -> 'dining table' (the station's recorded `type` when known)."""
    t = (stations or {}).get(obj_id, {}).get("type") or obj_id.split("|")[0]
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", t).lower()


def episode_events(ep: dict) -> list:
    """The episode's interaction events (`meta["events"]`, chronological) as
    [(frame, caption)]: "Pick up apple", "Put apple in fridge" / "on dining
    table" (in/on from the move's `dst_container`; the item is the last pickup
    before the put), "Open cabinet", "Close cabinet"."""
    meta = ep["meta"]; stations = {st["objectId"]: st for st in meta.get("stations", [])}
    moves = {(m["item"], m["to"]): m for m in meta.get("moves", [])}
    out, held = [], None
    for e in sorted(meta.get("events", []), key=lambda e: e["frame"]):
        code = THOR_ACTIONS.get(e.get("action"))
        if code is None:
            continue
        obj = nice_type(e["objectId"], stations)
        if code == "pickup":
            held, text = e["objectId"], f"Pick up {obj}"
        elif code == "put":
            m = moves.get((held, e["objectId"])); item = nice_type(held, stations) if held else "item"
            text, held = f"Put {item} {'in' if m and m.get('dst_container') else 'on'} {obj}", None
        else:
            text = f"{code.capitalize()} {obj}"
        out.append((int(e["frame"]), text))
    return out


def arrival_frame(ep: dict, e: dict) -> int:
    """Frame of CLOSEST APPROACH to the station within the revisit's segment.
    A revisit's `frames` window is a segment of the eval phase and the segments
    partition it — frames[0] is when the agent LEAVES the previous station, so
    anchoring there shows the wrong place. Arrival is purely geometric, so it
    works for a source station whose object is now gone."""
    f0, f1 = e["frames"]
    sel = np.flatnonzero((ep["idx_all"] >= f0) & (ep["idx_all"] <= f1))
    if not len(sel):
        return f0
    d = np.linalg.norm(ep["pose_all"][sel, :2] - station_xyz(e["objectId"])[None, :2], axis=1)
    return int(ep["idx_all"][sel[int(np.argmin(d))]])


def station_final(ep: dict, obj_id: str):
    """The station's FINAL state after the exploration-phase moves, walked IN ORDER so a
    chain resolves correctly (a sofa that received the laptop and then lost it is
    EMPTY, not "holds + empty"): ("gone", item) for a moved item's own station,
    ("empty", item) for a site the item was taken from, ("holds", item) for the site
    it was last put on; None when no move touched the station (observed only).
    `moves` lists only the chains whose pickup FIRED; a free-standing pickup records
    `src` as the object itself, so the object's own station IS the source."""
    state = None
    for mv in ep["meta"].get("moves", []):
        item = mv.get("item")
        if mv.get("src") == obj_id:
            state = ("gone", item) if item == obj_id else ("empty", item)
        if mv.get("to") == obj_id:
            state = ("holds", item)
    return state


def station_role(ep: dict, obj_id: str) -> str:
    """What the exploration phase did to this station, for the revisit menu
    (`station_final`, in words)."""
    st = station_final(ep, obj_id)
    return {None: "UNCHANGED — observed only (control)",
            "gone": "SOURCE — object was here, should now be GONE",
            "empty": "SOURCE — object taken from here, should now be EMPTY",
            "holds": "DESTINATION — object placed here, should now HOLD it"}[st and st[0]]


def station_state(ep: dict, obj_id: str) -> str:
    """`station_final` as a short caption for the video: "Apple should be gone" (the
    moved item's own station), "Counter top should be empty" (a site the item was
    taken from), "Apple should be here" (where it was last put), "Unchanged"."""
    stations = {st["objectId"]: st for st in ep["meta"].get("stations", [])}
    st = station_final(ep, obj_id)
    if st is None:
        return "Unchanged"
    kind, item = st; item = nice_type(item, stations).capitalize()
    return {"gone": f"{item} should be gone", "empty": f"{nice_type(obj_id, stations).capitalize()} should be empty",
            "holds": f"{item} should be here"}[kind]


def before_counterpart(ep: dict, frames, pos_tol=0.02, yaw_tol=np.deg2rad(1.0), pitch_tol=np.deg2rad(1.0), min_peak=0.05) -> dict:
    """{eval frame: exploration-leg frame showing the BEFORE state, or None} for
    each frame in `frames` — the change-mask builder's `smallest` reference: among
    the EXACT-view counterparts (closures.npz pairs within `pos_tol` m /
    `yaw_tol` / `pitch_tol`, one side in each phase) whose latent differs at all
    (peak token error > `min_peak`), the one with the smallest mean error."""
    from far.data.backends import _thor_pitch_from_quat
    d, es = ep["ep_path"], int(ep["eval_start"])
    lat = np.load(os.path.join(d, "latents.npy"), mmap_mode="c"); T = int(lat.shape[0])
    pz = np.load(os.path.join(d, "poses.npz"))
    pos, yaw, pit = np.asarray(pz["pos"])[:T], np.asarray(pz["heading"])[:T], _thor_pitch_from_quat(pz["quat"])[:T]
    cz = np.load(os.path.join(d, "closures.npz")); i, j = np.asarray(cz["i"]), np.asarray(cz["j"])
    ok = (i < T) & (j < T); i, j = i[ok], j[ok]
    d_pos = np.linalg.norm(pos[i] - pos[j], axis=1)
    d_yaw = np.abs(np.arctan2(np.sin(yaw[i] - yaw[j]), np.cos(yaw[i] - yaw[j]))); d_pit = np.abs(pit[i] - pit[j])
    exact = (d_pos < pos_tol) & (d_yaw < yaw_tol) & (d_pit < pitch_tol) & (((i < es) & (j >= es)) | ((j < es) & (i >= es)))
    ev, ctx = np.maximum(i, j)[exact], np.minimum(i, j)[exact]
    out = {}
    for f in frames:
        cs = ctx[ev == f]
        if not len(cs):
            out[f] = None; continue
        x = lat[f].astype(np.float32)
        errs = {int(c): ((x - lat[c].astype(np.float32)) ** 2).mean(0) for c in cs}
        nz = [c for c, e in errs.items() if float(e.max()) > min_peak]
        out[f] = min(nz, key=lambda c: float(errs[c].mean())) if nz else None
    return out


def _grow_box(y0, y1, x0, x1, n=32, min_len=8):
    """Grow a token box to at least `min_len` cells a side (LPIPS's AlexNet collapses
    crops narrower than ~64 px), sliding to stay inside the grid."""
    def side(lo, hi):
        if hi - lo < min_len:
            lo = max(0, lo - (min_len - (hi - lo)) // 2); hi = min(n, lo + min_len); lo = max(0, hi - min_len)
        return lo, hi
    (y0, y1), (x0, x1) = side(y0, y1), side(x0, x1)
    return y0, y1, x0, x1


def exclude_from_pool(ep, spec, window=None, event=None, pad_s=1.0, verbose=True):
    """COUNTERFACTUAL ablation: make one or more intervals of the observation
    history UNRETRIEVABLE, so the arms must answer the reveal without them.

    Call AFTER `apply_window` (it intersects with the pool phase rather than
    replacing it). Excluded frames are cleared in `ep["ctx_mask"]`, which every
    memory module already honours as padding, so none of them can be selected —
    no other code path changes.

    `window`: one interval or a LIST of them, excluded as [a, b):
        `(a, b)` in EPISODE frame indices when the bounds are ints, or in
        SECONDS when either bound is a float; `[(a1, b1), (a2, b2), ...]` for
        several. Overlaps are fine, the union is taken.
    `event`:  one index or a LIST of indices into the episode's exploration-leg
        interaction FIRINGS (the order `inspect_trajectory` prints, negatives
        count from the end); each firing's frame is excluded padded by `pad_s`
        seconds each side. Use this to delete "the moment the container was
        opened / the object was moved" without hunting for frame numbers.
    Both may be given together; the exclusions are unioned.
    Returns the resolved `[(a, b), ...]` in frame indices, or None.

    The point of the ablation: an arm that still renders the CHANGED state
    after its evidence is gone is not reading it from memory.
    """
    def _windows(x):
        """None -> []; `(a, b)` -> one interval; `[(a, b), ...]` -> several."""
        if x is None:
            return []
        seq = list(x)
        return [tuple(v) for v in seq] if seq and isinstance(seq[0], (list, tuple)) else [tuple(seq)]

    def _events(x):
        """None -> []; an int -> one; a list/tuple of ints -> several."""
        if x is None:
            return []
        return [int(v) for v in x] if isinstance(x, (list, tuple)) else [int(x)]

    idx = np.asarray(ep["ctx_idx"])
    wins = []
    for w in _windows(window):
        a, b = w
        if isinstance(a, float) or isinstance(b, float):
            a, b = round(float(a) * spec.fps), round(float(b) * spec.fps)
        wins.append((int(a), int(b)))
    ev_list = _events(event)
    if ev_list:
        fires = np.flatnonzero(np.asarray(ep["ctx_fire"]).sum(1) > 0)
        assert len(fires), "no interaction firings in this episode's observation history"
        pad = int(round(pad_s * spec.fps))
        for e in ev_list:
            assert -len(fires) <= e < len(fires), f"event {e} out of range (0..{len(fires) - 1})"
            f = int(idx[fires[e]])
            wins.append((f - pad, f + pad + 1))
    if not wins:
        return None

    keep = np.asarray(ep["ctx_mask"], dtype=bool)
    fire = np.asarray(ep["ctx_fire"]) if "ctx_fire" in ep else None
    drop_all = np.zeros(len(idx), dtype=bool)
    for a, b in wins:
        d = (idx >= a) & (idx < b) & keep
        d[-1] = False                                      # never drop the CURRENT frame (the pool tail)
        if verbose:
            n_fire = int(fire[d].sum()) if fire is not None else 0
            note = "" if d.any() else "   !! does not overlap the retrievable pool"
            print(f"EXCLUDE [{a}, {b}) = {(b - a) / spec.fps:.1f}s: {int(d.sum())} pool frames, "
                  f"{n_fire} interaction firings{note}")
        drop_all |= d
    ep["ctx_mask"] = keep & ~drop_all
    n_before, n_after = int(keep.sum()) - 1, int(ep["ctx_mask"].sum()) - 1
    if verbose:
        left = ep["ctx_idx"][ep["ctx_mask"]][:-1]
        n_fire = int(fire[drop_all].sum()) if fire is not None else 0
        print(f"EXCLUDE total ({len(wins)} window{'s' if len(wins) != 1 else ''}): dropped {int(drop_all.sum())} "
              f"pool frames, {n_fire} firings | RETRIEVABLE {n_before} -> {n_after}, of which "
              f"{int((left < ep['eval_start']).sum())} exploration")
    # Over-exclusion is the failure mode that MASQUERADES as a broken model: with
    # nothing retrievable every memory falls back to padding frames, so the arms
    # roll out with no usable context and look like they ignore memory entirely.
    # Easy to hit by accident because FLOAT bounds are SECONDS: (0.0, 43.0) is the
    # whole pool of a 43 s clip. Say so loudly rather than let it read as a result.
    if n_after == 0:
        print("  !! EVERY retrievable frame was excluded — the arms will roll out WITHOUT context "
              "(memories fall back to padding). If that is not the ablation you meant, check the "
              "window units: FLOATS are SECONDS, ints are frames.")
    elif n_after < 0.1 * max(n_before, 1):
        print(f"  !! only {n_after} retrievable frames remain ({n_after / max(n_before, 1):.0%} of the pool) — "
              f"retrieval is near-degenerate; check the window units (floats = seconds, ints = frames).")
    return wins

def print_revisits(ep: dict, spec) -> list:
    """Print the eval-phase revisit menu (one row per `inspections` entry with
    phase 'eval') and return it."""
    meta = ep["meta"]
    stations = {st["objectId"]: st for st in meta.get("stations", [])}
    revisits = [e for e in meta.get("inspections", []) if e["phase"] == "eval"]
    curr0 = int(ep["ctx_idx"][-1])
    print(f"revisits in the eval phase (current frame = {curr0}, "
          f"{meta.get('k_realized', '?')} of {meta.get('k_target', '?')} chains fired):")
    for i, e in enumerate(revisits):
        st = stations.get(e["objectId"], {})
        arr, op = arrival_frame(ep, e), e.get("open_frames")
        print(f"  [{i}] {st.get('type', '?'):10s} station {st.get('idx', '?')} kind={e['kind']:4s}"
              f"  segment={e['frames']}  ARRIVES {arr} (+{(arr - curr0) / spec.fps:5.1f}s)"
              + (f"  opens {op[0]}" if op else "") + f"  |  {station_role(ep, e['objectId'])}")
    if not revisits:
        print("  (none — this episode logs no eval-phase revisits)")
    return revisits


def resolve_window(ep: dict, revisits: list, spec, revisit, sec_before, sec_after, stride,
                   start_offset, max_steps):
    """REVISIT -> (START_OFFSET, MAX_STEPS, anchor). None keeps the manual values.
    The window brackets the ARRIVAL at the chosen station: `sec_before` before it
    to `sec_after` after it, clamped to the eval leg's first frame."""
    if revisit is None:
        return start_offset, max_steps, None
    assert revisits, "REVISIT set but this episode has no eval-phase revisits"
    assert 0 <= revisit < len(revisits), f"REVISIT {revisit} out of range (0..{len(revisits) - 1})"
    e = revisits[revisit]
    anchor = arrival_frame(ep, e)
    curr0 = int(ep["ctx_idx"][-1])
    f_start = anchor - int(round(sec_before * spec.fps))
    f_end = anchor + int(round(sec_after * spec.fps))
    so = f_start - curr0
    if so < 0:
        print(f"  !! arrival - {sec_before:g}s = frame {f_start} precedes the eval leg (starts at {curr0}); clamping START_OFFSET to 0")
        so, f_start = 0, curr0
    ms = max(1, int(round((f_end - f_start) / stride)))
    stations = {st["objectId"]: st for st in ep["meta"].get("stations", [])}
    print(f"  -> REVISIT {revisit} ({stations.get(e['objectId'], {}).get('type', '?')}): arrival {anchor}, "
          f"generating [{f_start}..{f_end}] = START_OFFSET {so}, MAX_STEPS {ms} "
          f"({sec_before:g}s before .. {sec_after:g}s after arrival @ stride {stride})")
    if f_end > e["frames"][1]:
        print(f"     note: window runs {(f_end - e['frames'][1]) / spec.fps:.1f}s past this segment — it spills into the next revisit")
    return so, ms, anchor


def apply_window(ep: dict, spec, start_offset: int, pool_phase: str, stride: int, max_steps) -> list:
    """Move the rollout start `start_offset` frames into the eval leg and build
    the step targets. Mutates `ep` in place; returns `targets` (positions into
    the remaining goal arrays).

    POOL_PHASE decides what the retrievable history contains:
      "exploration" — the frames walked before the start do NOT join the pool;
          only the start frame is appended as the CURRENT observation, so every
          RETRIEVABLE frame predates eval_start (the headline setting: answer
          "what is here now?" from memory of the room BEFORE it was rearranged).
      "traversed"   — the walked eval-phase frames join the pool (ablation).
    Also derives the interaction bookkeeping: per-frame firings are the row
    difference of the CUMULATIVE counts; any (a, b] window is events[b] − events[a]."""
    if start_offset > 0:
        assert start_offset < len(ep["goal_idx"]), "START_OFFSET exceeds the sampled trajectory"
        absorb = slice(0, start_offset) if pool_phase == "traversed" else slice(start_offset - 1, start_offset)
        k = start_offset if pool_phase == "traversed" else 1
        ep["ctx_lat"] = torch.cat([ep["ctx_lat"], ep["goal_lat"][absorb]])
        ep["ctx_pose_g"] = np.concatenate([ep["ctx_pose_g"], ep["goal_pose_g"][absorb]])
        ep["ctx_idx"] = np.concatenate([ep["ctx_idx"], ep["goal_idx"][absorb]])
        ep["ctx_mask"] = np.concatenate([ep["ctx_mask"], np.ones(k, dtype=bool)])
        ep["ctx_ev"] = np.concatenate([ep["ctx_ev"], ep["goal_ev"][absorb]])
        if ep.get("ctx_state") is not None:
            ep["ctx_state"] = np.concatenate([ep["ctx_state"], ep["goal_state"][absorb]])
            ep["goal_state"] = ep["goal_state"][start_offset:]
        for key in ("goal_lat", "goal_pose_g", "goal_idx", "goal_ev"):
            ep[key] = ep[key][start_offset:]
    v = ep["ctx_idx"][ep["ctx_mask"]]
    retr = v[:-1]                                              # the memory treats the last slot as "now"
    n_eval_retr = int((retr >= ep["eval_start"]).sum())
    print(f"pool ({pool_phase}): {len(v)} valid frames = {int((v < ep['eval_start']).sum())} exploration + "
          f"{int((v >= ep['eval_start']).sum())} eval-phase | RETRIEVABLE: {len(retr)}, of which {n_eval_retr} eval-phase")
    eff_len = int(np.searchsorted(ep["goal_idx"], ep["goal_idx"][-1])) + 1   # sampler repeat-pads short clips
    steps = eff_len // stride
    assert steps >= 1, "trajectory too short for this ROLLOUT_STRIDE/START_OFFSET"
    if max_steps is not None:
        steps = min(steps, max_steps)
    targets = list(range(stride - 1, steps * stride, stride))
    ep["eff_len"], ep["stride"], ep["targets"] = eff_len, stride, targets
    ep["ev_curr0"] = ep["ctx_ev"][-1]
    ep["state_curr0"] = None if ep.get("ctx_state") is None else ep["ctx_state"][-1]
    ep["ctx_fire"] = np.diff(np.concatenate([ep["ctx_ev"][:1], ep["ctx_ev"]]), axis=0)
    ep["goal_fire"] = np.diff(np.concatenate([ep["ctx_ev"][-1:], ep["goal_ev"]]), axis=0)
    ctx_rows = np.flatnonzero(ep["ctx_fire"].sum(1) > 0)
    eval_rows = np.flatnonzero(ep["goal_fire"][:eff_len].sum(1) > 0)
    in_win = [r for r in eval_rows if r <= targets[-1]]
    print(f"clip {ep['video_id']}: pool={ep['n_pool']} curr_idx={ep['ctx_idx'][-1]} rollout frames "
          f"[{ep['goal_idx'][targets[0]]}..{ep['goal_idx'][targets[-1]]}] ({len(targets)} steps @ stride {stride} = {stride / spec.fps:.1f}s)")
    print("interactions  outbound: " + ", ".join(f"{c}={int(x)}" for c, x in zip(INTERACT_CODES, ep["ev_curr0"])) + f"  [{len(ctx_rows)} frames]")
    print("interactions  eval leg: " + ", ".join(f"{c}={int(x)}" for c, x in zip(INTERACT_CODES, ep["goal_fire"][:eff_len].sum(0)))
          + f"  [{len(eval_rows)} frames, {len(in_win)} inside the rollout window]")
    if not in_win:
        print("  !! no interaction fires inside the rollout window — raise MAX_STEPS / SEC_AFTER or pick another clip")
    return targets


# ── decoding ──────────────────────────────────────────────────────────────────
class Decoder:
    """SDXL-VAE decode in chunks -> images in [0, 1], on CPU."""

    def __init__(self, tokenizer, device, bs=8):
        self.tok, self.device, self.bs = tokenizer, device, bs

    @torch.no_grad()
    def __call__(self, lat):
        out = [self.tok.detokenize(lat[i:i + self.bs].to(self.device)).clamp(-1, 1).cpu()
               for i in range(0, lat.shape[0], self.bs)]
        return torch.cat(out) * 0.5 + 0.5


def _u8(img):
    return (img.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def _label_u8(frame_u8, text, color=(255, 255, 255)):
    from PIL import Image, ImageDraw
    im = Image.fromarray(frame_u8)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 10 + 7 * len(text), 22], fill=(0, 0, 0))
    d.text((5, 5), text, fill=color)
    return np.asarray(im)


QUICKSAND_URL = "https://raw.githubusercontent.com/google/fonts/main/ofl/quicksand/Quicksand%5Bwght%5D.ttf"


def _quicksand_path():
    """The Quicksand variable font (OFL, Google Fonts) under `$FAR_MODELS_ROOT/cache/fonts`,
    downloaded once like the SwinIR weights; None if it is neither there nor fetchable."""
    from far.constants import ROOT
    from far.constants import CACHE_ROOT
    path = os.path.join(CACHE_ROOT, "fonts", "Quicksand[wght].ttf")
    if not os.path.isfile(path):
        try:
            import urllib.request
            os.makedirs(os.path.dirname(path), exist_ok=True); urllib.request.urlretrieve(QUICKSAND_URL, path)
        except Exception:
            return None
    return path


def _ttf(size, bold=False):
    """The video label font at `size` px: Quicksand (weight 700 bold / 500 regular),
    falling back to Liberation Sans, then DejaVu Sans (shipped with matplotlib),
    then PIL's bitmap font."""
    from PIL import ImageFont
    qs = _quicksand_path()
    if qs:
        try:
            f = ImageFont.truetype(qs, size=int(size)); f.set_variation_by_axes([700 if bold else 500]); return f
        except Exception:
            pass
    try:
        from matplotlib.font_manager import findfont, FontProperties
        fp = FontProperties(family=["Liberation Sans", "DejaVu Sans"], weight="bold" if bold else "normal")
        return ImageFont.truetype(findfont(fp), size=int(size))
    except Exception:
        return ImageFont.load_default()


def _tag_u8(frame_u8, text, color=(255, 255, 255), size=None, margin=8, bottom=False):
    """A salient label: bold text on a translucent dark pill in the tile's top-left
    corner (`bottom=True`: bottom-left). `size` defaults to ~6.5% of the tile height
    (17 px on a 256 px tile) and shrinks (to 12 px) until the pill fits the tile."""
    from PIL import Image, ImageDraw
    h, tw = frame_u8.shape[:2]; size = size or max(13, round(h * 0.065))
    im = Image.fromarray(frame_u8).convert("RGBA"); ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    while True:
        font = _ttf(size, bold=True); px, py = round(size * 0.55), round(size * 0.3)
        x0, y0, x1, y1 = d.textbbox((0, 0), text, font=font); w, hh = (x1 - x0) + 2 * px, (y1 - y0) + 2 * py
        if margin + w <= tw - margin or size <= 12:
            break
        size -= 1
    my = h - margin - hh if bottom else margin
    d.rounded_rectangle([margin, my, margin + w, my + hh], radius=round(hh * 0.3), fill=(0, 0, 0, 165))
    d.text((margin + px - x0, my + py - y0), text, fill=(*color, 255), font=font)
    return np.asarray(Image.alpha_composite(im, ov).convert("RGB"))


def _phase_card_u8(frame_u8, title, sub, title_col=(255, 210, 120), dim=0.35):
    """`frame_u8` dimmed to `dim`, with `title` / `sub` centred on it — the one-shot
    card shown when the video crosses into the evaluation phase."""
    from PIL import Image, ImageDraw
    im = Image.fromarray((frame_u8.astype(np.float32) * dim).astype(np.uint8)); d = ImageDraw.Draw(im); W, H = im.size
    f1, f2 = _ttf(round(W * 0.043), bold=True), _ttf(round(W * 0.025))
    for txt, font, dy, col in ((title, f1, -round(W * 0.033), title_col), (sub, f2, round(W * 0.023), (255, 255, 255))):
        x0, y0, x1, y1 = d.textbbox((0, 0), txt, font=font)
        d.text(((W - (x1 - x0)) // 2 - x0, H // 2 + dy - (y1 - y0) // 2 - y0), txt, fill=col, font=font)
    return np.asarray(im)


def _err_tint_u8(tile_u8, pred_t, gt_t, thresh=0.1, patch=16, alpha=0.2):
    """Red ERROR PANES on an arm tile: non-overlapping `patch` x `patch` squares (native
    256-px units, scaled with the tile) tinted red where the mean |pred - GT| over the
    square exceeds the absolute `thresh` (images in [0, 1]) — ss_rollout's overlay."""
    diff = (pred_t - gt_t).abs().mean(0).cpu().numpy(); h, w = diff.shape; P = max(1, round(patch * h / 256))
    gh, gw = h // P, w // P
    grid = diff[:gh * P, :gw * P].reshape(gh, P, gw, P).mean((1, 3)) > thresh
    mask = np.zeros(tile_u8.shape[:2], bool); mask[:gh * P, :gw * P] = np.kron(grid, np.ones((P, P), bool))
    out = tile_u8.copy(); out[mask] = (out[mask] * (1 - alpha) + np.array([255, 0, 0]) * alpha).round().astype(np.uint8)
    return out


def _round_u8(frame_u8, radius, bg):
    """The tile with its corners rounded onto `bg` (antialiased: the mask is drawn at
    4x and downsampled). Draw boxes / pills BEFORE rounding so they are not clipped."""
    from PIL import Image, ImageDraw
    h, w = frame_u8.shape[:2]
    m = Image.new("L", (w * 4, h * 4), 0); ImageDraw.Draw(m).rounded_rectangle([0, 0, w * 4 - 1, h * 4 - 1], radius=radius * 4, fill=255)
    m = np.asarray(m.resize((w, h), Image.LANCZOS)).astype(np.float32)[..., None] / 255
    return (frame_u8 * m + np.array(bg, np.float32) * (1 - m)).round().astype(np.uint8)


def _pill_u8(frame_u8, text, color, glyph=None, size=None, margin=None):
    """Bottom-left pill on a tile: translucent black fill, an optional check / cross
    glyph in `color`, coloured bold text. Sizes scale with the tile (17 px text on 256)."""
    from PIL import Image, ImageDraw
    h = frame_u8.shape[0]; size = size or max(12, round(h * 0.066)); margin = margin or round(h * 0.04)
    im = Image.fromarray(frame_u8).convert("RGBA"); ov = Image.new("RGBA", im.size, (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    f = _ttf(size, bold=True); x0, y0, x1, y1 = d.textbbox((0, 0), text, font=f); ind = round(h * 0.1) if glyph else 0
    px, py = round(size * 0.65), round(size * 0.4); pw, ph = x1 - x0 + 2 * px + ind, y1 - y0 + 2 * py; my = h - margin - ph
    d.rounded_rectangle([margin, my, margin + pw, my + ph], radius=ph // 2, fill=(0, 0, 0, 170))
    if glyph:
        cx, cy, r = margin + round(h * 0.066), my + ph // 2, round(h * 0.031); g = max(2, round(r * 0.5))
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(*color, 255)); dark = (20, 20, 20, 255)
        if glyph == "check":
            d.line([(cx - g, cy), (cx - g // 3, cy + g), (cx + g, cy - g)], fill=dark, width=2)
        else:
            d.line([(cx - g, cy - g), (cx + g, cy + g)], fill=dark, width=2); d.line([(cx + g, cy - g), (cx - g, cy + g)], fill=dark, width=2)
    d.text((margin + px + ind - x0, my + py - y0), text, fill=(*color, 255), font=f)
    return np.asarray(Image.alpha_composite(im, ov).convert("RGB"))


def _box_u8(frame_u8, box_tok, color=(255, 255, 255), width=None, grid=32, text=None, size=None):
    """Outline `box_tok` = (y0, y1, x0, x1) in latent-grid tokens on the tile (scaled
    to its pixel size); `text` = a filled pill in the same colour sitting on the
    box's top edge (inside it when the box touches the tile's top)."""
    from PIL import Image, ImageDraw
    h, w = frame_u8.shape[:2]; sy, sx = h / grid, w / grid
    y0, y1, x0, x1 = box_tok; X0, Y0, X1, Y1 = round(x0 * sx), round(y0 * sy), round(x1 * sx) - 1, round(y1 * sy) - 1
    im = Image.fromarray(frame_u8); d = ImageDraw.Draw(im); lw = width or max(2, round(h / 96))
    d.rectangle([X0, Y0, X1, Y1], outline=color, width=lw)
    if text:
        size = size or max(12, round(h * 0.055))                                  # 14 px on a 256 px tile, scales with it
        font = _ttf(size, bold=True); tx0, ty0, tx1, ty1 = d.textbbox((0, 0), text, font=font); px, py = round(size * 0.45), round(size * 0.25)
        pw, ph = (tx1 - tx0) + 2 * px, (ty1 - ty0) + 2 * py
        ly = Y0 - ph if Y0 - ph >= 0 else Y0 + lw; lx = min(X0, w - pw)
        d.rounded_rectangle([lx, ly, lx + pw, ly + ph], radius=round(ph * 0.25), fill=color)
        d.text((lx + px - tx0, ly + py - ty0), text, fill=(20, 20, 20), font=font)
    return np.asarray(im)


def _write_mp4(path, frames_u8, fps, audio=None):
    """H.264 mp4; `fps` may be a Fraction (an exact step rate); `audio` = ((channels,
    samples) int16, sample_rate), one contiguous track already cut to the frames,
    muxed as AAC."""
    import av
    from fractions import Fraction
    container = av.open(path, "w")
    stream = container.add_stream("h264", rate=Fraction(fps))               # every stream is added BEFORE the first mux
    stream.height, stream.width = frames_u8[0].shape[:2]
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "20"}
    if audio is not None:
        pcm, sr = audio; layout = "stereo" if pcm.shape[0] == 2 else "mono"
        astream = container.add_stream("aac", rate=int(sr), layout=layout)
        resampler = av.AudioResampler(format="fltp", layout=layout, rate=int(sr))
    for f in frames_u8:
        for pkt in stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="rgb24")):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    if audio is not None:
        for i in range(0, pcm.shape[1], 1024):
            af = av.AudioFrame.from_ndarray(np.ascontiguousarray(pcm[:, i:i + 1024]), format="s16p", layout=layout)
            af.sample_rate, af.pts = int(sr), i
            for rf in resampler.resample(af):
                for pkt in astream.encode(rf):
                    container.mux(pkt)
        for rf in resampler.resample(None):
            for pkt in astream.encode(rf):
                container.mux(pkt)
        for pkt in astream.encode():
            container.mux(pkt)
    container.close()


# ── inspection ────────────────────────────────────────────────────────────────
def tint_mask(img_u8, mask, color=(1.0, 0.0, 0.0), alpha=0.4):
    """Blend `color` into `img_u8` (H, W, 3) where `mask` (h, w bool, e.g. the 32x32
    latent-grid silhouette rows) is set; the mask is nearest-upsampled to the image."""
    img = img_u8.astype(np.float32) / 255.0
    m = np.asarray(mask, bool)
    fy, fx = img.shape[0] // m.shape[0], img.shape[1] // m.shape[1]
    m = np.kron(m, np.ones((fy, fx), bool))[:img.shape[0], :img.shape[1]]
    w = alpha * m[..., None].astype(np.float32)
    out = img * (1 - w) + np.asarray(color, np.float32)[None, None] * w
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def inspect_trajectory(ep: dict, spec, decode: Decoder, n_show: int = 8,
                       tint_masks=None, tint_color=(1.0, 0.0, 0.0), tint_alpha=0.4):
    """Top-down map (context path, GT eval leg, rollout window, interaction
    sites: outbound = hollow squares, eval leg = green stars) plus the current
    frame and evenly spaced GT targets, tagged with the interactions firing in
    their (prev target, target] window. No generation yet.

    `tint_masks`: optional per-EPISODE-frame masks, `(T, h, w)` bool indexed by the
    episode frame index (e.g. the dyn corpus's second-agent silhouette rows,
    `ep["_agent_mask"]` from `dyn_rollout.attach`); where set, the current and GT
    frames are tinted with `tint_color` at `tint_alpha`."""
    import matplotlib.pyplot as plt
    targets, eff = ep["targets"], ep["eff_len"]
    n_show = min(n_show, len(targets))
    show = np.linspace(0, len(targets) - 1, n_show).round().astype(int)
    imgs = decode(torch.cat([ep["ctx_lat"][-1:], ep["goal_lat"][[targets[s] for s in show]]]))
    fig = plt.figure(figsize=(16, 6.5))
    gs = fig.add_gridspec(2, 7, wspace=0.05, hspace=0.15)
    ax = fig.add_subplot(gs[:, :2])
    cp, gp = ep["ctx_pose_g"], ep["goal_pose_g"]
    valid_xy, traj_xy, win_xy = cp[ep["ctx_mask"]][:, :2], gp[:eff][:, :2], gp[targets][:, :2]
    ax.plot(valid_xy[:, 0], valid_xy[:, 1], ":", color="dimgray", lw=1.2, label="context path")
    ax.plot(traj_xy[:, 0], traj_xy[:, 1], "-", color="lightcoral", lw=1.2, label="GT eval leg")
    ax.plot(win_xy[:, 0], win_xy[:, 1], "-", color="tab:red", lw=2.0, label="rollout window")
    ctx_rows = np.flatnonzero(ep["ctx_fire"].sum(1) > 0)
    eval_rows = np.flatnonzero(ep["goal_fire"][:eff].sum(1) > 0)
    if len(ctx_rows):
        ax.scatter(*cp[ctx_rows][:, :2].T, marker="s", s=70, facecolors="none", edgecolors="tab:orange", lw=1.6, zorder=4, label="interaction (outbound)")
    if len(eval_rows):
        ax.scatter(*gp[eval_rows][:, :2].T, marker="*", s=150, color="tab:green", edgecolors="black", zorder=5, label="interaction (eval leg)")
    ax.plot(*cp[-1, :2], marker="*", ms=13, color="black", label="current")
    L = 0.06 * max(np.ptp(traj_xy[:, 0]), np.ptp(traj_xy[:, 1]), 1.0)
    ax.arrow(*cp[-1, :2], L * np.cos(cp[-1, 4]), L * np.sin(cp[-1, 4]), head_width=0.25 * L, color="black")
    all_xy = np.concatenate([valid_xy, traj_xy, cp[-1:, :2]]); lo, hi = all_xy.min(0), all_xy.max(0)
    c, half = (lo + hi) / 2, 0.55 * max(hi[0] - lo[0], hi[1] - lo[1], 2.0)
    ax.set_xlim(c[0] - half, c[0] + half); ax.set_ylim(c[1] - half, c[1] + half)
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="best"); ax.set_title("top-down", fontsize=12)
    def _frame(img_t, fi):
        u8 = tensor_to_image(img_t)
        if tint_masks is not None and 0 <= int(fi) < len(tint_masks) and np.any(tint_masks[int(fi)]):
            u8 = tint_mask(u8, tint_masks[int(fi)], tint_color, tint_alpha)
        return u8
    cells = [(r, col) for r in range(2) for col in range(2, 7)]
    axc = fig.add_subplot(gs[cells[0]]); axc.imshow(_frame(imgs[0], ep["ctx_idx"][-1])); axc.set_title(f"current (idx {ep['ctx_idx'][-1]})", fontsize=10); axc.axis("off")
    for k, s in enumerate(show):
        axc = fig.add_subplot(gs[cells[k + 1]]); axc.imshow(_frame(imgs[k + 1], ep["goal_idx"][targets[s]]))
        dt = (ep["goal_idx"][targets[s]] - ep["ctx_idx"][-1]) / spec.fps
        lo_r = 0 if s == 0 else targets[show[k - 1]] + 1 if k else 0
        fired = ep["goal_fire"][lo_r:targets[s] + 1].sum(0)
        tag = " ".join(f"{c}x{int(v)}" for c, v in zip(INTERACT_CODES, fired) if v > 0)
        axc.set_title(f"GT step {s + 1} (+{dt:.1f}s)" + (f"\n{tag}" if tag else ""), fontsize=9, color="tab:green" if tag else "black")
        axc.axis("off")
    plt.suptitle(f"{ep['video_id']} — trajectory inspection (no generation yet)", fontsize=13)
    plt.show()


def write_episode_video(ep: dict, spec, decode: Decoder, path: str, history_speed: int = 4, eval_speed: int = 1, width: int = 800, **kw):
    """The WHOLE clip, GROUND TRUTH ONLY, in the rollout video's style: one tight
    panel (no arm rows, no context slots) — "History / evaluation in N s" with the
    action captions and agent sightings, the transition card, then "Ground truth /
    evaluation" with the expected-state captions, boxes and reveal holds. The
    history plays at `history_speed` x real time and the eval leg at `eval_speed`
    x (integers; the video is encoded at their product times the frame rate).
    Every frame comes from the episode's `latents.npy` (not the sampled pool), at
    `ep["stride"]` spacing. Other `write_rollout_video` knobs pass through `**kw`."""
    lat = np.load(os.path.join(ep["ep_path"], "latents.npy"), mmap_mode="c")
    es, n, stride = int(ep["eval_start"]), int(lat.shape[0]), int(ep["stride"])
    ep2 = dict(ep)
    ep2.update(ctx_idx=np.arange(0, es), ctx_lat=torch.from_numpy(np.asarray(lat[:es], dtype=np.float32)), ctx_mask=np.ones(es, bool),
               goal_idx=np.arange(es, n), goal_lat=torch.from_numpy(np.asarray(lat[es:], dtype=np.float32)), eff_len=n - es,
               targets=np.arange(0, n - es, stride), ctx_fire=np.zeros((es, len(INTERACT_CODES))))
    K = len(ep2["targets"]); gt_img = decode(ep2["goal_lat"][ep2["targets"]])
    write_rollout_video(ep2, spec, decode, gt_img, {}, {}, "{}", "", np.zeros((K, len(INTERACT_CODES)), int), {}, {}, path,
                        include_interaction_phase=True, history_speed=history_speed, eval_speed=eval_speed, width=width, **kw)


# ── rollout ───────────────────────────────────────────────────────────────────
class SubsampledPool(torch.nn.Module):
    """Restricts the inner memory's retrieval pool to fixed PAST positions
    (worldmem cap); the current frame is always forwarded."""

    def __init__(self, memory, pos, n_base):
        super().__init__()
        self.memory, self.pos, self.n_base = memory, pos, n_base

    def forward(self, frames, poses, times, action, stride, context_mask=None, **kw):
        T = frames.shape[1]
        idx = torch.cat([self.pos[self.pos < T - 1], torch.arange(self.n_base, max(self.n_base, T - 1), dtype=torch.long),
                         torch.tensor([T - 1], dtype=torch.long)]).to(frames.device)
        cm = None if context_mask is None else context_mask[:, idx]
        for k in ("context_keys", "context_agent_state"):
            if kw.get(k) is not None:
                kw[k] = kw[k][:, idx]
        return self.memory(frames[:, idx], poses[:, idx], times[:, idx], action, stride, cm, **kw)


def _picks_recorder(last: dict):
    """Forward hook on the BASE memory: record WHICH frames it retrieved (times,
    seconds from the pool origin) and, where emitted, their per-pick scores,
    into `last["times"]` / `last["scores"]`. Set-aware memories return a K-set
    axis — collapse to set 0, the set the generator conditions on."""
    def hook(mod, args, kwargs, out):
        t = out[2][0, 0]
        last["times"] = (t[0] if t.dim() > 1 else t).detach().float().cpu().numpy()
        sc = out[4]
        if sc is None:
            last["scores"] = None
        else:
            v = sc[0, 0]
            last["scores"] = (v[0] if v.dim() > 1 else v).detach().float().cpu().numpy()
    return hook


def rollout(model, label, ep, spec, stats_np, device, sampler="ddim", seed=0, insert_generated=False,
            feed_interact=True, worldmem_pool_size=None, refresh_every=1, lookahead_steps=0):
    """AR-roll one model along the episode's targets. Each step generates the
    next frame via `sample_latent`; the generated latent REPLACES the single tail
    slot (history stays 100% real) unless `insert_generated`. `action_interact`
    is the (current, target] multi-hot, fed rather than generated: pose deltas
    cannot express "the container opens".

    RETRIEVAL CADENCE (`refresh_every`): the memory is queried on step 0 and
    every `refresh_every`-th step after; in between, the SAME retrieved frames
    (cached as pool slots) are re-used with only the current frame swapped —
    the core forwards a raw context untouched when no memory is attached, and
    poses/times are re-centred on the new current frame each step. 1 =
    re-retrieve every step (the battery protocol).

    LOOKAHEAD RETRIEVAL (`lookahead_steps` = k > 0): the memory is queried with
    the pose / stride / interaction label of target t+k (clamped to the
    trajectory end) instead of the next step's, while the GENERATOR keeps the
    true next-step action. Two reasons: (i) at stride 2 a next-step query is a
    0.2 s hop — a horizon the retriever barely saw in training (goals up to
    len_traj_pred = 3.2 s), and one where the pose-matched CURRENT view is
    always the best answer (the stale-copy regime); a lookahead query is back
    inside the trained horizon; (ii) during an approach the lookahead goal IS
    the inspection pose, so the fresh dwell frames enter the context before
    the turn instead of after it. Combine with `refresh_every = k` to hold the
    set for the k steps it was chosen for. Only the memory's query looks
    ahead; retrieval then goes through the same cached-context path as the
    cadence (memory detached, raw context forwarded).

    AGENT-STATE OBSERVATION (ai2thor_dyn, `ep["ctx_agent"]` present): every pool
    frame's `[vi, dx, dy, vi', dx', dy']` detection is passed as
    `context_agent_state`, to the generator's predictor / adaLN conditioning
    and (when the retriever's metadata scorer carries the agent-state cue) to
    the retriever — exactly as training and the in-training eval do. The
    dynamic-agent generators are trained with that conditioning on EVERY step,
    so omitting it runs the DiT off-distribution (flat or blown-up latents).
    Real pool frames carry their
    ground-truth detections; a GENERATED frame carries the PREDICTOR'S OWN
    output for it (`core.last_agent_pred`, binarised to the backend's format:
    vi = logit > 0, position zeroed when absent) with the primed half looked up
    at `agent_pair_s` earlier among real + generated frames — the loop stays
    self-consistent with what the model just painted, and no detector runs on
    generated pixels.

    Returns (preds, interacts, pick_times, pick_scores): the picks are recorded
    per step from the memory's top set (times in seconds from the pool origin,
    and per-pick scores where the memory emits them); on cached steps they are
    the cached set."""
    mem_orig = model.core.memory
    while hasattr(mem_orig, "memory"):                # shed wrappers left by a crashed run
        mem_orig = mem_orig.memory
    mem = mem_orig
    ctx_mask = ep["ctx_mask"]
    if worldmem_pool_size is not None and mem is not None and type(mem).__name__ == "WorldMemRetrieval":
        valid_pos = np.flatnonzero(ctx_mask)
        if worldmem_pool_size < len(valid_pos):
            sub = np.sort(np.random.default_rng(seed).choice(valid_pos, size=worldmem_pool_size, replace=False))
            mem = SubsampledPool(mem, torch.as_tensor(sub, dtype=torch.long), n_base=len(ctx_mask)).to(device)
            print(f"[{label}] worldmem pool capped: {len(valid_pos)} -> {worldmem_pool_size}")
    model.core.memory = mem
    last = {}
    handle = mem_orig.register_forward_hook(_picks_recorder(last), with_kwargs=True) if mem_orig is not None else None
    pool_lat, pool_pose_g = [f for f in ep["ctx_lat"]], ep["ctx_pose_g"].copy()
    pool_idx, pool_mask = list(ep["ctx_idx"]), list(ctx_mask)
    n_real, origin_idx, ev_curr = len(pool_lat), pool_idx[0], ep["ev_curr0"].copy()
    st_curr = None if ep.get("state_curr0") is None else ep["state_curr0"].copy()
    ag_real = ep.get("ctx_agent")                                   # (n_real, 6) or None
    pool_agent = None if ag_real is None else [np.asarray(r, dtype=np.float32).copy() for r in ag_real]
    pair_s = float(ep.get("agent_pair_s", 0.0))
    real_t = (np.asarray(pool_idx[:n_real]) - origin_idx) / spec.fps  # real pool frames' times (fixed)
    gen_t, gen_tri = [], []                                          # generated frames: time, predicted [vi, dx, dy]
    mem_takes_agent = bool(getattr(mem_orig, "uses_agent_state", False))
    def _gen_state(tri, t_gen):
        """6-vector for a generated frame: its predicted triple + the triple of
        the frame `pair_s` earlier (nearest in time among real + generated)."""
        cand_t = np.concatenate([real_t, np.asarray(gen_t, dtype=np.float64)])
        cand_s = [r[:3] for r in pool_agent[:n_real]] + gen_tri
        j = int(np.argmin(np.abs(cand_t - (t_gen - pair_s))))
        return np.concatenate([tri, cand_s[j]]).astype(np.float32)
    preds, interacts, pick_times, pick_scores = [], [], [], []
    cached = None                                     # pool slots of the retrieved memories (cadence > 1 / lookahead)
    la = max(0, int(lookahead_steps))
    mem_takes_interact = getattr(getattr(mem_orig, "query_vit", None), "interact_dim", 0) > 0
    pbar = tqdm(ep["targets"], desc=label, unit="frame", leave=False)
    try:
        for step, tpos in enumerate(pbar):
            tgt_idx, tgt_pose_g = int(ep["goal_idx"][tpos]), ep["goal_pose_g"][tpos]
            local = global_to_local_pose(np.concatenate([pool_pose_g, tgt_pose_g[None]]), pool_pose_g[-1])
            local[:, :3] /= spec.meters_per_waypoint
            ctx_poses = torch.as_tensor(local[:-1], dtype=torch.float32)
            action = local[-1].copy(); action[:3] = normalize_data(action[None, :3], stats_np)[0]
            ctx_times = torch.as_tensor((np.asarray(pool_idx) - origin_idx) / spec.fps, dtype=torch.float32)
            stride_nl = (tgt_idx - origin_idx) / spec.max_frame_offset
            ai = np.clip(ep["goal_ev"][tpos] - ev_curr, 0, 1).astype(np.float32); interacts.append(ai.copy())
            ai_t = torch.as_tensor(ai if feed_interact else np.zeros_like(ai))[None, None].to(device)
            # STATE basis for the generator (`_generator_interact` appends its
            # leading columns when interact_dim > 4): [door fraction AT the
            # target, 0, pickup/put onsets in (current, target]]. Absolute
            # state — repeated labels cannot re-toggle a door.
            as_t = None
            if st_curr is not None:
                a_state = ep["goal_state"][tpos].copy()
                a_state[2:] = np.clip(a_state[2:] - st_curr[2:], 0, 1)
                as_t = torch.as_tensor(a_state if feed_interact else np.zeros_like(a_state))[None, None].to(device)
            refresh = mem_orig is None or cached is None or refresh_every <= 1 or step % refresh_every == 0
            if la > 0 and mem_orig is not None and refresh:
                # Query the memory DIRECTLY for target t+k; the generator below
                # then runs on the cached set with the true next-step action.
                tq = ep["targets"][min(step + la, len(ep["targets"]) - 1)]
                q_local = global_to_local_pose(np.concatenate([pool_pose_g, ep["goal_pose_g"][tq][None]]), pool_pose_g[-1])
                q_local[:, :3] /= spec.meters_per_waypoint
                q_action = q_local[-1].copy(); q_action[:3] = normalize_data(q_action[None, :3], stats_np)[0]
                q_stride = (int(ep["goal_idx"][tq]) - origin_idx) / spec.max_frame_offset
                q_ai = np.clip(ep["goal_ev"][tq] - ev_curr, 0, 1).astype(np.float32)
                mem_kw = {}
                if mem_takes_interact:
                    mem_kw["action_interact"] = torch.as_tensor(q_ai if feed_interact else np.zeros_like(q_ai))[None, None].to(device)
                if mem_takes_agent and pool_agent is not None:
                    mem_kw["context_agent_state"] = torch.as_tensor(np.stack(pool_agent))[None].to(device)
                with torch.no_grad():
                    mem(torch.stack(pool_lat)[None].to(device), ctx_poses[None].to(device), ctx_times[None].to(device),
                        torch.as_tensor(q_action, dtype=torch.float32)[None, None].to(device),
                        torch.tensor([[q_stride]], dtype=torch.float32).to(device),
                        torch.as_tensor(np.asarray(pool_mask))[None].to(device),
                        num_memories=model.core.num_memories, **mem_kw)
                t_all = ctx_times.numpy()
                cached = np.array([int(np.argmin(np.abs(t_all - t))) for t in last["times"]])
            use_mem = refresh and not (la > 0 and mem_orig is not None)          # lookahead never retrieves inside sample_latent
            slots = np.arange(len(pool_lat)) if use_mem else np.concatenate([cached, [len(pool_lat) - 1]])
            model.core.memory = mem if use_mem else None          # cached step: raw context, no retrieval
            cst = None if pool_agent is None else torch.as_tensor(np.stack(pool_agent)[slots])[None].to(device)
            with torch.no_grad():
                pred = model.core.sample_latent(
                    torch.stack([pool_lat[j] for j in slots])[None].to(device), torch.as_tensor(action, dtype=torch.float32)[None].to(device),
                    torch.tensor([stride_nl], dtype=torch.float32).to(device), num_goals=1,
                    context_poses=ctx_poses[slots][None].to(device), context_times=ctx_times[slots][None].to(device),
                    context_mask=torch.as_tensor(np.asarray(pool_mask)[slots])[None].to(device),
                    action_interact=ai_t, action_state=as_t,
                    use_ddim=(sampler == "ddim"), context_agent_state=cst)[0].float().cpu()
            preds.append(pred)
            if pool_agent is not None:                            # the generated frame's state = the predictor's decision
                ap = getattr(model.core, "last_agent_pred", None)
                tri = np.zeros(3, np.float32)
                if ap is not None:
                    ap = ap[0].float().cpu().numpy(); vi = float(ap[0] > 0)
                    tri = np.array([vi, ap[1] * vi, ap[2] * vi], np.float32)
                t_gen = (tgt_idx - origin_idx) / spec.fps
                new_state = _gen_state(tri, t_gen); gen_t.append(t_gen); gen_tri.append(tri)
            if use_mem and mem_orig is not None:                      # cache the top set's pool slots by their times
                t_all = ctx_times.numpy()
                cached = np.array([int(np.argmin(np.abs(t_all - t))) for t in last["times"]])
            if mem_orig is not None:
                pick_times.append(last["times"].copy()); pick_scores.append(None if last["scores"] is None else last["scores"].copy())
            if insert_generated or len(pool_lat) == n_real:
                pool_lat.append(pred); pool_pose_g = np.concatenate([pool_pose_g, tgt_pose_g[None]])
                pool_idx.append(tgt_idx); pool_mask.append(True)
                if pool_agent is not None: pool_agent.append(new_state)
            else:
                pool_lat[-1], pool_pose_g[-1], pool_idx[-1] = pred, tgt_pose_g, tgt_idx
                if pool_agent is not None: pool_agent[-1] = new_state
            ev_curr = ep["goal_ev"][tpos].copy()
            if st_curr is not None:
                st_curr = ep["goal_state"][tpos].copy()
            pbar.set_postfix(target=tgt_idx, act="".join(c[0].upper() if v else "." for c, v in zip(INTERACT_CODES, ai)),
                             ret=("new+la" if la > 0 else "new") if refresh else "cached")
    finally:
        model.core.memory = mem_orig
        if handle is not None:
            handle.remove()
    return torch.stack(preds), np.stack(interacts), pick_times, pick_scores


def run_rollouts(models, ep, spec, stats_np, device, **kw):
    """Roll every arm on the SAME clip with the same noise seed. Returns
    (preds, pick_times, pick_scores, step_interact)."""
    preds, pick_times, pick_scores, step_interact = {}, {}, {}, None
    for label, model in models.items():
        torch.manual_seed(kw.get("seed", 0))
        preds[label], step_interact, pt, ps = rollout(model, label, ep, spec, stats_np, device, **kw)
        if pt:
            pick_times[label], pick_scores[label] = pt, ps
    print("steps with an interaction in (current, target]:", int((step_interact.sum(1) > 0).sum()), "/", len(ep["targets"]))
    return preds, pick_times, pick_scores, step_interact


# ── metrics ───────────────────────────────────────────────────────────────────
def _lpips(device):
    import lpips
    return lpips.LPIPS(net="alex", verbose=False).to(device).eval()


def _dreamsim(device):
    try:                       # the peft/torchao guard eval_rollout.py uses
        import peft.import_utils as piu, peft.tuners.lora.torchao as plt_
        piu.is_torchao_available = lambda *a, **k: False; plt_.is_torchao_available = lambda *a, **k: False
    except Exception:
        pass
    from dreamsim import dreamsim
    from far.constants import hub_cache_dir
    return dreamsim(pretrained=True, device=str(device), cache_dir=hub_cache_dir())[0]


def metric_curves(metric, preds, pred_img, gt_lat, gt_img, ep, device, revisits=None, revisit=None,
                  center_frac=0.5):
    """Per-step metric per arm -> (mvals, fmt, unit, better).

    Whole-frame metrics ("psnr" / "lpips" / "dreamsim") answer "how similar is
    the frame overall" and get outvoted by fidelity-everywhere-else when the
    error is one small object (the stale-pot case). The rest target that:
      "latent_top32"   — mean of the 32 worst latent positions (decode-free;
                         worst-REGION, not object-region).
      "station" / "station_latent" — LPIPS / latent MSE on a crop around the
                         REVISIT station projected via pose + FOV (camera = base
                         + 0.675 m; NaN when off-screen). Needs `revisit`.
      "center_lpips" / "center_dreamsim" — fixed CENTER crop (interactions are
                         gaze-centred, a revisiting agent faces the station)."""
    fmt, unit, better = "{:.3f}", "", "lower"
    if metric == "psnr":
        mvals = {l: (-10 * torch.log10(((img - gt_img) ** 2).flatten(1).mean(1))).numpy() for l, img in pred_img.items()}
        fmt, unit, better = "{:.1f}", "dB", "higher"
    elif metric in ("lpips", "center_lpips"):
        fn = _lpips(device); crop = _center_crop(gt_img, center_frac) if metric == "center_lpips" else (lambda x: x)
        with torch.no_grad():
            mvals = {l: torch.cat([fn(crop(img[i:i + 8]).to(device) * 2 - 1, crop(gt_img[i:i + 8]).to(device) * 2 - 1).flatten().cpu()
                                   for i in range(0, img.shape[0], 8)]).numpy() for l, img in pred_img.items()}
    elif metric in ("dreamsim", "center_dreamsim"):
        ds = _dreamsim(device)
        if metric == "center_dreamsim":
            cc = _center_crop(gt_img, center_frac)
            crop = lambda x: torch.nn.functional.interpolate(cc(x), size=(224, 224), mode="bilinear", align_corners=False)
        else:
            crop = lambda x: x
        with torch.no_grad():
            mvals = {l: torch.cat([ds(crop(gt_img[i:i + 8]).to(device), crop(img[i:i + 8]).to(device)).flatten().cpu()
                                   for i in range(0, img.shape[0], 8)]).numpy() for l, img in pred_img.items()}
    elif metric == "latent_top32":
        mvals = {l: ((lat - gt_lat) ** 2).sum(1).flatten(1).topk(32, dim=1).values.mean(1).numpy() for l, lat in preds.items()}
        fmt = "{:.2f}"
    elif metric in ("station", "station_latent"):
        assert revisit is not None, f"METRIC={metric!r} needs a REVISIT selection"
        st = station_xyz(revisits[revisit]["objectId"])
        boxes = [_station_box(st, ep["goal_pose_g"][t]) for t in ep["targets"]]
        print(f"station in view on {sum(b is not None for b in boxes)}/{len(boxes)} steps ({revisits[revisit]['objectId'].split('|')[0]})")
        mvals = {}
        if metric == "station":
            fn = _lpips(device)
            with torch.no_grad():
                for l, img in pred_img.items():
                    mvals[l] = np.array([np.nan if b is None else float(fn(img[k:k + 1, :, b[0]:b[1], b[2]:b[3]].to(device) * 2 - 1,
                                                                               gt_img[k:k + 1, :, b[0]:b[1], b[2]:b[3]].to(device) * 2 - 1).flatten()[0].cpu())
                                         for k, b in enumerate(boxes)])
        else:
            for l, lat in preds.items():
                vals = []
                for k, b in enumerate(boxes):
                    if b is None:
                        vals.append(np.nan); continue
                    y0, y1, x0, x1 = b
                    ly0, ly1, lx0, lx1 = y0 // 8, max(y0 // 8 + 1, -(-y1 // 8)), x0 // 8, max(x0 // 8 + 1, -(-x1 // 8))
                    vals.append(float(((lat[k] - gt_lat[k]) ** 2).sum(0)[ly0:ly1, lx0:lx1].mean()))
                mvals[l] = np.array(vals)
            fmt = "{:.2f}"
    else:
        raise ValueError(f"METRIC={metric!r}")
    return mvals, fmt, unit, better


def _center_crop(ref, frac):
    H, W = ref.shape[-2:]
    ch, cw = int(H * frac / 2), int(W * frac / 2)
    return lambda x: x[:, :, H // 2 - ch:H // 2 + ch, W // 2 - cw:W // 2 + cw]


def _station_box(st_xyz, pr, W=256, H=256, radius=0.35, cam_off=0.675, fov_half=np.deg2rad(45.0)):
    """Pixel box around the station projected into a frame at pose `pr`
    (camera = agent base + cam_off, Unity-derived camera-right axis); None when
    behind / off-screen."""
    f = (W / 2) / np.tan(fov_half)
    d = st_xyz - pr[:3]; d[2] -= cam_off
    yaw, pitch = pr[4], pr[3]
    xf = d[0] * np.cos(yaw) + d[1] * np.sin(yaw); yr = -d[0] * np.sin(yaw) + d[1] * np.cos(yaw)
    xp = xf * np.cos(pitch) + d[2] * np.sin(pitch); zp = -xf * np.sin(pitch) + d[2] * np.cos(pitch)
    if xp < 0.2:
        return None
    u, v = W / 2 + f * yr / xp, H / 2 - f * zp / xp
    if not (0 <= u < W and 0 <= v < H):
        return None
    hs = int(np.clip(f * radius / xp, 20, 110))
    return int(max(0, v - hs)), int(min(H, v + hs)), int(max(0, u - hs)), int(min(W, u + hs))


# ── figures ───────────────────────────────────────────────────────────────────
def plot_compare(ep, spec, cur_img, gt_img, pred_img, mvals, step_interact, fmt, unit, title, save_path, n_show=8):
    """GT row + one row per arm at evenly spaced steps; per-step metric stamped
    on each frame; interaction steps marked green in the GT row."""
    import matplotlib.pyplot as plt
    targets = ep["targets"]
    rows = [("GT", gt_img)] + list(pred_img.items())
    n_show = min(n_show, len(targets)); show = np.linspace(0, len(targets) - 1, n_show).round().astype(int)
    fig, axes = plt.subplots(len(rows), n_show + 1, figsize=(2.6 * (n_show + 1), 2.7 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, (name, imgs) in enumerate(rows):
        axes[r, 0].imshow(tensor_to_image(cur_img)); axes[r, 0].set_title("current" if r == 0 else "")
        axes[r, 0].set_ylabel(name, fontsize=12); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        for c, s in enumerate(show):
            ax = axes[r, c + 1]; ax.imshow(tensor_to_image(imgs[s])); ax.axis("off")
            tag = " ".join(k[0].upper() for k, v in zip(INTERACT_CODES, step_interact[s]) if v)
            if r == 0:
                dt = (ep["goal_idx"][targets[s]] - ep["ctx_idx"][-1]) / spec.fps
                ax.set_title(f"+{dt:.1f}s" + (f"  [{tag}]" if tag else ""), fontsize=10, color="tab:green" if tag else "black")
            else:
                sv = mvals[name][s]
                ax.set_title("–" if not np.isfinite(sv) else f"{fmt.format(sv)} {unit}".strip(), fontsize=10)
    plt.suptitle(title, fontsize=13)
    plt.tight_layout(); plt.savefig(save_path, dpi=110, bbox_inches="tight"); plt.show()


def plot_metric_curve(ep, spec, mvals, step_interact, metric, fmt, unit, better, save_path):
    """Per-step metric per arm vs seconds into the eval leg; green lines = an
    interaction fires in that step's window (the discriminating steps)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 3.2))
    tsec = (ep["goal_idx"][ep["targets"]] - ep["ctx_idx"][-1]) / spec.fps
    for label, v in mvals.items():
        ax.plot(tsec, v, marker="o", ms=3.5, lw=1.6, label=f"{label} (mean {fmt.format(np.nanmean(v))}{' ' + unit if unit else ''})")
    for s in np.flatnonzero(step_interact.sum(1) > 0):
        ax.axvline(tsec[s], color="tab:green", alpha=0.35, lw=1.2, zorder=0)
    ax.set_xlabel("seconds into the eval leg"); ax.set_ylabel(f"{metric.upper()}{' (' + unit + ')' if unit else ''}  ({better} = better)")
    ax.grid(alpha=0.3); ax.legend(fontsize=9, frameon=True, fancybox=False, edgecolor="black")
    plt.tight_layout(); plt.savefig(save_path, dpi=110, bbox_inches="tight"); plt.show()


def change_region_diagnostic(ep, spec, preds, pred_img, gt_lat, gt_img, pick_times, pick_scores, device,
                             max_frac=0.10, pad=2, save_path=None, mask_name="latents.change.npy"):
    """Does the model render what CHANGED, or copy the best-matching context?
    Restricts the error to the GT change region (`latents.change` sidecar: the
    latent tokens that differ from the SAME viewpoint earlier in the episode —
    the mask the mask-weighted loss trains on). Per arm: in-/out-mask latent
    MSE, LPIPS on the region's bbox crop, and the COPY signature — in-region
    distance from the prediction to the TOP-1 retrieved frame vs to the GT
    (ratio < 1 = closer to what memory handed it). Steps with no mask, or a mask
    above `max_frac` of the grid, are skipped (the loss's gate)."""
    import matplotlib.pyplot as plt
    chg_path = os.path.join(ep["ep_path"], mask_name)      # any packed 32x32 mask family (change / union / agent silhouette)
    if not os.path.isfile(chg_path):
        print(f"no mask sidecar at {chg_path} — build it (build_masks_ai2thor.py / build_agent_masks_dyn.py)"); return None
    packed = np.load(chg_path, mmap_mode="c")
    chg = np.unpackbits(np.asarray(packed), axis=1).reshape(len(packed), 32, 32).astype(bool)
    lat_all = np.load(os.path.join(ep["ep_path"], "latents.npy"), mmap_mode="c")
    origin, fn = int(ep["ctx_idx"][0]), _lpips(device)

    def top1_frame(label, k):
        t, sc = pick_times[label][k], pick_scores.get(label, [None] * len(pick_times[label]))[k]
        j = int(np.argmax(sc)) if sc is not None and len(sc) == len(t) else 0
        return origin + int(round(float(t[j]) * spec.fps))

    @torch.no_grad()
    def crop_lpips(a, b, box):
        y0, y1, x0, x1 = [v * 8 for v in box]
        return float(fn(a[None, :, y0:y1, x0:x1].to(device) * 2 - 1, b[None, :, y0:y1, x0:x1].to(device) * 2 - 1).flatten()[0])

    rows, used, skipped = {l: [] for l in preds}, [], 0
    for k, tpos in enumerate(ep["targets"]):
        f = int(ep["goal_idx"][tpos])
        m = chg[f] if f < len(chg) else None
        if m is None or m.mean() == 0 or m.mean() >= max_frac:
            skipped += 1; continue
        used.append(k); ys, xs = np.nonzero(m)

        def _min_side(lo, hi, n=32, min_len=8):
            # LPIPS's AlexNet collapses crops narrower than ~64 px mid-stack;
            # grow tiny boxes (a mug is 2-3 latent cells) symmetrically to at
            # least min_len cells (x8 px), sliding to stay inside the grid.
            if hi - lo < min_len:
                lo = max(0, lo - (min_len - (hi - lo)) // 2)
                hi = min(n, lo + min_len)
                lo = max(0, hi - min_len)
            return lo, hi

        _y = _min_side(max(ys.min() - pad, 0), min(ys.max() + 1 + pad, 32))
        _x = _min_side(max(xs.min() - pad, 0), min(xs.max() + 1 + pad, 32))
        box = (_y[0], _y[1], _x[0], _x[1])
        mt, gt_l = torch.as_tensor(m), gt_lat[k].float()
        for label in preds:
            pr_l = preds[label][k].float()
            ctx_l = torch.from_numpy(np.asarray(lat_all[top1_frame(label, k)])).float()
            e_gt, e_ctx = ((pr_l - gt_l) ** 2).mean(0), ((pr_l - ctx_l) ** 2).mean(0)
            rows[label].append(dict(step=k, frac=float(m.mean()), in_mse=float(e_gt[mt].mean()), out_mse=float(e_gt[~mt].mean()),
                                    d_ctx=float(e_ctx[mt].mean()), lp=crop_lpips(pred_img[label][k], gt_img[k], box)))
    if not used:
        print(f"no usable change region in this rollout window ({skipped}/{len(ep['targets'])} steps skipped)"); return None
    print(f"change-region diagnostic over {len(used)}/{len(ep['targets'])} steps ({skipped} skipped: no mask or > {max_frac:.0%} of the grid) | "
          f"region = {np.mean([r['frac'] for r in rows[list(preds)[0]]]):.1%} of latent tokens\n")
    print(f"  {'arm':22s} {'in-mask MSE':>12s} {'out-mask':>10s} {'in/out':>7s} {'chg LPIPS':>10s} {'copy ratio':>11s} {'copy%':>7s}")
    for label, rr in rows.items():
        a = {k: np.array([r[k] for r in rr]) for k in rr[0]}; ratio = a["d_ctx"] / np.maximum(a["in_mse"], 1e-9)
        print(f"  {label:22s} {a['in_mse'].mean():12.4f} {a['out_mse'].mean():10.4f} {a['in_mse'].mean() / max(a['out_mse'].mean(), 1e-9):7.2f} "
              f"{a['lp'].mean():10.4f} {np.median(ratio):11.2f} {np.mean(ratio < 1):7.0%}")
    print("\n  copy ratio < 1 and copy% high => the prediction sits closer to the RETRIEVED frame than to the truth, inside the changed region.")
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.0))
    st = [r["step"] for r in rows[list(preds)[0]]]
    for label, rr in rows.items():
        a = {k: np.array([r[k] for r in rr]) for k in rr[0]}
        axes[0].plot(st, a["in_mse"], "-o", ms=3, label=label); axes[1].plot(st, a["lp"], "-o", ms=3, label=label)
        axes[2].plot(st, a["d_ctx"] / np.maximum(a["in_mse"], 1e-9), "-o", ms=3, label=label)
    axes[0].set_ylabel("in-mask latent MSE"); axes[1].set_ylabel("change-region LPIPS"); axes[2].set_ylabel("dist(pred, retrieved) / dist(pred, GT)")
    axes[2].axhline(1.0, color="black", ls="--", lw=1); axes[2].annotate("below = copies memory", (0.02, 0.06), xycoords="axes fraction", fontsize=8)
    for ax in axes:
        ax.set_xlabel("rollout step"); ax.grid(alpha=0.3, ls="--"); ax.legend(fontsize=7)
    plt.suptitle(f"{ep['video_id']} — change-region diagnostic", fontsize=12); plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.show()
    return rows


def write_rollout_video(ep, spec, decode, gt_img, pred_img, mvals, fmt, unit, step_interact, pick_times, pick_scores,
                        path, pick_top_n=4, width=1280, save_frame=None, include_interaction_phase=False,
                        change_box=True, change_max_frac=0.10, change_pad=2, change_min_tokens=4, change_max_box_frac=0.5,
                        action_label_seconds=2.0, state_label=True, phase_card_seconds=1.5, gutter=6, canvas=(24, 24, 24),
                        bottom_pad=64, outer_pad=None, verdict=True, verdict_margin=0.02, device=None, reveal_hold_seconds=2.0,
                        layout="panels", ours=None, history_speed=1, agent_px_min=100, agent_hold_seconds=2.0, agent_hold_at="peak",
                        preds=None, detector=None, eval_speed=1, verdict_lpips_max=0.30, verdict_sep_min=0.05, verdict_binary=True,
                        error_overlay=False, error_thresh=0.1, error_patch=16, error_alpha=0.2, audio=None):
    """GT vs each arm side by side, one video frame per rollout step, and — per
    arm — the context frames it actually retrieved (best-ranked first). Labels:
    "Ground truth" / the arm name on the frames, "Context #<rank> from <offset>
    sec" (the pick's time offset from the target) on the context thumbnails. Picks map back
    to REAL frames via t*fps + origin.

    `change_box`: outline the GT target frame's CHANGE REGIONS (the packed 32x32
    `latents.change.npy` mask: tokens that differ from the same viewpoint earlier
    in the episode) on the GT tile and every arm's tile — ONE BOX PER CONNECTED
    REGION (8-connected components of at least `change_min_tokens` tokens, so
    stray single-token specks from lighting / decoder noise do not blow one box
    up to the whole frame), each the raw token bbox plus `change_pad` tokens, no
    size floor; boxes that overlap after padding are merged. No boxes when the
    frame's mask is empty, when it covers >= `change_max_frac` of the grid (the
    loss gate's "too large to judge" — a closing door changes the whole view), or
    when the surviving boxes still cover >= `change_max_box_frac` of the grid.
    Boxes are drawn ONLY while the GT is AT a station during an eval-phase
    REVISIT: from the revisit's ARRIVAL frame (`arrival_frame` — the closest
    approach, as in the menu `print_revisits` prints) to the end of its segment.
    The segments partition the eval leg and each starts when the agent leaves
    the previous station, so the walk between stations gets no boxes.
    Silently off without the sidecar.

    `include_interaction_phase`: prepend the episode from its first frame up to
    the first rollout target, GT tile only (every `stride`-th frame, so it plays
    at the rollout's speed; arm / pick tiles stay blank). False (default) = the
    rollout steps only.

    `action_label_seconds`: every interaction event of the episode
    (`episode_events`: "Pick up apple", "Put apple on dining table", "Open
    cabinet", ...) is captioned on the GT tile — a green pill, bottom-left —
    from its frame for this many seconds (prelude and rollout steps alike; 0
    disables). `state_label`: at every eval-phase revisit — arrival frame to
    the end of its segment, the change-box window — the same slot instead
    carries the EXPECTED STATE in coral (`station_state`: "Apple should be
    gone" / "Apple should be here"; observed-only stations get NO caption — a
    drift there is left to the viewer); it takes precedence over an action
    caption when both are live (ctrl clips re-opening a station).

    `gt_img` / `pred_img` may be pre-upscaled (e.g. SwinIR, `far.analysis.upscale.sr_tensor`):
    the grid takes their tile size and everything decoded here — context picks,
    the history prelude, the verdicts' before-views — is bicubic-fitted to it, so
    only the rollout frames carry the super-resolution.

    `verdict`: at those same CHANGED-station revisits each arm is judged on
    whether it rendered the AFTER state. Per step, inside the change box (LPIPS
    on the crop, >= 64 px a side): `d_a` to the GT, `d_b` to the exact-view
    BEFORE frame (`before_counterpart`), `sep` between those two. The step is
    CORRECT when `d_a < verdict_lpips_max` (an absolute ceiling — a render far
    from everything fails it, which the relative test alone cannot see) AND
    (`d_b - d_a > verdict_margin` OR `sep < verdict_sep_min` — where the two
    states are perceptually identical there is nothing to be stale about, so the
    relative test is waived). A revisit's verdict is the MAJORITY over its steps
    (no per-step flicker): "Correct state" (green box + pill) or, with
    `verdict_binary`, "Wrong state" (red) for everything else; with
    `verdict_binary=False` a tie or a revisit with no usable step gets no label.
    Every per-step number is printed, for calibrating the ceiling. The GT
    tile's boxes, and every box without a verdict, are white — red only ever
    means a verdict. Steps with no box or no before-counterpart are skipped;
    the per-revisit margins are printed. `device` for LPIPS (default: the
    decoder's).

    PHASE: the GT tag reads "Ground truth · history" (white) before eval_start
    and "Ground truth · evaluation" (amber) from it on. With
    `include_interaction_phase`, the first frame at or after eval_start is
    preceded by a TRANSITION CARD — that frame dimmed under "Evaluation phase /
    models now predict from memory" — held for `phase_card_seconds` (0 disables).

    LAYOUT: tiles sit on a `canvas`-coloured grid with `gutter` px between
    them and around the edge (empty slots are canvas-coloured cells, not black
    holes); 0 = edge-to-edge as before. `bottom_pad` extra canvas rows under the
    grid keep a player's control bar off the bottom tiles. `outer_pad` (px, default
    None) replaces BOTH the gutter at the frame edge and `bottom_pad` with one
    canvas margin on all four sides — 0 puts the panel / grid edge-to-edge while
    the inner gutter (tile insets, row spacing) is untouched.

    `layout`: "panels" (default) wraps every row in a rounded panel with a NAME
    CARD on the left — the arm name (`ours` = the label drawn in amber with
    "(ours)"; the GT card reads "History / evaluation in N s" before eval_start
    and "Ground truth / evaluation" from it on) — and the panel's left edge in the
    verdict colour (grey without one). The action / expected-state caption
    (GT) and the verdict with its check / cross glyph (arms) are pills in the
    BOTTOM-LEFT corner of the prediction tiles; every tile's corners are rounded
    with the panels' radius. A "Prediction / Recalled memories" header sits
    under the GT panel. "grid" = the plain tile grid with
    the tags / captions / verdict pills drawn on the tiles.

    `history_speed` / `eval_speed`: playback speed of the history prelude and of
    the rollout steps, x real time, WITHOUT skipping frames. Any positive value
    (snapped to a fraction with denominator <= 10, e.g. 1.5 = 3/2, 0.5 = slow
    motion): with `history_speed = a/b`, `eval_speed = c/d` and `L = lcm(a, c)`
    the video is encoded at `L` x the step rate, prelude frames are repeated
    `L*b/a` times and rollout frames `L*d/c` times; holds and the transition
    card keep their durations in seconds.

    SECOND AGENT (ai2thor-dyn): when the episode carries `agent2.npz` (px on
    screen per frame; `ep["_px"]` from `dyn_rollout.attach`, else loaded here),
    every run of frames with px >= `agent_px_min` is a SIGHTING: the GT tile
    gets a light-blue box around the silhouette (`latents.agent.npy` /
    `ep["_agent_mask"]`, when present) and an "Agent observed" pill for the
    span, and the first frame of each sighting — history and evaluation alike —
    is frozen for `agent_hold_seconds` (0 disables) — at `agent_hold_at` = "peak"
    (the sighting's frame with the MOST agent pixels: the agent fully in view) or
    "onset" (its first frame). Inert without the sidecar.

    ai2thor-dyn REVEALS: the eval `look_A` / `look_B` legs re-view the corridor
    ends; the agent ended at `meta["final"]`, so the GT tile reads "Agent should
    be here" at that end (from the first GT frame that shows it) and "Agent
    shouldn't be here" at the other (after the turn's last glimpse of the final
    end), with the GT silhouette boxed. With `preds` ({arm: (steps, 4, 32, 32)
    generated latents}) each arm is judged by the agent-cell `detector`
    (`dyn_rollout.AgentCellDetector`, built from `models/agent_cell_detector.pt`
    when None): it PAINTS the agent when >= `PAINT_CELLS` cells fire; correct
    iff paints == should; majority over the reveal's steps -> "Correct state" /
    "Wrong state" (a tie: no label). The detector's cells are boxed on the arm's
    tile in the verdict colour. `reveal_hold_seconds` freezes these reveals too.

    `error_overlay`: red ERROR PANES on every arm's prediction tile — `error_patch`
    x `error_patch` squares (native 256-px units) tinted red at `error_alpha`
    where the mean |pred - GT| exceeds `error_thresh` (`_err_tint_u8`, the
    SoundSpaces video's overlay). Drawn under the boxes and pills.

    `audio`: True = mux the episode's `audio.wav` (SoundSpaces: binaural, sample-
    aligned with the frames), or a path to one. Each frame carries the sound of
    the `stride` frames it shows (the video is then encoded at the EXACT
    fps/stride rate); the transition card and every hold are silent. Off, with
    a note, when `history_speed` or `eval_speed` != 1 — the track would fall
    out of sync — or when the file is missing.

    `reveal_hold_seconds`: the MIDDLE step of every changed-station revisit
    (the state-caption window) is frozen for this long so the viewer can
    inspect the reveal (0 disables). `save_frame` still indexes rollout steps.

    `save_frame`: rollout-step index (int, negative OK) or list of them — each
    step's composed grid is ALSO written as `<path minus .mp4>_step<k>.png`,
    for pulling a still out of the video (prelude frames do not count)."""
    idx2lat = {int(ep["ctx_idx"][k]): ep["ctx_lat"][k] for k in np.flatnonzero(ep["ctx_mask"])}
    for j in range(ep["eff_len"]):
        idx2lat.setdefault(int(ep["goal_idx"][j]), ep["goal_lat"][j])
    t0, cache = int(ep["ctx_idx"][0]), {}

    tile_hw = tuple(gt_img.shape[-2:])                # GT / arm tiles may arrive pre-upscaled (SwinIR); everything decoded
                                                      # here (context picks, prelude, before-views) is fitted to that size
    def fit(img):
        """(N, 3, h, w) decoded frames -> the GT tile size (bicubic, antialiased), if they differ."""
        import torch.nn.functional as F
        return img if tuple(img.shape[-2:]) == tile_hw else F.interpolate(img, size=tile_hw, mode="bicubic", antialias=True).clamp(0, 1)

    def pick_img(fi):
        if fi not in cache:
            lat = idx2lat.get(fi); cache[fi] = None if lat is None else fit(decode(lat[None]))[0]
        return cache[fi]

    def picks_for(label, k):
        t = np.atleast_1d(np.asarray(pick_times[label][k], dtype=float)); fi = (np.round(t * spec.fps) + t0).astype(int)
        sc = pick_scores[label][k]
        sc = np.atleast_1d(np.asarray(sc, dtype=float)) if sc is not None and np.ndim(sc) and len(np.atleast_1d(sc)) == len(fi) else [None] * len(fi)
        out = list(zip(fi.tolist(), sc)); return out[:pick_top_n] if pick_top_n else out

    K = gt_img.shape[0]; n_pick = max(len(picks_for(l, 0)) for l in pick_times) if pick_times else 0
    ncols, blank = 1 + n_pick, np.full_like(_u8(gt_img[0]), canvas, dtype=np.uint8); frames = []
    es, stride = int(ep["eval_start"]), int(ep["stride"])
    chg = None
    if change_box:
        cp = os.path.join(ep["ep_path"], "latents.change.npy")
        if os.path.isfile(cp):
            packed = np.load(cp, mmap_mode="c"); chg = np.unpackbits(np.asarray(packed), axis=1).reshape(len(packed), 32, 32).astype(bool)
        else:
            print(f"change boxes off: no {cp} (build_masks_ai2thor.py --family change)")

    at_station = [(arrival_frame(ep, e), int(e["frames"][1])) for e in ep["meta"].get("inspections", []) if e.get("phase") == "eval"]
    events, ACTION_COL, PHASE_COL, STATE_COL = episode_events(ep) if action_label_seconds > 0 else [], (120, 255, 120), (255, 210, 120), (255, 120, 110)
    state_caps = [(a, b, txt) for (a, b), e in zip(at_station, [e for e in ep["meta"].get("inspections", []) if e.get("phase") == "eval"])
                  for txt in [station_state(ep, e["objectId"])] if txt != "Unchanged"] if state_label else []   # observed-only stations: no caption
    video_fps = max(1, round(spec.fps / ep["stride"]))
    from fractions import Fraction
    from math import lcm
    hs, se = (max(Fraction(float(x)).limit_denominator(10), Fraction(1, 10)) for x in (history_speed, eval_speed))
    L = lcm(hs.numerator, se.numerator)                                        # encoder rate = L x the step rate;
    enc_fps = video_fps * L; r_pre, r_step = L * hs.denominator // hs.numerator, L * se.denominator // se.numerator   # frame repeats
    n_card = round(phase_card_seconds * enc_fps) if include_interaction_phase else 0

    WHITE, GREY, BOX_COL = (255, 255, 255), (110, 110, 110), (255, 255, 255)
    VERDICT_COLS = {"Correct state": (120, 255, 120), "Wrong state": (255, 80, 80)}

    # second-agent sightings (ai2thor-dyn): px on screen per frame + silhouette masks
    AGENT_COL = (130, 200, 255)
    px = ep.get("_px"); a_mask = ep.get("_agent_mask")
    if px is None and os.path.isfile(os.path.join(ep["ep_path"], "agent2.npz")):
        px = np.load(os.path.join(ep["ep_path"], "agent2.npz"))["px"]
        mp = os.path.join(ep["ep_path"], "latents.agent.npy")
        a_mask = (np.unpackbits(np.asarray(np.load(mp, mmap_mode="c")), axis=1)[:, :1024].reshape(-1, 32, 32).astype(bool)
                  if os.path.isfile(mp) else None)
    a_vis = np.asarray(px) >= agent_px_min if px is not None else None
    a_onsets = []                                             # one hold frame per sighting: its onset, or its pixel-count peak
    if a_vis is not None:
        edges = np.flatnonzero(np.diff(np.concatenate([[0], a_vis.astype(int), [0]])))
        for on, off in zip(edges[::2], edges[1::2]):
            a_onsets.append(int(on) if agent_hold_at == "onset" else int(on + np.argmax(np.asarray(px)[on:off])))
    n_agent = round(agent_hold_seconds * enc_fps) if a_onsets else 0

    # ai2thor-dyn reveal windows (see the docstring): appended to the state captions.
    legs = {L["name"]: (int(L["start"]), int(L["end"])) for L in ep["meta"].get("legs", []) if "start" in L}
    final, dyn_wins = ep["meta"].get("final"), []
    if state_label and a_vis is not None and final in ("A", "B"):
        for X in ("A", "B"):
            if f"look_{X}" not in legs:
                continue
            a, b = legs[f"look_{X}"]; seen = np.flatnonzero(a_vis[a:b]) + a
            lo = (int(seen[0]) if len(seen) else a) if final == X else (int(seen[-1]) + 1 if len(seen) else a)
            if lo <= b - 1:
                dyn_wins.append((lo, b - 1, "Agent should be here" if final == X else "Agent shouldn't be here", final == X))
    n_tour = len(state_caps); state_caps = state_caps + [(a, b, txt) for a, b, txt, _ in dyn_wins]

    def agent_seen(fi):
        return a_vis is not None and 0 <= fi < len(a_vis) and bool(a_vis[fi])

    def agent_box(tile, fi):
        """The GT tile with the second agent's silhouette boxed while it is on screen."""
        if not agent_seen(fi) or a_mask is None or fi >= len(a_mask) or not a_mask[fi].any() or any(a <= fi <= b for a, b, _, _ in dyn_wins):
            return tile                                        # inside a reveal window the verdict boxes take over
        ys, xs = np.nonzero(a_mask[fi])
        return _box_u8(tile, (max(ys.min() - 1, 0), min(ys.max() + 2, 32), max(xs.min() - 1, 0), min(xs.max() + 2, 32)), AGENT_COL)

    def caption_for(fi):
        """(text, colour) of the GT caption at frame `fi`: the expected state while at a
        changed-station revisit, else the latest interaction event still live, else None."""
        state = [text for a, b, text in state_caps if a <= fi <= b]
        live = [text for f, text in events if f <= fi < f + action_label_seconds * spec.fps]
        return ((state[0], STATE_COL) if state else (live[-1], ACTION_COL) if live
                else ("Agent observed", AGENT_COL) if agent_seen(fi) else None)

    def gt_tile(tile, fi):
        """GRID layout: the GT tile tagged with its phase and captioned. PANELS: bare (the card carries both)."""
        if layout == "panels":
            return tile
        tile = _tag_u8(tile, "Ground truth · history") if fi < es else _tag_u8(tile, "Ground truth · evaluation", PHASE_COL)
        cap = caption_for(fi)
        return _tag_u8(tile, cap[0], cap[1], size=round(tile.shape[0] * 0.085), bottom=True) if cap else tile

    def with_card(frame, fi):
        """[card x n_card, frame] the first time `fi` reaches eval_start after history was shown, else [frame]."""
        if n_card and fi >= es and with_card.armed:
            with_card.armed = False
            return [_phase_card_u8(frame, "Evaluation phase", "models now predict from memory")] * n_card + [frame]
        with_card.armed = with_card.armed or fi < es
        return [frame]
    with_card.armed = False

    def in_revisit(fi):
        return any(a <= fi <= b for a, b in at_station)

    def change_box_for(fi):
        """-> [(y0, y1, x0, x1) in tokens, ...] — one per connected region; [] outside a
        station revisit, or when the mask is empty, gated (too large), or the boxes
        would cover most of the grid."""
        from scipy import ndimage
        m = chg[fi] if chg is not None and fi < len(chg) and in_revisit(fi) else None
        if m is None or not m.any() or m.mean() >= change_max_frac:
            return []
        lab, n = ndimage.label(m, structure=np.ones((3, 3), dtype=int))
        boxes = []
        for c in range(1, n + 1):
            ys, xs = np.nonzero(lab == c)
            if len(ys) < change_min_tokens:
                continue
            boxes.append([max(int(ys.min()) - change_pad, 0), min(int(ys.max()) + 1 + change_pad, 32),
                          max(int(xs.min()) - change_pad, 0), min(int(xs.max()) + 1 + change_pad, 32)])
        merged = True                                     # merge boxes that overlap after padding
        while merged and len(boxes) > 1:
            merged = False
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    a, b = boxes[i], boxes[j]
                    if a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]:
                        boxes[i] = [min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3])]
                        del boxes[j]; merged = True; break
                if merged: break
        cover = np.zeros((32, 32), bool)
        for y0, y1, x0, x1 in boxes:
            cover[y0:y1, x0:x1] = True
        return [] if not boxes or cover.mean() >= change_max_box_frac else [tuple(b) for b in boxes]

    def padc(a, top, bot, left, right):
        return np.stack([np.pad(a[..., c], ((top, bot), (left, right)), constant_values=canvas[c]) for c in range(3)], -1)

    def even(a):
        """Even frame dimensions for the encoder."""
        return padc(a, 0, a.shape[0] % 2, 0, 0)[:, :a.shape[1] - a.shape[1] % 2]

    def compose(rows):
        """GRID layout: tile the grid; with a gutter every tile and the whole grid are
        padded in the canvas colour, then `bottom_pad` canvas rows go under it."""
        half = gutter // 2
        if gutter:
            grid = np.concatenate([np.concatenate([padc(a, half, half, half, half) for a in r], axis=1) for r in rows], axis=0)
        else:
            grid = np.concatenate([np.concatenate(r, axis=1) for r in rows], axis=0)
        if outer_pad is not None:
            return even(padc(grid, outer_pad, outer_pad, outer_pad, outer_pad))
        return even(padc(grid, half, half + bottom_pad, half, half))

    # PANELS layout: sizes follow the tile so pre-upscaled frames keep the proportions.
    T, TW = tile_hw; PANEL_COL = (34, 34, 38); card_w, edge_w, rad = round(T * 0.98), max(4, round(T / 42)), round(T / 21)   # tile height / width
    blank_p = np.full((T, TW, 3), PANEL_COL, np.uint8)

    def card_u8(name, name_col, line2, line2_col):
        """The name card: the name and an optional second line ("(ours)" / the GT phase)."""
        from PIL import Image, ImageDraw
        im = Image.new("RGB", (card_w, T), PANEL_COL); d = ImageDraw.Draw(im); m = round(T * 0.086)
        y = round(T * 0.09); d.text((m, y), name, fill=name_col, font=_ttf(round(T * 0.086), bold=True))
        if line2:
            d.text((m, y + round(T * 0.11)), line2, fill=line2_col, font=_ttf(round(T * 0.0625)))
        return np.asarray(im)

    def arm_card(lab):
        return card_u8(lab, PHASE_COL if lab == ours else WHITE, "(ours)" if lab == ours else None, PHASE_COL)

    def gt_card(fi):
        """History phase: "History" + a countdown to eval_start; evaluation: "Ground truth" + "evaluation"."""
        if fi < es:
            return card_u8("History", WHITE, f"evaluation in {max(1, round((es - fi) / spec.fps))} s", (170, 170, 170))
        return card_u8("Ground truth", WHITE, "evaluation", (170, 170, 170))

    def gt_pill(tile, fi):
        """PANELS: the GT tile with its caption (expected state / live action) as a bottom-left pill."""
        cap = caption_for(fi)
        return _pill_u8(tile, cap[0], cap[1]) if cap else tile

    def panel_u8(tiles, edge_col, card):
        from PIL import Image, ImageDraw
        h, w = T + 2 * gutter, edge_w + card_w + (TW + gutter) * len(tiles) + gutter
        im = Image.new("RGB", (w, h), canvas); d = ImageDraw.Draw(im)
        d.rounded_rectangle([0, 0, w - 1, h - 1], radius=rad, fill=PANEL_COL)
        d.rounded_rectangle([0, 0, edge_w + rad, h - 1], radius=rad, fill=edge_col); d.rectangle([edge_w, 0, edge_w + rad, h - 1], fill=PANEL_COL)
        a = np.asarray(im).copy(); a[gutter:gutter + T, edge_w:edge_w + card_w] = card
        for n, tl in enumerate(tiles):
            x = edge_w + card_w + gutter + n * (TW + gutter); a[gutter:gutter + T, x:x + TW] = _round_u8(tl, rad, PANEL_COL)
        return a

    def compose_panels(panels):
        """`panels` = [(tiles, edge_colour, card), ...], GT first; the column header goes under the GT panel."""
        from PIL import Image, ImageDraw
        rows = [panel_u8(*pn) for pn in panels]; W = rows[0].shape[1]; hh = round(T * 0.13)
        hdr = Image.new("RGB", (W, hh), canvas); d = ImageDraw.Draw(hdr); f = _ttf(round(T * 0.06), bold=True)
        x = edge_w + card_w + gutter + round(T * 0.025)
        d.text((x, round(hh * 0.3)), "Prediction", fill=(190, 190, 190), font=f)
        d.text((x + TW + gutter, round(hh * 0.3)), "Recalled memories  (best-ranked first)", fill=(190, 190, 190), font=f)
        gap = np.full((gutter, W, 3), canvas, np.uint8); parts = [rows[0]] + ([gap, np.asarray(hdr)] if len(rows) > 1 else [])
        for r in rows[1:]:
            parts += [r, gap]
        if len(rows) > 1:
            parts.pop()                                       # the trailing gap is the outer margin, added below
        body = np.concatenate(parts, 0)
        if outer_pad is not None:
            return even(padc(body, outer_pad, outer_pad, outer_pad, outer_pad))
        return even(padc(body, gutter, (gutter if len(rows) > 1 else 0) + bottom_pad, gutter, gutter))

    def ctx_tiles(lab, k, tgt_fi, fill):
        """The arm's retrieved-context tiles for step `k`, tagged, padded with `fill` to n_pick."""
        out = []
        for n, (fi, _sc) in enumerate(picks_for(lab, k) if lab in pick_times else []):
            img = pick_img(fi)
            out.append(_tag_u8(fill.copy(), f"Context #{n + 1}  n/a") if img is None
                       else _tag_u8(_u8(img), f"Context #{n + 1} from {(fi - tgt_fi) / spec.fps:+.1f} sec"))
        return out + [fill] * (n_pick - len(out))

    n_pre, shown = 0, []                                      # per encoded frame: the episode frame it shows (None = card / hold)
    if include_interaction_phase:
        # GT-only prelude: the episode from its first frame to the first rollout target,
        # every `stride`-th frame (the rollout's own playback speed), each mapped to the
        # nearest frame we hold a latent for (pool + eval-leg frames).
        first_tgt = int(ep["goal_idx"][ep["targets"][0]])
        have = np.array(sorted(idx2lat)); want = np.arange(t0, first_tgt, stride)
        pos = np.clip(np.searchsorted(have, want), 0, len(have) - 1)
        pos = np.where((pos > 0) & (np.abs(have[np.maximum(pos - 1, 0)] - want) < np.abs(have[pos] - want)), pos - 1, pos)
        pre_fi = [int(f) for f, w in zip(have[pos], want) if abs(int(f) - int(w)) <= stride // 2 + 1]
        pre_fi = sorted(set(pre_fi))
        agent_hold_pre = {min(pre_fi, key=lambda f: abs(f - on)) for on in a_onsets if pre_fi and on < first_tgt}   # nearest prelude frame
        wait = [[_tag_u8(blank.copy(), lab)] + [blank] * n_pick for lab in pred_img]
        wait_p = [([blank_p] * ncols, GREY, arm_card(lab)) for lab in pred_img]
        for s0 in range(0, len(pre_fi), 16):
            chunk = pre_fi[s0:s0 + 16]
            imgs = fit(decode(torch.stack([idx2lat[f] for f in chunk])))
            for fi, im in zip(chunk, imgs):
                if layout == "panels":
                    frame = compose_panels([([gt_pill(agent_box(_u8(im), fi), fi)] + [blank_p] * n_pick, PHASE_COL, gt_card(fi))] + wait_p)
                else:
                    frame = compose([[gt_tile(agent_box(_u8(im), fi), fi)] + [blank] * n_pick] + wait)
                cards = with_card(frame, fi)[:-1]
                frames.extend(cards + [frame] * r_pre); shown.extend([None] * len(cards) + [fi] * r_pre)
                if fi in agent_hold_pre:
                    frames.extend([frame] * n_agent); shown.extend([None] * n_agent)   # freeze at the sighting's onset
        n_pre = len(frames)

    # STATE VERDICTS: one per (changed-station revisit, arm) from the before/after comparison.
    tgt_of = [int(ep["goal_idx"][ep["targets"][k]]) for k in range(K)]
    win_of = {k: w for w, (a, b, _) in enumerate(state_caps) for k in range(K) if a <= tgt_of[k] <= b}
    verdicts = {}
    gt_boxes, arm_boxes = {}, {}                              # dyn reveals: GT silhouette box / detector box per (step, arm)
    tour_k = {k: w for k, w in win_of.items() if w < n_tour}; dyn_k = {k: w for k, w in win_of.items() if w >= n_tour}
    for k, w in dyn_k.items():                                # GT silhouette box on the reveals (with or without arms)
        f = tgt_of[k]
        if a_mask is not None and f < len(a_mask) and a_mask[f].any():
            ys, xs = np.nonzero(a_mask[f]); gt_boxes[k] = (max(ys.min() - 1, 0), min(ys.max() + 2, 32), max(xs.min() - 1, 0), min(xs.max() + 2, 32))
    if verdict and tour_k and chg is not None and pred_img:
        device = device or getattr(decode, "device", "cpu"); fn = _lpips(device)
        lat_all = np.load(os.path.join(ep["ep_path"], "latents.npy"), mmap_mode="c")
        before = before_counterpart(ep, sorted({tgt_of[k] for k in tour_k}))

        @torch.no_grad()
        def crop(a, b, box):
            y0, y1, x0, x1 = [round(v * tile_hw[0] / 32) for v in box]      # tokens -> tile pixels
            return float(fn(a[None, :, y0:y1, x0:x1].to(device) * 2 - 1, b[None, :, y0:y1, x0:x1].to(device) * 2 - 1).flatten()[0])
        rows = {}
        for k, w in sorted(tour_k.items()):
            boxes, cp = change_box_for(tgt_of[k]), before.get(tgt_of[k])
            if not boxes or cp is None:
                continue
            box = _grow_box(min(b[0] for b in boxes), max(b[1] for b in boxes), min(b[2] for b in boxes), max(b[3] for b in boxes))
            before_img = fit(decode(torch.from_numpy(np.asarray(lat_all[cp], dtype=np.float32))[None]))[0]
            sep = crop(gt_img[k], before_img, box)                  # how distinguishable the two states are here
            for lab in pred_img:
                d_a, d_b = crop(pred_img[lab][k], gt_img[k], box), crop(pred_img[lab][k], before_img, box)
                ok = d_a < verdict_lpips_max and (d_b - d_a > verdict_margin or sep < verdict_sep_min)
                rows.setdefault((w, lab), []).append((bool(ok), d_a, d_b, sep))
        print(f"state verdicts — LPIPS in the change box: correct step = d_a < {verdict_lpips_max} AND "
              f"(d_b - d_a > {verdict_margin} OR sep < {verdict_sep_min}); majority over the revisit's steps"
              f"{' (binary)' if verdict_binary else ''}:")
        for (w, lab), rr in sorted(rows.items()):
            frac = float(np.mean([r[0] for r in rr]))
            v = "Correct state" if frac > 0.5 else ("Wrong state" if verdict_binary or frac < 0.5 else None)
            verdicts[(w, lab)] = v
            print(f"  {state_caps[w][2]:24s} {lab:18s} {sum(r[0] for r in rr)}/{len(rr)} steps  "
                  f"d_a {np.mean([r[1] for r in rr]):.3f}  d_b {np.mean([r[2] for r in rr]):.3f}  sep {np.mean([r[3] for r in rr]):.3f}  -> {v or '(no label)'}")
        if not rows:
            print("  (no step had both a change box and a before-counterpart)")
    if verdict and dyn_k and preds is not None and pred_img:
        if detector is None:
            from far.analysis.dyn_rollout import AgentCellDetector
            detector = AgentCellDetector(device or getattr(decode, "device", "cpu"))
        votes = {}
        for k, w in sorted(dyn_k.items()):
            should = dyn_wins[w - n_tour][3]
            for lab in pred_img:
                if lab not in preds:
                    continue
                pm = detector(preds[lab][k:k + 1])[0]; n_cells = int(pm.sum()); paints = n_cells >= detector.PAINT_CELLS
                votes.setdefault((w, lab), []).append((paints == should, n_cells))
                if paints:
                    ys, xs = np.nonzero(pm); arm_boxes[(k, lab)] = (max(ys.min() - 1, 0), min(ys.max() + 2, 32), max(xs.min() - 1, 0), min(xs.max() + 2, 32))
        print(f"agent verdicts — agent-cell detector on each arm's generation ('paints' = >= {detector.PAINT_CELLS} cells), majority over the reveal's steps:")
        for (w, lab), vv in sorted(votes.items()):
            frac = float(np.mean([c for c, _ in vv]))
            v = "Correct state" if frac > 0.5 else ("Wrong state" if verdict_binary or frac < 0.5 else None)
            verdicts[(w, lab)] = v
            print(f"  {state_caps[w][2]:26s} {lab:18s} correct {sum(c for c, _ in vv)}/{len(vv)} steps, cells {[n for _, n in vv]}  -> {v or '(no label)'}")
    elif verdict and dyn_k and pred_img:
        print("agent verdicts skipped: pass preds={arm: generated latents} (the rollout's `preds`) to judge the reveals")

    n_hold = round(reveal_hold_seconds * enc_fps)
    hold_k = {ks[len(ks) // 2] for w in set(win_of.values()) for ks in [sorted(k for k, ww in win_of.items() if ww == w)]} if n_hold else set()

    def boxed(tile, boxes, color=BOX_COL, text=None):
        for n, b in enumerate(sorted(boxes)):                 # the label rides the topmost box
            tile = _box_u8(tile, b, color, text=text if n == 0 else None)
        return tile

    agent_hold_k = {int(np.argmin([abs(t - on) for t in tgt_of])) for on in a_onsets if on >= tgt_of[0]}   # nearest rollout step
    n_box = n_steps_boxed = n_rev = 0; step_frames = []
    for k in range(K):
        tgt_fi, w = tgt_of[k], win_of.get(k)
        boxes = change_box_for(tgt_fi); n_box += len(boxes); n_steps_boxed += bool(boxes); n_rev += in_revisit(tgt_fi)
        if layout == "panels":
            gt_b = boxes or ([gt_boxes[k]] if k in gt_boxes else [])
            panels = [([gt_pill(agent_box(boxed(_u8(gt_img[k]), gt_b), tgt_fi), tgt_fi)] + [blank_p] * n_pick, PHASE_COL, gt_card(tgt_fi))]
            for lab in pred_img:
                v = verdicts.get((w, lab)) if w is not None else None
                pred = _u8(pred_img[lab][k])
                if error_overlay:
                    pred = _err_tint_u8(pred, pred_img[lab][k], gt_img[k], error_thresh, error_patch, error_alpha)
                pred = boxed(pred, boxes or ([arm_boxes[(k, lab)]] if (k, lab) in arm_boxes else []), VERDICT_COLS[v] if v else BOX_COL)
                if v:
                    pred = _pill_u8(pred, v, VERDICT_COLS[v], "check" if v == "Correct state" else "cross")
                panels.append(([pred] + ctx_tiles(lab, k, tgt_fi, blank_p), VERDICT_COLS[v] if v else GREY, arm_card(lab)))
            frame = compose_panels(panels)
        else:
            gt_b = boxes or ([gt_boxes[k]] if k in gt_boxes else [])
            rows = [[gt_tile(agent_box(boxed(_u8(gt_img[k]), gt_b), tgt_fi), tgt_fi)] + [blank] * n_pick]
            for lab in pred_img:
                v = verdicts.get((w, lab)) if w is not None else None
                ab = boxes or ([arm_boxes[(k, lab)]] if (k, lab) in arm_boxes else [])
                pred = _u8(pred_img[lab][k])
                if error_overlay:
                    pred = _err_tint_u8(pred, pred_img[lab][k], gt_img[k], error_thresh, error_patch, error_alpha)
                rows.append([_tag_u8(boxed(pred, ab, VERDICT_COLS[v] if v else BOX_COL, v), lab)] + ctx_tiles(lab, k, tgt_fi, blank))
            frame = compose(rows)
        cards = with_card(frame, tgt_fi)[:-1]; step_frames.append(frame)
        frames.extend(cards + [frame] * r_step); shown.extend([None] * len(cards) + [tgt_fi] * r_step)   # card (if any) + the step
        if k in agent_hold_k:
            frames.extend([frame] * n_agent); shown.extend([None] * n_agent)   # freeze at the sighting's onset
        if k in hold_k:
            frames.extend([frame] * n_hold); shown.extend([None] * n_hold)     # freeze mid-reveal
    track, enc_rate = None, enc_fps
    if audio:
        wav = os.path.join(ep["ep_path"], "audio.wav") if audio is True else str(audio)
        if hs != 1 or se != 1:
            print("audio off: history_speed / eval_speed != 1 (the track would fall out of sync)")
        elif not os.path.isfile(wav):
            print(f"audio off: no {wav}")
        else:
            import wave
            from fractions import Fraction
            w = wave.open(wav); sr, ch = w.getframerate(), w.getnchannels()
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, ch).T; w.close()
            enc_rate = Fraction(int(round(spec.fps)), stride)     # exact: every frame owns stride/fps s of sound
            spf = int(round(sr * stride / spec.fps)); segs = []
            for fi in shown:
                seg = np.zeros((ch, spf), np.int16)
                if fi is not None:
                    a = int(round(fi * sr / spec.fps)); chunk = pcm[:, a:a + spf]; seg[:, :chunk.shape[1]] = chunk
                segs.append(seg)
            track = (np.concatenate(segs, 1), sr)
            print(f"audio: {os.path.basename(wav)} muxed — {len(shown) - shown.count(None)} frames with sound, {shown.count(None)} silent (card / holds)")
    _write_mp4(path, frames, enc_rate, track)
    pre_tag = f"{n_pre} GT-only prelude frames + " if include_interaction_phase else ""
    box_tag = (f", {n_box} change boxes on {n_steps_boxed}/{n_rev} at-station steps of {K} (regions >= {change_min_tokens} tokens; "
               f"empty, >= {change_max_frac:.0%}-of-grid masks and >= {change_max_box_frac:.0%}-of-grid box covers skipped)") if chg is not None else ""
    hold_tag = f", {len(hold_k)} reveals frozen {reveal_hold_seconds:g} s" if hold_k else ""
    hold_tag += f", {len(a_onsets)} agent sightings frozen {agent_hold_seconds:g} s" if a_onsets and n_agent else ""
    pre_tag = pre_tag.replace("prelude frames", f"prelude frames @ {float(hs):g}x") if hs != 1 and pre_tag else pre_tag
    step_tag = f" @ {float(se):g}x" if se != 1 else ""
    print(f"rollout video: {path}  ({pre_tag}{K} steps @ {video_fps} fps{step_tag}{f' (encoded at {enc_fps} fps)' if enc_fps != video_fps else ''}, "
          f"{1 + len(pred_img)}x{ncols} {layout}{box_tag}{hold_tag})")
    _save_step_pngs(path, step_frames, ep, save_frame)
    _show_video(path, min(width, frames[0].shape[1]))


def _show_video(path, width):
    """Embed the written video inline when running in a notebook (Jupyter, VS Code, Colab);
    a no-op in scripts and terminal IPython, where `display` would only print the object."""
    try:
        from IPython import get_ipython
    except ImportError:
        return
    ip = get_ipython()
    if ip is None or getattr(ip, "kernel", None) is None:
        return
    from IPython.display import Video, display
    display(Video(path, embed=True, width=width))


def _save_step_pngs(path, frames, ep, save_frame):
    """`save_frame` (rollout-step index, negative OK, or a list) -> each step's
    composed frame written as `<path minus .mp4>_step<k>.png`."""
    if save_frame is None:
        return
    from PIL import Image
    for sf in ([save_frame] if np.isscalar(save_frame) else list(save_frame)):
        k = int(sf) % len(frames)                     # negative indices count from the end
        png = os.path.splitext(path)[0] + f"_step{k}.png"
        Image.fromarray(frames[k]).save(png)
        print(f"frame saved:   {png}  (step {k}, target idx {int(ep['goal_idx'][ep['targets'][k]])})")


def write_rollout_strip(ep, spec, gt_img, pred_img, mvals, fmt, unit, step_interact, path, width=1280, save_frame=None):
    """GT and each arm's generation side by side in ONE row per rollout step, no
    retrieved-context thumbnails — `write_rollout_video` minus the pick columns,
    for the upscaled demo video (only the frames shown pass through SwinIR, so no
    `SRDecoder` is needed). Labels / metric tags / `save_frame` as there."""
    K, frames = gt_img.shape[0], []
    for k in range(K):
        tgt_fi = int(ep["goal_idx"][ep["targets"][k]])
        fired = [c for c, v in zip(INTERACT_CODES, step_interact[k]) if v]
        ftag, fcol = ("  [" + ",".join(fired) + "]") if fired else "", (120, 255, 120) if fired else (255, 255, 255)
        tiles = [_label_u8(_u8(gt_img[k]), f"GT idx {tgt_fi}{ftag}", fcol)]
        for lab in pred_img:
            mv = mvals[lab][k]; mtag = "–" if not np.isfinite(mv) else f"{fmt.format(mv)}{unit}"
            tiles.append(_label_u8(_u8(pred_img[lab][k]), f"{lab} {mtag}{ftag}", fcol))
        frames.append(np.concatenate(tiles, axis=1))
    video_fps, ncols = max(1, round(spec.fps / ep["stride"])), 1 + len(pred_img)
    _write_mp4(path, frames, video_fps)
    print(f"rollout strip: {path}  ({len(frames)} steps @ {video_fps} fps, 1x{ncols} row, {frames[0].shape[0]}px tiles)")
    _save_step_pngs(path, frames, ep, save_frame)
    _show_video(path, min(width, 300 * ncols))


def plot_retrieval_geography(ep, spec, pick_times, pick_scores, step_interact, save_path=None):
    """WHERE in time each arm reads from, per step: x = target time, y = picked
    frame time (seconds from the pool origin; colour = score where available).
    Shaded = exploration phase; green = interaction step. On a discriminating
    step, a pick from the outbound visit to the station is a STALE view."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(pick_times), figsize=(5.6 * len(pick_times), 4.2), squeeze=False)
    t_curr, t_end = (ep["ctx_idx"][-1] - ep["ctx_idx"][0]) / spec.fps, (ep["eval_start"] - ep["ctx_idx"][0]) / spec.fps
    t_tgt = (ep["goal_idx"][ep["targets"]] - ep["ctx_idx"][0]) / spec.fps
    for ax, (label, rows) in zip(axes[0], pick_times.items()):
        ax.axhspan(0, t_end, color="tab:orange", alpha=0.10, zorder=0); ax.axhline(t_end, color="tab:orange", lw=1, ls=":", label="eval phase begins")
        for si, picks in enumerate(rows):
            p = np.atleast_1d(np.asarray(picks, dtype=float)); sc = pick_scores[label][si]
            if sc is not None and np.ndim(sc) and len(np.atleast_1d(sc)) == len(p):
                v = np.atleast_1d(np.asarray(sc, dtype=float)); rng = np.ptp(v)
                col = plt.cm.viridis((v - v.min()) / rng if rng > 0 else np.full_like(v, 0.5))
            else:
                col = "tab:blue"
            ax.scatter(np.full_like(p, t_tgt[si]), p, s=26, c=col, edgecolors="none", zorder=3)
        ax.plot(t_tgt, t_tgt, "--", color="gray", lw=1, label="= target time"); ax.axhline(t_curr, color="black", lw=1, label="current frame")
        for s in np.flatnonzero(step_interact.sum(1) > 0):
            ax.axvline(t_tgt[s], color="tab:green", alpha=0.35, lw=1.2, zorder=1)
        ax.set_xlabel("target time (s from pool origin)"); ax.set_ylabel("retrieved frame time (s)")
        ax.set_title(f"{label} — picks per step" + ("  (colour = score)" if pick_scores[label][0] is not None else ""))
        ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")
    plt.suptitle("retrieval geography (shaded = exploration phase; green = interaction step)", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.show()


def export_frames(ep, targets, decode, gt_img, pred_img, cur_img, save_dir, scope="episode",
                  ctx_frames=False, pick_times=None, spec=None):
    """Dump the rollout's frames as stand-alone PNGs under `<save_dir>/frames/`:
      current.png                    the current frame the rollout starts from
      gt/f<idx>.png                  GROUND-TRUTH frames, `scope` selects how many:
                                       "episode" — the sampled clip in temporal order, the same
                                                   span `write_episode_video` renders: the
                                                   outbound/exploration leg AND the eval leg.
                                                   NOTE it covers what the EPISODE DICT holds, so
                                                   under START_OFFSET > 0 with POOL_PHASE
                                                   "exploration" the eval frames walked BEFORE the
                                                   window are absent (they joined neither the pool
                                                   nor the sliced trajectory);
                                       "clip"    — every frame of `latents.npy`, window-independent
                                                   (use this when the rollout starts mid-trajectory
                                                   and the frames before it are still wanted);
                                       "eval"    — the eval leg only;
                                       None      — skip.
      gt_targets/step<k>_f<idx>.png  the GT at each rollout target (stride-aligned with the arms)
      <arm>/step<k>_f<idx>.png       each arm's generated frame per step
      <arm>_top1/step<k>_f<idx>_ctx<fi>.png
                                     the TOP-1 context frame the arm retrieved for that step
                                     (`fi` = its episode frame index; needs `pick_times` +
                                     `spec`; resolved to REAL pool frames via t*fps + origin,
                                     exactly as `write_rollout_video`; arms without picks skip)
      context/f<idx>.png             the retrievable pool frames only (ctx_frames)
    `<idx>` is the episode frame index (0-based, the same axis as meta.json
    events / eval_start), `<k>` the rollout step. "episode" decodes the full clip
    (~1-2 min, a few hundred PNGs) and its exploration half is what the pool is
    drawn from, so it covers the frames the memory could have retrieved.
    Returns the frames root."""
    from far.analysis.figure_io import save_frames, slug
    root = os.path.join(save_dir, "frames")
    goal_idx = np.asarray(ep["goal_idx"])
    tgt_idx = goal_idx[np.asarray(targets)]
    save_frames(root, {"current": cur_img})
    save_frames(os.path.join(root, "gt_targets"),
                {f"step{k:03d}_f{int(fi):06d}": gt_img[k] for k, fi in enumerate(tgt_idx)})
    for label, imgs in pred_img.items():
        save_frames(os.path.join(root, slug(label)),
                    {f"step{k:03d}_f{int(fi):06d}": imgs[k] for k, fi in enumerate(tgt_idx)})
    if scope == "clip":
        # Window-independent: decode the clip's own latents, so the dump does not
        # depend on what `apply_window` sliced away.
        lat = np.load(os.path.join(ep["ep_path"], "latents.npy"), mmap_mode="c")
        es, out = ep["eval_start"], {}
        for s0 in range(0, len(lat), 16):
            imgs = decode(torch.from_numpy(np.asarray(lat[s0:s0 + 16], dtype=np.float32)))
            out.update({f"f{s0 + j:06d}_{'eval' if s0 + j >= es else 'explore'}": im
                        for j, im in enumerate(imgs)})
        save_frames(os.path.join(root, "gt"), out)
    elif scope:
        assert scope in ("episode", "eval"), scope
        # Frames in temporal order. "episode" walks the exploration leg (the pool,
        # deduplicated and sorted by episode index) and then the eval leg, the same
        # playlist `write_episode_video` renders; "eval" is the eval leg alone.
        rows = []
        if scope == "episode":
            seen = set()
            for k in np.argsort(np.asarray(ep["ctx_idx"])):
                fi = int(ep["ctx_idx"][k])
                if fi not in seen:
                    seen.add(fi); rows.append(("ctx", int(k), fi))
            rows += [("goal", j, int(goal_idx[j])) for j in range(ep["eff_len"]) if int(goal_idx[j]) not in seen]
        else:
            rows = [("goal", j, int(goal_idx[j])) for j in range(len(goal_idx))]
        out, es = {}, ep["eval_start"]
        for s in range(0, len(rows), 16):                       # decode in batches, as the video writer does
            batch = rows[s:s + 16]
            for src, lat in (("ctx", ep["ctx_lat"]), ("goal", ep["goal_lat"])):
                sel = [r for (t, r, _f) in batch if t == src]
                if sel:
                    imgs = decode(lat[sel])
                    for (t, r, fi), im in zip([b for b in batch if b[0] == src], imgs):
                        out[f"f{fi:06d}_{'eval' if fi >= es else 'explore'}"] = im
        save_frames(os.path.join(root, "gt"), out)
    if ctx_frames:
        ctx_idx = np.asarray(ep["ctx_idx"]); ctx_mask = np.asarray(ep["ctx_mask"]).astype(bool)
        imgs = decode(ep["ctx_lat"][ctx_mask])
        save_frames(os.path.join(root, "context"),
                    {f"f{int(fi):06d}": imgs[k] for k, fi in enumerate(ctx_idx[ctx_mask])})
    n_top = 0
    if pick_times:
        assert spec is not None, "export_frames: `spec` is needed to resolve pick times to frames"
        idx2lat = {int(ep["ctx_idx"][k]): ep["ctx_lat"][k] for k in np.flatnonzero(np.asarray(ep["ctx_mask"]).astype(bool))}
        t0 = int(ep["ctx_idx"][0]); top1 = {}
        for label in pred_img:
            rows = pick_times.get(label) or []
            if not len(rows):
                continue
            fis = []
            for k in range(len(tgt_idx)):
                t = np.atleast_1d(np.asarray(rows[k], dtype=float))
                fis.append(int(round(float(t.reshape(-1)[0]) * spec.fps)) + t0)
            top1[label] = fis
        uniq = sorted({fi for fis in top1.values() for fi in fis if fi in idx2lat})
        dec = decode(torch.stack([idx2lat[fi] for fi in uniq])) if uniq else torch.zeros(0)
        img_of = {fi: dec[k] for k, fi in enumerate(uniq)}
        for label, fis in top1.items():
            kept = {f"step{k:03d}_f{int(tgt_idx[k]):06d}_ctx{int(fi):06d}": img_of[fi]
                    for k, fi in enumerate(fis) if fi in img_of}
            save_frames(os.path.join(root, slug(label) + "_top1"), kept); n_top += len(kept)
    print(f"frames -> {root}/  (arms: {', '.join(slug(l) for l in pred_img)}"
          + (f"; gt scope '{scope}'" if scope else "") + (f"; {n_top} top-1 context frames" if n_top else "") + ")")
    return root


def export_records(ep, spec, targets, pred_img, pick_times, save_dir, metrics=None,
                   step_interact=None, name="meta.json", **header):
    """Write `<save_dir>/<name>`: the per-step numbers that go with the PNGs
    `export_frames` just wrote. Shared by the three rollout notebooks and
    `scripts/teaser/build_ingredients.py`, so a clip run by hand and a clip run
    by the batch script produce interchangeable folders.

    Per step, per arm: the metric values passed in `metrics`
    (`{"lpips": mvals}` — one array per arm), the retrieved pick times (seconds
    from the pool origin, rank order, exactly what the memory hook recorded),
    and those picks resolved to EPISODE FRAME INDICES the way `export_frames`
    resolves them (`round(t * fps) + ctx_idx[0]`). A pick index names a file in
    `frames/gt/` — `f<idx>_explore.png` below `eval_start`, `f<idx>_eval.png`
    at or above it — so the frames behind the whole retrieved SET are on disk
    under `scope="episode"`, not just the top-1 copy.

    `header` is merged into the record's top level (run dirs, sampler, seed …).
    Returns the dict it wrote."""
    goal_idx = np.asarray(ep["goal_idx"])
    tgt_idx = goal_idx[np.asarray(targets)]
    t0, stride = int(ep["ctx_idx"][0]), int(ep.get("stride", 1) or 1)
    fi_of = lambda t: int(round(float(t) * spec.fps)) + t0
    metrics = metrics or {}

    def picks(label, k):
        rows = (pick_times or {}).get(label) or []
        if k >= len(rows) or rows[k] is None:
            return []
        t = np.asarray(rows[k], dtype=float)
        return [float(v) for v in np.atleast_1d(t[0] if t.ndim > 1 else t).reshape(-1)]

    steps = []
    for k, fi in enumerate(tgt_idx):
        arms = {}
        for label in pred_img:
            ps = picks(label, k)
            fis = [fi_of(t) for t in ps]
            row = {m: (None if label not in vals or k >= len(vals[label])
                       or not np.isfinite(vals[label][k]) else float(vals[label][k]))
                   for m, vals in metrics.items()}
            row.update(picks_s=[round(t, 3) for t in ps], picks_frames=fis,
                       top1_frame=fis[0] if fis else None,
                       top1_offset_s=None if not fis else round((fis[0] - int(fi)) / spec.fps, 2))
            arms[label] = row
        rec = dict(k=k, frame=int(fi), t_s=round(k * stride / spec.fps, 3), arms=arms)
        if step_interact is not None and k < len(step_interact):
            rec["interact"] = [int(v) for v in step_interact[k]]
        steps.append(rec)

    out = dict(video_id=ep.get("video_id"), ep_path=ep.get("ep_path"), fps=float(spec.fps),
               stride=stride, n_steps=len(steps), origin_frame=t0,
               current_frame=int(ep["ctx_idx"][-1]), eval_start=int(ep.get("eval_start", goal_idx[0])),
               arms=list(pred_img), metrics=list(metrics), steps=steps, **header)
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, name)

    def plain(o):                                   # numpy -> json
        if isinstance(o, dict):
            return {str(k): plain(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [plain(v) for v in o]
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return None if np.isnan(o) else float(o)
        if isinstance(o, np.ndarray):
            return plain(o.tolist())
        if isinstance(o, torch.Tensor):
            return plain(o.detach().cpu().numpy())
        return o

    with open(path, "w") as f:
        json.dump(plain(out), f, indent=1)
    print(f"records -> {path}  ({len(steps)} steps x {len(pred_img)} arm(s)"
          + (f"; metrics {', '.join(metrics)}" if metrics else "") + ")")
    return out
