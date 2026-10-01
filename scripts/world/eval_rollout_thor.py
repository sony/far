"""Per-frame rollout evaluation for AI2-THOR — DATA COLLECTION ONLY.

The THOR sibling of `eval_rollout.py`: same battery pattern (one compressed
.npz record per (arm, clip), existing records skipped -> resumable, clips
sharded across GPUs via shard=K num_shards=N), but everything episode- and
model-shaped is DELEGATED to `far.analysis.thor_rollout` — the audited
notebook machinery (`build_models` / `load_episode` / `apply_window` /
`rollout` with the `action_interact` + `action_state` feeds) — so battery
records replay bit-for-bit in `rollout_ai2thor.ipynb`.

Per (arm, clip): one contiguous replace-tail AR rollout along the EVAL leg
(start at eval_start + start_offset, `rollout_stride` GT frames per step,
exploration-phase pool), then per step:
  - frame-wise PSNR / LPIPS / DreamSim vs GT (same keys as the loopnav /
    soundspaces records — existing aggregation code reads these unchanged);
  - event covariates: the interact multi-hot + door fraction actually fed,
    signed seconds to the nearest eval-leg reveal + its station type;
  - IN-MASK block from the `.change.npy` sidecar at the target frame:
    mask_frac / mask_valid (nonzero AND <= mask_max_frac), in/out latent MSE,
    LPIPS on the mask bbox crop (64-px min-side floor), and the COPY
    SIGNATURE d_ctx_in (in-mask latent MSE to the TOP-1 retrieved frame) +
    that pick's time offset.
Records hold per-step scalars only (the eligibility scan holds no episode
tensors — see eval_rollout.py's host-RAM freeze note).

`use_ctrl_set=true` evaluates the seed-matched `_ctrl` siblings instead
(same CLIP_IDX ordering; reveals with UNCHANGED interiors) and suffixes the
record dirs `<arm>_ctrl` — run it into a SEPARATE output_dir, the eligibility
cache is protocol-specific. Presentation/aggregation is a separate analysis
step over the saved records.

    CUDA_VISIBLE_DEVICES=0 python scripts/world/eval_rollout_thor.py shard=0 num_shards=4 &
    CUDA_VISIBLE_DEVICES=1 python scripts/world/eval_rollout_thor.py shard=1 num_shards=4 &
"""

import contextlib
import io
import json
import os
import time

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from far.constants import CONFIG_ROOT, ROOT
from far.analysis import thor_rollout as R

STATION_CODE = {"Fridge": 1, "Microwave": 2, "Cabinet": 3, "Drawer": 4}


def _quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def _min_side(lo, hi, n=32, min_len=8):
    """Grow [lo, hi) to >= min_len cells, sliding to stay inside the grid —
    LPIPS's AlexNet collapses crops narrower than ~64 px mid-stack."""
    if hi - lo < min_len:
        lo = max(0, lo - (min_len - (hi - lo)) // 2)
        hi = min(n, lo + min_len)
        lo = max(0, hi - min_len)
    return lo, hi


def _top1_frame(pick_times, pick_scores, k, fps, origin):
    """change_region_diagnostic semantics: argmax score where scores align
    with the pick list, else the first (rank-0) pick."""
    t = np.atleast_1d(np.asarray(pick_times[k], dtype=float))
    sc = pick_scores[k] if pick_scores is not None else None
    j = (int(np.argmax(sc)) if sc is not None and np.ndim(sc)
         and len(np.atleast_1d(sc)) == len(t) else 0)
    return origin + int(round(float(t[j]) * fps))


@hydra.main(config_path=CONFIG_ROOT, config_name="eval_rollout_thor", version_base=None)
def main(cfg):
    os.chdir(ROOT)
    os.environ.setdefault("FAR_MODELS_ROOT", os.path.join(ROOT, "models"))
    os.environ.setdefault("FAR_RESULT_ROOT", os.path.join(ROOT, "results"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_root = cfg.output_dir
    os.makedirs(out_root, exist_ok=True)
    dn = str(cfg.dataset)
    ctrl = bool(cfg.use_ctrl_set)

    # ── Models + dataset (notebook machinery — audited conditioning path) ────
    runs = {label: (arm.run_dir, str(arm.ckpt)) for label, arm in cfg.arms.items()}
    models, data_cfg = R.build_models(runs, device)
    if ctrl:
        R.use_ctrl_index(data_cfg, dn)
    dataset, spec, stats_np, trained_pool = R.build_dataset(data_cfg, dn, int(cfg.context_pool_size))
    R.install_eval_knobs(models, data_cfg, str(cfg.sampler), int(cfg.sampling_steps),
                         None if cfg.pnp_chunk_size is None else int(cfg.pnp_chunk_size),
                         None if cfg.context_size is None else int(cfg.context_size))
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
        return torch.cat(out) * 0.5 + 0.5                    # (N, 3, H, W), [0, 1]

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

    @torch.no_grad()
    def crop_lpips(pred_lat, gt_lat, box):
        y0, y1, x0, x1 = [v * 8 for v in box]
        p = detok(pred_lat[None])[..., y0:y1, x0:x1]
        g = detok(gt_lat[None])[..., y0:y1, x0:x1]
        return float(lpips_fn(p * 2 - 1, g * 2 - 1).flatten()[0])

    # ── Protocol + eligibility (shard 0 scans, others wait) ──────────────────
    protocol = OmegaConf.to_container(cfg, resolve=True)
    protocol.pop("arms")
    with open(os.path.join(out_root, "protocol.json"), "w") as f:
        json.dump(protocol, f, indent=2, default=str)

    elig_params = dict(dataset=dn, use_ctrl_set=ctrl,
                       num_clips=None if cfg.num_clips is None else int(cfg.num_clips),
                       start_offset=int(cfg.start_offset), rollout_stride=int(cfg.rollout_stride),
                       min_steps=max(1, int(cfg.min_steps)),
                       max_steps=None if cfg.max_steps is None else int(cfg.max_steps),
                       context_pool_size=int(cfg.context_pool_size),
                       pool_phase=str(cfg.pool_phase), seed=int(cfg.seed))
    elig_path = os.path.join(out_root, "eligible_clips.json")

    def build_ep(idx):
        """Episode + window exactly as the notebook builds them; None when the
        eval leg yields too few steps."""
        ep = _quiet(R.load_episode, dataset, idx, int(cfg.seed) + idx)
        try:
            _quiet(R.apply_window, ep, spec, int(cfg.start_offset), str(cfg.pool_phase),
                   int(cfg.rollout_stride), None if cfg.max_steps is None else int(cfg.max_steps))
        except Exception:
            return None
        if len(ep["targets"]) < max(1, int(cfg.min_steps)):
            return None
        return ep

    def load_elig():
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
        # metadata-only retention: (idx, n_steps, video_id); episodes are
        # rebuilt one at a time in the record loop (host-RAM hygiene).
        order = np.random.default_rng(int(cfg.seed)).permutation(len(dataset))
        eligible, scanned = [], 0
        for idx in order:
            if cfg.num_clips is not None and len(eligible) >= int(cfg.num_clips):
                break
            idx = int(idx)
            scanned += 1
            ep = build_ep(idx)
            if ep is not None:
                eligible.append([idx, len(ep["targets"]), ep["video_id"]])
            del ep
        print(f"[clips] {len(eligible)} eligible episodes (scanned {scanned} of {len(dataset)})")
        tmp = f"{elig_path}.tmp{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump({"params": elig_params, "clips": eligible, "scanned": scanned}, f)
        os.replace(tmp, elig_path)
    else:
        print(f"[clips] loaded {len(eligible)} eligible episodes from cache")

    # ── Pending work for THIS shard, clip-major ──────────────────────────────
    arm_dir = {label: label + ("_ctrl" if ctrl else "") for label in models}
    for label in models:
        os.makedirs(os.path.join(out_root, arm_dir[label]), exist_ok=True)
    my = [e for i, e in enumerate(eligible) if i % int(cfg.num_shards) == int(cfg.shard)]
    todo = []
    for clip_idx, n_steps, _vid in my:
        missing = [l for l in models if not os.path.exists(
            os.path.join(out_root, arm_dir[l], f"{clip_idx:05d}.npz"))]
        if missing:
            todo.append((int(clip_idx), int(n_steps), missing))
    n_pending = sum(len(ls) for _, _, ls in todo)
    total_steps = sum(n * len(ls) for _, n, ls in todo)
    print(f"[shard {int(cfg.shard)}] {n_pending} records pending ({total_steps} steps)")

    done_recs, done_steps, t0 = 0, 0, time.time()
    for clip_idx, n_steps, labels in todo:
        ep = build_ep(clip_idx)
        assert ep is not None and len(ep["targets"]) == n_steps, \
            f"clip {clip_idx}: episode rebuild diverged from the eligibility scan"
        targets = ep["targets"]
        tg_frames = ep["goal_idx"][targets].astype(np.int64)
        origin = int(ep["ctx_idx"][0])
        fps = spec.fps

        # per-clip sidecars: GT latents (for d_ctx) + the change-mask family
        lat_all = np.load(os.path.join(ep["ep_path"], "latents.npy"), mmap_mode="c")
        chg = None
        mpath = os.path.join(ep["ep_path"], "latents" + str(cfg.mask_suffix))
        if os.path.isfile(mpath):
            packed = np.load(mpath, mmap_mode="c")
            chg = np.unpackbits(np.asarray(packed), axis=1).reshape(len(packed), 32, 32).astype(bool)

        # per-clip covariates (arm-independent)
        n = len(targets)
        door_frac = (np.array([float(ep["goal_state"][t][0]) for t in targets], np.float32)
                     if ep.get("goal_state") is not None else np.full(n, np.nan, np.float32))
        ev_opens = [(e["frame"], e["objectId"].split("|")[0]) for e in ep["meta"]["events"]
                    if e["action"] == "OpenObject" and e["frame"] >= ep["eval_start"]]
        if ev_opens:
            of = np.array([f for f, _ in ev_opens])
            near = np.argmin(np.abs(tg_frames[:, None] - of[None]), axis=1)
            sec_to_reveal = (tg_frames - of[near]) / fps
            reveal_station = np.array([STATION_CODE.get(ev_opens[j][1], 0) for j in near], np.int8)
        else:
            sec_to_reveal = np.full(n, np.nan, np.float32)
            reveal_station = np.zeros(n, np.int8)

        for label in labels:
            rec_path = os.path.join(out_root, arm_dir[label], f"{clip_idx:05d}.npz")
            if os.path.exists(rec_path):
                continue
            torch.manual_seed(int(cfg.seed) + clip_idx)
            with contextlib.redirect_stdout(io.StringIO()):
                preds, interacts, pick_times, pick_scores = R.rollout(
                    models[label], label, ep, spec, stats_np, device,
                    sampler=str(cfg.sampler), seed=int(cfg.seed) + clip_idx,
                    feed_interact=bool(cfg.feed_interact),
                    worldmem_pool_size=(None if cfg.worldmem_pool_size is None
                                        else int(cfg.worldmem_pool_size)))
            pred_lat = preds                       # rollout returns (n_steps, C, H, W) stacked
            gt_lat = ep["goal_lat"][targets]
            psnr, lp, dsim = frame_metrics(pred_lat, gt_lat)

            # ── in-mask block ────────────────────────────────────────────────
            mask_valid = np.zeros(n, bool)
            mask_frac = np.full(n, np.nan, np.float32)
            in_mse = np.full(n, np.nan, np.float32)
            out_mse = np.full(n, np.nan, np.float32)
            clpips = np.full(n, np.nan, np.float32)
            d_ctx_in = np.full(n, np.nan, np.float32)
            pick_dt = np.full(n, np.nan, np.float32)
            for k in range(n):
                fi = _top1_frame(pick_times, pick_scores, k, fps, origin)
                pick_dt[k] = (fi - int(tg_frames[k])) / fps
                f = int(tg_frames[k])
                if chg is None or f >= len(chg):
                    continue
                m = chg[f]
                mask_frac[k] = m.mean()
                if m.sum() == 0 or m.mean() >= float(cfg.mask_max_frac):
                    continue
                mask_valid[k] = True
                mt = torch.as_tensor(m)
                pr, g = pred_lat[k].float(), gt_lat[k].float()
                e_gt = ((pr - g) ** 2).mean(0)
                in_mse[k] = float(e_gt[mt].mean())
                out_mse[k] = float(e_gt[~mt].mean())
                if 0 <= fi < len(lat_all):
                    ctx_l = torch.from_numpy(np.asarray(lat_all[fi])).float()
                    d_ctx_in[k] = float((((pr - ctx_l) ** 2).mean(0))[mt].mean())
                ys, xs = np.nonzero(m)
                pad = int(cfg.mask_pad)
                y = _min_side(max(ys.min() - pad, 0), min(ys.max() + 1 + pad, 32))
                x = _min_side(max(xs.min() - pad, 0), min(xs.max() + 1 + pad, 32))
                clpips[k] = crop_lpips(pred_lat[k], gt_lat[k], (y[0], y[1], x[0], x[1]))

            payload = dict(
                # frame-wise (shared record keys)
                psnr=psnr, lpips=lp, dreamsim=dsim,
                frac=np.arange(1, n + 1) / n,
                t_rel=(tg_frames - (ep["eval_start"] + 1)) / fps,
                frame_idx=tg_frames,
                # event covariates
                interact_fed=np.asarray(interacts, dtype=np.float32),
                door_frac_fed=door_frac,
                sec_to_reveal=sec_to_reveal.astype(np.float32),
                reveal_station=reveal_station,
                # in-mask block
                mask_valid=mask_valid, mask_frac=mask_frac,
                in_mse=in_mse, out_mse=out_mse, crop_lpips=clpips,
                d_ctx_in=d_ctx_in, pick_dt=pick_dt,
                # scalars
                video_id=np.array(ep["video_id"]), kind=np.array("ctrl" if ctrl else "act"),
                eval_start=np.array(ep["eval_start"]), n_steps=np.array(n),
                ckpt_steps=np.array(int(getattr(models[label], "train_steps", -1))),
            )
            np.savez_compressed(rec_path, **payload)
            done_recs += 1
            done_steps += n
            el = time.time() - t0
            eta = (total_steps - done_steps) * el / max(done_steps, 1)
            nv = int(mask_valid.sum())
            print(f"[{arm_dir[label]}] clip {clip_idx:5d} ({ep['video_id']}): {n} steps  "
                  f"lpips {lp.mean():.3f}  in-mask steps {nv}"
                  + (f"  in_mse {np.nanmean(in_mse):.4f}  d_ctx {np.nanmean(d_ctx_in):.4f}" if nv else "")
                  + f"  | shard {done_recs}/{n_pending}  ETA {eta/60:.0f} min")
            del preds, pred_lat, gt_lat
        del ep, lat_all, chg
    print("done")


if __name__ == "__main__":
    main()
