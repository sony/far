"""Analysis over `eval_rollout_thor.py` record batteries — one function per
cell of `notebooks/rollout_ai2thor_analysis.ipynb`.

Records are per-(arm, clip) .npz files of per-step scalars (see the collector's
docstring): frame-wise PSNR/LPIPS/DreamSim, event covariates (fed labels,
signed seconds-to-reveal, station), and the in-mask block from `.change.npy`
(in/out latent MSE, bbox-crop LPIPS, copy signature `d_ctx_in` vs the top-1
retrieved frame). All aggregation here is over the COMMON clip set (paired
across arms); partial batteries are fine. Any clip index printed here replays
in `rollout_ai2thor.ipynb` (same CLIP_IDX ordering).
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ARM_COL = {"temporal": "#9e9e9e", "worldmem": "#4d4d4d", "pnp_vmz": "#D55E00"}
PRETTY = {"temporal": "Temporal", "worldmem": "WorldMem", "pnp_vmz": "FAR -- Multi-Cue"}
STATION = {0: "none", 1: "Fridge", 2: "Microwave", 3: "Cabinet", 4: "Drawer"}


def _col(arm):
    return ARM_COL.get(arm, "tab:blue")


def load_records(eval_dir, arms=None):
    """-> {arm: {clip_idx: record}} restricted to the clips COMMON to every
    arm (paired analysis). Prints coverage and checkpoint provenance; warns
    when an arm's records mix checkpoint steps (a battery resumed across a
    `latest` refresh)."""
    arms = arms or sorted(d for d in os.listdir(eval_dir)
                          if os.path.isdir(os.path.join(eval_dir, d)) and d != "stale_ref")   # stale_ref/ = the arm-independent reference
    recs = {}
    for a in arms:
        recs[a] = {int(Path(p).stem): dict(np.load(p, allow_pickle=False))
                   for p in sorted(glob.glob(os.path.join(eval_dir, a, "*.npz")))}
    common = sorted(set.intersection(*(set(r) for r in recs.values())))
    for a in arms:
        cks = sorted({int(r["ckpt_steps"]) for r in recs[a].values()})
        drop = len(recs[a]) - len(common)
        recs[a] = {c: recs[a][c] for c in common}
        tag = f"ckpt {cks[0]}" if len(cks) == 1 else f"MIXED ckpts {cks} — battery crossed a refresh!"
        print(f"{a:>10}: {len(recs[a]) + drop} records ({drop} outside the common set) | {tag}")
    n_steps = sum(int(r["n_steps"]) for r in recs[arms[0]].values())
    print(f"common clips: {len(common)} | steps per arm: {n_steps}")
    return recs


def _clip_mean(rec, key, valid_only=False):
    v = rec[key].astype(float)
    if valid_only:
        v = v[rec["mask_valid"].astype(bool)]
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) else np.nan


def boot_ci(vals, B=2000, seed=0):
    """95% bootstrap CI of the mean over clips (NaN-safe)."""
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs = rng.choice(v, size=(B, len(v)), replace=True).mean(axis=1)
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def summary_table(recs):
    """Whole-battery means (clip-mean then arm-mean, 95% CI) — frame-wise
    metrics over ALL steps, in-mask metrics over mask-VALID steps, and the
    COPY RATE (fraction of valid steps with d_ctx_in < in_mse: the prediction
    is closer to the top-1 retrieved frame than to GT in the change region)."""
    rows = [("psnr", False), ("lpips", False), ("dreamsim", False),
            ("in_mse", True), ("out_mse", True), ("crop_lpips", True), ("d_ctx_in", True)]
    print(f"{'':>12}" + "".join(f"{PRETTY.get(a, a):>30}" for a in recs))
    for key, vo in rows:
        line = f"{key:>12}"
        for a, rr in recs.items():
            m, lo, hi = boot_ci([_clip_mean(r, key, vo) for r in rr.values()])
            line += f"{m:.4f} [{lo:.3f}, {hi:.3f}]".rjust(30)
        print(line)
    line = f"{'copy rate':>12}"
    for a, rr in recs.items():
        per_clip = []
        for r in rr.values():
            mv = r["mask_valid"].astype(bool) & np.isfinite(r["d_ctx_in"]) & np.isfinite(r["in_mse"])
            if mv.any():
                per_clip.append(float((r["d_ctx_in"][mv] < r["in_mse"][mv]).mean()))
        m, lo, hi = boot_ci(per_clip)
        line += f"{m:.3f} [{lo:.3f}, {hi:.3f}]".rjust(30)
    print(line)
    print("\ncopy rate > ~0.5: the arm tends to REPRODUCE what its memory handed it "
          "rather than the ground-truth change region.")


def plot_frac_curves(recs, metric="lpips", n_bins=25, save_path=None):
    """Per-arm mean +- 95% CI of `metric` vs NORMALIZED retrace progress
    (each clip interpolated onto a common frac grid)."""
    U = np.linspace(0.02, 1.0, n_bins)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for a, rr in recs.items():
        mat = np.stack([np.interp(U, r["frac"], r[metric].astype(float)) for r in rr.values()])
        mu = np.nanmean(mat, 0)
        se = np.nanstd(mat, 0) / np.sqrt(mat.shape[0])
        ax.plot(U, mu, color=_col(a), lw=2, label=PRETTY.get(a, a))
        ax.fill_between(U, mu - 1.96 * se, mu + 1.96 * se, color=_col(a), alpha=0.18, lw=0)
    ax.set_xlabel("retrace progress (fraction of rolled window)")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()


def plot_event_aligned(recs, metrics=("lpips", "in_mse"), window=6.0, bin_s=1.0, save_path=None):
    """Metrics re-indexed on SIGNED seconds to the nearest eval-leg reveal
    (negative = approaching, 0 = the open, positive = after). The reveal is
    where state has to be rendered — a spike here that the frac curves smear
    out. `in_mse` uses mask-valid steps only."""
    edges = np.arange(-window, window + bin_s, bin_s)
    mids = (edges[:-1] + edges[1:]) / 2
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.0 * len(metrics), 4.2), squeeze=False)
    for ax, met in zip(axes[0], metrics):
        vo = met in ("in_mse", "out_mse", "crop_lpips", "d_ctx_in")
        for a, rr in recs.items():
            per_bin = [[] for _ in mids]
            for r in rr.values():
                s = r["sec_to_reveal"].astype(float)
                v = r[met].astype(float)
                ok = np.isfinite(s) & np.isfinite(v)
                if vo:
                    ok &= r["mask_valid"].astype(bool)
                for j in np.flatnonzero(ok):
                    b = np.searchsorted(edges, s[j], side="right") - 1
                    if 0 <= b < len(mids):
                        per_bin[b].append(v[j])
            mu = np.array([np.mean(b) if b else np.nan for b in per_bin])
            se = np.array([np.std(b) / np.sqrt(len(b)) if len(b) > 1 else np.nan for b in per_bin])
            ax.plot(mids, mu, color=_col(a), lw=2, label=PRETTY.get(a, a))
            ax.fill_between(mids, mu - 1.96 * se, mu + 1.96 * se, color=_col(a), alpha=0.18, lw=0)
        ax.axvline(0, color="tab:green", lw=1.2, alpha=0.6)
        ax.set_xlabel("seconds to nearest reveal (0 = open)")
        ax.set_ylabel(met)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()


def plot_copy_signature(recs, save_path=None):
    """Per mask-valid step: in-mask distance to GT (x) vs to the TOP-1
    retrieved frame (y). Below the diagonal = the prediction reproduces the
    retrieved (possibly stale) content instead of the true change."""
    n = len(recs)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.4), squeeze=False)
    for ax, (a, rr) in zip(axes[0], recs.items()):
        xs, ys = [], []
        for r in rr.values():
            mv = r["mask_valid"].astype(bool) & np.isfinite(r["d_ctx_in"]) & np.isfinite(r["in_mse"])
            xs.append(r["in_mse"][mv])
            ys.append(r["d_ctx_in"][mv])
        x, y = np.concatenate(xs), np.concatenate(ys)
        ax.scatter(x, y, s=4, alpha=0.15, color=_col(a), edgecolors="none")
        lim = np.nanpercentile(np.concatenate([x, y]), 99)
        ax.plot([0, lim], [0, lim], "--", color="black", lw=1)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        cr = float((y < x).mean())
        ax.set_title(f"{PRETTY.get(a, a)}  (copy rate {cr:.2f})", fontsize=11)
        ax.set_xlabel("in-mask MSE vs GT")
        ax.set_ylabel("in-mask MSE vs top-1 pick")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()


def station_table(recs):
    """In-mask MSE and copy rate split by the nearest reveal's station type —
    which container class each arm fails on."""
    codes = sorted({int(c) for rr in recs.values() for r in rr.values()
                    for c in np.unique(r["reveal_station"])})
    print(f"{'station':>10}" + "".join(f"{PRETTY.get(a, a)[:24]:>26}" for a in recs)
          + "      (in_mse | copy rate | n steps)")
    for c in codes:
        line = f"{STATION.get(c, c):>10}"
        for a, rr in recs.items():
            vals, copy, n = [], [], 0
            for r in rr.values():
                mv = (r["reveal_station"] == c) & r["mask_valid"].astype(bool) \
                    & np.isfinite(r["in_mse"]) & np.isfinite(r["d_ctx_in"])
                vals.extend(r["in_mse"][mv])
                copy.extend(r["d_ctx_in"][mv] < r["in_mse"][mv])
                n += int(mv.sum())
            line += (f"{np.mean(vals):>10.4f} | {np.mean(copy):>4.2f} | {n:>5}" if n
                     else f"{'—':>22}").rjust(26)
        print(line)


def _state_step_mask(r, swing_margin=0.0):
    """Mask-valid steps with a finite change-region crop-LPIPS, minus the
    door-ANIMATION band `sec_to_reveal` in [-swing_margin, 0) (see
    `state_error_rate`)."""
    mv = r["mask_valid"].astype(bool) & np.isfinite(r["crop_lpips"])
    if swing_margin > 0:
        s = r["sec_to_reveal"].astype(float)
        swing = np.isfinite(s) & (s >= -swing_margin) & (s < 0)
        mv &= ~swing
    return mv


def _container_mask(r, reveal_window=3.0):
    """CONTAINER stratum: steps within +-`reveal_window` s of an eval-leg
    reveal at a container station (`reveal_station` > 0)."""
    s = r["sec_to_reveal"].astype(float)
    near = np.zeros(len(s), bool)
    ok = np.isfinite(s)
    near[ok] = np.abs(s[ok]) <= reveal_window
    return near & (r["reveal_station"] > 0)


def state_error_rate(recs, tau=0.10, taus=None, reveal_window=3.0, swing_margin=0.0, save_path=None):
    """Binary state-error proxy: a mask-valid step FAILS when its
    change-region crop-LPIPS exceeds `tau` (the change was not rendered).
    Left: per-arm error rate vs tau (pooled over steps) — read the threshold
    choice off this sweep. Tables at `tau`: overall clip-mean rate (95% CI), a
    CONTAINER vs SURFACE stratification (container = within +-`reveal_window`
    seconds of an eval-leg open at a container station — the interior-reveal
    regime; surface = every other masked change, e.g. relocated items), and
    the per-station split of the container stratum. The proxy is monotone in
    tau, so ARM ORDERING is threshold-stable wherever the curves don't cross.

    `swing_margin` (seconds) excludes steps in the door-ANIMATION band just
    before the settled open (`sec_to_reveal` in [-margin, 0)): a wrongly drawn
    swing sweeping through the bbox crop inflates crop-LPIPS without saying
    anything about the rendered STATE. 0 = count everything; ~0.8 covers the
    3-6-frame spans. The whole-frame animation spike stays visible in the
    event-aligned curves regardless."""
    taus = np.linspace(0.02, 0.5, 25) if taus is None else np.asarray(taus)
    _base_mask = lambda r: _state_step_mask(r, swing_margin)

    step_vals = {}
    for a, rr in recs.items():
        v = np.concatenate([r["crop_lpips"][_base_mask(r)] for r in rr.values()])
        step_vals[a] = v[np.isfinite(v)]
    if swing_margin > 0:
        print(f"(swing_margin = {swing_margin}s: door-animation steps excluded everywhere below)")

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for a, v in step_vals.items():
        ax.plot(taus, [(v > t).mean() for t in taus], color=_col(a), lw=2,
                label=PRETTY.get(a, a))
    ax.axvline(tau, color="black", ls="--", lw=1)
    ax.annotate(f"tau = {tau}", (tau, 0.97), fontsize=9, ha="left")
    ax.set_xlabel("tau (change-region crop-LPIPS threshold)")
    ax.set_ylabel("state-error rate (fraction of mask-valid steps)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()

    print(f"state-error rate at tau = {tau} (clip-mean, 95% CI over clips):")
    for a, rr in recs.items():
        per_clip = []
        for r in rr.values():
            mv = _base_mask(r)
            if mv.any():
                per_clip.append(float((r["crop_lpips"][mv] > tau).mean()))
        m, lo, hi = boot_ci(per_clip)
        print(f"  {PRETTY.get(a, a):>24}: {m:.3f} [{lo:.3f}, {hi:.3f}]   "
              f"({len(step_vals[a])} steps, {len(per_clip)} clips)")

    _cont_mask = lambda r: _container_mask(r, reveal_window)

    print(f"\nCONTAINER (within +-{reveal_window:.0f}s of a reveal) vs SURFACE strata, "
          f"clip-mean error rate at tau = {tau}:")
    for name, keep in (("container", True), ("surface", False)):
        line = f"  {name:>10}:"
        for a, rr in recs.items():
            per_clip, n = [], 0
            for r in rr.values():
                mv = _base_mask(r)
                mv &= _cont_mask(r) if keep else ~_cont_mask(r)
                if mv.any():
                    per_clip.append(float((r["crop_lpips"][mv] > tau).mean()))
                    n += int(mv.sum())
            m, lo, hi = boot_ci(per_clip)
            line += f"  {PRETTY.get(a, a)[:20]}: {m:.3f} [{lo:.3f},{hi:.3f}] (n={n})"
        print(line)

    print(f"\ncontainer stratum by station (step-pooled error rate at tau = {tau}):")
    codes = sorted({int(c) for rr in recs.values() for r in rr.values()
                    for c in np.unique(r["reveal_station"]) if c > 0})
    print(f"{'station':>10}" + "".join(f"{PRETTY.get(a, a)[:22]:>24}" for a in recs))
    for c in codes:
        line = f"{STATION.get(c, c):>10}"
        for a, rr in recs.items():
            vals = []
            for r in rr.values():
                mv = (r["reveal_station"] == c) & _cont_mask(r) & _base_mask(r)
                vals.extend(r["crop_lpips"][mv] > tau)
            line += (f"{np.mean(vals):>18.3f} (n={len(vals)})" if vals else f"{'—':>24}").rjust(24)
        print(line)


# ── STALE-REFERENCE diagnostic ───────────────────────────────────────────────
# d(generated, GT) < d(GT, stale) on the change-region crop, where `stale` is
# the exact-view exploration-leg counterpart the `.change` mask itself was
# derived from (the frame that still shows the OLD state). d(GT, stale)
# depends only on the episode, the frame and the crop — never on an arm — so
# it is built ONCE per evaluated (clip, step) into `<eval_dir>/stale_ref/` and
# read back for every arm and every later battery on the same protocol.

def _grow_box(lo, hi, n=32, min_len=8):
    """The crop floor of eval_rollout_thor / change_region_diagnostic: grow
    boxes narrower than `min_len` cells symmetrically (LPIPS collapses crops
    under ~64 px), sliding to stay inside the grid."""
    if hi - lo < min_len:
        lo = max(0, lo - (min_len - (hi - lo)) // 2)
        hi = min(n, lo + min_len)
        lo = max(0, hi - min_len)
    return lo, hi


def _stale_counterparts(ep_dir, lat, pos_tol=0.02, yaw_tol=np.radians(1.0), pitch_tol=np.radians(1.0),
                        min_peak=0.05, reference="smallest"):
    """{eval frame: stale frame} — the `change`-family pairing of
    scripts/data/build_masks_ai2thor.py (exact-view closure pairs, one side per
    phase, `reference` = the counterpart the mask was diffed against). Eval
    frames whose mask came only from the item-role PNG union have no exact
    counterpart and are absent."""
    ep_dir = Path(ep_dir)
    meta = json.loads((ep_dir / "meta.json").read_text()); T = int(lat.shape[0]); es = int(meta.get("eval_start", T))
    pz = np.load(ep_dir / "poses.npz"); pos, yaw = np.asarray(pz["pos"])[:T], np.asarray(pz["heading"])[:T]
    q = np.asarray(pz["quat"], np.float64)[:T]; pit = np.arcsin(np.clip(2.0 * (q[:, 2] * q[:, 3] - q[:, 0] * q[:, 1]), -1.0, 1.0))
    cz = np.load(ep_dir / "closures.npz"); i, j = np.asarray(cz["i"]), np.asarray(cz["j"])
    ok = (i < T) & (j < T); i, j = i[ok], j[ok]
    d_pos = np.linalg.norm(pos[i] - pos[j], axis=1)
    d_yaw = np.abs(np.arctan2(np.sin(yaw[i] - yaw[j]), np.cos(yaw[i] - yaw[j]))); d_pit = np.abs(pit[i] - pit[j])
    exact = (d_pos < pos_tol) & (d_yaw < yaw_tol) & (d_pit < pitch_tol)
    exact &= ((i < es) & (j >= es)) | ((j < es) & (i >= es))
    cands = {}
    for k in np.flatnonzero(exact):
        a, b = int(i[k]), int(j[k]); cands.setdefault(max(a, b), []).append((min(a, b), float(d_pos[k] + d_yaw[k] + d_pit[k])))
    err = lambda x, y: ((lat[x].astype(np.float32) - lat[y].astype(np.float32)) ** 2).mean(0)
    out = {}
    for ev, cs in cands.items():
        if reference == "tightest":
            ctx = min(cs, key=lambda c: c[1])[0]
            if float(err(ev, ctx).max()) > 1e-8: out[ev] = ctx
            continue
        errs = {ctx: err(ev, ctx) for ctx, _ in cs}; peaks = {c: float(e.max()) for c, e in errs.items()}
        if reference == "earliest":
            ctx = min(errs)
            if peaks[ctx] > 1e-8: out[ev] = ctx
        else:
            nz = [c for c in errs if peaks[c] > min_peak]
            if nz: out[ev] = min(nz, key=lambda c: float(errs[c].mean()))
    return out


def _v3_tokenizer(device):
    """The SDXL VAE the ai2thor_v3 latents were encoded with (same fallback as
    thor_rollout.build_models when the local folder is absent)."""
    from far.tokenizers.sd_vae import SD_VAE
    ckpt = "models/sdxl-vae-fp16-fix"
    if not os.path.isdir(ckpt): ckpt = "madebyollin/sdxl-vae-fp16-fix"
    tok = SD_VAE(image_height=256, image_width=256, vae_scale=0.13025); tok.load_weights(ckpt)
    return tok.to(device).eval()


def build_stale_ref(eval_dir, device="cuda", arm=None, force=False, log_every=100):
    """Build (once) and return the stale reference for `eval_dir`: per common
    clip, per step, `d_stale` = crop-LPIPS between the decoded GT frame and its
    decoded stale counterpart on the SAME crop the battery scored
    `crop_lpips` on (protocol `mask_pad`, 8-cell floor, x8 px). NaN where the
    step is not mask-valid or the frame has no exact counterpart. Resumable:
    clips with a file in `<eval_dir>/stale_ref/` are skipped unless `force`."""
    import torch
    out_dir = Path(eval_dir) / "stale_ref"; out_dir.mkdir(exist_ok=True)
    proto = json.loads((Path(eval_dir) / "protocol.json").read_text())
    pad = int(proto.get("mask_pad", 2)); suffix = proto.get("mask_suffix", ".change.npy")
    root = Path("datasets") / proto["dataset"].replace("_latent", "") / "latent"
    arm = arm or sorted(d for d in os.listdir(eval_dir) if os.path.isdir(os.path.join(eval_dir, d)) and d != "stale_ref")[0]
    files = sorted(glob.glob(os.path.join(eval_dir, arm, "*.npz")))
    todo = [f for f in files if force or not (out_dir / os.path.basename(f)).exists()]
    print(f"stale reference: {len(files)} clips from arm '{arm}', {len(todo)} to build -> {out_dir}")
    if todo:
        import lpips as lpips_pkg
        dev = torch.device(device); tok = _v3_tokenizer(dev); fn = lpips_pkg.LPIPS(net="alex", verbose=False).to(dev).eval()
        @torch.no_grad()
        def detok(lat_np):
            out = []
            for k in range(0, len(lat_np), 16):
                out.append(tok.detokenize(torch.as_tensor(np.asarray(lat_np[k:k + 16], np.float32), device=dev)).clamp(-1, 1))
            return torch.cat(out) * 0.5 + 0.5
        for n, f in enumerate(todo, 1):
            r = np.load(f, allow_pickle=False); vid = str(r["video_id"]); ep = root / vid
            lat = np.load(ep / "latents.npy", mmap_mode="c"); rows = np.load(ep / ("latents" + suffix), mmap_mode="c")
            pairs = _stale_counterparts(ep, lat)
            fi, mv = r["frame_idx"].astype(int), r["mask_valid"].astype(bool)
            stale = np.full(len(fi), -1, int); d = np.full(len(fi), np.nan, np.float32)
            need = sorted({int(x) for k, x in enumerate(fi) if mv[k] and x in pairs} | {pairs[int(x)] for k, x in enumerate(fi) if mv[k] and int(x) in pairs})
            if need:
                img = dict(zip(need, detok(lat[need]).cpu()))
                with torch.no_grad():
                    for k, x in enumerate(fi):
                        x = int(x)
                        if not mv[k] or x not in pairs: continue
                        m = np.unpackbits(np.asarray(rows[x]))[:1024].reshape(32, 32).astype(bool)
                        ys, xs = np.nonzero(m)
                        y = _grow_box(max(ys.min() - pad, 0), min(ys.max() + 1 + pad, 32)); xx = _grow_box(max(xs.min() - pad, 0), min(xs.max() + 1 + pad, 32))
                        y0, y1, x0, x1 = y[0] * 8, y[1] * 8, xx[0] * 8, xx[1] * 8
                        g, st = img[x][None, :, y0:y1, x0:x1].to(dev), img[pairs[x]][None, :, y0:y1, x0:x1].to(dev)
                        stale[k] = pairs[x]; d[k] = float(fn(g * 2 - 1, st * 2 - 1).flatten()[0])
            np.savez(out_dir / os.path.basename(f), video_id=vid, frame_idx=fi, stale_idx=stale, d_stale=d)
            if n % log_every == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} clips", flush=True)
    return load_stale_ref(eval_dir)


def load_stale_ref(eval_dir):
    """-> {clip_idx: {frame_idx, stale_idx, d_stale}} from `<eval_dir>/stale_ref/`."""
    ref = {int(Path(p).stem): dict(np.load(p, allow_pickle=False)) for p in sorted(glob.glob(os.path.join(eval_dir, "stale_ref", "*.npz")))}
    n = sum(len(v["d_stale"]) for v in ref.values()); k = sum(int(np.isfinite(v["d_stale"]).sum()) for v in ref.values())
    print(f"stale reference: {len(ref)} clips, {k}/{n} steps with a counterpart")
    return ref


def _ref_thresholds(rec, ref_clip):
    """Per-step threshold d(GT, stale) aligned to `rec`'s steps (NaN = none)."""
    thr = np.full(len(rec["frame_idx"]), np.nan, np.float32)
    if ref_clip is None: return thr
    lut = dict(zip(ref_clip["frame_idx"].astype(int).tolist(), ref_clip["d_stale"].astype(np.float32).tolist()))
    for k, f in enumerate(rec["frame_idx"].astype(int)):
        thr[k] = lut.get(int(f), np.nan)
    return thr


def stale_margin_table(recs, ref, reveal_window=3.0, swing_margin=0.0, min_ref=0.0,
                       bins=(0.0, 0.02, 0.05, 0.15, np.inf)):
    """Success = d(generated, GT) < d(GT, stale) on the change-region crop
    (the arm rendered the change better than the old state would): coverage,
    clip-mean success rate (95% CI over clips), container vs surface strata,
    per station, and the median ratio d(gen, GT) / d(GT, stale). `min_ref`
    drops steps whose stale counterpart is nearly identical to GT in the crop
    (d(GT, stale) < min_ref: a change too small to render or to judge — the
    test degenerates to decoder noise there); the `bins` breakdown shows the
    success rate as a function of how visible the change is."""
    def _rows(a, keep=None, lo=None, hi=None):
        per, ratios, n = [], [], 0
        for c, r in recs[a].items():
            thr = _ref_thresholds(r, ref.get(c)); mv = _state_step_mask(r, swing_margin) & np.isfinite(thr) & (thr >= min_ref)
            if lo is not None: mv &= (thr >= lo) & (thr < hi)
            if keep is not None: mv &= keep(r)
            if mv.any():
                per.append(float((r["crop_lpips"][mv] < thr[mv]).mean())); ratios.extend((r["crop_lpips"][mv] / np.maximum(thr[mv], 1e-6)).tolist()); n += int(mv.sum())
        return per, ratios, n
    arms = list(recs); a0 = arms[0]
    tot = sum(int(_state_step_mask(r, swing_margin).sum()) for r in recs[a0].values())
    thr_all = np.concatenate([_ref_thresholds(r, ref.get(c))[_state_step_mask(r, swing_margin)] for c, r in recs[a0].items()])
    cov = int(np.isfinite(thr_all).sum()); kept = int((thr_all >= min_ref).sum())
    print(f"steps with a stale counterpart: {cov}/{tot} mask-valid steps ({cov / max(tot, 1):.0%}); "
          f"d(GT, stale) q10/50/90 = {np.nanpercentile(thr_all, [10, 50, 90]).round(3)}; min_ref {min_ref} keeps {kept} ({kept / max(tot, 1):.0%})")
    print(f"\nsuccess rate d(gen, GT) < d(GT, stale)  — clip mean [95% CI] (n steps) | median ratio d(gen,GT)/d(GT,stale)")
    strata = [("all", None), ("container", lambda r: _container_mask(r, reveal_window)), ("surface", lambda r: ~_container_mask(r, reveal_window))]
    codes = sorted({int(c) for rr in recs.values() for r in rr.values() for c in np.unique(r["reveal_station"]) if c > 0})
    strata += [(STATION.get(c, str(c)), (lambda c: (lambda r: _container_mask(r, reveal_window) & (r["reveal_station"] == c)))(c)) for c in codes]
    for name, keep in strata:
        line = f"  {name:>10}:"
        for a in arms:
            per, ratios, n = _rows(a, keep); m, lo, hi = boot_ci(per)
            line += f"  {PRETTY.get(a, a)[:20]}: {m:.3f} [{lo:.3f},{hi:.3f}] (n={n}) r={np.median(ratios) if ratios else float('nan'):.2f}"
        print(line)
    print("\nby the size of the change, d(GT, stale) bin — step-pooled success rate (n):")
    for lo_, hi_ in zip(bins[:-1], bins[1:]):
        if hi_ <= min_ref: continue                                   # entirely below the floor
        line = f"  [{max(lo_, min_ref):.2f}, {hi_ if np.isfinite(hi_) else 'inf'}):".ljust(16)
        for a in arms:
            per, ratios, n = _rows(a, None, lo_, hi_); rate = np.mean(np.concatenate([np.atleast_1d(v) for v in per])) if per else float("nan")
            line += f"  {PRETTY.get(a, a)[:20]}: {rate:.3f} (n={n})"
        print(line)


def plot_state_error_by_station(recs, tau=0.15, reveal_window=3.0, swing_margin=0.0,
                                per_clip=True, ref=None, min_ref=0.0, save_path=None):
    """State-error rate as a grouped bar chart: one group per container station
    type (Fridge / Microwave / Cabinet / Drawer — the container stratum of
    `state_error_rate`, same `tau`, `reveal_window`, `swing_margin`, so the
    numbers match its table) plus the SURFACE stratum (every other masked
    change, e.g. relocated items); bars in the order temporal, worldmem, then
    the remaining arms.
    `per_clip=True`: each clip's failure rate over its steps at that station,
    bar = clip mean, whisker = 95% bootstrap CI over clips (clips without a
    step at the station are skipped); False: step-pooled rate, no whisker.
    `ref` (from `build_stale_ref` / `load_stale_ref`): failure = crop-LPIPS
    >= d(GT, stale) per step instead of > tau (steps without a stale
    counterpart are excluded). Prints the table under the figure."""
    codes = sorted({int(c) for rr in recs.values() for r in rr.values()
                    for c in np.unique(r["reveal_station"]) if c > 0})
    groups = [(c, STATION.get(c, str(c))) for c in codes] + [(-1, "Surface")]
    arms = [a for a in ("temporal", "worldmem") if a in recs] + [a for a in recs if a not in ("temporal", "worldmem")]
    stats = {}                                                # (code, arm) -> (mean, lo, hi, n_steps, n_clips)
    for c, _ in groups:
        for a in arms:
            per, pooled, n = [], [], 0
            for r in recs[a].values():
                base, cont = _state_step_mask(r, swing_margin), _container_mask(r, reveal_window)
                thr = None
                if ref is not None:
                    thr = _ref_thresholds(r, ref.get(next(k for k, v in recs[a].items() if v is r)))
                    base = base & np.isfinite(thr) & (thr >= min_ref)
                mv = (base & ~cont) if c < 0 else (base & cont & (r["reveal_station"] == c))
                if mv.any():
                    fail = (r["crop_lpips"][mv] >= thr[mv]) if thr is not None else (r["crop_lpips"][mv] > tau)
                    per.append(float(fail.mean())); pooled.extend(fail.tolist()); n += int(mv.sum())
            if per_clip:
                m, lo, hi = boot_ci(per)
            else:
                m = float(np.mean(pooled)) if pooled else np.nan; lo = hi = np.nan
            stats[(c, a)] = (m, lo, hi, n, len(per))

    fig, ax = plt.subplots(figsize=(1.6 * len(groups) + 2.0, 4.0))
    width = 0.8 / len(arms)
    for j, a in enumerate(arms):
        xs = np.arange(len(groups)) + (j - (len(arms) - 1) / 2) * width
        ms = np.array([stats[(c, a)][0] for c, _ in groups])
        if per_clip:
            lo = np.array([stats[(c, a)][1] for c, _ in groups]); hi = np.array([stats[(c, a)][2] for c, _ in groups])
            err = np.vstack([ms - lo, hi - ms])
            ax.bar(xs, ms, width, yerr=np.nan_to_num(err), capsize=3, color=_col(a),
                   label=PRETTY.get(a, a), error_kw=dict(lw=1))
        else:
            ax.bar(xs, ms, width, color=_col(a), label=PRETTY.get(a, a))
    # every arm is scored on the same clips/steps, so the counts go in the group labels
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels([f"{name}\n({stats[(c, arms[0])][3]} steps, {stats[(c, arms[0])][4]} clips)" for c, name in groups], fontsize=8.5)
    ax.set_ylabel("state-error rate (d(gen,GT) >= d(GT,stale))" if ref is not None else f"state-error rate (crop-LPIPS > {tau})")
    ax.set_ylim(0, 1.16)                                      # headroom for the legend above bars near 1.0
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_title(f"containers: +-{reveal_window:.0f}s of a reveal; surface: every other masked change"
                 + (f", swing {swing_margin}s excluded" if swing_margin > 0 else "")
                 + (" — clip mean, 95% CI" if per_clip else " — step-pooled"), fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper center", ncol=len(arms), frameon=False)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()

    print(f"{'station':>16}" + "".join(f"{PRETTY.get(a, a)[:22]:>34}" for a in arms)
          + ("      (clip-mean [95% CI], n steps / clips)" if per_clip else "      (step-pooled rate, n steps)"))
    for c, name in groups:
        line = f"{name:>16}"
        for a in arms:
            m, lo, hi, n, k = stats[(c, a)]
            cell = (f"{m:.3f} [{lo:.3f},{hi:.3f}] {n}/{k}" if per_clip else f"{m:.3f} (n={n})") if n else "—"
            line += f"{cell:>34}"
        print(line)
    return stats


# ── stratification by the number of interactions in the clip ────────────────

def clip_k(recs, eval_dir, by="interactions"):
    """{clip_idx: count} per evaluated clip. `by="interactions"`: the number of
    interaction spans FIRED in the exploration leg (runs of a nonzero
    `actions.npz` interact code before eval_start — a move = a pickup + a
    put, so 2 per move); `by="k"`: `meta.json` k_realized, the number of
    station VISITS (the `_k<n>_` tag in the id — a k=3 clip usually holds one
    move plus observe-only visits)."""
    proto = json.loads((Path(eval_dir) / "protocol.json").read_text())
    root = Path("datasets") / proto["dataset"].replace("_latent", "") / "latent"
    a0 = next(iter(recs)); out = {}
    for c, r in recs[a0].items():
        ep = root / str(r["video_id"]); m = json.loads((ep / "meta.json").read_text())
        if by == "k":
            out[c] = int(m.get("k_realized", len(m.get("moves", []))))
        else:
            it = np.asarray(np.load(ep / "actions.npz")["interact"])[: int(m["eval_start"])]
            out[c] = int(((it != 0) & (np.r_[0, it[:-1]] != it)).sum())
    return out


def by_interactions(recs, eval_dir, ref=None, tau=0.15, reveal_window=3.0, swing_margin=0.0,
                    min_ref=0.05, by="interactions", bins=(2, 4, 6, 8), save_path=None):
    """Both state-error readouts stratified by the clip's memory load — `by`
    "interactions": interaction spans fired in the exploration leg, binned at
    `bins` (the last bin pools everything >= its value); "k": station visits
    (`clip_k`). Panels: the tau proxy (crop-LPIPS > tau, all mask-valid
    steps) and, with `ref`, the stale-margin failure rate (d(gen, GT) >=
    d(GT, stale), steps with d(GT, stale) >= `min_ref`). Clip-mean rates with
    95% CIs over clips; table + bar chart."""
    ks = clip_k(recs, eval_dir, by); arms = [a for a in ("temporal", "worldmem") if a in recs] + [a for a in recs if a not in ("temporal", "worldmem")]
    edges = list(bins); kmax = len(edges)
    kbin = lambda k: max((i + 1 for i, e in enumerate(edges) if k >= e), default=1)      # 1-based bin index
    labels = [str(e) for e in edges[:-1]] + [f"{edges[-1]}+"]; unit = "interactions" if by == "interactions" else "visits"
    def rate(a, kb, use_ref):
        per, n = [], 0
        for c, r in recs[a].items():
            if kbin(ks[c]) != kb: continue
            mv = _state_step_mask(r, swing_margin)
            if use_ref:
                thr = _ref_thresholds(r, ref.get(c)); mv &= np.isfinite(thr) & (thr >= min_ref)
                fail = r["crop_lpips"][mv] >= thr[mv]
            else:
                fail = r["crop_lpips"][mv] > tau
            if mv.any(): per.append(float(fail.mean())); n += int(mv.sum())
        m, lo, hi = boot_ci(per); return m, lo, hi, n, len(per)
    panels = [("tau proxy: crop-LPIPS > %.2f" % tau, False)] + ([("stale margin: d(gen,GT) >= d(GT,stale), d(GT,stale) >= %.2f" % min_ref, True)] if ref is not None else [])
    stats = {(a, kb, use_ref): rate(a, kb, use_ref) for a in arms for kb in range(1, kmax + 1) for _, use_ref in panels}
    fig, axes = plt.subplots(1, len(panels), figsize=(6.0 * len(panels), 4.0), squeeze=False)
    width = 0.8 / len(arms)
    for ax, (title, use_ref) in zip(axes[0], panels):
        for j, a in enumerate(arms):
            xs = np.arange(kmax) + (j - (len(arms) - 1) / 2) * width
            ms = np.array([stats[(a, kb, use_ref)][0] for kb in range(1, kmax + 1)])
            lo = np.array([stats[(a, kb, use_ref)][1] for kb in range(1, kmax + 1)]); hi = np.array([stats[(a, kb, use_ref)][2] for kb in range(1, kmax + 1)])
            ax.bar(xs, ms, width, yerr=np.nan_to_num(np.vstack([ms - lo, hi - ms])), capsize=3, color=_col(a), label=PRETTY.get(a, a), error_kw=dict(lw=1))
        ax.set_xticks(np.arange(kmax)); ax.set_xticklabels([f"{l} {unit}\n({stats[(arms[0], kb, use_ref)][4]} clips)" for l, kb in zip(labels, range(1, kmax + 1))], fontsize=8.5)
        ax.set_ylim(0, 1.16); ax.set_yticks(np.linspace(0, 1, 6)); ax.set_ylabel("state-error rate (clip mean, 95% CI)"); ax.set_title(title, fontsize=10); ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="upper center", ncol=len(arms), frameon=False)
    fig.tight_layout()
    if save_path: fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
    for title, use_ref in panels:
        print(f"\n{title} — clip-mean error rate [95% CI] (n steps / clips):")
        print(f"{unit[:5]:>5}" + "".join(f"{PRETTY.get(a, a)[:22]:>34}" for a in arms))
        for l, kb in zip(labels, range(1, kmax + 1)):
            line = f"{l:>5}"
            for a in arms:
                m, lo, hi, n, k = stats[(a, kb, use_ref)]; line += f"{(f'{m:.3f} [{lo:.3f},{hi:.3f}] {n}/{k}' if k else '—'):>34}"
            print(line)
    return stats


def plot_stale_margin_by_size(recs, ref, reveal_window=3.0, swing_margin=0.0,
                              edges=(0.02, 0.05, 0.10, 0.15, 0.20, 0.30, np.inf), save_path=None):
    """Stale-margin error rate (d(gen, GT) >= d(GT, stale)) vs the SIZE of the
    change d(GT, stale): two panels, SURFACE changes (left) and CONTAINER
    reveals (right, within +-`reveal_window` s of a container-station reveal).
    Steps are binned by d(GT, stale) at `edges` (below the first edge omitted —
    changes the sampler cannot resolve); each point sits at the bin's median
    d(GT, stale), y = clip-mean error with a 95% bootstrap band over clips.
    Arms in the order temporal, worldmem, then the rest."""
    arms = [a for a in ("temporal", "worldmem") if a in recs] + [a for a in recs if a not in ("temporal", "worldmem")]
    data = {}
    for a in arms:
        parts = []
        for c, r in recs[a].items():
            thr = _ref_thresholds(r, ref.get(c)); mv = _state_step_mask(r, swing_margin) & np.isfinite(thr)
            parts.append((thr[mv], r["crop_lpips"][mv], np.full(int(mv.sum()), c), _container_mask(r, reveal_window)[mv]))
        data[a] = tuple(np.concatenate(x) for x in zip(*parts))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    for ax, (title, keep) in zip(axes, (("surface changes", False), ("container reveals", True))):
        for a in arms:
            thr, cl, clip, cont = data[a]; xs, ms, los, his, ns = [], [], [], [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                b = (thr >= lo) & (thr < hi) & (cont == keep)
                if b.sum() == 0: continue
                fail = cl[b] >= thr[b]; per = [fail[clip[b] == c].mean() for c in np.unique(clip[b])]
                m, l, h = boot_ci(per); xs.append(float(np.median(thr[b]))); ms.append(m); los.append(l); his.append(h); ns.append(int(b.sum()))
            ax.plot(xs, ms, "o-", color=_col(a), lw=2, ms=4, label=PRETTY.get(a, a))
            ax.fill_between(xs, los, his, color=_col(a), alpha=0.18, lw=0)
            if a == arms[0]:
                for x, n in zip(xs, ns): ax.annotate(f"n={n}", (x, 0.0), ha="center", va="bottom", fontsize=7, color="gray", xytext=(0, 2), textcoords="offset points")
        ax.set_xlim(edges[0], None); ax.set_ylim(0, 1); ax.set_xlabel("size of the change: d(GT, stale), crop-LPIPS"); ax.set_title(title, fontsize=10); ax.grid(alpha=0.3)
    axes[0].set_ylabel("state-error rate: d(gen, GT) >= d(GT, stale)\n(clip mean, 95% CI)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(arms), fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    if save_path: fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()


def paired_deltas(recs, ref="temporal", metric="lpips", save_path=None):
    """Per-clip paired difference (arm − ref) of the clip-mean metric, with
    bootstrap CI — removes clip difficulty as a confound (loopnav-analysis
    style)."""
    base = {c: _clip_mean(r, metric) for c, r in recs[ref].items()}
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for k, (a, rr) in enumerate(x for x in recs.items() if x[0] != ref):
        d = [_clip_mean(rr[c], metric) - base[c] for c in rr]
        m, lo, hi = boot_ci(d)
        ax.errorbar([m], [k], xerr=[[m - lo], [hi - m]], fmt="o", color=_col(a),
                    capsize=4, ms=7, label=PRETTY.get(a, a))
        print(f"{PRETTY.get(a, a):>24} − {ref}:  Δ{metric} = {m:+.4f}  [{lo:+.4f}, {hi:+.4f}]"
              f"   (better on {int(np.mean(np.array(d) < 0) * 100)}% of clips)")
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks([])
    ax.set_xlabel(f"Δ{metric} vs {ref} (negative = better)")
    ax.grid(alpha=0.3, axis="x")
    ax.legend(fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()


def worst_clips(recs, arm="pnp_vmz", metric="in_mse", k=15, ref=None):
    """Exemplar mining: the clips where `arm` is worst on `metric` (in-mask
    metrics use valid steps) — or, with `ref`, where its LEAD over `ref` is
    most negative. Printed CLIP_IDX values replay in rollout_ai2thor.ipynb."""
    vo = metric in ("in_mse", "out_mse", "crop_lpips", "d_ctx_in")
    score = {c: _clip_mean(r, metric, vo) for c, r in recs[arm].items()}
    if ref is not None:
        score = {c: score[c] - _clip_mean(recs[ref][c], metric, vo) for c in score}
    order = sorted((c for c in score if np.isfinite(score[c])), key=lambda c: -score[c])[:k]
    tag = f"{metric}" + (f" − {ref}" if ref else "")
    print(f"worst {k} clips for {arm} by {tag}:")
    for c in order:
        r = recs[arm][c]
        st = [STATION.get(int(s), s) for s in np.unique(r["reveal_station"]) if s > 0]
        print(f"  CLIP_IDX {c:>5}  {str(r['video_id']):<30} {tag} {score[c]:.4f}  "
              f"steps {int(r['n_steps'])}  stations {','.join(st) or '—'}")
    return order
