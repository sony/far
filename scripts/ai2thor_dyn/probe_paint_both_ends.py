"""BOTH-ENDS painting probe (AI2-THOR-dyn): one memory-required trajectory -> TWO generated frames.

A draw anchored on ONE reveal span only asks "is an agent visible at this goal", which cannot
see HEDGING: a model that paints an agent at whichever end it is shown scores well on it. Here,
for a single current frame, both reveal ends are queried:

  * both reveal pseudo-spans START at the same fixed frame (eval_start + reveal_start_offset),
    so ONE anchored current frame is valid for both ends (THORDynBackend.reveal_spans);
  * goal_A = look_A onset + u_A, goal_B = look_B onset + u_B, each u drawn from the training
    `event_post` window, both within the anchored horizon `event_len_traj_pred` of the current;
  * goals are returned in FIXED [A, B] order (index 0 = end A, index 1 = end B).

Each end is generated SEPARATELY (num_goals=1) under the SAME manual seed, so the two frames
share their diffusion noise and differ only in the goal pose/time and in what the memory
retrieves for that goal. Conditioning is the model's OWN predictor (no forcing). The readout is
the latent agent-cell detector: >= MIN_CELLS firing cells counts as "painted".

Kept rows are memory-required by construction: the draw is anchored AND the current frame shows
no agent. Each row is one trajectory with a per-end verdict, so the aggregate can separate

  correct    painted exactly the end(s) that truly hold the agent
  hedged     painted BOTH ends while only one holds the agent
  missed     painted NEITHER while one holds the agent
  wrong_end  painted exactly one end -- the wrong one

usage: SHARD=i NSHARDS=n [EPOCHS=10] [EVAL_BS=8] [MIN_CELLS=10] [CKPT=0500000] [DETECTOR=models/agent_cell_detector.pt] \
       python scripts/ai2thor_dyn/probe_paint_both_ends.py <run_dir> <tag> <out_dir>

`run_dir` is a checkpoint bundle (models/ai2thor_dyn/<arm>), `tag` names the arm in the output
files (`<out_dir>/bothends_<tag>_shard<i>.json`); the detector comes from
scripts/ai2thor_dyn/train_agent_cell_detector.py. Aggregate with scripts/ai2thor_dyn/aggregate_paint_both_ends.py.
The launcher bash_scripts/ai2thor_dyn/probe_both_ends_sharded.sh runs every arm, one shard per GPU.
"""
import sys, os, json, time, random, numpy as np, torch, torch.nn as nn
from omegaconf import OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader, Subset
from far.constants import ROOT
from far.data.samplers import _build_context, _past_end

if len(sys.argv) != 4:
    sys.exit(__doc__)
RUN, TAG, OUTDIR = sys.argv[1], sys.argv[2], sys.argv[3]
SHARD, NSHARDS = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
EPOCHS, BS = int(os.environ.get("EPOCHS", "10")), int(os.environ.get("EVAL_BS", "8"))
MIN_CELLS = int(os.environ.get("MIN_CELLS", "10"))
DETECTOR = os.environ.get("DETECTOR", os.path.join(os.environ.get("FAR_MODELS_ROOT", os.path.join(ROOT, "models")), "agent_cell_detector.pt"))
os.makedirs(OUTDIR, exist_ok=True)
dev = "cuda"
CODE_A, CODE_B = 0, 1                      # THORDynBackend.REVEAL_CODES: 0 -> look_A, 1 -> look_B


class BothEndsSampler:
    """Wraps the test ContextGoalSampler: one anchored current, one goal per reveal end.

    Returns the sampler's usual contract plus goals in [A, B] order. When an episode cannot
    serve both ends from one current (a missing leg, or end B beyond the anchored horizon) it
    falls back to the wrapped sampler's UNIFORM draw and leaves `event_anchored` False, so the
    probe drops the row instead of silently scoring a single-end draw.
    """

    def __init__(self, inner):
        self.inner = inner
        self.n_both = 0
        self.n_drop = 0

    def __getattr__(self, k):                       # forward event_pre / pool_phase / ... to the real sampler
        return getattr(self.inner, k)

    def _fallback(self, sample):
        self.n_drop += 1
        out = self.inner({**sample, "force_event": False})
        out["event_anchored"] = False
        return out

    def __call__(self, sample):
        spans = sample.get("event_spans") or []
        num_frames = int(sample["num_frames"])
        by_code = {int(c): (int(a), int(b)) for c, a, b in spans}
        if CODE_A not in by_code or CODE_B not in by_code:
            return self._fallback(sample)
        start = max(by_code[CODE_A][0], by_code[CODE_B][0])        # both spans share this start
        onsets = {CODE_A: by_code[CODE_A][1], CODE_B: by_code[CODE_B][1]}

        # current: the training window, `event_pre` frames before the shared span start
        pre_lo, pre_hi = self.inner.event_pre
        lo_curr = int(sample["curr_frame_idx"]) if (                      # == _event_draw's bound
            self.inner.event_phase == "eval" and sample.get("curr_frame_idx") is not None) else 1
        hi_pre = min(pre_hi, start - lo_curr, start - 1)
        lo_pre = max(pre_lo, 0)                                           # pre >= 0 here: both spans
                                                                          # share `start`, so walking
                                                                          # INTO one span would leave
                                                                          # the other's window
        if hi_pre < lo_pre:
            return self._fallback(sample)
        curr = start - random.randint(lo_pre, hi_pre)

        # one goal per end, inside the training `event_post` window and the anchored horizon
        post_lo, post_hi = self.inner.event_post
        horizon = self.inner.event_len_traj_pred or self.inner.len_traj_pred
        goals = []
        for code in (CODE_A, CODE_B):
            g_lo = max(onsets[code] + post_lo, curr + 1)
            g_hi = min(onsets[code] + post_hi, num_frames - 1, curr + horizon)
            if g_hi < g_lo:
                return self._fallback(sample)
            goals.append(random.randint(g_lo, g_hi))

        context_indices, context_mask = _build_context(
            self.inner.context_pool_size, curr, _past_end(sample, curr, self.inner.pool_phase))
        self.n_both += 1
        return {"context_indices": context_indices,
                "context_mask": context_mask,
                "goal_indices": goals,                       # FIXED order: [end A, end B]
                "frame_indices": context_indices + goals,
                "event_anchored": True}


# ── model ────────────────────────────────────────────────────────────────────
cfg = OmegaConf.load(f"{RUN}/config.yaml")
if "sdxl-vae-fp16-fix" in str(cfg.model.tokenizer.ckpt_path) and not os.path.isdir(str(cfg.model.tokenizer.ckpt_path)):
    cfg.model.tokenizer.ckpt_path = "madebyollin/sdxl-vae-fp16-fix"
cfg.model.agent_cond_binary = cfg.model.get("agent_cond_binary", False)
model = instantiate(cfg.model).to(dev).eval()
CKPT = os.environ.get("CKPT", "0500000")                 # checkpoint stem: 0500000 | latest | ...
ck = torch.load(f"{RUN}/checkpoints/{CKPT}.pth.tar", map_location=dev, weights_only=False)
strip = lambda d: {k.replace("_orig_mod.", ""): v for k, v in d.items()}
model.core.generator.load_state_dict(strip(ck["ema"]), strict=True)
if "memory" in ck:                                        # temporal / worldmem have no trained memory state
    model.core.memory.load_state_dict(strip(ck["memory"]), strict=True)
model.core.diffusion = instantiate(cfg.model.diffusion, timestep_respacing="ddim20")
core = model.core

det = torch.load(DETECTOR, map_location=dev, weights_only=False)
net = nn.Sequential(nn.Conv2d(4, 48, 3, padding=1), nn.GELU(), nn.Conv2d(48, 48, 3, padding=1), nn.GELU(),
                    nn.Conv2d(48, 48, 3, padding=1), nn.GELU(), nn.Conv2d(48, 1, 1)).to(dev)
net.load_state_dict(det["net"]); net.eval()
mu, sd = torch.tensor(det["mu"], device=dev), torch.tensor(det["sd"], device=dev)
cells = lambda lat: (torch.sigmoid(net((lat - mu) / sd)[:, 0]) > 0.5).cpu().numpy()

# ── data: both-ends sampler, sharded by episode ──────────────────────────────
full = instantiate(cfg.ai2thor_dyn_latent.test)
base = getattr(full, "base_dataset", full)
base.sampler = BothEndsSampler(base.sampler)
base.event_frac = 0                                  # never substitute clips: we drive the draw ourselves
ds = Subset(full, list(range(len(full)))[SHARD::NSHARDS])
nb = (len(ds) + BS - 1) // BS
print(f"START tag={TAG} shard={SHARD}/{NSHARDS} episodes={len(ds)} batches/epoch={nb} epochs={EPOCHS} "
      f"total_batches={nb * EPOCHS} min_cells={MIN_CELLS} step={ck['train_steps']}", flush=True)

rows, t0, done, n_seen, n_kept = [], time.time(), 0, 0, 0
with torch.no_grad():
    for ep in range(EPOCHS):
        torch.manual_seed(9000 + 131 * ep + SHARD); random.seed(9000 + 131 * ep + SHARD); np.random.seed(9000 + 131 * ep + SHARD)
        for bi, batch in enumerate(DataLoader(ds, batch_size=BS, shuffle=False, num_workers=3)):
            g = lambda k: batch[k].to(dev) if batch.get(k) is not None else None
            cst = g("context_agent_state")
            anch = batch["event_anchored"].to(dev).bool()
            # memory-required by construction: both ends drawn AND no agent on screen now
            keep = anch & (cst[:, -1, 0] <= 0)
            n_seen += int(anch.shape[0])
            if not bool(keep.any()):
                done += 1; continue
            k = keep
            ctx_lat, act, strd = g("context_frames")[k], g("goal_poses_nl")[k], g("goal_times_nl")[k]
            cpose, ctime, cmask = g("context_poses")[k], g("context_times")[k], g("context_mask")[k]
            ckeys = g("context_keys"); ckeys = None if ckeys is None else ckeys[k]
            ai, ast = g("action_interact"), g("action_state")
            ai = None if ai is None else ai[k]; ast = None if ast is None else ast[k]
            cst_k, gst = cst[k], g("goal_agent_state")[k]
            B, G = act.shape[:2]
            assert G == 2, f"expected 2 goals (one per end), got {G}"
            n_kept += B
            glm = batch.get("goal_loss_mask")
            gtm = None
            if glm is not None:
                glm = glm[keep.cpu()].numpy()
                gtm = (np.unpackbits(glm.reshape(B * G, -1).astype(np.uint8), axis=1)[:, :1024].reshape(B, G, 32, 32).astype(bool)
                       if glm.shape[-1] == 128 else (glm.reshape(B, G, 32, 32) > 0.5))
            out = {}
            for e, end in ((0, "A"), (1, "B")):
                torch.manual_seed(4321 + 17 * ep + bi)          # SAME noise for both ends
                x0 = core.sample_latent(ctx_lat, act[:, e], strd[:, e], num_goals=1,
                                        context_poses=cpose, context_times=ctime, context_mask=cmask,
                                        context_keys=ckeys,
                                        action_interact=None if ai is None else ai[:, e:e + 1],
                                        action_state=None if ast is None else ast[:, e:e + 1],
                                        use_ddim=True, context_agent_state=cst_k).float()
                out[end] = (cells(x0), torch.sigmoid(core.last_agent_pred[:, 0].float()).cpu().numpy())
            gt = (gst[..., 0] > 0.5).cpu().numpy()               # (B, 2) agent visible at each end's goal
            dt = strd.float().cpu().numpy()
            for j in range(B):
                r = {"gt_A": bool(gt[j, 0]), "gt_B": bool(gt[j, 1]),
                     "dt_A": float(dt[j, 0]), "dt_B": float(dt[j, 1])}
                for e, end in ((0, "A"), (1, "B")):
                    pm, pr = out[end]
                    r[f"cells_{end}"] = int(pm[j].sum())
                    r[f"paint_{end}"] = bool(int(pm[j].sum()) >= MIN_CELLS)
                    r[f"p_{end}"] = float(pr[j])
                    if gtm is not None and gtm[j, e].any():
                        r[f"iou_{end}"] = float((pm[j] & gtm[j, e]).sum() / max((pm[j] | gtm[j, e]).sum(), 1))
                n_true = int(r["gt_A"]) + int(r["gt_B"])
                n_paint = int(r["paint_A"]) + int(r["paint_B"])
                if r["paint_A"] == r["gt_A"] and r["paint_B"] == r["gt_B"]:
                    r["outcome"] = "correct"
                elif n_true == 1 and n_paint == 2:
                    r["outcome"] = "hedged"
                elif n_true == 1 and n_paint == 0:
                    r["outcome"] = "missed"
                elif n_true == 1 and n_paint == 1:
                    r["outcome"] = "wrong_end"
                else:
                    r["outcome"] = f"other(true={n_true},paint={n_paint})"
                rows.append(r)
            done += 1; el = time.time() - t0
            print(f"PROGRESS {TAG} shard={SHARD} {done}/{nb * EPOCHS} ep={ep+1}/{EPOCHS} "
                  f"elapsed={el:.0f}s eta={(nb*EPOCHS-done)/max(done/el,1e-9):.0f}s traj={len(rows)} "
                  f"kept={n_kept}/{n_seen} peak={torch.cuda.max_memory_allocated()/2**30:.1f}GB", flush=True)
        json.dump(rows, open(f"{OUTDIR}/bothends_{TAG}_shard{SHARD}.json", "w"))
json.dump(rows, open(f"{OUTDIR}/bothends_{TAG}_shard{SHARD}.json", "w"))
sm = {}
for r in rows:
    sm[r["outcome"]] = sm.get(r["outcome"], 0) + 1
print(f"DONE {TAG} shard={SHARD} trajectories={len(rows)} kept={n_kept}/{n_seen} "
      f"both_ends_draws={base.sampler.n_both} dropped_draws={base.sampler.n_drop} "
      f"outcomes={sm} elapsed={time.time()-t0:.0f}s", flush=True)
