"""Per-cell agent silhouette detector on SD-VAE latents (4x32x32 -> 32x32 mask) for the
AI2-THOR-dyn both-ends painting probe: trained on train-split frames (`latents.agent.npy`
silhouette rows as targets), validated on test-split REAL frames. It is the "does the generation
paint the second agent" readout of scripts/ai2thor_dyn/probe_paint_both_ends.py.

usage: python scripts/ai2thor_dyn/train_agent_cell_detector.py [out_dir=models]   -> <out_dir>/agent_cell_detector.pt
Run from the repository root (reads results/indices/ai2thor_dyn and datasets/ai2thor_dyn/latent).
"""
import json, os, sys, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from far.constants import ROOT
os.chdir(ROOT)
S = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("FAR_MODELS_ROOT", os.path.join(ROOT, "models"))
os.makedirs(S, exist_ok=True); dev = "cuda"; random.seed(0); np.random.seed(0); torch.manual_seed(0)
def frames(split, n_ep, per_ep):
    ids = json.load(open(f"results/indices/ai2thor_dyn/ai2thor_dyn_latent_{split}.json"))["sample_ids"]; ids = ids[:: max(1, len(ids) // n_ep)][:n_ep]
    X, Y = [], []
    for vid in ids:
        d = Path(f"datasets/ai2thor_dyn/latent/{vid}"); m = json.loads((d/"meta.json").read_text()); es = int(m["eval_start"]); T = int(m["n_frames"])
        lat = np.load(d/"latents.npy", mmap_mode="c"); rows = np.load(d/"latents.agent.npy", mmap_mode="c"); px = np.load(d/"agent2.npz")["px"]
        ev = np.arange(es, T); watch_vis = np.flatnonzero(px[:es] > 0); watch_no = np.setdiff1d(np.arange(es), watch_vis)
        sel = np.concatenate([ev, np.random.choice(watch_vis, min(per_ep, len(watch_vis)), replace=False) if len(watch_vis) else [], np.random.choice(watch_no, min(per_ep, len(watch_no)), replace=False)]).astype(int)
        X.append(np.asarray(lat[sel], np.float32)); Y.append(np.stack([np.unpackbits(np.asarray(rows[i]))[:1024].reshape(32, 32) for i in sel]).astype(np.float32))
    return np.concatenate(X), np.concatenate(Y)
Xtr, Ytr = frames("train", 120, 60); Xte, Yte = frames("test", 60, 60)
print(f"train frames {len(Xtr)} (cells positive {Ytr.mean():.4f}) | test frames {len(Xte)}", flush=True)
mu, sd = Xtr.mean((0, 2, 3), keepdims=True), Xtr.std((0, 2, 3), keepdims=True) + 1e-6
net = nn.Sequential(nn.Conv2d(4, 48, 3, padding=1), nn.GELU(), nn.Conv2d(48, 48, 3, padding=1), nn.GELU(), nn.Conv2d(48, 48, 3, padding=1), nn.GELU(), nn.Conv2d(48, 1, 1)).to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
Xt, Yt = torch.tensor((Xtr - mu) / sd), torch.tensor(Ytr); pw = torch.tensor((1 - Ytr.mean()) / Ytr.mean(), device=dev).sqrt()
for ep in range(12):
    perm = torch.randperm(len(Xt)); tot = 0
    for i in range(0, len(Xt), 128):
        idx = perm[i:i+128]; x, y = Xt[idx].to(dev), Yt[idx].to(dev); opt.zero_grad()
        l = F.binary_cross_entropy_with_logits(net(x)[:, 0], y, pos_weight=pw); l.backward(); opt.step(); tot += l.item() * len(idx)
    if ep % 3 == 2: print(f"  epoch {ep+1}: loss {tot/len(Xt):.4f}", flush=True)
net.eval()
with torch.no_grad(): pr = torch.sigmoid(net(torch.tensor((Xte - mu) / sd).to(dev))[:, 0]).cpu().numpy()
pm = pr > 0.5; gt = Yte > 0.5
tp = (pm & gt).sum(); print(f"test cells: precision {tp / max(pm.sum(), 1):.3f} recall {tp / max(gt.sum(), 1):.3f} | frame IoU (frames with agent) {np.mean([(a & b).sum() / max((a | b).sum(), 1) for a, b in zip(pm, gt) if b.any()]):.3f}")
has = gt.reshape(len(gt), -1).any(1); cells = pm.reshape(len(pm), -1).sum(1)
for thr in (4, 6, 10): print(f"  frame-level 'paints' (>= {thr} cells): recall {np.mean(cells[has] >= thr):.3f}  FPR {np.mean(cells[~has] >= thr):.3f}  | agent frames mean cells {cells[has].mean():.1f}, empty {cells[~has].mean():.1f}")
torch.save({"net": net.state_dict(), "mu": mu, "sd": sd}, f"{S}/agent_cell_detector.pt"); print("saved")
