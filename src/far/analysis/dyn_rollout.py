"""Rollout eyeballing on the door-corridor DYNAMIC-AGENT corpus
(`ai2thor_dyn_latent`) — the corpus-specific pieces that
`notebooks/rollout_ai2thor_dyn.ipynb` needs on top of `thor_rollout`
(which supplies the models, dataset, windows, rollouts, metrics and video):
the episode's leg / crossing structure, reveal-leg windows, and a reveal panel
showing every arm's prediction where the agent should (or should not) be.
"""
import json
import os

import numpy as np
import torch

from far.constants import ROOT


def _hwc(img):
    """thor_rollout.Decoder output (C,H,W) float [0,1] -> (H,W,C) for imshow."""
    return np.asarray(torch.as_tensor(img).permute(1, 2, 0).clamp(0, 1))


def attach(ep: dict) -> dict:
    """Attach the corpus sidecars to a `thor_rollout.load_episode` episode:
    agent px per frame, the per-frame silhouette masks (`_agent_mask`, (T, 32, 32)
    bool, for tinting), legs, the merged pre-eval crossings (last = the
    direction-bearing one); print the structure."""
    d = ep["ep_path"]; m = ep["meta"]
    ep["_px"] = np.load(os.path.join(d, "agent2.npz"))["px"]
    mp = os.path.join(d, "latents.agent.npy")             # A2's silhouette on the 32x32 latent grid, packed rows
    ep["_agent_mask"] = (np.unpackbits(np.asarray(np.load(mp, mmap_mode="c")), axis=1)[:, :1024].reshape(-1, 32, 32).astype(bool)
                        if os.path.isfile(mp) else None)
    ep["_legs"] = {L["name"]: (L["start"], L["end"]) for L in m["legs"]}
    es = m["eval_start"]; on = ep["_px"][:es] > 0
    e = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0])); runs = [(int(a), int(b)) for a, b in zip(e[::2], e[1::2])]
    merged = []
    for a, b in runs:
        if merged and a - merged[-1][1] < 20: merged[-1] = (merged[-1][0], b)
        else: merged.append((a, b))
    ep["_pulses"] = merged
    ev = ", ".join(f"{n} {a}-{b}" for n, (a, b) in ep["_legs"].items() if n in ("go_door", "look_A", "look_B"))
    print(f"{ep['video_id']} | scene {m['scene']} | K={m['K']} spawn {m['spawn']} -> FINAL {m['final']} | eval_start {es} | "
          f"crossings {merged} (last = direction) | eval legs: {ev}")
    return ep


def leg_of(ep, f):
    for n, (a, b) in ep["_legs"].items():
        if a <= f < b:
            return n
    return "other"


def window(ep: dict, which: str = "full", stride: int = 4, sec_before: float = 0.0, fps: float = 10.0):
    """-> (start_offset, max_steps) for `thor_rollout.apply_window`: 'full' =
    the whole eval leg from eval_start; 'look_A' / 'look_B' = that reveal leg
    (optionally starting `sec_before` seconds earlier, still inside the eval
    leg)."""
    es = ep["eval_start"]
    if which == "full":
        return 0, None
    a, b = ep["_legs"][which]
    start = max(0, a - es - int(sec_before * fps))
    return start, int(np.ceil((b - es - start) / stride))


class AgentCellDetector:
    """Per-cell second-agent silhouette detector on SD-VAE latents (4x32x32 -> 32x32
    bool), the readout the painting probes use. A 3-layer CNN trained on REAL
    train-split latents against the `latents.agent.npy` silhouette rows; on real
    TEST frames: per-cell precision 0.75 / recall 0.95, frame IoU 0.75, and at the
    >= 10-cell 'paints' threshold recall 0.91 / false-positive rate 0.07 — its own
    error floor, so treat a few stray cells on an empty frame as noise. Weights:
    `agent_cell_detector.pt` under `$FAR_MODELS_ROOT` (default `models/`).
    Drawn as the red box in `reveal_panel`."""
    PAINT_CELLS = 10

    def __init__(self, device, path=None):
        import torch.nn as nn
        path = path or os.path.join(os.environ.get("FAR_MODELS_ROOT", os.path.join(ROOT, "models")), "agent_cell_detector.pt")
        det = torch.load(path, map_location=device, weights_only=False)
        self.net = nn.Sequential(nn.Conv2d(4, 48, 3, padding=1), nn.GELU(), nn.Conv2d(48, 48, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(48, 48, 3, padding=1), nn.GELU(), nn.Conv2d(48, 1, 1)).to(device)
        self.net.load_state_dict(det["net"]); self.net.eval()
        self.mu, self.sd, self.device = torch.as_tensor(det["mu"], device=device), torch.as_tensor(det["sd"], device=device), device

    @torch.no_grad()
    def __call__(self, lat):
        """(N, 4, 32, 32) latents -> (N, 32, 32) bool cell mask."""
        x = torch.as_tensor(lat).to(self.device).float()
        return (torch.sigmoid(self.net((x - self.mu) / self.sd)[:, 0]) > 0.5).cpu().numpy()


def reveal_panel(ep, spec, gt_img, pred_img, targets, preds=None, detector=None, save_path=None):
    """One row per reveal leg inside the window: GT at the leg's midpoint (agent
    mask outlined in GREEN when the sidecar exists) followed by every arm's
    prediction at that step. The question each row asks: is the agent rendered
    at this end iff it arrived here (FINAL = {ep['meta']['final']}).

    With `preds` ({arm: (steps, 4, 32, 32) generated latents}) and a `detector`
    (`AgentCellDetector`), each arm's panel gets the detector's cells drawn as a
    RED box and its title reports the cell count, PAINTS / no paint at the
    >= 10-cell threshold, and the IoU against the GT silhouette (when the GT
    frame has one) — the same readout as the painting probes."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    px, fin = ep["_px"], ep["meta"]["final"]
    mask = None
    mp = os.path.join(ep["ep_path"], "latents.agent.npy")
    if os.path.isfile(mp):
        packed = np.load(mp, mmap_mode="c"); mask = np.unpackbits(np.asarray(packed), axis=1).reshape(len(packed), 32, 32).astype(bool)
    gidx = np.asarray(ep["goal_idx"])[targets]
    rows = []
    for leg in ("look_A", "look_B"):
        if leg not in ep["_legs"]:
            continue
        a, b = ep["_legs"][leg]; inside = np.flatnonzero((gidx >= a) & (gidx < b))
        if len(inside) == 0:
            continue
        # FINAL end: the step whose GT shows the agent most. The OTHER end: the
        # leg's LAST step — the settled view (early look frames still catch the
        # agent at the final end in the periphery while the camera turns).
        k = int(inside[np.argmax(px[gidx[inside]])]) if leg[-1] == fin else int(inside[-1])
        rows.append((leg, k))
    if not rows:
        print("no reveal-leg step inside the window"); return
    arms = list(pred_img); n = 1 + len(arms)
    row_h = 2.9 if detector is not None else 2.4          # 3-line titles need the room
    fig, axes = plt.subplots(len(rows), n, figsize=(2.3 * n, row_h * len(rows)), squeeze=False)
    for r, (leg, k) in enumerate(rows):
        f = int(gidx[k]); here = leg[-1] == fin
        axes[r, 0].imshow(_hwc(gt_img[k])); axes[r, 0].set_title(f"GT f{f} {leg} · agent {'HERE' if here else 'elsewhere'} (px {int(px[f])})", fontsize=8,
                                                         color="tab:green" if here else "black")
        if mask is not None and f < len(mask) and mask[f].any():
            ys, xs = np.nonzero(mask[f])
            axes[r, 0].add_patch(Rectangle((xs.min() * 8, ys.min() * 8), (xs.max() - xs.min() + 1) * 8, (ys.max() - ys.min() + 1) * 8,
                                           fill=False, ec="tab:green", lw=2))
        gt_m = mask[f] if (mask is not None and f < len(mask)) else None
        for j, arm in enumerate(arms, start=1):
            axes[r, j].imshow(_hwc(pred_img[arm][k])); title = f"{arm}\nstep {k}"
            pl = preds.get(arm) if preds is not None else None
            if pl is not None and detector is not None:
                pm = detector(pl[k:k + 1])[0]; n_pred = int(pm.sum())
                verdict = "PAINTS" if n_pred >= detector.PAINT_CELLS else "no paint"
                title += f"\ndet {n_pred} cells · {verdict}"
                if gt_m is not None and gt_m.any():
                    inter, union = int((pm & gt_m).sum()), int((pm | gt_m).sum())
                    title += f" · IoU {inter / union if union else 0.0:.2f}"
                if n_pred:
                    ys, xs = np.nonzero(pm)
                    axes[r, j].add_patch(Rectangle((xs.min() * 8, ys.min() * 8), (xs.max() - xs.min() + 1) * 8, (ys.max() - ys.min() + 1) * 8,
                                                   fill=False, ec="tab:red", lw=2))
            axes[r, j].set_title(title, fontsize=8, color=("tab:green" if (pl is not None and detector is not None and (n_pred >= detector.PAINT_CELLS) == here) else "black"))
        for ax in axes[r]:
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{ep['video_id']} — reveals: the agent arrived at end {fin}; green box = GT silhouette"
                 + ("; red box = agent-cell detector on the generation (green title = paints iff the agent is here)" if detector is not None else ""), fontsize=10)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.show()
