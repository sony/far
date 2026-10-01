"""Aggregate the both-ends probe: per-trajectory outcomes with cluster-bootstrap CIs.

One row = one memory-required trajectory carrying BOTH reveal ends. Rows are appended in episode
order (1 row per episode per epoch), so the episode is recoverable as shard:(i % n_ep) and the
bootstrap resamples EPISODES, not trajectories.

usage: python scripts/ai2thor_dyn/aggregate_paint_both_ends.py <rows_dir> [epochs=10] [arm1,arm2,...]
"""
import json, glob, re, sys, numpy as np
D, EPOCHS = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10
ARMS = sys.argv[3].split(",") if len(sys.argv) > 3 else ["temporal", "worldmem", "A_nocue", "B_rag"]
rng = np.random.default_rng(0)
OUTCOMES = ["correct", "hedged", "missed", "wrong_end"]

def load(arm):
    rows, clust = [], []
    for f in sorted(glob.glob(f"{D}/bothends_{arm}_shard*.json")):
        sh = re.search(r"shard(\d+)", f).group(1); r = json.load(open(f))
        n_ep = max(len(r) // EPOCHS, 1)
        rows += r; clust += [f"{sh}:{i % n_ep}" for i in range(len(r))]
    return rows, np.array(clust)

def boot(rows, clust, fn, n=2000):
    keys = np.unique(clust); idx = {k: np.flatnonzero(clust == k) for k in keys}
    base = fn(rows)
    if base is None or not np.isfinite(base): return (base, float("nan"), float("nan"))
    v = []
    for _ in range(n):
        pick = np.concatenate([idx[k] for k in rng.choice(keys, len(keys), replace=True)])
        x = fn([rows[i] for i in pick])
        if x is not None and np.isfinite(x): v.append(x)
    return (base, float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
pct = lambda t: f"{t[0]:5.1%} [{t[1]:.1%}, {t[2]:.1%}]" if np.isfinite(t[0]) else "n/a"

summary = {}
for arm in ARMS:
    rows, clust = load(arm)
    if not rows: print(f"{arm}: no rows"); continue
    one = [r for r in rows if int(r["gt_A"]) + int(r["gt_B"]) == 1]      # the discriminating case
    print(f"\n{'='*92}\n=== {arm} | {len(rows)} memory-required trajectories over {len(np.unique(clust))} episodes x {EPOCHS} epochs")
    n_true = np.array([int(r["gt_A"]) + int(r["gt_B"]) for r in rows])
    print(f"    ends truly holding the agent: exactly one {np.mean(n_true==1):.0%}, neither {np.mean(n_true==0):.0%}, both {np.mean(n_true==2):.0%}")
    print(f"{'='*92}")
    idx1 = np.flatnonzero(n_true == 1); c1 = clust[idx1]
    print(f"\n  EXACTLY-ONE-END trajectories ({len(one)}): the model must pick a side")
    for o in OUTCOMES:
        t = boot(one, c1, lambda x, o=o: float(np.mean([r["outcome"] == o for r in x])))
        print(f"    {o:10s} {pct(t)}")
    acc = boot(one, c1, lambda x: float(np.mean([r["outcome"] == "correct" for r in x])))
    # side-blind check: per-end accuracy, as the earlier one-end-at-a-time sweep measured it
    per_end = boot(one, c1, lambda x: float(np.mean([[r["paint_A"] == r["gt_A"], r["paint_B"] == r["gt_B"]] for r in x])))
    pred = boot(one, c1, lambda x: float(np.mean([((r["p_A"] > 0.5) == r["gt_A"]) and ((r["p_B"] > 0.5) == r["gt_B"]) for r in x])))
    hedge_p = boot(one, c1, lambda x: float(np.mean([r["p_A"] > 0.5 and r["p_B"] > 0.5 for r in x])))
    print(f"\n    trajectory correct (BOTH ends right)   {pct(acc)}")
    print(f"    per-end accuracy (side-blind, as before) {pct(per_end)}")
    print(f"    predictor alone, both ends right         {pct(pred)}")
    print(f"    predictor says 'present' at BOTH ends    {pct(hedge_p)}")
    cells_t = np.mean([r["cells_A"] if r["gt_A"] else r["cells_B"] for r in one])
    cells_f = np.mean([r["cells_B"] if r["gt_A"] else r["cells_A"] for r in one])
    iou = [r.get("iou_A") if r["gt_A"] else r.get("iou_B") for r in one]
    iou = [x for x in iou if x is not None]
    print(f"    detector cells: true end {cells_t:.1f} vs empty end {cells_f:.1f} | IoU at the true end {np.mean(iou) if iou else float('nan'):.2f}")
    if len(rows) - len(one):
        rest = [r for r in rows if int(r["gt_A"]) + int(r["gt_B"]) != 1]
        print(f"\n  OTHER trajectories ({len(rest)}): painted nothing {np.mean([not r['paint_A'] and not r['paint_B'] for r in rest]):.0%}, "
              f"painted something {np.mean([r['paint_A'] or r['paint_B'] for r in rest]):.0%}")
    summary[arm] = (acc, per_end)
if len(summary) >= 2:
    print(f"\n{'='*92}\nSUMMARY (exactly-one-end trajectories; 95% CI over episodes)")
    print(f"  {'arm':10s} {'trajectory correct':>28s} {'per-end (side-blind)':>28s}")
    for arm, (acc, per) in summary.items():
        print(f"  {arm:10s} {pct(acc):>28s} {pct(per):>28s}")
