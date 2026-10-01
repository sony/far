"""FAR quickstart: roll out FAR and a baseline on one AI2-THOR v3 test clip.

    python scripts/quickstart.py                              # FAR vs WorldMem on clip 350
    python scripts/quickstart.py --arms far_multicue temporal worldmem --clip 341
    python scripts/quickstart.py --max-steps 40               # a shorter window (faster)

On the first run it downloads the chosen checkpoint bundles (~0.5 GB each) and the
AI2-THOR v3 demo tier (1.1 GB, 114 test clips) with `scripts/download_release.py`; the
SDXL VAE decoder and the LPIPS weights are fetched automatically on first use. Later runs
reuse everything.

The episode: during the exploration leg the agent opens containers and moves objects;
on the evaluation leg it retraces the tour and revisits the changed stations. Every arm
generates the evaluation leg frame by frame, retrieving context from the exploration
leg, with the same generator architecture and noise seed. Outputs, in `--out`:
  rollout.mp4   ground truth and each arm side by side, with the frames each arm
                recalled, change boxes and a correct / wrong-state verdict at revisits
  compare.png   a still grid of evenly spaced steps with per-step LPIPS
The first run takes about 8 minutes on one H100 GPU, downloads included (2.4 GB);
each arm rolls the 141-step evaluation leg of the default clip in about 2.5 minutes.
Needs a CUDA GPU (about 14 GB of memory with two arms); the device is `--device`, or
`$FAR_DEVICE`, default `cuda:0`. For LoopNav, SoundSpaces and AI2-THOR-dyn, and for every
knob, use the rollout notebooks in `notebooks/`.
"""
import argparse
import json
import os
import subprocess
import sys
import time

ARMS = {"far_multicue": "FAR", "far_meta": "FAR (metadata only)", "worldmem": "WorldMem", "temporal": "Temporal"}
STEP = "0700000"
CORPUS, GROUP = "ai2thor_v3", "ai2thor_v3_latent"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", type=int, default=350, help="test clip index; must be in the demo tier (default 350)")
    ap.add_argument("--arms", nargs="+", default=["far_multicue", "worldmem"], choices=sorted(ARMS),
                    help="checkpoint bundles to compare (default: far_multicue worldmem)")
    ap.add_argument("--max-steps", type=int, default=None, help="cap on generation steps (default: the whole evaluation leg)")
    ap.add_argument("--out", default=None, help="output folder (default: results/quickstart/clip<clip>)")
    ap.add_argument("--device", default=None, help="torch device (default: $FAR_DEVICE or cuda:0)")
    ap.add_argument("--no-download", action="store_true", help="skip the download step (assets already in place)")
    a = ap.parse_args()

    from far.constants import ROOT, MODELS_ROOT
    os.chdir(ROOT)
    t0 = time.time()
    if not a.no_download:
        cmd = [sys.executable, os.path.join(ROOT, "scripts", "download_release.py"), "--no-assets", "--data", "demo"]
        for arm in a.arms:
            cmd += ["--arm", f"{CORPUS}/{arm}"]
        print("[1/4] downloading checkpoints and the AI2-THOR v3 demo tier (skips what is already present)", flush=True)
        subprocess.run(cmd, check=True)

    import torch
    import matplotlib
    matplotlib.use("Agg")
    from far.analysis import thor_rollout as R

    device = a.device or os.environ.get("FAR_DEVICE", "cuda:0")
    if not torch.cuda.is_available() or not device.startswith("cuda"):
        sys.exit("the quickstart needs a CUDA GPU (diffusion rollouts on CPU take hours)")
    test_ids = json.load(open(os.path.join("results", "indices", CORPUS, f"{GROUP}_test.json")))["sample_ids"]
    have = [i for i, e in enumerate(test_ids) if os.path.isfile(os.path.join("datasets", CORPUS, "latent", e, "latents.npy"))]
    if a.clip not in have:
        sys.exit(f"clip {a.clip} is not on disk. Clips in the demo tier: {have}\n"
                 f"(or fetch the whole test split: python scripts/download_release.py --no-models --data test --corpus {CORPUS})")
    out = a.out or os.path.join("results", "quickstart", f"clip{a.clip}")
    os.makedirs(out, exist_ok=True)

    print(f"[2/4] loading {', '.join(ARMS[x] for x in a.arms)} and clip {a.clip} on {device}", flush=True)
    runs = {ARMS[arm]: (os.path.join(MODELS_ROOT, CORPUS, arm), STEP) for arm in a.arms}
    models, data_cfg = R.build_models(runs, device)
    dataset, spec, stats_np, _ = R.build_dataset(data_cfg, GROUP, 1000)
    decode = R.Decoder(next(iter(models.values())).tokenizer, device)
    R.install_eval_knobs(models, data_cfg, "ddim", 20, 10, 12)
    ep = R.load_episode(dataset, a.clip, 0)
    revisits = R.print_revisits(ep, spec)
    targets = R.apply_window(ep, spec, 0, "exploration", 1, a.max_steps)

    print(f"[3/4] rolling out {len(targets)} steps per arm", flush=True)
    preds, pick_times, pick_scores, step_interact = R.run_rollouts(models, ep, spec, stats_np, device, sampler="ddim", seed=0)
    gt_lat = ep["goal_lat"][targets]
    gt_img = decode(gt_lat)
    cur_img = decode(ep["ctx_lat"][-1:])[0]
    pred_img = {label: decode(p) for label, p in preds.items()}

    print("[4/4] writing the comparison grid and video", flush=True)
    mvals, fmt, unit, better = R.metric_curves("lpips", preds, pred_img, gt_lat, gt_img, ep, device, revisits)
    R.plot_compare(ep, spec, cur_img, gt_img, pred_img, mvals, step_interact, fmt, unit,
                   title=f"{ep['video_id']}: FAR quickstart (LPIPS per step, lower is better)",
                   save_path=os.path.join(out, "compare.png"))
    ours = "FAR" if "FAR" in pred_img else None
    R.write_rollout_video(ep, spec, decode, gt_img, pred_img, mvals, fmt, unit, step_interact, pick_times, pick_scores,
                          os.path.join(out, "rollout.mp4"), pick_top_n=4, include_interaction_phase=True,
                          gutter=10, bottom_pad=64, device=device, layout="panels", ours=ours)
    print("\nThe state verdicts above say, per changed-station revisit, whether each arm rendered the state the")
    print("exploration leg left behind (correct) or a stale one (wrong). Whole-frame LPIPS is near zero for most")
    print("steps (each step predicts 0.1 s ahead of a real frame); the revisits are where memory matters.")
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {os.path.abspath(out)}/rollout.mp4, compare.png")


if __name__ == "__main__":
    main()
