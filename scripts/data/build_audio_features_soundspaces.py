"""Pre-compute binaural audio features for the SoundSpaces latent episodes.

For each episode dir under --dataset-root (the latent dataset produced by
`build_latent_dataset_soundspaces.py`, which carries the copied `audio.wav`),
computes a 4-plane time-frequency feature array and stores it as
`audio_feat.npy` inside the episode dir:

    (4, n_mels, T_hops) float16, planes = [logmel_L, logmel_R, cos_IPD, sin_IPD]

  - log-mel per ear from the UNNORMALIZED waveform (absolute level = the
    distance cue; ILD = the plane difference — no per-window/per-channel gain
    normalization, ever). log(x + eps) floors silence.
  - IPD (interaural phase difference) mel-banded CIRCULARLY: the complex
    cross-spectrum S_L * conj(S_R) is banded FIRST, then the angle is taken —
    never average raw angles (they wrap), and the banding is magnitude-
    weighted so silent bins' meaningless phase is suppressed. cos/sin encode
    the angle smoothly. Broadband energy needs no extra plane — pool it from
    the log-mel planes.

Alignment contract (fps 10, sr 16 kHz, hop 160 = 10 ms → 100 hops/s):
frame i sits at hop index 10*i (torch.stft center=True: hop h is centered at
h*10 ms). A causal W-second window for frame i is the slice
`feat[..., 10*i - 100*W : 10*i]`, left-padded for the first W seconds
(log-floor in planes 0-1, zeros in planes 2-3). Consumers read the exact
parameters from `<dataset-root>/audio_feat_params.json`, written by this
script; re-running with DIFFERENT parameters against an existing params file
is a hard error (the features, the cached keys, and the live query path must
all share one bit-identical convention).

After the feature pass, dataset-GLOBAL log-mel mean/std (the only
normalization consumers should ever apply) are estimated over a sample of
episodes and stored in the params file.

Resume-safe (existing audio_feat.npy skipped), atomic writes, CPU-only:
    python scripts/data/build_audio_features_soundspaces.py \\
        --dataset-root datasets/soundspaces/latent --workers 16
"""

import argparse
import json
import os
import random
import time
import wave
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

PARAMS = {
    "sr": 16000,
    "n_fft": 512,
    "win_length": 400,     # 25 ms
    "hop_length": 160,     # 10 ms -> 100 hops/s, frame i at hop 10*i (fps 10)
    "n_mels": 64,
    "fmin": 0.0,
    "fmax": 8000.0,
    "log_eps": 1e-10,
    "planes": ["logmel_L", "logmel_R", "cos_ipd", "sin_ipd"],
    "recommended_window_hops": 50,   # 0.5 s causal windows
}


def mel_filterbank(sr, n_fft, n_mels, fmin, fmax) -> np.ndarray:
    """HTK-style triangular mel filterbank, (n_mels, n_fft//2 + 1)."""
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)

    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    freqs = np.linspace(0.0, sr / 2.0, n_fft // 2 + 1)
    fb = np.zeros((n_mels, freqs.shape[0]), dtype=np.float32)
    for m in range(n_mels):
        lo, c, hi = pts[m], pts[m + 1], pts[m + 2]
        fb[m] = np.clip(np.minimum((freqs - lo) / (c - lo),
                                   (hi - freqs) / (hi - c)), 0.0, None)
    return fb


def read_wav_stereo(path: Path) -> np.ndarray:
    """(2, N) float32 in [-1, 1]; asserts stereo 16-bit at the contract sr."""
    with wave.open(str(path)) as w:
        assert w.getnchannels() == 2, f"{path}: expected stereo, got {w.getnchannels()}ch"
        assert w.getframerate() == PARAMS["sr"], \
            f"{path}: sr {w.getframerate()} != {PARAMS['sr']}"
        assert w.getsampwidth() == 2, f"{path}: expected int16"
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return (raw.reshape(-1, 2).T.astype(np.float32)) / 32768.0


def compute_features(wav: np.ndarray, fb: np.ndarray) -> np.ndarray:
    """(2, N) waveform -> (4, n_mels, T_hops) float16 feature planes."""
    x = torch.from_numpy(wav)
    S = torch.stft(x, n_fft=PARAMS["n_fft"], hop_length=PARAMS["hop_length"],
                   win_length=PARAMS["win_length"],
                   window=torch.hann_window(PARAMS["win_length"]),
                   center=True, return_complex=True)          # (2, F, T)
    fbt = torch.from_numpy(fb)                                 # (M, F)
    power = S.abs().square()                                   # (2, F, T)
    logmel = torch.log(torch.einsum("mf,cft->cmt", fbt, power)
                       + PARAMS["log_eps"])                    # (2, M, T)
    cross = S[0] * S[1].conj()                                 # (F, T) complex
    banded = torch.einsum("mf,ft->mt", fbt.to(torch.complex64), cross)
    ipd = torch.angle(banded)                                  # (M, T)
    feat = torch.cat([logmel, torch.cos(ipd)[None], torch.sin(ipd)[None]])
    return feat.to(torch.float16).numpy()                      # (4, M, T)


def atomic_save_npy(arr, path: Path):
    for stale in path.parent.glob(path.name + ".tmp*"):
        stale.unlink()
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp, "wb") as f:                                 # no ".npy" appending
        np.save(f, arr)
    os.replace(tmp, path)


def process_episode(ep_dir: str) -> tuple[str, int, bool]:
    """Returns (name, T_hops, computed?)."""
    torch.set_num_threads(1)
    d = Path(ep_dir)
    out = d / "audio_feat.npy"
    if out.exists():
        return d.name, 0, False
    fb = _FB if _FB is not None else mel_filterbank(
        PARAMS["sr"], PARAMS["n_fft"], PARAMS["n_mels"],
        PARAMS["fmin"], PARAMS["fmax"])
    wav = read_wav_stereo(d / "audio.wav")
    feat = compute_features(wav, fb)
    meta = json.loads((d / "meta.json").read_text())
    # Frame i must index hop 10*i: with center=True, T_hops = N//hop + 1 =
    # 10*n_frames + 1, so the last frame's hop (10*(n_frames-1)) always exists.
    need = 10 * (meta["n_frames"] - 1) + 1
    assert feat.shape[2] >= need, \
        f"{d.name}: {feat.shape[2]} hops < {need} needed for {meta['n_frames']} frames"
    assert np.isfinite(feat).all(), f"{d.name}: non-finite features"
    atomic_save_npy(feat, out)
    return d.name, feat.shape[2], True


_FB = None


def _init_worker():
    global _FB
    torch.set_num_threads(1)
    _FB = mel_filterbank(PARAMS["sr"], PARAMS["n_fft"], PARAMS["n_mels"],
                         PARAMS["fmin"], PARAMS["fmax"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", default="datasets/soundspaces/latent")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N episodes (smoke runs)")
    parser.add_argument("--stats-episodes", type=int, default=100,
                        help="Episodes sampled for the global log-mel stats")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    params_path = root / "audio_feat_params.json"
    if params_path.exists():
        stored = json.loads(params_path.read_text())
        stored_core = {k: v for k, v in stored.items() if k in PARAMS}
        if stored_core != PARAMS:
            raise SystemExit(
                f"PARAMS MISMATCH vs {params_path}:\n stored: {stored_core}\n"
                f" script: {PARAMS}\nRefusing to mix feature conventions — "
                "delete existing audio_feat.npy files AND the params file to rebuild.")

    episodes = sorted(str(d) for d in root.iterdir()
                      if d.is_dir() and (d / "audio.wav").is_file())
    if args.limit is not None:
        episodes = episodes[: args.limit]
    print(f"{len(episodes)} episodes, {args.workers} workers")

    t0, done, computed = time.time(), 0, 0
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init_worker) as pool:
        for name, hops, was_computed in pool.map(process_episode, episodes,
                                                 chunksize=8):
            done += 1
            computed += int(was_computed)
            if done % 500 == 0 or done == len(episodes):
                rate = done / max(time.time() - t0, 1e-9)
                print(f"{done}/{len(episodes)} ({computed} computed, "
                      f"{rate:.1f} eps/s)", flush=True)

    # Dataset-global log-mel stats (planes 0-1) — the ONLY normalization
    # consumers should apply. Written once; kept if already present.
    stored = json.loads(params_path.read_text()) if params_path.exists() else dict(PARAMS)
    if "logmel_mean" not in stored:
        rng = random.Random(0)
        sample = rng.sample(episodes, min(args.stats_episodes, len(episodes)))
        acc = [np.load(Path(e) / "audio_feat.npy", mmap_mode="c")[:2].astype(np.float32)
               for e in sample if (Path(e) / "audio_feat.npy").exists()]
        flat = np.concatenate([a.reshape(-1) for a in acc])
        stored["logmel_mean"] = float(flat.mean())
        stored["logmel_std"] = float(flat.std())
        tmp = params_path.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(stored, indent=1))
        os.replace(tmp, params_path)
        print(f"params + stats written: logmel mean {stored['logmel_mean']:.2f} "
              f"std {stored['logmel_std']:.2f} -> {params_path}")
    print(f"DONE: {computed} computed, {done - computed} skipped "
          f"in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
