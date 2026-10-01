"""Latent-upsampling decode — the SAME SD-VAE latent decoded at a higher resolution.

The SDXL VAE decoder is fully convolutional: a (4, 32, 32) latent decodes to 256x256 and an
upsampled (4, 64, 64) copy of it to 512x512, with no retraining. Against upscaling the 256x256
decode in PIXEL space, the decoder then interpolates in a space it models, so edges, wall
seams and object outlines come out cleaner; it invents no texture (nothing generative). It is
deterministic and applies identically to ground truth and generations. Machinery for
`notebooks/upscale.ipynb`; every number in the paper stays on the native 256x256 decodes.
"""
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ── tokenizer / latents ───────────────────────────────────────────────────────
def load_tokenizer(run_dir: str, device):
    """The SDXL VAE of a run (built through the run's config so the checkpoint-path
    fallback and the builder's key stripping apply); the DiT it comes with is dropped."""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
    p = str(cfg.model.tokenizer.get("ckpt_path"))
    if "sdxl-vae-fp16-fix" in p and not os.path.isdir(p):
        cfg.model.tokenizer.ckpt_path = "madebyollin/sdxl-vae-fp16-fix"
    model = instantiate(cfg.model)
    tok = model.tokenizer.to(device).eval()
    for q in tok.parameters():
        q.requires_grad_(False)
    return tok


def load_latent(ep_dir: str, frame: int) -> torch.Tensor:
    """(1, 4, 32, 32) float latent of one episode frame (already at the VAE scale)."""
    lat = np.load(os.path.join(ep_dir, "latents.npy"), mmap_mode="c")[int(frame)]
    return torch.as_tensor(np.asarray(lat, dtype=np.float32))[None]


def mask_box(packed_rows_path: str, frame: int, pad: float = 0.08, fallback=(0.3, 0.3, 0.7, 0.7)):
    """Zoom box (x0, y0, x1, y1) in FRAME FRACTIONS around a packed 32x32 latent-grid mask
    row (`latents.agent.npy`, `.change.npy`); `fallback` when the row is empty / missing."""
    if not os.path.isfile(packed_rows_path):
        return fallback
    rows = np.load(packed_rows_path, mmap_mode="c")
    if int(frame) >= len(rows):
        return fallback
    m = np.unpackbits(np.asarray(rows[int(frame)]))[:1024].reshape(32, 32).astype(bool)
    if not m.any():
        return fallback
    ys, xs = np.nonzero(m)
    x0, x1 = xs.min() / 32 - pad, (xs.max() + 1) / 32 + pad
    y0, y1 = ys.min() / 32 - pad, (ys.max() + 1) / 32 + pad
    side = max(x1 - x0, y1 - y0, 0.2)                       # square, >= 20% of the frame
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0, y0 = np.clip(cx - side / 2, 0, 1 - side), np.clip(cy - side / 2, 0, 1 - side)
    return (float(x0), float(y0), float(x0 + side), float(y0 + side))


# ── the two upscaling routes ─────────────────────────────────────────────────
def upsample_latent(lat: torch.Tensor, scale: int, mode: str = "bicubic") -> torch.Tensor:
    """Resample a (N, 4, h, w) latent by an integer `scale` before decoding."""
    if scale == 1:
        return lat
    if mode == "nearest":
        return F.interpolate(lat, scale_factor=scale, mode="nearest")
    return F.interpolate(lat, scale_factor=scale, mode=mode, align_corners=False)


@torch.no_grad()
def decode(tok, lat: torch.Tensor, scale: int = 1, mode: str = "bicubic", chunk: int = 4) -> np.ndarray:
    """Decode (N, 4, h, w) latents -> (N, H, W, 3) uint8, at `scale` x the native size by
    upsampling the LATENT with `mode` first (nearest | bilinear | bicubic)."""
    dev = next(tok.parameters()).device
    out = []
    for i in range(0, lat.shape[0], chunk):
        z = upsample_latent(lat[i:i + chunk].to(dev).float(), scale, mode)
        x = (tok.detokenize(z).clamp(-1, 1) * 0.5 + 0.5).permute(0, 2, 3, 1).cpu().numpy()
        out.append((x * 255).round().astype(np.uint8))
    return np.concatenate(out)


_PIL = {"nearest": Image.NEAREST, "bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC, "lanczos": Image.LANCZOS}


def pixel_resize(img_u8: np.ndarray, size: int, mode: str = "bicubic") -> np.ndarray:
    """Resize an (H, W, 3) uint8 image to `size` x `size` in PIXEL space (the baseline route)."""
    return np.asarray(Image.fromarray(img_u8).resize((size, size), _PIL[mode]))


def variants(tok, lat: torch.Tensor, scale: int, latent_modes=("nearest", "bilinear", "bicubic"),
             pixel_modes=("bicubic", "lanczos")) -> dict:
    """{name: (H, W, 3) uint8} — the native decode, the pixel-space upscales of it, and the
    latent-upsampled decodes, all at `scale` x 256 except 'native'."""
    native = decode(tok, lat, 1)[0]
    out = {"native 256": native}
    for m in pixel_modes:
        out[f"pixel {m} x{scale}"] = pixel_resize(native, native.shape[0] * scale, m)
    for m in latent_modes:
        out[f"latent {m} x{scale}"] = decode(tok, lat, scale, m)[0]
    return out


# ── numbers ───────────────────────────────────────────────────────────────────
def sharpness(img_u8: np.ndarray) -> float:
    """Variance of the Laplacian of the gray image (higher = crisper edges)."""
    g = img_u8.astype(np.float64) @ np.array([0.299, 0.587, 0.114])
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def psnr(a_u8: np.ndarray, b_u8: np.ndarray) -> float:
    mse = np.mean((a_u8.astype(np.float64) - b_u8.astype(np.float64)) ** 2)
    return float(10 * np.log10(255.0 ** 2 / max(mse, 1e-12)))


def roundtrip_psnr(img_hi: np.ndarray, native: np.ndarray) -> float:
    """Does the upscale keep the CONTENT? PSNR between the hi-res image box-downsampled back
    to the native size and the native decode (high = same picture, only sharper)."""
    lo = np.asarray(Image.fromarray(img_hi).resize(native.shape[:2][::-1], Image.BOX))
    return psnr(lo, native)


def metrics_table(tok, samples, scales=(2, 4), latent_modes=("nearest", "bilinear", "bicubic"),
                  pixel_modes=("bicubic", "lanczos")):
    """Print sharpness + round-trip PSNR for every route on every sample; returns the rows."""
    rows = []
    print(f"{'sample':28s} {'route':22s} {'size':>6s} {'sharpness':>10s} {'roundtrip PSNR':>15s}")
    for title, lat, _box in samples:
        native = decode(tok, lat, 1)[0]
        print(f"{title[:28]:28s} {'native':22s} {native.shape[0]:6d} {sharpness(native):10.1f} {'-':>15s}")
        for s in scales:
            v = variants(tok, lat, s, latent_modes, pixel_modes)
            for name, img in v.items():
                if name == "native 256":
                    continue
                r = dict(sample=title, route=name, size=img.shape[0], sharpness=sharpness(img), roundtrip=roundtrip_psnr(img, native))
                rows.append(r)
                print(f"{'':28s} {name:22s} {img.shape[0]:6d} {r['sharpness']:10.1f} {r['roundtrip']:15.2f}")
    return rows


# ── figure ────────────────────────────────────────────────────────────────────
def _crop(img_u8: np.ndarray, box, out: int) -> np.ndarray:
    H, W = img_u8.shape[:2]; x0, y0, x1, y1 = box
    c = img_u8[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)]
    return np.asarray(Image.fromarray(c).resize((out, out), Image.NEAREST))   # NEAREST: show the pixels as they are


def panel(tok, samples, scale: int, latent_modes=("nearest", "bilinear", "bicubic"), pixel_modes=("bicubic", "lanczos"),
          zoom_out: int = 256, save_path: str = None, dpi: int = 110):
    """Two rows per sample: the full frame per route, then the zoom box per route (nearest-
    neighbour enlarged so each route's own pixels are visible). `samples` = [(title, lat, box)]."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    per = [(title, variants(tok, lat, scale, latent_modes, pixel_modes), box) for title, lat, box in samples]
    names = list(per[0][1]); ncol = len(names)
    fig, axes = plt.subplots(2 * len(per), ncol, figsize=(2.3 * ncol, 2.45 * 2 * len(per)), squeeze=False)
    for r, (title, v, box) in enumerate(per):
        for c, name in enumerate(names):
            img = v[name]; H = img.shape[0]
            ax = axes[2 * r][c]; ax.imshow(img, interpolation="none")
            x0, y0, x1, y1 = box
            ax.add_patch(Rectangle((x0 * H, y0 * H), (x1 - x0) * H, (y1 - y0) * H, fill=False, ec="#00E5FF", lw=1.2))
            ax.set_title(f"{name}\n{H}x{H}", fontsize=8)
            axz = axes[2 * r + 1][c]; axz.imshow(_crop(img, box, zoom_out), interpolation="none")
            if c == 0:
                ax.set_ylabel(title, fontsize=8); axz.set_ylabel("zoom", fontsize=8)
            for a in (ax, axz):
                a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"Same latent, decoded natively vs upscaled: pixel-space routes vs LATENT-upsampled decodes (x{scale})", fontsize=10)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True); fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()
    return fig


# ── a generated latent, to check the effect off the GT manifold ──────────────
@torch.no_grad()
def generate_one(run_dir: str, device, ckpt: str = "latest", group: str = None, seed: int = 0):
    """One DDIM-20 generation from a run: the first formatted test pair whose draw is
    reveal-anchored with no agent on screen (falls back to the first pair). Returns
    (gt_latent (1,4,32,32), generated_latent (1,4,32,32), label)."""
    import random
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from far.analysis.thor_rollout import build_model
    model, cfg = build_model(run_dir, ckpt, device)
    model.core.diffusion = instantiate(cfg.model.diffusion, timestep_respacing="ddim20")
    group = group or str(cfg.datasets[0])
    ds = instantiate(cfg[group].test); torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    b = next(iter(DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)))
    g = lambda k: b[k].to(device) if b.get(k) is not None else None
    cst = g("context_agent_state"); anch = b.get("event_anchored")
    keep = 0
    if cst is not None and anch is not None:
        cand = torch.nonzero(anch.bool().to(device) & (cst[:, -1, 0] <= 0)).flatten()
        keep = int(cand[0]) if len(cand) else 0
    sl = slice(keep, keep + 1)
    act, strd = g("goal_poses_nl")[sl], g("goal_times_nl")[sl]
    ai, ast = g("action_interact"), g("action_state")
    torch.manual_seed(seed)
    x0 = model.core.sample_latent(
        g("context_frames")[sl], act[:, 0], strd[:, 0], num_goals=1,
        context_poses=g("context_poses")[sl], context_times=g("context_times")[sl], context_mask=g("context_mask")[sl],
        context_keys=None if g("context_keys") is None else g("context_keys")[sl],
        action_interact=None if ai is None else ai[sl, :1], action_state=None if ast is None else ast[sl, :1],
        use_ddim=True, context_agent_state=None if cst is None else cst[sl]).float().cpu()
    gt = b["goal_frames"][sl][:, 0].float()
    return gt, x0, f"{os.path.basename(run_dir)[:40]} @ {ckpt}"


# ── pixel-space super-resolution: SwinIR (released weights, vendored network) ─
# name -> (weight file, task, scale, large?). Classical = PSNR-trained on bicubic
# degradation (faithful, mild); real-SR = BSRGAN-style degradations, PSNR flavour
# (faithful) or GAN flavour (sharper, may add texture); L = the large real-SR model.
SWINIR_MODELS = {
    "classical x2":  ("001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth",          "classical_sr", 2, False),
    "classical x4":  ("001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth",          "classical_sr", 4, False),
    "real PSNR x2":  ("003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_PSNR.pth",     "real_sr",      2, False),
    "real GAN x2":   ("003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth",      "real_sr",      2, False),
    "real PSNR x4":  ("003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_PSNR.pth",     "real_sr",      4, False),
    "real GAN x4":   ("003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth",      "real_sr",      4, False),
    "real-L GAN x4": ("003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",  "real_sr",      4, True),
}
SWINIR_URL = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"


def load_swinir(name: str, device, weights_dir: str = None):
    """Build the SwinIR net for `name` (a `SWINIR_MODELS` key) exactly as the official test
    script does, load the released weights (downloaded into `weights_dir` if missing)."""
    import urllib.request
    from far.constants import ROOT
    from far.third_party.network_swinir import SwinIR
    fname, task, scale, large = SWINIR_MODELS[name]
    from far.constants import CACHE_ROOT
    weights_dir = weights_dir or os.path.join(CACHE_ROOT, "swinir")
    path = os.path.join(weights_dir, fname)
    if not os.path.isfile(path):
        os.makedirs(weights_dir, exist_ok=True); urllib.request.urlretrieve(SWINIR_URL + fname, path)
    common = dict(upscale=scale, in_chans=3, window_size=8, img_range=1., mlp_ratio=2)
    if task == "classical_sr":
        net = SwinIR(img_size=64, depths=[6] * 6, embed_dim=180, num_heads=[6] * 6, upsampler="pixelshuffle",
                     resi_connection="1conv", **common); key = "params"
    elif not large:
        net = SwinIR(img_size=64, depths=[6] * 6, embed_dim=180, num_heads=[6] * 6, upsampler="nearest+conv",
                     resi_connection="1conv", **common); key = "params_ema"
    else:
        net = SwinIR(img_size=64, depths=[6] * 9, embed_dim=240, num_heads=[8] * 9, upsampler="nearest+conv",
                     resi_connection="3conv", **common); key = "params_ema"
    sd = torch.load(path, map_location="cpu", weights_only=False)
    net.load_state_dict(sd[key] if key in sd else sd, strict=True)
    return net.to(device).eval(), scale


@torch.no_grad()
def swinir_upscale(net, img_u8: np.ndarray, scale: int, window: int = 8) -> np.ndarray:
    """Run SwinIR on one (H, W, 3) uint8 image -> (scale*H, scale*W, 3) uint8. Mirror-pads
    to a multiple of the window size and crops back, as the official test script does."""
    dev = next(net.parameters()).device
    x = torch.as_tensor(img_u8).float().permute(2, 0, 1)[None].to(dev) / 255.0
    h, w = x.shape[-2:]
    hp, wp = (-h) % window, (-w) % window
    if hp or wp:
        x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :h + hp]
        x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :w + wp]
    y = net(x)[..., :h * scale, :w * scale]
    return (y.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


def swinir_variants(nets: dict, native: np.ndarray, scale: int) -> dict:
    """{'SwinIR <name>': image} for every loaded net whose scale matches."""
    return {f"SwinIR {n}": swinir_upscale(net, native, s) for n, (net, s) in nets.items() if s == scale}


def sr_panel(tok, samples, scale: int, nets: dict, pixel_modes=("lanczos",), zoom_out: int = 256,
             save_path: str = None, dpi: int = 110):
    """Like `panel`, but the routes are: native, pixel-space baselines, and the SwinIR nets at
    this scale (no latent routes — those failed, see the notebook's verdict)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    per = []
    for title, lat, box in samples:
        native = decode(tok, lat, 1)[0]
        v = {"native 256": native}
        for m in pixel_modes:
            v[f"pixel {m} x{scale}"] = pixel_resize(native, native.shape[0] * scale, m)
        v.update(swinir_variants(nets, native, scale))
        per.append((title, v, box))
    names = list(per[0][1]); ncol = len(names)
    fig, axes = plt.subplots(2 * len(per), ncol, figsize=(2.3 * ncol, 2.45 * 2 * len(per)), squeeze=False)
    for r, (title, v, box) in enumerate(per):
        for c, name in enumerate(names):
            img = v[name]; H = img.shape[0]; x0, y0, x1, y1 = box
            ax = axes[2 * r][c]; ax.imshow(img, interpolation="none")
            ax.add_patch(Rectangle((x0 * H, y0 * H), (x1 - x0) * H, (y1 - y0) * H, fill=False, ec="#00E5FF", lw=1.2))
            ax.set_title(f"{name}\n{H}x{H}", fontsize=8)
            axz = axes[2 * r + 1][c]; axz.imshow(_crop(img, box, zoom_out), interpolation="none")
            if c == 0:
                ax.set_ylabel(title, fontsize=8); axz.set_ylabel("zoom", fontsize=8)
            for a in (ax, axz):
                a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"Pixel-space super-resolution of the native decode: Lanczos vs SwinIR (x{scale})", fontsize=10)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True); fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()
    return per


def sr_metrics_table(per):
    """Sharpness + round-trip PSNR for the routes in a `sr_panel` result."""
    print(f"{'sample':28s} {'route':24s} {'size':>6s} {'sharpness':>10s} {'roundtrip PSNR':>15s}")
    for title, v, _ in per:
        native = v["native 256"]
        print(f"{title[:28]:28s} {'native':24s} {native.shape[0]:6d} {sharpness(native):10.1f} {'-':>15s}")
        for name, img in v.items():
            if name != "native 256":
                print(f"{'':28s} {name:24s} {img.shape[0]:6d} {sharpness(img):10.1f} {roundtrip_psnr(img, native):15.2f}")


# ── upscaling decoded frame batches / a drop-in decoder for the video writers ─
@torch.no_grad()
def sr_tensor(net, imgs: torch.Tensor, scale: int, pbar=None) -> torch.Tensor:
    """SwinIR on a (N, 3, H, W) float [0, 1] batch (the `Decoder` output format) ->
    (N, 3, scale*H, scale*W) float [0, 1] on CPU, frame by frame. `pbar`: an
    optional tqdm-like object advanced by one per frame, so a caller can run ONE
    bar across several batches (GT + every arm); None = silent (`SRDecoder`)."""
    out = []
    for im in imgs:
        u8 = (im.permute(1, 2, 0).cpu().float().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
        out.append(torch.from_numpy(swinir_upscale(net, u8, scale)).permute(2, 0, 1).float() / 255.0)
        if pbar is not None:
            pbar.update(1)
    return torch.stack(out)


class SRDecoder:
    """`decode` (latents -> (N, 3, H, W) [0, 1]) followed by SwinIR: a drop-in for
    `thor_rollout.Decoder` / `ss_rollout.Decoder` so the video writers' context thumbnails
    (`decode(lat)`) come out at the same size as the upscaled GT / generated frames."""
    def __init__(self, decode, net, scale: int):
        self.decode, self.net, self.scale = decode, net, scale

    def __call__(self, lat):
        return sr_tensor(self.net, self.decode(lat), self.scale)
