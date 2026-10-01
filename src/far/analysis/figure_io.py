"""Small I/O helpers for the paper-figure notebooks (figures.ipynb).

`save_frames` writes the individual frames that a composite figure was
assembled from (current / GT / per-arm top-1 context / generation) as
stand-alone PNGs, so they can be re-laid-out in the paper or slides without
re-running the models.
"""
import os
import re

import numpy as np
from PIL import Image


def slug(label: str) -> str:
    """Filesystem-safe arm name: 'FAR -- Multi-Cue' -> 'far_multi_cue'."""
    s = re.sub(r"[^0-9a-zA-Z]+", "_", label.strip().lower()).strip("_")
    return s or "arm"


def _to_pil(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img
    a = np.asarray(img)
    if a.ndim == 3 and a.shape[0] in (1, 3, 4) and a.shape[-1] not in (1, 3, 4):
        a = np.transpose(a, (1, 2, 0))                      # CHW -> HWC
    if a.dtype != np.uint8:
        a = (np.clip(a, 0.0, 1.0) * 255).round().astype(np.uint8)
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]
    return Image.fromarray(a)


def save_frames(out_dir: str, frames: dict) -> list:
    """Write `{name: image}` as `<out_dir>/<name>.png`. Images may be HWC/CHW
    uint8 or float-in-[0,1] arrays / tensors, or PIL images. Returns the paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, img in frames.items():
        if hasattr(img, "detach"):                          # torch tensor
            img = img.detach().cpu().numpy()
        p = os.path.join(out_dir, f"{name}.png")
        _to_pil(img).save(p)
        paths.append(p)
    print(f"saved {len(paths)} frames -> {out_dir}/")
    return paths


def render_with_inset(img, cue, rect=(0.62, 0.04, 0.36, 0.30), size=512,
                      cmap="magma", edge="white", edge_lw=0.8) -> np.ndarray:
    """Composite a frame with the binaural-cue inset EXACTLY as the SoundSpaces
    figure draws it (log-mel of the L channel, `magma`, origin lower, white
    spines, bottom-right `rect` in axes fractions) and return it as an HWC
    uint8 array of `size` x `size` px. Rendered off-screen (Agg), so it never
    touches the notebook's inline backend. `cue` may be (2, n_mel, n_t) or
    (n_mel, n_t); a leading channel axis selects channel 0."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    if hasattr(cue, "detach"):
        cue = cue.detach().cpu().numpy()
    cue = np.asarray(cue)
    if cue.ndim == 3:
        cue = cue[0]
    frame = np.asarray(_to_pil(img).convert("RGB"))
    dpi = 100
    fig = Figure(figsize=(size / dpi, size / dpi), dpi=dpi)
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(frame, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ins = ax.inset_axes(list(rect))
    ins.imshow(cue, origin="lower", aspect="auto", cmap=cmap)
    ins.set_xticks([]); ins.set_yticks([])
    for s in ins.spines.values():
        s.set_edgecolor(edge); s.set_linewidth(edge_lw * size / 256)
    fig.canvas.draw()
    out = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    return out
