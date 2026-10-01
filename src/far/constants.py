"""Repository roots.

Derived from this file's location (``src/far/constants.py`` sits two levels
below the repo root), which is right for the editable install the README asks
for (``pip install -e .``). Override any of them with ``FAR_ROOT``,
``FAR_CONFIG_ROOT`` and ``FAR_RESULT_ROOT`` when the configs or results live
elsewhere.
"""
import os

ROOT = os.environ.get("FAR_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_ROOT = os.environ.get("FAR_CONFIG_ROOT") or os.path.join(ROOT, "configs")
RESULT_ROOT = os.environ.get("FAR_RESULT_ROOT") or os.path.join(ROOT, "results")

# Run configs saved before the package was renamed interpolate
# ``${oc.env:PMEM_MODELS_ROOT}`` / ``${oc.env:PMEM_RESULT_ROOT}``; current code
# uses the ``FAR_*`` names. Keep both defined and equal so every checkpoint
# bundle loads regardless of which name its config carries.
for _old, _new, _default in (("PMEM_MODELS_ROOT", "FAR_MODELS_ROOT", os.path.join(ROOT, "models")),
                             ("PMEM_RESULT_ROOT", "FAR_RESULT_ROOT", RESULT_ROOT)):
    _v = os.environ.get(_new) or os.environ.get(_old) or _default
    os.environ.setdefault(_new, _v)
    os.environ.setdefault(_old, _v)
del _old, _new, _default, _v

MODELS_ROOT = os.environ["FAR_MODELS_ROOT"]
# Library downloads (DreamSim + torch hub, which also serves LPIPS / FID weights; SwinIR;
# fonts) live under models/cache/ so the released bundles and assets are the only other
# things in models/. `cache/` is disposable: every file in it is re-fetched on demand.
CACHE_ROOT = os.path.join(MODELS_ROOT, "cache")
os.environ.setdefault("TORCH_HOME", CACHE_ROOT)              # torch hub dir = $TORCH_HOME/hub


def hub_cache_dir():
    """`models/cache/hub`: DreamSim's cache_dir, which it also makes the torch hub dir."""
    return os.path.join(CACHE_ROOT, "hub")
