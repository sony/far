import os
from copy import deepcopy
from hydra.utils import instantiate


def instantiate_with_optional_ckpt(cfg, **kwargs):
    cfg = deepcopy(cfg)
    ckpt_path = cfg.pop("ckpt_path", None)

    obj = instantiate(cfg, **kwargs)

    if ckpt_path is not None:
        # Only local-looking paths are checked: a Hugging Face model id (e.g. the SDXL VAE's
        # "madebyollin/sdxl-vae-fp16-fix") has no extension and is resolved by load_weights.
        p = str(ckpt_path)
        looks_local = os.path.isabs(p) or os.path.splitext(p)[1] != ""
        if looks_local and not os.path.isfile(p):
            raise FileNotFoundError(
                f"checkpoint not found: {ckpt_path} (cwd {os.getcwd()}). Released checkpoints, including the "
                f"pre-trained retrievers the FAR launchers start from, download with: "
                f"bash bash_scripts/download.sh checkpoints <corpus>")
        obj.load_weights(ckpt_path)

    return obj