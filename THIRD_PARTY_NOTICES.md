# Third-party notices

FAR is released under the Creative Commons Attribution-NonCommercial 4.0
International license (see `LICENSE`). It builds on the following third-party
code, weights and data; their licenses apply to the corresponding parts.

## Code adapted into this repository

| Component | Where | Upstream | License |
|---|---|---|---|
| Navigation World Models (NWM) — CDiT generator, diffusion utilities, distributed / experiment helpers, pose transforms | `src/far/generators/cdit_mod_*.py`, `src/far/generators/diffusion/`, `src/far/distributed.py`, `src/far/experiment.py`, `src/far/data/transforms.py` | https://github.com/facebookresearch/nwm | CC BY-NC 4.0, Copyright (c) Meta Platforms, Inc. and affiliates |
| DiT (Scalable Diffusion Models with Transformers), the ancestor of NWM's generator and diffusion code | same files | https://github.com/facebookresearch/DiT | CC BY-NC 4.0, Copyright (c) Meta Platforms, Inc. and affiliates |
| OpenAI guided-diffusion / GLIDE / improved-diffusion (the diffusion utilities NWM modified) | `src/far/generators/diffusion/` | https://github.com/openai/guided-diffusion, https://github.com/openai/glide-text2im | MIT |
| MAE (masked autoencoders; ViT blocks and positional-embedding helpers) | `src/far/tokenizers/vit_vae.py`, generator ViT blocks | https://github.com/facebookresearch/mae | CC BY-NC 4.0, Copyright (c) Meta Platforms, Inc. and affiliates |
| taming-transformers (VQGAN) | `src/far/tokenizers/vit_vae.py` | https://github.com/CompVis/taming-transformers | MIT, Copyright (c) 2020 Patrick Esser, Robin Rombach, Björn Ommer |
| rotary-embedding-torch | `src/far/tokenizers/rotary_embedding.py` | https://github.com/lucidrains/rotary-embedding-torch | MIT, Copyright (c) 2021 Phil Wang |
| SwinIR (vendored network definition, unchanged apart from the removed `__main__` smoke test) | `src/far/third_party/network_swinir.py`, license text in `src/far/third_party/LICENSE-SwinIR` | https://github.com/JingyunLiang/SwinIR | Apache License 2.0, Copyright (c) Jingyun Liang / Ze Liu |
| WorldMem — the FOV-overlap greedy retrieval used as a baseline is an independent re-implementation of `_generate_condition_indices`; no WorldMem code is copied | `src/far/memories/worldmem.py` | https://github.com/xizaoqu/WorldMem | S-Lab License 1.0 (non-commercial) |

## Pretrained weights downloaded at run time

| Weights | Used for | Source | License |
|---|---|---|---|
| SDXL VAE (fp16 fix) | latent tokenizer for the AI2-THOR and SoundSpaces corpora | https://huggingface.co/madebyollin/sdxl-vae-fp16-fix | MIT |
| SD VAE ft-EMA | alternative latent tokenizer | https://huggingface.co/stabilityai/sd-vae-ft-ema | MIT |
| Oasis 500M ViT-VAE | latent tokenizer for the LoopNav corpus | https://github.com/etched-ai/open-oasis | MIT |
| SwinIR released weights | optional super-resolution of demo videos | https://github.com/JingyunLiang/SwinIR/releases | Apache License 2.0 |
| LPIPS (AlexNet backbone), DreamSim, DINO | evaluation metrics and retriever encoders (pip dependencies) | https://github.com/richzhang/PerceptualSimilarity, https://github.com/ssundaram21/dreamsim, https://github.com/facebookresearch/dino | BSD-2-Clause, MIT, Apache License 2.0 |

## Datasets

| Dataset | Source | Terms |
|---|---|---|
| LoopNav (Minecraft loop-navigation videos) | https://huggingface.co/datasets/kevinLian/LoopNav | see the dataset card |
| SoundSpaces 2.0 (audio-visual navigation on Matterport3D scenes) | https://github.com/facebookresearch/sound-spaces | SoundSpaces: CC BY 4.0; Matterport3D scenes: the Matterport3D terms of use |
| AI2-THOR (iTHOR scenes) | https://ai2thor.allenai.org | Apache License 2.0 |

The full text of the licenses named above is available at the linked
repositories; the MIT, BSD and Apache texts are standard and unmodified.
