"""Download the released FAR checkpoint bundles (and, with --data, the latent corpora) from Hugging Face.

Checkpoints go to `$FAR_MODELS_ROOT` (default `<repo>/models`). The HF repo mirrors that
directory (`<corpus>/<arm>/config.yaml` + `<corpus>/<arm>/checkpoints/<step>.pth.tar`, plus the
tokenizer and detector weights), so files land exactly where the configs, launchers and
notebooks expect them. `bundles.json` in the repo lists every bundle (corpus, arm, step, role,
size, sha256) and is what this script selects from.

    python scripts/download_release.py                        # every released bundle + assets (~13 GB)
    python scripts/download_release.py --corpus ai2thor_v3    # one corpus (repeatable)
    python scripts/download_release.py --arm loopnav/far_multicue --arm loopnav/temporal
    python scripts/download_release.py --list                 # show what is available, download nothing
    python scripts/download_release.py --verify               # also SHA-256 check every downloaded file

`--data <tier>` additionally fetches the latent corpora of the selected corpora from the
datasets repo: tar shards are downloaded to `$FAR_RESULT_ROOT/release/downloads/`, size-checked
(SHA-256 with --verify), and extracted ONCE into the repo root (`datasets/<corpus>/latent/...`;
a `<shard>.extracted` marker skips the extraction next time); the manifests / indices /
`latent_params.json` are placed at their repo-relative paths. Tiers: `demo` = the rollout-demo
clips, `test` = the test split (a superset of demo), `train` = test + train.

    python scripts/download_release.py --data demo --corpus ai2thor_v3        # 1.1 GB
    python scripts/download_release.py --data train --corpus ai2thor_v3 --remove-shards   # 84 GB, delete the tars after extraction

The SoundSpaces corpora (`--corpus soundspaces_v1 | soundspaces_v2`) are rendered from Matterport3D
scenes and live in a GATED repo: accept the Matterport3D Terms of Use on its Hugging Face page,
then `hf auth login` before running the script. Without `--corpus`, only the ungated corpora
(AI2-THOR v3 / dyn, LoopNav) are fetched. LoopNav (`--corpus loopnav`) is the May 2026 release of
kevinLian/LoopNav, redistributed with its authors' permission; please cite their paper
(arXiv:2505.22976) when you use it.

Downloads resume and skip files already present. Pin a release with `--revision v1.0`.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

REPO_ID = "1202kbs/FAR-Checkpoints"
DATA_REPO_ID = "1202kbs/FAR-Datasets-AI2THOR"
SS_DATA_REPO_ID = "1202kbs/FAR-Datasets-SoundSpaces"            # gated: Matterport3D Terms of Use
LOOPNAV_DATA_REPO_ID = "1202kbs/FAR-Datasets-LoopNav"           # redistributed with the LoopNav authors' permission
# release corpus name -> (datasets repo, corpus name inside it = the datasets/<name>/ folder)
DATA_SOURCES = {"ai2thor_v3": (DATA_REPO_ID, "ai2thor_v3"), "ai2thor_dyn": (DATA_REPO_ID, "ai2thor_dyn"),
                "soundspaces_v1": (SS_DATA_REPO_ID, "soundspaces"), "soundspaces": (SS_DATA_REPO_ID, "soundspaces"),
                "soundspaces_v2": (SS_DATA_REPO_ID, "soundspaces_v2"), "loopnav": (LOOPNAV_DATA_REPO_ID, "loopnav")}
UNGATED_DATA_REPOS = [DATA_REPO_ID, LOOPNAV_DATA_REPO_ID]        # fetched when no --corpus is given
DATA_TIERS = {"demo": ["demo"], "test": ["test"], "train": ["test", "train"]}


def _roots():
    try:
        from far.constants import ROOT, RESULT_ROOT
    except ImportError:                                          # not installed yet: this file lives in <repo>/scripts
        ROOT = os.environ.get("FAR_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        RESULT_ROOT = os.environ.get("FAR_RESULT_ROOT") or os.path.join(ROOT, "results")
    return ROOT, RESULT_ROOT, os.environ.get("FAR_MODELS_ROOT") or os.path.join(ROOT, "models")


def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _check(p, entry, verify):
    if not os.path.isfile(p) or os.path.getsize(p) != entry["size_bytes"]:
        print(f"  PROBLEM size/missing: {p}"); return 1
    if verify and _sha256(p) != entry["sha256"]:
        print(f"  PROBLEM sha256 mismatch: {p}"); return 1
    return 0


def download_models(a, dest):
    from huggingface_hub import hf_hub_download, snapshot_download
    os.makedirs(dest, exist_ok=True)
    bf = a.bundles_file or hf_hub_download(a.repo, "bundles.json", revision=a.revision)
    bundles = [b for b in json.load(open(bf))["bundles"] if b.get("release", True)]
    want_arms = set(a.arm or []); want_corpora = set(a.corpus or []) | {x.split("/")[0] for x in want_arms}
    sel = []
    for b in bundles:
        if b["role"] == "asset":
            if not a.no_assets: sel.append(b)
            continue
        name = f"{b['corpus']}/{b['arm']}"
        if want_arms and name not in want_arms: continue
        if not want_arms and want_corpora and b["corpus"] not in want_corpora: continue
        sel.append(b)
    if not sel:
        sys.exit("nothing selected — check --corpus / --arm against --list")
    print(f"{'bundle':32s} {'step':8s} {'role':9s} {'size':>8s}")
    for b in sel:
        name = f"{b['corpus']}/{b['arm']}" if b["corpus"] else b["path"]
        print(f"  {name:30s} {str(b['step'] or '-'):8s} {b['role']:9s} {b['size_bytes'] / 2**30:6.2f} GB")
    print(f"  {len(sel)} entries, {sum(b['size_bytes'] for b in sel) / 2**30:.1f} GB -> {dest}")
    if a.list:
        return 0
    patterns = ["bundles.json"]
    for b in sel:
        patterns.append(b["path"])
        if b["corpus"]: patterns.append(f"{b['corpus']}/{b['arm']}/config.yaml")
    snapshot_download(a.repo, revision=a.revision, local_dir=dest, allow_patterns=patterns)
    bad = 0
    for b in sel:
        bad += _check(os.path.join(dest, b["path"]), b, a.verify)
        if b["corpus"] and not os.path.isfile(os.path.join(dest, b["corpus"], b["arm"], "config.yaml")):
            print(f"  PROBLEM missing config.yaml for {b['corpus']}/{b['arm']}"); bad += 1
    return bad


def download_data(a, root, result_root):
    named = set(a.corpus or []) | {x.split("/")[0] for x in (a.arm or [])}
    groups = {}
    if named:
        for c in sorted(named):
            if c not in DATA_SOURCES:
                print(f"  no data release for corpus '{c}' (skipped)"); continue
            repo, data_corpus = DATA_SOURCES[c]
            groups.setdefault(a.data_repo or repo, set()).add(data_corpus)
    else:
        for repo in ([a.data_repo] if a.data_repo else UNGATED_DATA_REPOS):
            groups[repo] = None                                  # every corpus of the ungated repos
    return sum(_download_repo(a, repo, corpora, root, result_root) for repo, corpora in groups.items())


def _download_repo(a, repo, corpora, root, result_root):
    from huggingface_hub import hf_hub_download, snapshot_download
    dl = os.path.join(result_root, "release", "downloads", repo.split("/")[-1]); os.makedirs(dl, exist_ok=True)
    try:
        listing = json.load(open(hf_hub_download(repo, "datasets.json", repo_type="dataset", revision=a.data_revision)))
    except Exception as e:                                       # gated / unauthenticated / missing
        print(f"  cannot read {repo}: {type(e).__name__}. " + ("Accept the Matterport3D Terms of Use on "
              f"https://huggingface.co/datasets/{repo}, then `hf auth login`." if repo == SS_DATA_REPO_ID else ""))
        return 1
    corpora = corpora or {s["corpus"] for s in listing["shards"]}
    tiers = DATA_TIERS[a.data]
    shards = [s for s in listing["shards"] if s["corpus"] in corpora and s["tier"] in tiers]
    files = [f for f in listing["files"] if f["corpus"] in corpora]
    if not shards:
        print(f"  no data shards in {repo} for corpora {sorted(corpora)} at tier '{a.data}'"); return 1
    print(f"{repo}\n{'shard':34s} {'tier':6s} {'episodes':>8s} {'size':>8s}")
    for s in shards:
        print(f"  {s['path']:32s} {s['tier']:6s} {s['episodes']:8d} {s['size_bytes'] / 2**30:6.2f} GB")
    print(f"  {len(shards)} shards, {sum(s['size_bytes'] for s in shards) / 2**30:.1f} GB -> {dl}, extracted into {root}")
    if a.list:
        return 0
    snapshot_download(repo, repo_type="dataset", revision=a.data_revision, local_dir=dl,
                      allow_patterns=[s["path"] for s in shards] + [f["path"] for f in files])
    bad = 0
    for f in files:                                              # results/... -> RESULT_ROOT, datasets/... -> ROOT
        src = os.path.join(dl, f["path"]); bad += _check(src, f, a.verify)
        rel = f["path"].split("/", 1)
        dst = os.path.join(result_root, rel[1]) if rel[0] == "results" else os.path.join(root, f["path"])
        os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy2(src, dst)
    for s in shards:
        p = os.path.join(dl, s["path"]); marker = p + ".extracted"
        if os.path.isfile(marker) and open(marker).read().strip() == s["sha256"]:
            print(f"  already extracted: {s['path']}"); continue
        if _check(p, s, a.verify):
            bad += 1; continue
        print(f"  extracting {s['path']} ({s['episodes']} episodes) -> {root}", flush=True)
        subprocess.run(["tar", "-xf", p, "-C", root, "--keep-directory-symlink"], check=True)
        open(marker, "w").write(s["sha256"] + "\n")
        if a.remove_shards:
            os.remove(p)
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=REPO_ID)
    ap.add_argument("--revision", default=None, help="checkpoint repo revision / tag (default: latest)")
    ap.add_argument("--corpus", action="append", help="loopnav | soundspaces_v1 | soundspaces_v2 | ai2thor_v3 | ai2thor_dyn (repeatable)")
    ap.add_argument("--arm", action="append", help="<corpus>/<arm> (repeatable); implies that corpus")
    ap.add_argument("--no-assets", action="store_true", help="skip the tokenizer / detector weights")
    ap.add_argument("--no-models", action="store_true", help="skip the checkpoint bundles (data only)")
    ap.add_argument("--list", action="store_true"); ap.add_argument("--verify", action="store_true")
    ap.add_argument("--bundles-file", default=None, help="use a local bundles.json instead of fetching it (testing)")
    ap.add_argument("--data", choices=sorted(DATA_TIERS), default=None, help="also fetch the latent corpora at this tier")
    ap.add_argument("--data-repo", default=None, help="override the datasets repo of every selected corpus")
    ap.add_argument("--data-revision", default=None, help="datasets repo revision / tag (default: latest)")
    ap.add_argument("--remove-shards", action="store_true", help="delete each tar shard once it is extracted")
    a = ap.parse_args()
    root, result_root, models_root = _roots()
    bad = 0
    if not a.no_models:
        bad += download_models(a, models_root)
    if a.data:
        bad += download_data(a, root, result_root)
    if a.list:
        return
    print("done" if not bad else f"done with {bad} problems", "(sha256 verified)" if a.verify and not bad else "")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
