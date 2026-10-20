"""Upload the web app to the public Hugging Face Static Space, and its networks to a public model repo.

Usage:
    .venv/Scripts/python deploy_space.py --models   # upload checkpoints/*.onnx to the model repo (after export_onnx.py --all)
    .venv/Scripts/python deploy_space.py            # upload app/dist (build first: `npm run build` in app/)
Needs a Write token saved by `.venv/Scripts/hf.exe auth login`.

Why two repos: a Space holds at most 1 GB, and the 31 ONNX files are 1.4 GB. The networks live in a model repo
instead, and the site downloads each one from there when the visitor picks it (MODEL_BASE in make_app_data.py).
Each app upload makes one commit that adds the new build and deletes files the build no longer has
(Vite renames assets on every content change, so old files would pile up otherwise).
"""

import argparse

from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

SPACE_ID = "VictorCKW/bird-species-classifier"
MODEL_REPO_ID = "VictorCKW/bird-species-models"  # the ONNX files; make_app_data.py points the app here
ROOT = Path(__file__).parent
DIST = ROOT / "app" / "dist"
CHECKPOINTS = ROOT / "checkpoints"

# The model repo's card: what the files are and how the site uses them.
MODEL_README = """---
license: mit
tags:
- image-classification
- onnx
---

ONNX files for the bird species classifier at https://huggingface.co/spaces/VictorCKW/bird-species-classifier
(COS30082 Assignment 1, CUB-200, 200 species). One file per training run; the final model is
`7-final-convnext.onnx` (ConvNeXt-Tiny). Input: one photo, shape (1, 3, 448, 448), pixel values 0-1.
Output: 200 probabilities, or 200 SVM scores for the `1-frozen-svm-*` runs. Weights are stored as float16.
"""

# The Space's README front matter is its settings: `sdk: static` serves the files as they are,
# with no build on Hugging Face's side (https://huggingface.co/docs/hub/spaces-sdks-static).
README = """---
title: Bird Species Classifier
emoji: 🐦
colorFrom: gray
colorTo: yellow
sdk: static
app_file: index.html
pinned: false
short_description: CNN that names 200 bird species, run in your browser
---

COS30082 Assignment 1. The final model, a ConvNeXt-Tiny CNN, runs in the visitor's browser
through onnxruntime-web. Every other run of the experiments can be picked too, down to the baselines.
"""


def upload_models() -> None:
    """Upload every checkpoints/<run>.onnx to the public model repo, in one commit, and delete remote .onnx files
    with no local copy (renamed or removed runs). Files already there with the same content are skipped by the Hub,
    so a rerun only sends what changed."""
    files = sorted(CHECKPOINTS.glob("*.onnx"))
    if not files:
        raise SystemExit("checkpoints/ has no .onnx files: run export_onnx.py --all first.")
    api = HfApi()
    api.create_repo(MODEL_REPO_ID, repo_type="model", private=False, exist_ok=True)  # public: the site downloads them
    ops = [CommitOperationAdd(path_in_repo=f.name, path_or_fileobj=str(f)) for f in files]
    ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=MODEL_README.encode("utf-8")))
    local = {f.name for f in files}
    ops += [CommitOperationDelete(path_in_repo=name) for name in api.list_repo_files(MODEL_REPO_ID, repo_type="model")
            if name.endswith(".onnx") and name not in local]
    info = api.create_commit(MODEL_REPO_ID, operations=ops, repo_type="model", commit_message="Upload ONNX networks")
    print(f"Committed {len(files)} networks:", info.commit_url)


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload the app to the Space, or the networks to the model repo.")
    parser.add_argument("--models", action="store_true", help="upload checkpoints/*.onnx to the model repo instead")
    if parser.parse_args().models:
        upload_models()
        return
    if not (DIST / "index.html").exists():
        raise SystemExit("app/dist/index.html is missing: run `npm run build` in app/ first.")
    api = HfApi()
    # Create the Space once; exist_ok makes later runs reuse it. private=False: the brief needs a public site.
    api.create_repo(SPACE_ID, repo_type="space", space_sdk="static", private=False, exist_ok=True)

    # Every built file, keyed by its path inside the Space (forward slashes, relative to dist/).
    local = {p.relative_to(DIST).as_posix(): p for p in DIST.rglob("*") if p.is_file()}
    ops: list[CommitOperationAdd | CommitOperationDelete] = [
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path)) for name, path in local.items()
    ]
    ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=README.encode("utf-8")))

    # Delete remote files from older builds, but keep the two files Hugging Face itself manages.
    keep = set(local) | {"README.md", ".gitattributes"}
    remote = api.list_repo_files(SPACE_ID, repo_type="space")
    ops += [CommitOperationDelete(path_in_repo=name) for name in remote if name not in keep]

    # One commit; large files (the ONNX networks, the wasm engine) go through Hugging Face's large-file storage automatically.
    info = api.create_commit(SPACE_ID, operations=ops, repo_type="space", commit_message="Deploy app/dist")
    print("Committed:", info.commit_url)
    print("Site:", f"https://huggingface.co/spaces/{SPACE_ID}")


if __name__ == "__main__":
    main()
