"""Upload the built web app (app/dist) to the public Hugging Face Static Space.

Usage: build first (`npm run build` in app/), then `.venv/Scripts/python deploy_space.py`.
Needs a Write token saved by `.venv/Scripts/hf.exe auth login`. Run it again after any app change:
each run makes one commit that adds the new build and deletes files the build no longer has
(Vite renames assets on every content change, so old files would pile up otherwise).
"""

from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

SPACE_ID = "VictorCKW/bird-species-classifier"
DIST = Path(__file__).parent / "app" / "dist"

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


def main() -> None:
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
