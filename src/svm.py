"""Frozen features + SVM: SVMs decide the species from a frozen pretrained network's features.

run.py calls fit_svm(cfg) in place of train.fit(cfg) when cfg.svm_kernel is set. One run:
    1. extract_features - the frozen ImageNet network turns every photo into a feature list once:
       512 numbers (ResNet-18), 2,048 (ResNet-50) or 768 (ConvNeXt-Tiny), cached in features/
    2. for every C in cfg.svm_c_values: standardise the features, fit one SVM per species, score the validation photos
    3. keep the C with the best validation Top-1, then score the test photos once
    4. write results/<name>.json in the same layout as train.py, with "c_search" in place of the per-epoch "history"
    5. save the chosen SVM to checkpoints/<name>.joblib, so the Gradio app can reuse it
There are no epochs: an SVM solves its cost exactly in one go, because that cost is convex (W4 s37, W5 s42).
"""

import time
from dataclasses import asdict

import joblib
import numpy as np
import torch
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn

from src.config import RunConfig
from src.data import make_loaders
from src.metrics import score
from src.models import build_model, set_head
from src.train import CHECKPOINT_DIR, RESULTS_DIR, ROOT, autocast, set_seed, tidy_json, to_device

# The features of every photo, saved once so later runs skip the network (gitignored).
FEATURES_DIR = ROOT / "features"


@torch.no_grad()  # nothing learns here, so no gradients are needed
def features_of(network: nn.Module, loader, cfg: RunConfig, device) -> dict[str, np.ndarray]:
    """Run every photo in the loader through the frozen network.
    Returns "x" (one row of features per photo) and "y" (the species of each row)."""
    network.eval()  # batch normalisation uses its stored ImageNet averages
    x, y = [], []
    for images, labels in loader:
        images, labels = to_device(images, labels, device)
        with autocast(cfg, device):
            x.append(network(images).float().cpu().numpy())  # features, copied back from the GPU as float32
        y.append(labels.cpu().numpy())
    return {"x": np.concatenate(x), "y": np.concatenate(y)}


def extract_features(cfg: RunConfig) -> dict[str, np.ndarray]:
    """The features and species numbers of the train, val and test photos: x_train, y_train, x_val, ...
    The first run computes them on the GPU (a few minutes); later runs read the cache file in a second."""
    # The seed decides which photos are in train and val; the size changes the features.
    cache = FEATURES_DIR / f"{cfg.arch}-{cfg.image_size}-seed{cfg.seed}.npz"
    if cache.exists():
        return dict(np.load(cache))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = build_model(cfg).to(device, memory_format=torch.channels_last)  # the frozen ImageNet network
    # Remove the decider: the network now outputs its features, not 200 scores. For ResNet these are the
    # averaged-and-flattened layer4 outputs; for ConvNeXt they come after its LayerNorm and Flatten.
    set_head(network, nn.Identity())

    data = {}
    for split, loader in zip(("train", "val", "test"), make_loaders(cfg)):
        out = features_of(network, loader, cfg, device)
        data[f"x_{split}"], data[f"y_{split}"] = out["x"], out["y"]
    FEATURES_DIR.mkdir(exist_ok=True)
    np.savez(cache, **data)
    return data


def build_svm(kernel: str, c: float) -> Pipeline:
    """A two-step decider: put every feature on the same ruler, then fence off each species.

    StandardScaler - each feature gets mean 0 and spread 1 (W5 s80). It learns the mean and spread
                     from the training photos only, so validation and test photos leak nothing into training.
    SVC            - a large-margin classifier (W4 s3-6). C sets how strict it is about training photos on the
                     wrong side of the fence (W4 s18-21). gamma="scale" sets the RBF bubble size from the data:
                     after standardising it is 1/(number of features), which is 1/(2 sigma^2) in the slides' kernel (W4 s29).
    OneVsRest      - an SVM separates only 2 groups, so we fit 200 of them, "this species vs all others",
                     and pick the species whose SVM is most sure (one-vs-all, W3). n_jobs=-1 uses every CPU core.
    """
    return make_pipeline(
        StandardScaler(),
        OneVsRestClassifier(SVC(kernel=kernel, C=c, gamma="scale"), n_jobs=-1),
    )


def summarise(y_true: np.ndarray, y_pred: np.ndarray, per_class: bool) -> dict:
    """Both scores, rounded, in the layout train.py uses (no loss: an SVM has no probabilities to take a log of)."""
    s = score(y_true.tolist(), y_pred.tolist())
    out = {"top1": round(s["top1"], 4), "mean_per_class": round(s["mean_per_class"], 4)}
    if per_class:
        out["per_class"] = s["per_class"]
    return out


def fit_svm(cfg: RunConfig) -> dict:
    """Choose C on the validation photos, score the test photos once with that C, and save the results."""
    if not cfg.frozen or not cfg.pretrained or cfg.augment:
        raise ValueError("The SVM needs fixed features: a pretrained, frozen network (frozen=True), no augmentation.")
    set_seed(cfg.seed)
    start = time.time()
    data = extract_features(cfg)

    c_search, best = [], None  # best = (C, validation Top-1, fitted SVM)
    for c in cfg.svm_c_values:
        fit_start = time.time()
        svm = build_svm(cfg.svm_kernel, c).fit(data["x_train"], data["y_train"])  # learns from training photos only
        train_top1 = score(data["y_train"].tolist(), svm.predict(data["x_train"]).tolist())["top1"]  # homework score
        val = summarise(data["y_val"], svm.predict(data["x_val"]), per_class=False)  # practice-exam score
        c_search.append({
            "c": c,
            "train_top1": round(train_top1, 4),
            "val_top1": val["top1"],
            "val_mean_per_class": val["mean_per_class"],
            "seconds": round(time.time() - fit_start, 1),
        })
        print(f"[{cfg.name}] C={c:g}  train acc {train_top1:.3f}  val acc {val['top1']:.3f}  "
              f"({c_search[-1]['seconds']} s)", flush=True)
        if best is None or val["top1"] > best[1]:  # on a tie the smaller, less strict C stays
            best = (c, val["top1"], svm)

    # Final scores. The test photos are used here only, once per run, never to choose C.
    best_c, _, svm = best
    result = {
        "config": asdict(cfg),
        "best_c": best_c,
        "minutes": round((time.time() - start) / 60, 1),
        "val": summarise(data["y_val"], svm.predict(data["x_val"]), per_class=False),
        "test": summarise(data["y_test"], svm.predict(data["x_test"]), per_class=True),
        "c_search": c_search,
    }
    print(f"[{cfg.name}] final (C={best_c:g}): val top-1 {result['val']['top1']:.3f}  "
          f"test top-1 {result['test']['top1']:.3f}  "
          f"test mean per-class {result['test']['mean_per_class']:.3f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"{cfg.name}.json").write_text(tidy_json(result) + "\n")
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    joblib.dump(svm, CHECKPOINT_DIR / f"{cfg.name}.joblib")  # the scaler + 200 SVMs in one file, for the app
    return result
