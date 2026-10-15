"""Builds the data files the web app (app/) needs, from the dataset, the result files and the ONNX files.

Usage, from the project root (after export_onnx.py):
    .venv/Scripts/python make_app_data.py

Writes:
    app/src/data/species.json    - the 200 species names, index = species number (tracked)
    app/src/data/results.json    - scores, settings and learning curves of every run in results/ (tracked)
    app/src/data/examples.json   - the example photos: file name and true species number (tracked)
    app/src/data/confidence.json - how often the final model is right when it is unsure and when it is sure (tracked)
    app/public/examples/*.jpg    - 8 random test photos, copied unchanged (tracked)
    app/public/models/*.onnx     - copies of checkpoints/8-final-*.onnx (gitignored, 104.7 MB)
The app reads every number it shows from results.json, so each score on the site traces back to a file in results/.
"""

import json
import random
import shutil
from pathlib import Path

import numpy as np

from src.data import DATA_DIR, NUM_CLASSES, read_labels
from src.train import CHECKPOINT_DIR, RESULTS_DIR

ROOT = Path(__file__).resolve().parent
APP_DIR = ROOT / "app"
EXAMPLE_COUNT = 8  # small enough to hang in one row along the bottom of the first screen
EXAMPLE_SEED = 42  # examples are drawn at random, not hand-picked, so they show the model as it is
MODELS = ("8-final-resnet50", "8-final-convnext")  # the two members of 8-final-ensemble
UNSURE_BELOW = 0.3  # the app calls an answer "unsure" when its top probability is under this


def confidence_check() -> dict:
    """How often 8-final-ensemble's first guess is right on the test photos, below and above UNSURE_BELOW.
    Label smoothing (0.1) trains the networks never to be fully sure, so their probabilities run low; this tells the
    visitor what a low number actually means. Reads the saved test probabilities (checkpoints/<name>-probs.npz)."""
    tables = [np.load(CHECKPOINT_DIR / f"{name}-probs.npz") for name in MODELS]
    probs = np.mean([t["test"] for t in tables], axis=0)  # the ensemble: equal-weight average, as in src/ensemble.py
    right = probs.argmax(axis=1) == tables[0]["test_labels"]
    unsure = probs.max(axis=1) < UNSURE_BELOW
    return {
        "threshold": UNSURE_BELOW,
        "unsure": {"photos": int(unsure.sum()), "right": round(float(right[unsure].mean()), 4)},
        "sure": {"photos": int((~unsure).sum()), "right": round(float(right[~unsure].mean()), 4)},
    }


def tidy_name(raw: str) -> str:
    """'Black_footed_Albatross' -> 'Black-footed Albatross', 'Black_and_white_Warbler' -> 'Black-and-white Warbler'.
    The file names lost the hyphens; in these names a lowercase word always belonged to the word before it."""
    words: list[str] = []
    for word in raw.split("_"):
        if words and word[0].islower():
            words[-1] += "-" + word  # glue to the previous word with the lost hyphen
        else:
            words.append(word)
    return " ".join(words)


def species_names() -> list[str]:
    """Species names from the training file names: 'Black_footed_Albatross_0019_416160254.jpg' -> 'Black-footed Albatross'.
    The last two underscore parts are a photo number and a Flickr id, so they are dropped."""
    names: dict[int, str] = {}
    for file_name, label in read_labels(DATA_DIR / "train.txt"):
        names.setdefault(label, tidy_name(file_name.rsplit("_", 2)[0]))
    assert len(names) == NUM_CLASSES, f"expected {NUM_CLASSES} species, found {len(names)}"
    return [names[label] for label in range(NUM_CLASSES)]


def settings(config: dict, result: dict) -> dict:
    """The training settings the app's experiments table explains, from one trained run's saved config.
    Every key is always present (None when it does not apply), so app/src/experiments.ts can type one shape."""
    return {
        "arch": config["arch"],
        "pretrained": config["pretrained"],
        "frozen": config["frozen"],
        "imageSize": config["image_size"],
        "batchSize": config["batch_size"],
        "optimizer": config["optimizer"],
        "momentum": config["momentum"],
        "lrNew": config["lr_new"],
        "lrPretrained": config["lr_pretrained"],
        "schedule": config["schedule"],
        "warmupEpochs": config["warmup_epochs"],
        "epochs": config["epochs"],
        "augment": config["augment"],
        "augmentStrength": config.get("augment_strength", "basic"),  # runs before experiment 5 lack the key: basic
        "dropout": config["dropout"],
        "weightDecay": config["weight_decay"],
        "labelSmoothing": config["label_smoothing"],
        "trainOnVal": config["train_on_val"],
        "svmKernel": config["svm_kernel"],
        "epochKept": result.get("final_epoch"),  # the epoch whose weights were scored (None for the SVM runs)
        "minutes": result["minutes"],
        "peakGpuGb": result.get("peak_vram_gb"),  # most graphics memory the run used (None for the SVM runs)
    }


def all_scores() -> dict[str, dict]:
    """Every run in results/, keyed by run name: scores, settings, members and learning curve.

    val: validation Top-1, or None when the run trained on the validation photos (experiment 8), because then
         the score only shows memory. test, perClass: test Top-1 and average per-class accuracy.
    settings: see settings(); None for a combination (experiment 7 and 8-final-ensemble), which trains nothing.
    members, flip: the runs a combination averages, and whether it adds mirror images; None for a trained run.
    curve: training and validation Top-1 per epoch (val None without honest validation photos); None when
           nothing trained epoch by epoch (combinations, SVMs).
    The ONNX check file has a different layout and is skipped."""
    raw = {path.stem: json.loads(path.read_text()) for path in sorted(RESULTS_DIR.glob("*.json"))}
    raw = {name: result for name, result in raw.items() if "test" in result}  # drops 8-final-ensemble-onnx.json

    def trained_on_val(name: str) -> bool:
        config = raw[name]["config"]
        # A combination saw the validation photos if any of its members did.
        return any(trained_on_val(m) for m in config["members"]) if "members" in config else config["train_on_val"]

    scores = {}
    for name, result in raw.items():
        config, history = result["config"], result.get("history") or []
        honest_val = not trained_on_val(name)
        scores[name] = {
            "val": result["val"]["top1"] if honest_val else None,
            "test": result["test"]["top1"],
            "perClass": result["test"]["mean_per_class"],
            "settings": None if "members" in config else settings(config, result),
            "members": config.get("members"),
            "flip": config.get("flip"),  # combinations only: True when each photo is also scored as its mirror image
            "curve": {
                "train": [round(epoch["train_top1"], 4) for epoch in history],
                "val": [round(epoch["val_top1"], 4) for epoch in history] if honest_val else None,
            } if history else None,
        }
    return scores


def write_examples() -> list[dict]:
    """Draw EXAMPLE_COUNT random test photos, copy them byte for byte, and return their labels.
    Copied, not re-saved: a resized JPEG copy moved ResNet-50's confidence on photo 0228 from 16.6% to 33.9%.
    The browser resizes the original instead, the same as it does for a visitor's own photo."""
    samples = random.Random(EXAMPLE_SEED).sample(read_labels(DATA_DIR / "test.txt"), EXAMPLE_COUNT)
    out_dir = APP_DIR / "public" / "examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    examples = []
    for file_name, label in samples:
        shutil.copyfile(DATA_DIR / "Test" / file_name, out_dir / file_name)
        examples.append({"file": file_name, "label": label})
    return examples


def main() -> None:
    data_dir = APP_DIR / "src" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "species.json").write_text(json.dumps(species_names(), indent=1) + "\n")
    (data_dir / "results.json").write_text(json.dumps(all_scores(), indent=1) + "\n")
    (data_dir / "examples.json").write_text(json.dumps(write_examples(), indent=1) + "\n")
    (data_dir / "confidence.json").write_text(json.dumps(confidence_check(), indent=1) + "\n")

    model_dir = APP_DIR / "public" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for name in MODELS:
        shutil.copyfile(CHECKPOINT_DIR / f"{name}.onnx", model_dir / f"{name}.onnx")
    print(f"wrote {data_dir} and {model_dir}")


if __name__ == "__main__":
    main()
