"""Scores a trained model on the 1,204 test photos without training it first, on any computer (no GPU needed).

Usage, from the project root:
    python evaluate.py                      # the final model, 8-final-convnext (ConvNeXt-Tiny), a few minutes
    python evaluate.py 5-combined           # any other run, by its name in src/config.py

The trained weights are not in this repository (1.4 GB). Every run was exported to ONNX (export_onnx.py) and
published in the public model repo https://huggingface.co/VictorCKW/bird-species-models . This script downloads the
run's file once into checkpoints/, runs every test photo through it with onnxruntime on the processor, and prints the
brief's two metrics next to the scores saved in results/<run>.json during training.

The photos get the same test transform as in training (src/data.py): short side resized to 512, middle 448 x 448
square kept. The ONNX file standardises the pixels and applies the softmax itself (export_onnx.py, BrowserModel).
The weights are stored as float16, so a handful of guesses may differ from the PyTorch run (rounding).
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms

from src.data import DATA_DIR, RESIZE_RATIO, read_labels
from src.metrics import score

ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "checkpoints"
MODEL_BASE = "https://huggingface.co/VictorCKW/bird-species-models/resolve/main"  # same address the web app uses
IMAGE_SIZE = 448  # every published run was trained and tested at 448 x 448


def fetch(name: str) -> Path:
    """The run's ONNX file in checkpoints/, downloaded from the model repo on first use (23 to 66 MB)."""
    path = CHECKPOINT_DIR / f"{name}.onnx"
    if not path.exists():
        CHECKPOINT_DIR.mkdir(exist_ok=True)
        print(f"Downloading {name}.onnx from {MODEL_BASE} ...", flush=True)
        part = path.with_suffix(".part")  # written under another name first, so a broken download is never used
        urllib.request.urlretrieve(f"{MODEL_BASE}/{name}.onnx", part)
        part.rename(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Score one trained run on the test photos (processor only).")
    parser.add_argument("run", nargs="?", default="8-final-convnext", help="run name (default: the final model)")
    name = parser.parse_args().run

    test_txt = DATA_DIR / "test.txt"
    if not test_txt.exists():
        raise SystemExit(f"{test_txt} is missing: place the dataset as the README's 'Dataset' section shows.")
    samples = read_labels(test_txt)
    to_pixels = transforms.Compose([
        transforms.Resize(round(IMAGE_SIZE * RESIZE_RATIO)),  # short side to 512, keeping the shape
        transforms.CenterCrop(IMAGE_SIZE),  # the middle 448 x 448 square
        transforms.ToTensor(),  # pixels 0-255 -> 0-1, shape (3, 448, 448)
    ])

    session = ort.InferenceSession(fetch(name), providers=["CPUExecutionProvider"])
    guesses, start = [], time.time()
    for i, (file_name, _) in enumerate(samples, 1):
        pixels = to_pixels(Image.open(DATA_DIR / "Test" / file_name).convert("RGB")).unsqueeze(0).numpy()
        outputs = session.run(None, {"pixels": pixels})[0][0]  # 200 probabilities (or 200 SVM scores)
        guesses.append(int(np.argmax(outputs)))  # the species with the highest output is the guess
        if i % 200 == 0:
            print(f"  {i} of {len(samples)} photos ({time.time() - start:.0f} s)", flush=True)

    result = score([label for _, label in samples], guesses)
    print(f"\n{name} on {len(samples)} test photos (ONNX, processor):")
    print(f"  Top-1 accuracy             {result['top1']:.1%}")
    print(f"  Average per-class accuracy {result['mean_per_class']:.1%}")
    saved = ROOT / "results" / f"{name}.json"
    if saved.exists():  # the scores the PyTorch run wrote during training, for comparison
        test = json.loads(saved.read_text())["test"]
        print(f"  Saved during training      {test['top1']:.1%} / {test['mean_per_class']:.1%} (results/{name}.json)")


if __name__ == "__main__":
    main()
