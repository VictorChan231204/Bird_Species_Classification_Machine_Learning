"""Draws the report's figures from saved results, with no training and no GPU.

Usage, from the project root, after the experiments have run:
    .venv/Scripts/python plot.py
It reads results/<name>.json (per-epoch history, written by src/train.py) and checkpoints/<name>-probs.npz (saved
probabilities) and writes:
    report/figures/learning-curves.png          validation Top-1 per epoch, two one-change comparisons (CURVE_PANELS)
    report/figures/confused-pairs.png           the species pairs the final model mixes up most: the wrong test photo
                                                beside a training photo of the species it guessed
    report/figures/example-predictions.png      test photos with the final model's top 3 guesses
    results/8-final-ensemble-analysis.json      the numbers behind the last two pictures
The last three are the qualitative results: not how often the final model is wrong, but where.
"""

import json
import random
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw straight to a file; no window opens
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.config import COMBINATIONS
from src.data import DATA_DIR, read_labels
from src.ensemble import average_probs

ROOT = Path(__file__).resolve().parent
FIGURES_DIR = ROOT / "report" / "figures"
FINAL = "8-final-ensemble"  # the final model: ResNet-50 and ConvNeXt-Tiny, probabilities averaged
# Learning-curve panels: each compares two runs that differ in one thing only, so the panel shows one cause.
CURVE_PANELS = {
    "Schedule: fixed rates against warm-up + cosine": {"2-control": "fixed rates", "3-adam-cosine": "warm-up + cosine"},
    "Regularisation: four tools added": {"3-sgd-lr3e-03": "without", "5-combined": "with the four tools"},
}
CURVE_FLOOR = 50  # the y axis starts here, so the 75-90% band where the runs differ is readable; epoch 1 sits below it
TOP_PAIRS = 4  # pairs in the confused-pairs picture
EXAMPLES = 3  # right and wrong examples each in the example-predictions picture
PHOTO_SHAPE = 4 / 3  # every photo is padded to this width/height, so the photos in a picture line up
BOX_GREY = "#f0f0f0"  # background of each pair's box in the confused-pairs picture


def plot_learning_curves() -> Path:
    """One panel per entry in CURVE_PANELS: validation Top-1 after every epoch for two runs (W5 s73).
    Training Top-1 is left out on purpose: train.py measures it on augmented photos while the weights still change,
    so it is not a fair partner for the validation line. A jumpy line means the steps are too big to settle."""
    fig, axes = plt.subplots(1, len(CURVE_PANELS), figsize=(9, 3.2), sharey=True)
    for ax, (title, runs) in zip(axes, CURVE_PANELS.items()):
        for name, label in runs.items():
            history = json.loads((ROOT / "results" / f"{name}.json").read_text())["history"]  # one row per epoch
            ax.plot([row["epoch"] for row in history], [100 * row["val_top1"] for row in history],
                    label=f"{name} ({label})")
        ax.set(title=title, xlabel="Epoch", ylim=(CURVE_FLOOR, 90))
        ax.title.set_fontsize(10)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("Validation Top-1 (%)")
    fig.tight_layout()
    path = FIGURES_DIR / "learning-curves.png"
    fig.savefig(path, dpi=200)
    return path


def species_names() -> dict[int, str]:
    """Species number -> readable name, taken from the training file names.
    'Black_footed_Albatross_0004_2731401028.jpg' -> 'Black footed Albatross' (the last two parts are photo ids)."""
    return {label: file_name.rsplit("_", 2)[0].replace("_", " ")
            for file_name, label in read_labels(DATA_DIR / "train.txt")}


def first_training_photo() -> dict[int, str]:
    """Species number -> the file name of its first training photo, used as that species' picture."""
    photos: dict[int, str] = {}
    for file_name, label in read_labels(DATA_DIR / "train.txt"):
        photos.setdefault(label, file_name)  # keep only the first photo seen for each species
    return photos


def confused_pairs(y_true: np.ndarray, y_pred: np.ndarray) -> list[tuple[int, int, int]]:
    """Count the mistakes for every pair of species, both ways round (A taken for B plus B taken for A).
    Returns (species A, species B, mistakes), most mistakes first."""
    counts = Counter(tuple(sorted((int(t), int(p)))) for t, p in zip(y_true, y_pred) if t != p)
    return [(a, b, n) for (a, b), n in counts.most_common()]


def bird_group(name: str) -> str:
    """The last word of a species name, a rough bird group: 'California Gull' -> 'Gull'."""
    return name.split()[-1]


def same_shape(photo: Image.Image, fill: str) -> Image.Image:
    """Pad a photo with a plain border up to the PHOTO_SHAPE width/height ratio, so photos of different shapes
    line up. Padding instead of cutting, so no part of the bird is lost."""
    width, height = photo.size
    new_width, new_height = max(width, round(height * PHOTO_SHAPE)), max(height, round(width / PHOTO_SHAPE))
    canvas = Image.new("RGB", (new_width, new_height), fill)
    canvas.paste(photo, ((new_width - width) // 2, (new_height - height) // 2))  # the photo in the middle
    return canvas


def plot_confused_pairs(cases, names, photos) -> Path:
    """Two pairs per row. Each pair sits in its own grey box: on the left, the test photo the model got wrong
    (its most confident mistake within the pair); on the right, a training photo of the species it guessed, for
    comparison. One line under both says how sure the model was and how many test photos the pair mixed up."""
    rows = (len(cases) + 1) // 2
    fig = plt.figure(figsize=(9, 2.6 * rows), layout="constrained")
    boxes = fig.subfigures(rows, 2, wspace=0.03, hspace=0.04)  # one box per pair
    for box, case in zip(boxes.flat, cases):
        box.set_facecolor(BOX_GREY)
        left, right = box.subplots(1, 2)
        left.imshow(same_shape(Image.open(DATA_DIR / "Test" / case["file"]).convert("RGB"), BOX_GREY))
        left.set_title(textwrap.fill(f"Test photo: {names[case['true']]}", 32), fontsize=8)
        right.imshow(same_shape(Image.open(DATA_DIR / "Train" / photos[case["guess"]]).convert("RGB"), BOX_GREY))
        right.set_title(textwrap.fill(f"Training photo: {names[case['guess']]}", 32), fontsize=8)
        for ax in (left, right):
            ax.set_xticks([]), ax.set_yticks([])  # photos need no axis numbers
        box.supxlabel(f"Guessed {names[case['guess']]} ({round(100 * case['probability'])}%). "
                      f"{case['pair_mistakes']} test photos mixed up in this pair.", fontsize=7.5)
    path = FIGURES_DIR / "confused-pairs.png"
    fig.savefig(path, dpi=200)
    return path


def plot_examples(examples, names) -> Path:
    """Test photos in two rows, right guesses on top, wrong ones below, each with the model's top 3 guesses."""
    fig, axes = plt.subplots(2, EXAMPLES, figsize=(2.6 * EXAMPLES, 5.6))
    for ax, ex in zip(axes.flat, examples):
        ax.imshow(same_shape(Image.open(DATA_DIR / "Test" / ex["file"]).convert("RGB"), "white"))
        ax.set_title(textwrap.fill(f"True: {names[ex['true']]}", 30), fontsize=8)
        # One line per guess: a tick if it is the true species, its probability, its name (wrapped, never cut short).
        guesses = [textwrap.fill(f"{'✓' if g == ex['true'] else '✗'} {round(100 * p)}% {names[g]}", 34,
                                 subsequent_indent="      ") for g, p in ex["top3"]]
        ax.set_xlabel("\n".join(guesses), fontsize=7, loc="left")
        ax.set_xticks([]), ax.set_yticks([])
    axes[0, 0].set_ylabel("Top guess right", fontsize=9)
    axes[1, 0].set_ylabel("Top guess wrong\n(most confident mistakes)", fontsize=9)
    fig.tight_layout()
    path = FIGURES_DIR / "example-predictions.png"
    fig.savefig(path, dpi=200)
    return path


def analyse_final() -> dict:
    """Find the final model's most confused pairs and example predictions from its saved test probabilities."""
    names, photos = species_names(), first_training_photo()
    probs, y_true = average_probs(COMBINATIONS[FINAL].members, "test", flip=False)  # same average as run.py
    files = [file_name for file_name, _ in read_labels(DATA_DIR / "test.txt")]  # same order as the saved rows
    y_pred = probs.argmax(axis=1)  # the first guess: the species with the highest probability
    wrong = np.flatnonzero(y_pred != y_true)
    pairs = confused_pairs(y_true, y_pred)
    ranked = np.argsort(probs, axis=1)[:, ::-1]  # every photo's species, most likely first

    def top3(i: int) -> list[tuple[int, float]]:
        return [(int(g), float(probs[i, g])) for g in ranked[i, :3]]

    # Wrong examples: the mistakes the model was most sure about. Right examples: a fixed random pick.
    most_sure_wrong = sorted(wrong, key=lambda i: -probs[i, y_pred[i]])[:EXAMPLES]
    right = random.Random(42).sample(sorted(set(range(len(files))) - set(wrong)), EXAMPLES)
    examples = [{"file": files[i], "true": int(y_true[i]), "top3": top3(i)} for i in [*right, *most_sure_wrong]]

    # For each of the most confused pairs, the actual test photo behind it: the pair's most confident mistake.
    cases = []
    for a, b, n in pairs[:TOP_PAIRS]:
        in_pair = [i for i in wrong if {int(y_true[i]), int(y_pred[i])} == {a, b}]
        i = max(in_pair, key=lambda i: probs[i, y_pred[i]])
        cases.append({"file": files[i], "true": int(y_true[i]), "guess": int(y_pred[i]),
                      "probability": float(probs[i, y_pred[i]]), "pair_mistakes": n})

    per_species = [(s, float((y_pred[y_true == s] == s).mean())) for s in range(len(names))]
    result = {
        "run": FINAL,
        "test_top1": round(1 - len(wrong) / len(files), 4),  # must match results/8-final-ensemble.json
        "test_top3": round(float(np.mean([y_true[i] in ranked[i, :3] for i in range(len(files))])), 4),
        "test_top5": round(float(np.mean([y_true[i] in ranked[i, :5] for i in range(len(files))])), 4),
        "mistakes": int(len(wrong)),
        # true and guessed species share a last word (both gulls, both sparrows): a near miss, not a wild guess
        "mistakes_within_same_group": int(sum(bird_group(names[y_true[i]]) == bird_group(names[y_pred[i]]) for i in wrong)),
        "species_all_right": sum(acc == 1.0 for _, acc in per_species),
        "top_pairs": [{"a": names[a], "b": names[b], "mistakes": n} for a, b, n in pairs[:15]],
        "pair_photos": [{**c, "true": names[c["true"]], "guess": names[c["guess"]],
                         "probability": round(c["probability"], 3)} for c in cases],
        "hardest_species": [{"species": names[s], "accuracy": round(acc, 3)}
                            for s, acc in sorted(per_species, key=lambda x: x[1])[:10]],
        "examples": [{**ex, "true": names[ex["true"]], "top3": [(names[g], round(p, 3)) for g, p in ex["top3"]]}
                     for ex in examples],
    }
    print(plot_confused_pairs(cases, names, photos))
    print(plot_examples(examples, names))
    (ROOT / "results" / f"{FINAL}-analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(plot_learning_curves())
    summary = analyse_final()
    print(json.dumps({k: v for k, v in summary.items() if k != "examples"}, indent=2))
