"""Draws the report's figures from saved results, with no training and no GPU.

Usage, from the project root, after the experiments have run:
    .venv/Scripts/python plot.py
It reads results/<name>.json (per-epoch history, written by src/train.py) and checkpoints/<name>-probs.npz (saved
probabilities) and writes:
    report/figures/progress.png                 test Top-1 of each baseline and of each step of the recipe
    report/figures/seed-noise.png               validation Top-1 of every recipe run on two seeds, and the gap between them
    report/figures/learning-curves.png          validation Top-1 per epoch, two one-change comparisons (CURVE_PANELS)
    report/figures/rate-scan.png                validation Top-1 against learning rate for ResNet-18 and ConvNeXt-Tiny
    report/figures/confidence.png               how often the final model is right, by how sure it is
    report/figures/confused-pairs.png           the species pairs the final model mixes up most: the wrong test photo
                                                beside a training photo of the species it guessed
    report/figures/example-predictions.png      test photos with the final model's top 3 guesses
    results/8-final-ensemble-analysis.json      the numbers behind the last two pictures
Confused pairs and example predictions are the qualitative results: not how often the final model is wrong, but where.
Charts share one look (CHART_STYLE): the report's Calibri font, a solid hairline grid, and a colour-blind-safe palette
(blue, orange, aqua, always in that order) checked with a palette validator.
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

# One look for every chart. Series colours go in a fixed order (blue, orange, aqua) and follow the run, never its rank.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY = "#9b9a95"  # bars and lines that give context rather than the result
INK, INK_SOFT, HAIRLINE = "#0b0b0b", "#52514e", "#dcdbd6"  # text, secondary text, grid and axis lines
CHART_STYLE = {
    "font.family": "Calibri", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "text.color": INK, "axes.labelcolor": INK_SOFT, "xtick.color": INK_SOFT, "ytick.color": INK_SOFT,
    "axes.edgecolor": HAIRLINE, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": HAIRLINE, "grid.linewidth": 0.6, "grid.linestyle": "-", "axes.axisbelow": True,
    "axes.prop_cycle": matplotlib.cycler(color=[BLUE, ORANGE, AQUA]), "lines.linewidth": 1.8,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
}
plt.rcParams.update(CHART_STYLE)


PHOTOS = {"val": 800, "test": 1204}  # photos per split


def score(name: str, split: str) -> float:
    """A run's Top-1 on "val" or "test" as a percentage, read from results/<name>.json.
    The file keeps 4 decimals (0.8405), which can round the wrong way at one decimal (84.0 instead of 84.1), so the
    share is turned back into a whole number of photos first: 0.8405 x 1,204 = 1,012 photos = 84.05%."""
    share = json.loads((ROOT / "results" / f"{name}.json").read_text())[split]["top1"]
    return 100 * round(share * PHOTOS[split]) / PHOTOS[split]


# Progress chart. Baselines: the finished recipe (5-combined) with one advantage taken away (experiment 1).
# Recipe path: the run each experiment carried forward, in order. Labels say what the step added.
BASELINES = {"1-random-start": "no ImageNet weights", "1-frozen-new-layer": "copied layers frozen",
             "5-combined": "full recipe, fine-tuned"}
RECIPE_PATH = {"2-control": "Adam, fixed rates", "3-adam-cosine": "+ warm-up and cosine",
               "3-sgd-lr3e-03": "SGD in place of Adam", "5-combined": "+ four regularisers",
               "7-ensemble": "+ ConvNeXt-Tiny, averaged", "8-final-ensemble": "+ all 4,829 photos"}
# Seed-noise chart: every recipe run on both seeds, in experiment order (seed 43 runs end in "-seed43").
SEEDED = ["3-adam-cosine", "3-sgd-lr1e-03", "3-sgd-lr3e-03", "4-batch24-lr3e-03", "5-combined",
          "6-convnext-lr3e-03", "7-ensemble"]
# Rate-scan chart: experiment 6, the 5-combined recipe at three copied-layer rates (new layer 10x).
RATE_SCAN = {"ResNet-18": "6-resnet18-lr{}", "ConvNeXt-Tiny": "6-convnext-lr{}"}
RATES = {1e-3: "1e-03", 3e-3: "3e-03", 1e-2: "1e-02"}
UNSURE_BELOW = 0.3  # the web app's "unsure" cut-off (make_app_data.py), marked on the confidence chart


def plot_progress() -> Path:
    """Two panels of horizontal bars, test Top-1 (seed 42). Left: what each advantage is worth (baselines).
    Right: the recipe path, one bar per step. One series per panel, so one colour; the result bar is blue and the
    others grey, so the eye lands on the result. Values sit at the bar tips."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.9), gridspec_kw={"width_ratios": [3, 5]})
    for ax, (title, runs, highlight) in zip(axes, (
            ("Baselines: one advantage removed", BASELINES, "5-combined"),
            ("Recipe path: one step at a time", RECIPE_PATH, FINAL))):
        names = list(runs)[::-1]  # first step at the top
        values = [score(n, "test") for n in names]
        ax.barh(range(len(names)), values, height=0.55, color=[BLUE if n == highlight else GREY for n in names])
        for y, v in enumerate(values):
            ax.text(v + 1.5, y, f"{v:.1f}%", va="center", fontsize=8.5, color=INK)  # value at the tip
        ax.set_yticks(range(len(names)), [f"{runs[n]}\n{n}" for n in names], fontsize=8)
        ax.set(xlim=(0, 100), title=title, xlabel="Test Top-1 (%)")
        ax.grid(axis="y", visible=False)  # only vertical gridlines help read bar length
        ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    path = FIGURES_DIR / "progress.png"
    fig.savefig(path)
    return path


def plot_seed_noise() -> Path:
    """A dumbbell per recipe: validation Top-1 on seed 42 and seed 43, joined by a line. The line's length is how far
    the score moved by luck alone; the longest (2.5 points) is the noise floor that every decision had to beat."""
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    names = SEEDED[::-1]  # experiment order, top to bottom
    seed42 = [score(n, "val") for n in names]
    seed43 = [score(f"{n}-seed43", "val") for n in names]
    for y, (a, b) in enumerate(zip(seed42, seed43)):
        ax.plot([a, b], [y, y], color="#c4c3bd", linewidth=2, zorder=1)
        ax.text(max(a, b) + 0.3, y, f"{abs(a - b):.1f}", va="center", fontsize=8, color=INK_SOFT)  # the gap
    ax.scatter(seed42, range(len(names)), s=40, color=BLUE, edgecolor="white", linewidth=1.2, zorder=2, label="seed 42")
    ax.scatter(seed43, range(len(names)), s=40, color=ORANGE, edgecolor="white", linewidth=1.2, zorder=2, label="seed 43")
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="Validation Top-1 (%). Number at the right: gap between the two seeds, in points", xlim=(82, 91.5))
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1), ncols=2)  # above the plot, clear of the dots
    fig.tight_layout()
    path = FIGURES_DIR / "seed-noise.png"
    fig.savefig(path)
    return path


def plot_rate_scan() -> Path:
    """Validation Top-1 (seed 42) against the copied layers' peak learning rate, log scale, one line per network.
    ResNet-50 has one point: its rates were scanned in experiment 3, on the recipe before the regularisers."""
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    for (network, pattern), colour in zip(RATE_SCAN.items(), (BLUE, ORANGE)):
        values = [score(pattern.format(tag), "val") for tag in RATES.values()]
        ax.plot(list(RATES), values, marker="o", markersize=6, color=colour, markeredgecolor="white", label=network)
        ax.annotate(f"{values[-1]:.1f}%", (1e-2, values[-1]), xytext=(7, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK)  # label the end point, where the rate is too big
    # A diamond, so the ResNet-50 point stays visible where it sits on the ConvNeXt-Tiny line
    ax.scatter([3e-3], [score("5-combined", "val")], s=60, marker="D", color=AQUA, edgecolor="white", zorder=4,
               label="ResNet-50 (5-combined)")
    ax.set_xscale("log")
    ax.set_xticks(list(RATES), ["1e-3", "3e-3", "1e-2"])
    ax.minorticks_off()
    ax.set(xlabel="Peak learning rate of the copied layers (new layer: 10x)", ylabel="Validation Top-1 (%)",
           ylim=(0, 100), xlim=(7e-4, 1.7e-2))
    ax.legend(loc="lower left")
    fig.tight_layout()
    path = FIGURES_DIR / "rate-scan.png"
    fig.savefig(path)
    return path


def plot_confidence() -> Path:
    """Reliability chart of the final model on the test photos. Photos are grouped by the top probability (0-10%,
    10-20%, ...); each bar is how often the first guess was right in that group. The grey line is where a perfectly
    calibrated model would sit (right 45% of the time when it says 45%). Bars above the line = underconfident."""
    probs, y_true = average_probs(COMBINATIONS[FINAL].members, "test", flip=False)
    top = probs.max(axis=1)  # how sure the model is about its first guess
    right = probs.argmax(axis=1) == y_true
    group = np.minimum((top * 10).astype(int), 9)  # 0 = 0-10%, ..., 9 = 90-100%
    centres = np.arange(10) * 10 + 5
    accuracy = [100 * right[group == g].mean() for g in range(10)]
    counts = [int((group == g).sum()) for g in range(10)]
    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    ax.bar(centres, accuracy, width=7, color=BLUE, label="first guess right", zorder=2)
    ax.plot([0, 100], [0, 100], color=GREY, linewidth=1, label="perfect calibration", zorder=3)
    ax.axvline(100 * UNSURE_BELOW, color=INK_SOFT, linewidth=1, zorder=3)
    ax.text(100 * UNSURE_BELOW - 1, 92, "app's 'unsure'\ncut-off (30%)", fontsize=8, color=INK_SOFT, ha="right")
    ax.set_xticks(centres, [f"{c - 5}-{c + 5}%\nn={n}" for c, n in zip(centres, counts)], fontsize=7.5)
    ax.set(xlim=(0, 100), ylim=(0, 100), xlabel="Top probability of the first guess (n = test photos in the group)",
           ylabel="First guess right (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1), ncols=2)  # above the plot, clear of the bars
    fig.tight_layout()
    path = FIGURES_DIR / "confidence.png"
    fig.savefig(path)
    return path


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
        left.set_title(textwrap.fill(f"Test photo: {names[case['true']]}", 30), fontsize=10)
        right.imshow(same_shape(Image.open(DATA_DIR / "Train" / photos[case["guess"]]).convert("RGB"), BOX_GREY))
        right.set_title(textwrap.fill(f"Training photo: {names[case['guess']]}", 30), fontsize=10)
        for ax in (left, right):
            ax.set_xticks([]), ax.set_yticks([])  # photos need no axis numbers
            ax.spines[:].set_visible(True)  # a thin frame on all four sides (the chart style hides top and right)
        box.supxlabel(f"Guessed {names[case['guess']]} ({round(100 * case['probability'])}%). "
                      f"{case['pair_mistakes']} test photos mixed up in this pair.", fontsize=9.5)
    path = FIGURES_DIR / "confused-pairs.png"
    fig.savefig(path, dpi=200)
    return path


def plot_examples(examples, names) -> Path:
    """Test photos in two rows, right guesses on top, wrong ones below, each with the model's top 3 guesses."""
    fig, axes = plt.subplots(2, EXAMPLES, figsize=(2.6 * EXAMPLES, 5.6))
    for ax, ex in zip(axes.flat, examples):
        ax.imshow(same_shape(Image.open(DATA_DIR / "Test" / ex["file"]).convert("RGB"), "white"))
        ax.set_title(textwrap.fill(f"True: {names[ex['true']]}", 26), fontsize=10)
        # One line per guess: a tick if it is the true species, its probability, its name (wrapped, never cut short).
        guesses = [textwrap.fill(f"{'✓' if g == ex['true'] else '✗'} {round(100 * p)}% {names[g]}", 28,
                                 subsequent_indent="      ") for g, p in ex["top3"]]
        # Segoe UI Symbol has both letters and the ✓ ✗ marks, which Calibri lacks. One font per label: mixing two
        # fonts in one label makes matplotlib drop most of the text.
        ax.set_xlabel("\n".join(guesses), fontsize=9, loc="left", fontfamily="Segoe UI Symbol")
        ax.set_xticks([]), ax.set_yticks([])
        ax.spines[:].set_visible(True)  # a thin frame on all four sides
    axes[0, 0].set_ylabel("Top guess right", fontsize=10)
    axes[1, 0].set_ylabel("Top guess wrong\n(most confident mistakes)", fontsize=10)
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
    for draw in (plot_progress, plot_seed_noise, plot_learning_curves, plot_rate_scan, plot_confidence):
        print(draw())
    summary = analyse_final()
    print(json.dumps({k: v for k, v in summary.items() if k != "examples"}, indent=2))
