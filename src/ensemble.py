"""Experiment 7: score flip and ensembles from saved probabilities, with no training and no GPU.

run.py calls score_combination(combo) for a name in COMBINATIONS (src/config.py). Each trained run saved its
probabilities for every validation and test photo, plain and flipped (checkpoints/<name>-probs.npz, written by
train.fit). An ensemble averages the members' rows: if ResNet-50 gives "sparrow" 0.6 and ConvNeXt-Tiny gives it
0.4, the ensemble gives 0.5. The most likely species after averaging is the guess.
Writes results/<name>.json with the same "val" and "test" layout as a training run.
"""

import numpy as np

from src.config import Combination
from src.train import CHECKPOINT_DIR, RESULTS_DIR, summarise, tidy_json


def average_probs(members: tuple[str, ...], split: str, flip: bool) -> tuple[np.ndarray, np.ndarray]:
    """Mean probability table of the members for one split ("val" or "test"), and the true labels.
    Stops if the members did not score the same photos in the same order (for example, different seeds)."""
    key = f"{split}_flip" if flip else split  # "val_flip" already holds the mean of the photo and its mirror
    tables, labels = [], None
    for member in members:
        saved = np.load(CHECKPOINT_DIR / f"{member}-probs.npz")
        if labels is not None and not np.array_equal(saved[f"{split}_labels"], labels):
            raise ValueError(f"{member} scored different {split} photos from {members[0]}: members need one seed")
        labels = saved[f"{split}_labels"]
        tables.append(saved[key])
    return np.mean(tables, axis=0), labels  # equal weights: each member counts the same


def score_combination(combo: Combination) -> dict:
    """Score one combination on validation (to choose) and test (once, to report), and save the results."""
    result = {"config": {"members": list(combo.members), "flip": combo.flip}}
    for split in ("val", "test"):
        probs, labels = average_probs(combo.members, split, combo.flip)
        result[split] = summarise(probs, labels, per_class=split == "test")  # per-species scores only for test
    print(f"{combo.name}: val top-1 {result['val']['top1']:.3f}  test top-1 {result['test']['top1']:.3f}  "
          f"test mean per-class {result['test']['mean_per_class']:.3f}")
    (RESULTS_DIR / f"{combo.name}.json").write_text(tidy_json(result) + "\n")
    return result
