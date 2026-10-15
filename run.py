"""The start button: runs one experiment by name.

Usage, from the project root:
    .venv/Scripts/python run.py check      # 2-epoch pipeline check (frozen ResNet-50 + new layer)
    .venv/Scripts/python run.py check-svm  # the same frozen ResNet-50 with a linear SVM as the decider
    .venv/Scripts/python run.py 7-ensemble # experiment 7: average saved probabilities, no training (seconds)
The experiment names and their settings live in src/config.py (EXPERIMENTS, and COMBINATIONS for experiment 7).
"""

import argparse

from src.config import COMBINATIONS, EXPERIMENTS
from src.ensemble import score_combination
from src.svm import fit_svm
from src.train import fit


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and score one bird species experiment.")
    parser.add_argument("experiment", choices=sorted(EXPERIMENTS | COMBINATIONS), help="name of the experiment in src/config.py")
    name = parser.parse_args().experiment
    if name in COMBINATIONS:  # experiment 7: no training, only averaging of saved probabilities
        score_combination(COMBINATIONS[name])
        return
    cfg = EXPERIMENTS[name]
    if cfg.svm_kernel:  # the frozen-features + SVM route has no epochs, so it takes its own path
        fit_svm(cfg)
    else:
        fit(cfg)


# Windows starts the data-loading workers as new Python processes that import this file.
# This guard stops each worker from starting its own training run.
if __name__ == "__main__":
    main()
