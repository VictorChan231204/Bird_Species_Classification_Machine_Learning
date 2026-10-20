"""The start button: runs one experiment by name.

Usage, from the project root:
    .venv/Scripts/python run.py check      # 2-epoch pipeline check (frozen ResNet-50 + new layer)
    .venv/Scripts/python run.py check-svm  # the same frozen ResNet-50 with a linear SVM as the decider
    .venv/Scripts/python run.py 7-final-convnext  # the final model, about 50 minutes
The experiment names and their settings live in src/config.py (EXPERIMENTS).
"""

import argparse

from src.config import EXPERIMENTS
from src.svm import fit_svm
from src.train import fit


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and score one bird species experiment.")
    parser.add_argument("experiment", choices=sorted(EXPERIMENTS), help="name of the experiment in src/config.py")
    name = parser.parse_args().experiment
    cfg = EXPERIMENTS[name]
    if cfg.svm_kernel:  # the frozen-features + SVM route has no epochs, so it takes its own path
        fit_svm(cfg)
    else:
        fit(cfg)


# Windows starts the data-loading workers as new Python processes that import this file.
# This guard stops each worker from starting its own training run.
if __name__ == "__main__":
    main()
