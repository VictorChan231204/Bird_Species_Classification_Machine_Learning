"""Shared pipeline for the bird species experiments.

Modules, in the order a run uses them:
    config.py  - the settings of each experiment (RunConfig, EXPERIMENTS)
    data.py    - reads the photos, makes the validation split, preprocessing, data loaders
    models.py  - builds the network, freezes the copied layers, sets the learning-rate groups
    train.py   - the training loop, learning-rate schedule, mixed precision, scoring, saving results
    metrics.py - Top-1 accuracy and average accuracy per class
    svm.py     - the frozen-features + SVM route (no backpropagation)

Start a run from the project root with: .venv/Scripts/python run.py <experiment name>
Measure GPU memory and speed with:      .venv/Scripts/python measure.py
"""
