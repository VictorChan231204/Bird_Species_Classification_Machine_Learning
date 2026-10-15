"""Experiment settings.

RunConfig holds every setting of one training run (the hyperparameters).
EXPERIMENTS maps a short name to its RunConfig; run.py looks the name up here.
The defaults below are placeholders. Each experiment overrides what it tests; they are not tuned values.
"""

from dataclasses import dataclass, replace
from typing import Literal


@dataclass(frozen=True)  # frozen: settings cannot change by accident during a run
class RunConfig:
    """All settings of one run. Every field has a default except the name."""

    name: str  # short name; also the file name of the saved results, weights and probabilities

    # --- Network (models.py) ---
    # Which torchvision network to build. ResNet-18 and ResNet-50 are the slides' ResNets (W5 s62-63);
    # ConvNeXt-Tiny is a modernised ResNet built only from standard CNN parts (https://arxiv.org/abs/2201.03545).
    arch: Literal["resnet18", "resnet50", "convnext_tiny"] = "resnet50"
    pretrained: bool = True  # True: start from ImageNet weights (W6 s21). False: random start (experiment 1 baseline).
    frozen: bool = False  # True: freeze all copied layers; only the new layer learns (W6 s22-25)
    dropout: float = 0.0  # chance to switch off each input of the new layer during training (W5 s54-55). 0 = off.

    # --- Data (data.py) ---
    image_size: int = 448  # side of the square photo the network reads. 448 keeps the small marks birds differ in.
    batch_size: int = 32  # photos per weight update (W3 s50-57)
    augment: bool = True  # random crop, flip and colour change of training photos (W5 s54, s74)
    # How strong the augmentation is (only used when augment=True). "basic": crop 50-100% of the photo, flip, small
    # colour change. "strong": also crop down to 35% and rotate up to 15 degrees either way (data.py build_transform).
    augment_strength: Literal["basic", "strong"] = "basic"
    train_on_val: bool = False  # True: the 800 validation photos also train (final retrain only; no honest validation left)
    seed: int = 42  # fixes the split, the photo order, the augmentation and the start weights of the new layer
    num_workers: int = 4  # helper processes per loader that read and prepare photos while the GPU trains

    # --- Training (train.py) ---
    epochs: int = 30  # passes through all training photos; every run trains all of them
    optimizer: Literal["sgd", "adam", "adamw"] = "sgd"  # SGD with momentum (W5 s43), Adam (W5 s49-50), or AdamW
    lr_new: float = 1e-2  # peak learning rate of the new layer (W6 s28, 30: lr_new > lr_pretrained)
    lr_pretrained: float = 1e-3  # peak learning rate of the copied, unfrozen layers
    momentum: float = 0.9  # SGD only: how much of the last step carries into the next (W5 s43). Nesterov is off.
    # Penalty on big weights (W2 s46-52, W5 s74). SGD and Adam add it to the gradient (L2); AdamW shrinks the
    # weights directly, separate from the gradient ("decoupled", https://arxiv.org/abs/1711.05101). 0 = off.
    weight_decay: float = 0.0
    # "constant": the learning rate never changes. "cosine": it falls along half a cosine curve to 0 by the last
    # batch (https://arxiv.org/abs/1608.03983), so the end of training takes small, settling steps.
    schedule: Literal["constant", "cosine"] = "cosine"
    warmup_epochs: int = 1  # epochs at the start where the learning rate climbs in a straight line from ~0 to its peak
    label_smoothing: float = 0.0  # share of the target spread over the other species (https://arxiv.org/abs/1512.00567). 0 = off.
    keep_best: bool = True  # True: at the end, restore the epoch with the best validation Top-1 (ignored with train_on_val)
    amp: bool = True  # mixed precision: most GPU maths in bfloat16, about half the memory and faster (CUDA only)

    # --- SVM decider (svm.py) ---
    # None: the new layer learns by backpropagation (train.py). "linear" or "rbf": SVMs decide from the frozen
    # network's features instead (svm.py). linear = a straight fence, rbf = a bendy fence (W4 s23-34).
    svm_kernel: Literal["linear", "rbf"] | None = None
    svm_c_values: tuple[float, ...] = (0.001, 0.01, 0.1, 1, 10, 100)  # SVM strictness C to try (W4 s18-21)


# Every runnable experiment, by name. Entries are added per experiment, each with a comment on what it changes.
EXPERIMENTS: dict[str, RunConfig] = {
    # A 2-epoch run that only proves the pipeline works end to end (a few minutes). The copied ResNet-50 is frozen,
    # so only the new layer learns, at 224 x 224 with a fixed Adam rate and no augmentation.
    "check": RunConfig(
        name="check", arch="resnet50", frozen=True, epochs=2, image_size=224, optimizer="adam", lr_new=1e-3,
        schedule="constant", warmup_epochs=0, augment=False,
    ),
    # The same frozen network, but a linear SVM decides instead of the new layer; one C only, to keep it quick.
    "check-svm": RunConfig(
        name="check-svm", arch="resnet50", frozen=True, epochs=2, image_size=224, optimizer="adam", lr_new=1e-3,
        schedule="constant", warmup_epochs=0, augment=False, svm_kernel="linear", svm_c_values=(0.01,),
    ),
    # Experiment 2 (control): a standard fine-tuning recipe on ResNet-50 (ImageNet V2 weights) at 448 x 448. Adam with
    # fixed rates (new layer 1e-3, copied layers 1e-4), no warm-up, basic augmentation, best validation epoch kept,
    # batch 12, mixed precision. The later experiments change this recipe one part at a time.
    "2-control": RunConfig(
        name="2-control", arch="resnet50", image_size=448, batch_size=12, epochs=30, optimizer="adam",
        lr_new=1e-3, lr_pretrained=1e-4, schedule="constant", warmup_epochs=0,
    ),
    # Experiment 3 (optimiser and schedule), one change at a time from 2-control:
    # 3-adam-cosine: same Adam rates, but 1 warm-up epoch then cosine decay to 0 (schedule is the only change).
    "3-adam-cosine": RunConfig(
        name="3-adam-cosine", arch="resnet50", image_size=448, batch_size=12, epochs=30, optimizer="adam",
        lr_new=1e-3, lr_pretrained=1e-4, schedule="cosine", warmup_epochs=1,
    ),
    # 3-sgd-lr*: SGD with momentum 0.9 (W5 s43) in place of Adam, same warm-up and cosine. SGD's step size works on
    # a different scale from Adam's, so three copied-layer rates are tried; the new layer keeps a 10x bigger rate
    # (lr_n > lr_p, W6 s28, s30). Validation picks the rate.
    **{
        f"3-sgd-lr{lr:.0e}": RunConfig(
            name=f"3-sgd-lr{lr:.0e}", arch="resnet50", image_size=448, batch_size=12, epochs=30, optimizer="sgd",
            lr_new=lr * 10, lr_pretrained=lr, schedule="cosine", warmup_epochs=1,
        )
        for lr in (1e-3, 3e-3, 1e-2)  # names: 3-sgd-lr1e-03, 3-sgd-lr3e-03, 3-sgd-lr1e-02
    },
}

# Noise check for experiment 3: the three runs that tied on validation (84.5-85.1%) are repeated with seed 43.
# The seed changes the random start of the new layer, the photo order, the augmentation AND which 4 photos per species
# form the validation set, so the gap between the two seeds shows how much a score moves by luck alone.
# The recipe with the best average validation score over both seeds wins experiment 3.
for _name in ("3-adam-cosine", "3-sgd-lr1e-03", "3-sgd-lr3e-03"):
    EXPERIMENTS[f"{_name}-seed43"] = replace(EXPERIMENTS[_name], name=f"{_name}-seed43", seed=43)

# Experiment 4 (batch size): the experiment 3 winner (3-sgd-lr3e-03) at batch 24 instead of 12. Batch normalisation
# averages over the photos in each batch (W5 s64-67), so 24 photos give steadier averages than 12. Batch 24 fits
# (measure.py: 4.3 GB). Batch 24 halves the steps per epoch (336 -> 168), so two rates are tried:
# 4-batch24-lr3e-03: same rates as the winner (batch size is the only change).
# 4-batch24-lr6e-03: rates doubled with the batch, the linear scaling rule (https://arxiv.org/abs/1706.02677).
# Seed 42 first, so the validation photos match the winner's seed-42 run (val 84.8%); seed 43 only if one comes close.
for _lr in (3e-3, 6e-3):
    _name = f"4-batch24-lr{_lr:.0e}"  # names: 4-batch24-lr3e-03, 4-batch24-lr6e-03
    EXPERIMENTS[_name] = replace(
        EXPERIMENTS["3-sgd-lr3e-03"], name=_name, batch_size=24, lr_pretrained=_lr, lr_new=_lr * 10,
    )
# Seed 42 results: same rates val 85.3%, doubled rates val 84.3% (lost on every measure, so not repeated).
# Noise check: the same-rate run repeats with seed 43. Rule fixed before it runs: the higher mean validation Top-1 over
# seeds 42 and 43 wins, batch 24 (this pair) against batch 12 (3-sgd-lr3e-03 and its -seed43 run, mean 84.6%).
EXPERIMENTS["4-batch24-lr3e-03-seed43"] = replace(
    EXPERIMENTS["4-batch24-lr3e-03"], name="4-batch24-lr3e-03-seed43", seed=43,
)

# Experiment 5 (regularisation): tools against memorising the training photos (train 99.8% vs val 84.6% right).
# Screening step: each tool alone on the experiment 3 winner, seed 42, against its seed-42 val of 84.8%. Values are
# common starting points, not tuned. Tools that do not hurt are then combined and run on seeds 42 and 43; the combined
# recipe wins only if its mean val beats the winner's 84.6%.
for _name, _change in {
    "5-label-smoothing": {"label_smoothing": 0.1},  # target 90% right species, 10% spread (https://arxiv.org/abs/1512.00567)
    "5-weight-decay": {"weight_decay": 1e-4},  # L2 penalty on big weights (W2 s46-52); 1e-4 as in torchvision's ResNet recipe
    "5-dropout": {"dropout": 0.3},  # switch off 30% of the new layer's inputs during training (W5 s54-55)
    "5-strong-augment": {"augment_strength": "strong"},  # wider crop range plus small rotations (W5 s54, s74)
}.items():
    EXPERIMENTS[_name] = replace(EXPERIMENTS["3-sgd-lr3e-03"], name=_name, **_change)

# Screening results (seed 42 val): label smoothing 85.5%, weight decay 85.5%, dropout 85.8%, strong augment 85.1%.
# None hurt (all >= the winner's 84.8%), so all four are combined. Rule fixed before these runs: the combined recipe
# wins if its mean val over seeds 42 and 43 beats the experiment 3 winner's 84.6%. If it does not, plan B (fixed
# 2026-10-14, before the runs): 5-dropout-seed43 runs, and dropout alone wins if its two-seed mean beats 84.6%;
# otherwise the experiment 3 winner stays.
EXPERIMENTS["5-combined"] = replace(
    EXPERIMENTS["3-sgd-lr3e-03"], name="5-combined",
    label_smoothing=0.1, weight_decay=1e-4, dropout=0.3, augment_strength="strong",
)
for _name in ("5-combined", "5-dropout"):
    EXPERIMENTS[f"{_name}-seed43"] = replace(EXPERIMENTS[_name], name=f"{_name}-seed43", seed=43)

# Experiment 6 (networks): the 5-combined recipe on ResNet-18 and ConvNeXt-Tiny. The network is the only change,
# plus a learning-rate scan per network, because a small and a large network can need different step sizes.
# Plan fixed 2026-10-14, before the runs:
# - Scan: copied rate 1e-3, 3e-3, 1e-2 (new layer 10x), full 30 epochs, seed 42. If the best rate sits at an edge
#   of the scan, one more run a step past that edge.
# - ResNet-50 reuses 5-combined (its rates were scanned in experiment 3); seed-42 val 86.1%, two-seed mean 87.4%.
# - A network's best rate is repeated with seed 43 only if its seed-42 val is >= 83.6% (within the 2.5-point noise
#   floor of ResNet-50's 86.1%). Winner: mean val over seeds 42 and 43; a challenger must beat 87.4% by more than
#   2.5 points, else ResNet-50 wins (tie rule).
# - ConvNeXt-Tiny keeps SGD (one change) and torchvision's built-in stochastic depth of 0.1 (random block skipping).
for _arch, _short in (("resnet18", "resnet18"), ("convnext_tiny", "convnext")):
    for _lr in (1e-3, 3e-3, 1e-2):
        _name = f"6-{_short}-lr{_lr:.0e}"  # e.g. 6-resnet18-lr3e-03, 6-convnext-lr1e-02
        EXPERIMENTS[_name] = replace(
            EXPERIMENTS["5-combined"], name=_name, arch=_arch, lr_pretrained=_lr, lr_new=_lr * 10,
        )

# Scan results (seed 42 val): ResNet-18 1e-3 82.3%, 3e-3 83.0%, 1e-2 11.6% (wiped the copied layers in epoch 1).
# ConvNeXt-Tiny 1e-3 86.8%, 3e-3 87.3%, 1e-2 75.4%. Best rates sit mid-scan, so no edge runs. ResNet-18's 83.0% is
# under the 83.6% bar: out. ConvNeXt-Tiny's 87.3% passes, so its best rate repeats with seed 43.
EXPERIMENTS["6-convnext-lr3e-03-seed43"] = replace(
    EXPERIMENTS["6-convnext-lr3e-03"], name="6-convnext-lr3e-03-seed43", seed=43,
)
# Decision: 6-convnext-lr3e-03-seed43 val 88.0%, so ConvNeXt-Tiny's two-seed mean is 87.6% against 5-combined's 87.4%.
# 0.3 points apart, inside the 2.5-point noise floor: a tie, and the tie rule keeps ResNet-50 (5-combined).

# Experiment 1 (baselines), planned 2026-10-15. Each run is 5-combined with one advantage taken away, so
# each gap to 5-combined (two-seed mean val 87.4%) shows what that advantage adds. Seed 42 only: the expected gaps are
# far bigger than the 2.5-point noise floor.
# - 1-random-start: no ImageNet weights (W6 s21), so every layer trains at the new-layer rate 3e-2 (models.py
#   param_groups). Rule fixed in advance: if training loss is still above 5.3 (ln 200, a blind guess) after epoch 2,
#   the rate was too big, and the run repeats at 3e-3 as 1-random-start-lr3e-03.
# - 1-frozen-new-layer: the copied layers never change; only the new layer learns (W6 s22-25). Shows what fine-tuning adds.
EXPERIMENTS["1-random-start"] = replace(EXPERIMENTS["5-combined"], name="1-random-start", pretrained=False)
EXPERIMENTS["1-random-start-lr3e-03"] = replace(
    EXPERIMENTS["1-random-start"], name="1-random-start-lr3e-03", lr_new=3e-3,  # fallback only, see the rule above
)
EXPERIMENTS["1-frozen-new-layer"] = replace(EXPERIMENTS["5-combined"], name="1-frozen-new-layer", frozen=True)
# Softmax layer against SVM on the same frozen features (W4, W6 s23-24). The SVM learns from one fixed, unaugmented
# view of each photo, so 1-frozen-new-layer-plain also drops augmentation, dropout and label smoothing: the decider
# is then the only difference. Weight decay stays: it plays the part of the SVM's C (both punish big weights).
EXPERIMENTS["1-frozen-new-layer-plain"] = replace(
    EXPERIMENTS["1-frozen-new-layer"], name="1-frozen-new-layer-plain", augment=False, dropout=0.0, label_smoothing=0.0,
)
for _kernel in ("linear", "rbf"):  # straight fence and bendy fence (W4 s23-34); validation picks C from 6 values
    _name = f"1-frozen-svm-{_kernel}"  # names: 1-frozen-svm-linear, 1-frozen-svm-rbf
    EXPERIMENTS[_name] = replace(EXPERIMENTS["1-frozen-new-layer-plain"], name=_name, svm_kernel=_kernel)
# Results (seed 42 val / test): 1-random-start 15.6 / 15.9 (epoch-2 training loss 5.27 < 5.3, so no fallback run);
# 1-frozen-new-layer 61.9 / 59.5; 1-frozen-new-layer-plain 59.6 / 57.9; 1-frozen-svm-linear 62.3 / 60.7 (C=0.1);
# 1-frozen-svm-rbf 60.0 / 59.8 (C=100). Pretraining adds about 70 points, fine-tuning about 25; the three deciders on
# frozen features sit within the 2.5-point noise floor of each other.


# Experiment 7 (flip and ensemble), planned 2026-10-15. No training: each combination averages the saved
# probabilities (checkpoints/<name>-probs.npz) of runs that already exist, then takes the most likely species.
# - Flip: the photo and its mirror, averaged. Ensemble: ResNet-50 (5-combined) and ConvNeXt-Tiny (6-convnext-lr3e-03),
#   equal weights. Weights are not tuned on validation, so validation stays a fair judge.
# - Members must share a seed: the seed picks the 800 validation photos, and a seed-43 model trained on some of the
#   seed-42 validation photos. So each combination exists once per seed (no suffix = seed 42, "-seed43" = seed 43).
# - Rule fixed before any result was seen: an extra is kept if it lifts the two-seed mean validation Top-1 by at least
#   1.0 point (8 photos per seed) over the setup without it, and lowers neither seed; otherwise the simpler setup wins.
#   Not the 2.5-point noise floor: that gap is between two training runs, and here the same trained models score the
#   same photos. If the ensemble wins, experiment 8 retrains both networks and the app runs both.
@dataclass(frozen=True)
class Combination:
    """One experiment 7 setup: which finished runs to average, and whether to add flip. run.py scores it with
    src/ensemble.py; it shares the EXPERIMENTS naming, so results land in results/<name>.json like any run."""

    name: str
    members: tuple[str, ...]  # EXPERIMENTS names whose saved probabilities are averaged; all with the same seed
    flip: bool  # True: use each member's flip probabilities (photo and mirror averaged) instead of the plain ones


COMBINATIONS: dict[str, Combination] = {}
for _seed_tag in ("", "-seed43"):
    _pair = (f"5-combined{_seed_tag}", f"6-convnext-lr3e-03{_seed_tag}")
    for _name, _members, _flip in (
        (f"7-flip{_seed_tag}", _pair[:1], True),  # ResNet-50 alone + flip; the reference is 5-combined itself
        (f"7-ensemble{_seed_tag}", _pair, False),
        (f"7-ensemble-flip{_seed_tag}", _pair, True),
    ):
        COMBINATIONS[_name] = Combination(_name, _members, _flip)
# Results (two-seed mean validation Top-1): 5-combined 87.4%; 7-flip 87.1% (seed 43 drops): rejected; 7-ensemble
# 89.1% (+1.7, both seeds up: 88.1% and 90.0%): kept; 7-ensemble-flip 88.9% (seed 43 drops against 7-ensemble):
# rejected. Mean test Top-1 / average per-class of 7-ensemble: 88.5% / 88.1%. Experiment 8 retrains both members.

# Experiment 8 (final model), planned 2026-10-15. Every choice is made, so the 800 validation photos join
# training (train_on_val): 4,829 photos instead of 4,029. Seed 42 only: nothing is chosen any more.
# - No honest validation is left, so each run keeps its last epoch (30). Safe: in 5-combined and 6-convnext-lr3e-03
#   (both seeds) epoch 30 sat at most 0.9 val points under the best epoch; cosine ends with a near-zero rate.
# - The validation scores in these results mean nothing: the networks trained on those photos.
# - Rule fixed before any result is seen: 8-final-ensemble is the final model, scored on test once, and its score is
#   reported whatever it is. Going back to 7-ensemble would let test choose between two models ("roll twice, keep the
#   best"). If its test Top-1 is under 85.6% (2.5 under 7-ensemble seed 42's 88.1%), look for a code mistake first;
#   only a found mistake allows a rerun.
EXPERIMENTS["8-final-resnet50"] = replace(EXPERIMENTS["5-combined"], name="8-final-resnet50", train_on_val=True)
EXPERIMENTS["8-final-convnext"] = replace(EXPERIMENTS["6-convnext-lr3e-03"], name="8-final-convnext", train_on_val=True)
COMBINATIONS["8-final-ensemble"] = Combination(  # equal-weight average of the two, no flip (the 7-ensemble setup)
    "8-final-ensemble", ("8-final-resnet50", "8-final-convnext"), flip=False,
)
# Results (test Top-1 / average per-class): 8-final-resnet50 87.5% / 87.4%; 8-final-convnext 89.2% / 89.0%;
# 8-final-ensemble 89.7% / 89.5%, above the 85.6% bug-check line, so it is the final model as agreed.
