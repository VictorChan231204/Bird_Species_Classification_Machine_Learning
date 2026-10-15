"""Training scheme: optimiser, learning-rate schedule, mixed precision, the training loop, scoring and saving.

run.py calls fit(cfg) once per experiment. fit() writes three files:
    results/<name>.json          - settings, the score of every epoch, the final validation and test scores (tracked in git)
    checkpoints/<name>.pt        - the trained weights, for the Gradio app and later analysis (gitignored, large)
    checkpoints/<name>-probs.npz - the final model's class probabilities for every val and test photo, plain and
                                   flipped, plus the true labels; later ensembles average these without re-running
svm.py reuses set_seed, tidy_json and the folder paths; measure.py reuses make_optimizer.
"""

import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import RunConfig
from src.data import make_loaders
from src.metrics import score
from src.models import build_model, param_groups, set_train_mode

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
CHECKPOINT_DIR = ROOT / "checkpoints"
GB = 1024**3  # bytes in one gigabyte (binary), for the GPU memory figure


def set_seed(seed: int) -> None:
    """Fix the random start values of the new layer and every other random choice in the main process.
    data.py seeds the photo order and the helper processes. GPU arithmetic is not fully repeatable,
    so scores can still differ slightly between repeats with the same seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds the GPU


def make_optimizer(model: nn.Module, cfg: RunConfig) -> torch.optim.Optimizer:
    """The optimiser that turns gradients into weight updates, with the two learning-rate groups from models.py.

    sgd   - plain gradient step plus momentum: each step keeps 90% of the last one, which smooths the path and
            speeds it up along steady slopes (W5 s43). weight_decay adds an L2 penalty to the gradient.
    adam  - a separate, self-adjusting step size for every weight (W5 s49-50). weight_decay adds the L2 penalty
            to the gradient ("coupled"); Adam then rescales it, which weakens it.
    adamw - Adam, but weight decay shrinks every weight directly by lr * weight_decay each step ("decoupled"),
            so the penalty acts at its full size (https://arxiv.org/abs/1711.05101).
    """
    groups = param_groups(model, cfg)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(groups, momentum=cfg.momentum, nesterov=False, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(groups, weight_decay=cfg.weight_decay)
    return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)


def lr_factor(step: int, total_steps: int, warmup_steps: int, schedule: str) -> float:
    """How much of its peak learning rate each group uses at this batch (0 to 1). Called once per batch.

    Warm-up: the factor climbs in a straight line from 1/warmup_steps (close to 0) to 1. The new layer starts
    random, so its first gradients are large and noisy; small first steps stop them from wrecking the copied
    weights (https://arxiv.org/abs/1706.02677).
    Then "constant" stays at 1, and "cosine" falls along half a cosine wave from 1 to 0 at the last batch:
    big steps early to travel, small steps late to settle (https://arxiv.org/abs/1608.03983).
    """
    if step < warmup_steps:
        return (step + 1) / warmup_steps  # step 0 -> 1/warmup_steps, last warm-up step -> 1
    if schedule == "constant":
        return 1.0
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)  # 0 at the end of warm-up, ~1 at the end
    return 0.5 * (1 + math.cos(math.pi * progress))  # cos(0) = 1 -> factor 1; cos(pi) = -1 -> factor 0


def autocast(cfg: RunConfig, device: torch.device):
    """Mixed precision for the forward pass: most GPU maths runs in bfloat16 (16-bit numbers) instead of float32.
    It roughly halves the memory of stored activations and uses the GPU's fast tensor cores.
    bfloat16 keeps float32's range of sizes (only fewer digits), so tiny gradients do not round to zero and,
    unlike float16, no loss scaling (GradScaler) is needed. The weights themselves stay float32."""
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.amp and device.type == "cuda")


def to_device(images: torch.Tensor, labels: torch.Tensor, device: torch.device):
    """Copy one batch to the GPU. channels_last stores each pixel's 3 colours side by side in memory,
    the layout the GPU's convolution kernels read fastest (the model is converted to match in fit())."""
    images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
    return images, labels.to(device, non_blocking=True)


def train_one_epoch(model, loader, loss_fn, optimiser, cfg, device, step: int, total_steps: int, warmup_steps: int):
    """One pass through all training photos. Returns the average loss, the training accuracy and the next step number."""
    set_train_mode(model, cfg)
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:  # one mini-batch of cfg.batch_size photos at a time
        images, labels = to_device(images, labels, device)

        factor = lr_factor(step, total_steps, warmup_steps, cfg.schedule)
        for group in optimiser.param_groups:
            group["lr"] = group["base_lr"] * factor  # warm-up / cosine: set this batch's learning rate
        step += 1

        with autocast(cfg, device):
            scores = model(images)  # forward pass: 200 raw scores per photo
        # Cross-entropy (W5 s36-37) in float32 for accuracy; with label smoothing the target is not a hard 1
        # for the right species but 1 - label_smoothing, with the rest spread over all species.
        loss = loss_fn(scores.float(), labels)

        optimiser.zero_grad(set_to_none=True)  # clear the gradients of the previous batch
        loss.backward()  # backpropagation: how much each weight caused the error (W5 s38-40)
        optimiser.step()  # the optimiser moves each weight a small step to reduce the loss (W5 s41-50)

        total_loss += loss.item() * len(labels)  # loss.item() is the batch average, so weight it by batch size
        correct += (scores.argmax(dim=1) == labels).sum().item()  # argmax = the species with the highest score
        seen += len(labels)
    return total_loss / seen, correct / seen, step


@torch.no_grad()  # no gradients needed when only scoring: faster and uses less GPU memory
def predict(model, loader: DataLoader, cfg: RunConfig, device, flip: bool) -> dict[str, np.ndarray]:
    """Class probabilities for every photo in the loader.

    Returns "probs" (softmax of the plain photo, one row of 200 per photo) and "labels". With flip=True also
    "probs_flip": the average of the softmax of the photo and of its left-right mirror (flip).
    A bird facing left is still the same bird, so two views give a slightly steadier guess for the cost of
    a second forward pass (https://arxiv.org/abs/1409.1556, section 3.2).
    """
    model.eval()  # dropout off; batch normalisation uses its stored averages
    probs, probs_flip, labels = [], [], []
    for images, batch_labels in loader:
        images, batch_labels = to_device(images, batch_labels, device)
        with autocast(cfg, device):
            plain = model(images).float().softmax(dim=1)  # softmax: scores -> probabilities that sum to 1 (W3 s43)
            if flip:
                mirrored = model(torch.flip(images, dims=[3])).float().softmax(dim=1)  # dim 3 = width: mirror
                probs_flip.append(((plain + mirrored) / 2).cpu())
        probs.append(plain.cpu())
        labels.append(batch_labels.cpu())
    out = {"probs": torch.cat(probs).numpy(), "labels": torch.cat(labels).numpy()}
    if flip:
        out["probs_flip"] = torch.cat(probs_flip).numpy()
    return out


def summarise(probs: np.ndarray, labels: np.ndarray, per_class: bool) -> dict:
    """Loss and both scores from a table of probabilities. The loss is plain cross-entropy, without label
    smoothing, so runs with and without smoothing can be compared: the mean of -log(probability of the right species)."""
    loss = float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, None)).mean())  # clip: no log(0)
    s = score(labels.tolist(), probs.argmax(axis=1).tolist())  # the guess is the most likely species
    out = {"loss": round(loss, 4), "top1": round(s["top1"], 4), "mean_per_class": round(s["mean_per_class"], 4)}
    if per_class:
        out["per_class"] = s["per_class"]  # 200 per-species accuracies, for the most-confused-species analysis
    return out


def fit(cfg: RunConfig) -> dict:
    """Train one experiment from start to end, score it on the test set once, and save the results."""
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()  # so peak_vram_gb measures this run only
    train_loader, val_loader, test_loader = make_loaders(cfg)

    model = build_model(cfg).to(device, memory_format=torch.channels_last)  # same memory layout as the batches
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)  # softmax + cross-entropy in one step (W5 s36-37)
    optimiser = make_optimizer(model, cfg)
    total_steps = cfg.epochs * len(train_loader)  # one step = one batch = one weight update
    warmup_steps = cfg.warmup_epochs * len(train_loader)
    keep_best = cfg.keep_best and not cfg.train_on_val  # with train_on_val there are no honest validation photos

    history = []  # one row per epoch, for the learning-curve graphs in the report
    best_val_top1, best_epoch, best_weights = -1.0, 0, None
    step = 0
    start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.time()
        train_loss, train_top1, step = train_one_epoch(
            model, train_loader, loss_fn, optimiser, cfg, device, step, total_steps, warmup_steps
        )
        val_out = predict(model, val_loader, cfg, device, flip=False)  # plain only each epoch; flip at the end
        val = summarise(val_out["probs"], val_out["labels"], per_class=False)
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_top1": round(train_top1, 4),
            "val_loss": val["loss"],
            "val_top1": val["top1"],
            "val_mean_per_class": val["mean_per_class"],
            "lr_new": float(f"{optimiser.param_groups[0]['lr']:.3g}"),  # the new layer's rate at the last batch of this epoch
            "seconds": round(time.time() - epoch_start, 1),
        })
        print(f"[{cfg.name}] epoch {epoch}/{cfg.epochs}  train loss {train_loss:.3f} acc {train_top1:.3f}  "
              f"val loss {val['loss']:.3f} acc {val['top1']:.3f}  lr {history[-1]['lr_new']:g}  "
              f"({history[-1]['seconds']} s)", flush=True)

        if val["top1"] > best_val_top1:  # a new best validation score
            best_val_top1, best_epoch = val["top1"], epoch
            if keep_best:
                # Remember the best weights on the CPU (the GPU has no room to spare). Full-budget training plus
                # "keep the best epoch" replaces early stopping, which cut bumpy curves off too soon (lessons.md).
                best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_weights is not None:
        model.load_state_dict(best_weights)  # restore the best validation epoch
    final_epoch = best_epoch if best_weights is not None else len(history)

    # Final scores, plain and with flip. The test photos are used here only, once per run,
    # never to choose settings.
    val_out = predict(model, val_loader, cfg, device, flip=True)
    test_out = predict(model, test_loader, cfg, device, flip=True)
    peak_vram_gb = round(torch.cuda.max_memory_allocated() / GB, 2) if device.type == "cuda" else 0.0
    result = {
        "config": asdict(cfg),
        "final_epoch": final_epoch,
        "minutes": round((time.time() - start) / 60, 1),
        "peak_vram_gb": peak_vram_gb,  # most GPU memory the run's tensors held at once
        "val": summarise(val_out["probs"], val_out["labels"], per_class=False),
        "val_flip": summarise(val_out["probs_flip"], val_out["labels"], per_class=False),
        "test": summarise(test_out["probs"], test_out["labels"], per_class=True),
        "test_flip": summarise(test_out["probs_flip"], test_out["labels"], per_class=True),
        "history": history,
    }
    print(f"[{cfg.name}] final (epoch {final_epoch}): val top-1 {result['val']['top1']:.3f} "
          f"(flip {result['val_flip']['top1']:.3f})  test top-1 {result['test']['top1']:.3f} "
          f"(flip {result['test_flip']['top1']:.3f})  test mean per-class {result['test']['mean_per_class']:.3f}  "
          f"peak VRAM {peak_vram_gb} GB")

    RESULTS_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"{cfg.name}.json").write_text(tidy_json(result) + "\n")  # tidy layout: see tidy_json() below
    torch.save(model.state_dict(), CHECKPOINT_DIR / f"{cfg.name}.pt")
    # Probabilities of the final model, for ensembles: average two models' rows, then take the most likely species.
    np.savez(
        CHECKPOINT_DIR / f"{cfg.name}-probs.npz",
        val=val_out["probs"], val_flip=val_out["probs_flip"], test=test_out["probs"], test_flip=test_out["probs_flip"],
        val_labels=val_out["labels"], test_labels=test_out["labels"],
    )
    return result


def tidy_json(value, indent: int = 0) -> str:
    """Write a results dict as JSON that is short to scroll through: one line per epoch,
    10 per-species scores per line, and small groups of plain values on one line.
    The output is still normal JSON, so json.load() reads it back unchanged."""
    pad, inner = "  " * indent, "  " * (indent + 1)
    one_line = json.dumps(value)
    if isinstance(value, dict) and (len(one_line) > 100 or any(isinstance(v, (dict, list)) for v in value.values())):
        # A big dict, or one holding lists or dicts: one key per line, each value laid out the same way.
        parts = [f"{inner}{json.dumps(key)}: {tidy_json(item, indent + 1)}" for key, item in value.items()]
        return "{\n" + ",\n".join(parts) + "\n" + pad + "}"
    if isinstance(value, list) and value and isinstance(value[0], dict):
        # A list of dicts, such as history: one epoch per line.
        return "[\n" + ",\n".join(inner + json.dumps(row) for row in value) + "\n" + pad + "]"
    if isinstance(value, list) and len(value) > 10:
        # A long list of numbers, such as the 200 per-species scores: 10 per line.
        rows = [", ".join(json.dumps(x) for x in value[i:i + 10]) for i in range(0, len(value), 10)]
        return "[\n" + ",\n".join(inner + row for row in rows) + "\n" + pad + "]"
    return one_line  # anything short fits on one line
