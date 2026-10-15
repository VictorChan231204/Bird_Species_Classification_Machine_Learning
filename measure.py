"""Measures GPU memory and speed of training settings, before any real run is planned.

Usage, from the project root:
    .venv/Scripts/python measure.py                                # default grid: 3 networks x size 448 x batch 12-32
    .venv/Scripts/python measure.py resnet50:448:24 convnext_tiny:384:32   # chosen settings, as arch:size:batch

For each setting it builds the network exactly as a full fine-tuning run would (pretrained, every layer learning,
bfloat16 mixed precision, channels_last, SGD), runs a few training steps on random tensors, and prints one row:
the peak GPU memory, the seconds per batch, and the estimated seconds per epoch for the 4,029 training photos.
It never reads the dataset, so photo loading time is not included: a real epoch is at least this long.
"""

import argparse
import gc
import math
import time

import torch
from torch import nn

from src.config import RunConfig
from src.data import NUM_CLASSES
from src.models import build_model
from src.train import GB, autocast, make_optimizer, set_train_mode

TRAIN_PHOTOS = 4029  # training photos per epoch (4,829 minus the 800 validation photos)
WARMUP_STEPS = 3  # untimed first steps: the GPU picks its convolution methods and fills its memory pool here
TIMED_STEPS = 10  # timed steps; seconds per batch is their average
DEFAULT_GRID = [(arch, 448, batch) for arch in ("resnet18", "resnet50", "convnext_tiny") for batch in (12, 16, 24, 32)]


def measure(arch: str, image_size: int, batch_size: int, device: torch.device) -> tuple[float, float]:
    """Train on random photos for a few steps. Returns (peak GPU memory in GB, seconds per batch).
    Raises torch.cuda.OutOfMemoryError when the setting does not fit in GPU memory."""
    cfg = RunConfig(name="measure", arch=arch, image_size=image_size, batch_size=batch_size)  # defaults: SGD, amp on
    model = build_model(cfg).to(device, memory_format=torch.channels_last)
    optimiser = make_optimizer(model, cfg)
    loss_fn = nn.CrossEntropyLoss()
    set_train_mode(model, cfg)  # fully trainable: every layer and its batch normalisation in training mode

    # One fixed random batch: noise photos and random species. Only the sizes matter for memory and speed.
    images = torch.randn(batch_size, 3, image_size, image_size, device=device).to(memory_format=torch.channels_last)
    labels = torch.randint(0, NUM_CLASSES, (batch_size,), device=device)

    torch.cuda.reset_peak_memory_stats()
    for step in range(WARMUP_STEPS + TIMED_STEPS):
        if step == WARMUP_STEPS:
            torch.cuda.synchronize()  # wait for the GPU to finish the untimed steps before starting the clock
            start = time.perf_counter()
        with autocast(cfg, device):
            loss = loss_fn(model(images).float(), labels)  # forward pass, as in train.py
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
    torch.cuda.synchronize()  # the GPU runs ahead of Python; wait until it has really finished
    seconds_per_batch = (time.perf_counter() - start) / TIMED_STEPS
    return torch.cuda.max_memory_allocated() / GB, seconds_per_batch


def parse_setting(text: str) -> tuple[str, int, int]:
    """'resnet50:448:24' -> ('resnet50', 448, 24)."""
    arch, size, batch = text.split(":")
    return arch, int(size), int(batch)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure GPU memory and speed of training settings.")
    parser.add_argument("settings", nargs="*", type=parse_setting, help="arch:size:batch, e.g. resnet50:448:24")
    grid = parser.parse_args().settings or DEFAULT_GRID
    device = torch.device("cuda")  # this script measures the GPU; it has no CPU fallback

    print(f"{'arch':<14} {'size':>5} {'batch':>6} {'peak GB':>8} {'s/batch':>8} {'s/epoch':>8}")
    for arch, image_size, batch_size in grid:
        try:
            peak_gb, seconds = measure(arch, image_size, batch_size, device)
            per_epoch = seconds * math.ceil(TRAIN_PHOTOS / batch_size)  # batches per epoch x seconds per batch
            print(f"{arch:<14} {image_size:>5} {batch_size:>6} {peak_gb:>8.2f} {seconds:>8.3f} {per_epoch:>8.0f}",
                  flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"{arch:<14} {image_size:>5} {batch_size:>6} {'OOM':>8}", flush=True)
        # Free everything this setting held before the next one, so one setting's memory cannot spill into the next.
        # (The model, optimiser and batch were local to measure(), so they are already unreferenced here.)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
