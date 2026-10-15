"""Data input, the train/validation split, preprocessing, and data loaders.

train.py and svm.py call make_loaders(cfg) once per run and get three loaders:
    train - photos the network learns from (4,029; 4,829 with train_on_val)
    val   - 4 photos per species held back from training (800), to watch progress and choose settings
    test  - the official test photos (1,204), scored once at the end of the run
"""

import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.config import RunConfig

# Folder with Train/, Test/, train.txt and test.txt (gitignored, 664 MB).
DATA_DIR = Path(__file__).resolve().parent.parent / "project_docs" / "Datasets"
NUM_CLASSES = 200  # species numbers 0-199
VAL_PER_CLASS = 4  # photos per species moved from training to validation

# Average colour and spread of the ImageNet photos, one value per colour channel (red, green, blue).
# Pretrained networks expect inputs standardised with these numbers (W5 s80: x_norm = (x - mean) / std).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RESIZE_RATIO = 256 / 224  # resize the short side to this times the crop size first: 224 -> 256, 448 -> 512

Sample = tuple[str, int]  # (photo file name, species number)


def read_labels(txt_path: Path) -> list[Sample]:
    """Read train.txt or test.txt. Each line is '<file name> <species number>'."""
    samples = []
    for line in txt_path.read_text().splitlines():
        if line.strip():  # skip empty lines
            file_name, label = line.rsplit(" ", 1)  # split at the last space: the label is the last word
            samples.append((file_name, int(label)))
    return samples


def split_train_val(samples: list[Sample], seed: int) -> tuple[list[Sample], list[Sample]]:
    """Move VAL_PER_CLASS random photos of every species into a validation set.

    Stratified: every species gets exactly the same number of validation photos,
    so the validation score treats all 200 species equally. The seed makes the split repeatable.
    """
    by_class: dict[int, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_class[sample[1]].append(sample)  # group the photos by species

    rng = random.Random(seed)  # own random generator, so other code cannot change the split
    train, val = [], []
    for label in sorted(by_class):  # sorted: same order every time, so the same seed gives the same split
        photos = by_class[label]
        rng.shuffle(photos)
        val.extend(photos[:VAL_PER_CLASS])
        train.extend(photos[VAL_PER_CLASS:])
    return train, val


def build_transform(augment: bool, image_size: int, strength: str = "basic") -> transforms.Compose:
    """Preprocessing that turns one photo file into the tensor the network reads.

    augment=False: resize the short side to image_size * 256/224, cut the centre image_size square.
    augment=True: a random crop, random mirror and random colour change, so the network
    never sees exactly the same photo twice (W5 s54, s74). Only ever used for training photos.
    strength="strong" (with augment=True): the crop can shrink to 35% of the photo and the photo first turns by up to
    15 degrees. The colour change stays small on purpose, because plumage colour tells species apart.
    """
    if augment:
        strong = strength == "strong"
        resize_and_crop = [
            # strong only: turn the photo by a random angle of -15 to +15 degrees; the corners it uncovers turn black
            *([transforms.RandomRotation(15, interpolation=transforms.InterpolationMode.BILINEAR)] if strong else []),
            # random square covering 35-100% (strong) or 50-100% (basic) of the photo, resized to image_size
            transforms.RandomResizedCrop(image_size, scale=(0.35 if strong else 0.5, 1.0)),
            transforms.RandomHorizontalFlip(),  # mirror left-right with a 50% chance
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # small random colour change
        ]
    else:
        resize_and_crop = [
            transforms.Resize(round(image_size * RESIZE_RATIO)),  # short side to 256 (or 512), keeping the shape
            transforms.CenterCrop(image_size),  # the middle 224 x 224 (or 448 x 448) square
        ]
    return transforms.Compose(
        resize_and_crop
        + [
            transforms.ToTensor(),  # pixels 0-255 -> numbers 0-1, shape (3 colours, image_size, image_size)
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),  # standardise each colour channel (W5 s80)
        ]
    )


class BirdDataset(Dataset):
    """A list of (photo, species) pairs that a DataLoader can read by index."""

    def __init__(self, image_dir: Path, samples: list[Sample], transform: transforms.Compose):
        self.image_dir = image_dir
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        file_name, label = self.samples[index]
        image = Image.open(self.image_dir / file_name).convert("RGB")  # RGB: a few photos may be greyscale
        return self.transform(image), label


def seed_worker(worker_id: int) -> None:
    """Runs once inside every loader helper process when it starts.

    PyTorch gives each helper its own torch seed, taken from the loader's Generator (see make_loaders).
    This copies that seed into Python's and NumPy's random generators too, so every random choice a
    helper makes (crop, flip, colour) replays exactly when the run is repeated with the same seed.
    """
    worker_seed = torch.initial_seed() % 2**32  # the helper's torch seed, cut down to the 32 bits NumPy accepts
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loaders(cfg: RunConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build the train, validation and test loaders for one run."""
    train_samples, val_samples = split_train_val(read_labels(DATA_DIR / "train.txt"), cfg.seed)
    if cfg.train_on_val:  # final retrain: all 4,829 training photos learn; the validation scores become meaningless
        train_samples = train_samples + val_samples
    test_samples = read_labels(DATA_DIR / "test.txt")

    train_set = BirdDataset(DATA_DIR / "Train", train_samples, build_transform(cfg.augment, cfg.image_size, cfg.augment_strength))
    val_set = BirdDataset(DATA_DIR / "Train", val_samples, build_transform(False, cfg.image_size))  # never augment scoring photos
    test_set = BirdDataset(DATA_DIR / "Test", test_samples, build_transform(False, cfg.image_size))

    # The loader's own random generator: it decides the photo order of every epoch and the seed of every helper.
    # Seeding it makes the order and the augmentation repeat. GPU arithmetic is still not exactly repeatable
    # (some GPU sums add numbers in a different order each time), so a repeat can differ slightly in its scores.
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)

    # num_workers: background processes read and resize photos while the GPU trains.
    # persistent_workers keeps them alive between epochs (only possible when there are helpers to keep);
    # pin_memory speeds up the copy to the GPU.
    options = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, persistent_workers=cfg.num_workers > 0,
                   pin_memory=True, worker_init_fn=seed_worker)
    return (
        # shuffle: a new random order every epoch (mini-batch GD, W3 s50-57), drawn from the seeded generator
        DataLoader(train_set, shuffle=True, generator=generator, **options),
        # Validation and test: read in the main process, no helpers. Each helper reserves about 1.7 GB of memory,
        # and 4 extra validation helpers crashed the first check run ("paging file is too small"). These photos are
        # never augmented, so skipping helpers only slows scoring a little: same photos, same order, same scores.
        DataLoader(val_set, shuffle=False, batch_size=cfg.batch_size, pin_memory=True),
        DataLoader(test_set, shuffle=False, batch_size=cfg.batch_size, pin_memory=True),
    )
