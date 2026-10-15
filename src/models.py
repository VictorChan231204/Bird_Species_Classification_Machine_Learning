"""Model architecture: builds the network, freezes the copied layers, and sets the learning-rate groups.

train.py calls, in order:
    build_model(cfg)           - the network with a new 200-species output layer
    param_groups(model, cfg)   - which weights the optimiser may change, and with which learning rate
    set_train_mode(model, cfg) - at the start of every epoch
svm.py and measure.py reuse build_model; svm.py also uses set_head to swap the new layer for "nothing".

Every network here has two parts. The copied layers turn a photo into a list of features (512 numbers for
ResNet-18, 2,048 for ResNet-50, 768 for ConvNeXt-Tiny). The last, linear layer turns those features into one
score per class (W6 s22-27). Transfer learning keeps the copied layers and swaps that last layer for the new layer.
In code, head() finds that last linear layer, whether it is still the old one or already the new layer.
"""

from torch import nn
from torchvision import models

from src.config import RunConfig
from src.data import NUM_CLASSES

# The pretrained ImageNet-1k weights for each network (downloaded once, then cached by torchvision).
# ResNet-50 uses V2: same network as V1, but trained with a newer recipe, 80.9% vs 76.1% ImageNet Top-1
# (https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html).
PRETRAINED_WEIGHTS = {
    "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
    "resnet50": models.ResNet50_Weights.IMAGENET1K_V2,
    "convnext_tiny": models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
}


def head(model: nn.Module) -> nn.Module:
    """The network's last linear layer: the old one before build_model, the new layer after. param_groups gives it lr_new.

    A ResNet calls it `fc` (W5 s62-63). ConvNeXt ends in `classifier` = LayerNorm2d, Flatten, Linear(768, 1000);
    only the Linear, `classifier[2]`, is replaced. The LayerNorm stays, because the copied features expect it.
    """
    return model.classifier[2] if isinstance(model, models.ConvNeXt) else model.fc


def set_head(model: nn.Module, layer: nn.Module) -> None:
    """Put `layer` where the network's last linear layer was (the partner of head())."""
    if isinstance(model, models.ConvNeXt):
        model.classifier[2] = layer
    else:
        model.fc = layer


def build_model(cfg: RunConfig) -> nn.Module:
    """Build the network, replace its last layer with the new layer, and freeze the copied layers if cfg.frozen."""
    if cfg.frozen and not cfg.pretrained:
        raise ValueError("Freezing random weights makes no sense: a random start must learn every layer.")

    weights = PRETRAINED_WEIGHTS[cfg.arch] if cfg.pretrained else None  # None = random start (experiment 1 baseline)
    model = getattr(models, cfg.arch)(weights=weights)  # e.g. models.resnet50(weights=...)

    # The old last layer guessed 1,000 ImageNet objects. Replace it with a new one for 200 species (W6 s27).
    # in_features is the length of the feature list: 512 (ResNet-18), 2,048 (ResNet-50) or 768 (ConvNeXt-Tiny).
    new_layer = nn.Linear(head(model).in_features, NUM_CLASSES)
    if cfg.dropout > 0:
        new_layer = nn.Sequential(nn.Dropout(cfg.dropout), new_layer)  # dropout acts on the features (W5 s54-55)
    set_head(model, new_layer)

    if cfg.frozen:
        head_ids = {id(w) for w in head(model).parameters()}
        for weight in model.parameters():
            if id(weight) not in head_ids:
                weight.requires_grad = False  # backpropagation no longer changes this weight: learning rate 0 (W6 s24)
    return model


def param_groups(model: nn.Module, cfg: RunConfig) -> list[dict]:
    """Split the trainable weights into two groups with their own learning rate, for the optimiser.

    Pretrained run: the new layer gets lr_new, the unfrozen copied layers get the smaller
    lr_pretrained, so fine-tuning only nudges the copied knowledge (W6 s28, 30).
    Random start: nothing is copied, so every weight gets lr_new.
    Each group also remembers its peak rate as "base_lr"; train.py scales it for warm-up and cosine decay.
    """
    new_weights = list(head(model).parameters())
    new_ids = {id(w) for w in new_weights}
    copied_weights = [w for w in model.parameters() if w.requires_grad and id(w) not in new_ids]

    copied_lr = cfg.lr_pretrained if cfg.pretrained else cfg.lr_new
    groups = [{"params": new_weights, "lr": cfg.lr_new, "base_lr": cfg.lr_new}]  # group 0 is always the new layer
    if copied_weights:  # empty when the copied layers are frozen (frozen=True)
        groups.append({"params": copied_weights, "lr": copied_lr, "base_lr": copied_lr})
    return groups


def set_train_mode(model: nn.Module, cfg: RunConfig) -> None:
    """Switch the network to training mode, but keep frozen copied layers in evaluation mode.

    Training mode turns on dropout (and ConvNeXt's random layer skipping) and makes batch normalisation
    (W5 s64-67) update its running averages. Frozen copied layers must not change at all, so they stay in
    evaluation mode and only the new layer trains.
    """
    if cfg.frozen:
        model.eval()  # the whole network, including its batch normalisation, as copied
        head(model).train()  # only the new layer (and its dropout, if any) trains
    else:
        model.train()
