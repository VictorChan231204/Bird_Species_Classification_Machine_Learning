"""Exports the two final networks to ONNX for the web app, and proves the exported ensemble scores like the original.

Usage, from the project root:
    .venv/Scripts/python export_onnx.py           # export both networks, then score the test photos through ONNX
    .venv/Scripts/python export_onnx.py --score   # skip the export; only score the files already in checkpoints/
    .venv/Scripts/python export_onnx.py --all     # export every run the app offers, each checked on 100 test photos

ONNX is a file format that any program can run, not only PyTorch. The web app runs these files in the visitor's
browser with onnxruntime-web, so no server is needed. This script writes, per network:
    checkpoints/<name>.onnx - the network, its weights stored as float16 (half the size of float32), gitignored

What goes in the file is more than the bare network (see BrowserModel): the file takes plain pixels (0-1) and
returns probabilities, so the browser only resizes, crops and divides by 255. The standardising (W5 s80) and the
softmax (W3 s43) happen inside the file, exactly as in training and scoring.

The proof: the test photos go through onnxruntime (the same engine as onnxruntime-web, on the processor), the two
networks' probabilities are averaged like 8-final-ensemble (src/ensemble.py), and the script prints Top-1 and
average per-class next to the PyTorch scores saved in checkpoints/<name>-probs.npz. It also counts the test photos
whose guess changed. Writes results/8-final-ensemble-onnx.json.
"""

import argparse
import time

import joblib

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import numpy_helper
from PIL import Image
from torch import nn
from torchvision import transforms

from src.config import COMBINATIONS, EXPERIMENTS
from src.data import DATA_DIR, IMAGENET_MEAN, IMAGENET_STD, RESIZE_RATIO, Sample, read_labels
from src.models import build_model, set_head
from src.svm import FEATURES_DIR
from src.train import CHECKPOINT_DIR, RESULTS_DIR, summarise, tidy_json

ENSEMBLE = COMBINATIONS["8-final-ensemble"]  # the final model: ResNet-50 + ConvNeXt-Tiny, equal weights, no flip


class BrowserModel(nn.Module):
    """The trained network wrapped so the exported file does the whole job from pixels to probabilities.

    Input: one photo, shape (1, 3 colours, 448, 448), pixel values 0-1.
    Output: shape (1, 200), the probability of each species; the row sums to 1.
    """

    def __init__(self, network: nn.Module):
        super().__init__()
        self.network = network
        # Stored as buffers so they travel into the ONNX file as constants. Shape (1, 3, 1, 1): one value per colour.
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        standardised = (pixels - self.mean) / self.std  # the same Normalize step as the test transform (W5 s80)
        return self.network(standardised).softmax(dim=1)  # scores -> probabilities that sum to 1 (W3 s43)


class SvmBrowserModel(nn.Module):
    """A frozen-features + SVM run (1-frozen-svm-*) as one network, so the browser runs it like any other file.

    The steps match src/svm.py: the frozen ImageNet ResNet-50 turns the photo into 2,048 features, StandardScaler
    puts each feature on the same ruler, and 200 one-vs-rest SVMs each give a score. The output is the 200 SVM scores
    (signed distance from each species' fence), not probabilities: an SVM has no softmax.

    Linear kernel: each SVM is one weight per feature plus a bias, so all 200 fold into one matrix multiply.
    RBF kernel: each SVM sums its support vectors' similarity to the photo, exp(-gamma * squared distance), times
    their learned weights (dual coefficients). The 200 SVMs share one table of support vectors, the union of theirs;
    a support vector one SVM does not use gets weight 0 in that SVM's column.
    """

    def __init__(self, trunk: nn.Module, svm, train_features: np.ndarray):
        super().__init__()
        self.trunk = trunk  # frozen ResNet-50 with its last layer removed: photo -> 2,048 features
        scaler, one_vs_rest = svm.steps[0][1], svm.steps[1][1]
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.register_buffer("feature_mean", torch.tensor(scaler.mean_, dtype=torch.float32))
        self.register_buffer("feature_scale", torch.tensor(scaler.scale_, dtype=torch.float32))
        machines = one_vs_rest.estimators_  # 200 binary SVCs, one per species, in species order
        self.kernel = machines[0].kernel
        bias = np.array([m.intercept_[0] for m in machines], dtype=np.float32)
        self.register_buffer("bias", torch.tensor(bias))
        if self.kernel == "linear":
            # (2048, 200): column s holds species s's weight for every feature
            weights = np.stack([m.coef_[0] for m in machines], axis=1).astype(np.float32)
            self.register_buffer("weights", torch.tensor(weights))
        else:
            used = np.unique(np.concatenate([m.support_ for m in machines]))  # training photos any SVM relies on
            column = {photo: row for row, photo in enumerate(used)}
            dual = np.zeros((len(used), len(machines)), dtype=np.float32)  # (support vectors, 200 species)
            for species, m in enumerate(machines):
                for photo, coef in zip(m.support_, m.dual_coef_[0]):
                    dual[column[photo], species] = coef
            # The support vectors in standardised units, as the SVMs saw them.
            vectors = scaler.transform(train_features[used]).astype(np.float32)
            self.register_buffer("vectors", torch.tensor(vectors))
            self.register_buffer("vector_norms", torch.tensor((vectors**2).sum(axis=1)))
            self.register_buffer("dual", torch.tensor(dual))
            self.gamma = float(machines[0]._gamma)  # gamma="scale" resolved at fit time: 1 / (2,048 * feature variance)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        features = self.trunk((pixels - self.mean) / self.std)  # (1, 2048)
        x = (features - self.feature_mean) / self.feature_scale  # StandardScaler
        if self.kernel == "linear":
            return x @ self.weights + self.bias  # (1, 200) SVM scores
        # Squared distance to every support vector, written as |x|^2 - 2 x.v + |v|^2 so it is one matrix multiply.
        distance = (x**2).sum(dim=1, keepdim=True) - 2 * x @ self.vectors.T + self.vector_norms
        return torch.exp(-self.gamma * distance) @ self.dual + self.bias  # (1, 200) SVM scores


def weights_to_float16(model: onnx.ModelProto) -> onnx.ModelProto:
    """Store every float32 weight as float16, and add a Cast node that turns it back into float32 when loaded.

    Float16 uses 2 bytes per number instead of 4, so the download halves. The arithmetic stays float32: onnxruntime
    runs the Casts once, when it loads the file, and then computes exactly as before. Rounding a weight to float16
    keeps about 3 significant digits (0.123456 -> 0.1235), which barely moves the probabilities.
    Why not compute in float16 too: onnxruntime's processor engine (onnxruntime-web's fallback when the browser has
    no WebGPU) lacks float16 versions of many layers, so a fully float16 file may not run there.
    """
    graph = model.graph
    casts = []
    for weight in list(graph.initializer):
        if weight.data_type != onnx.TensorProto.FLOAT:
            continue  # integer constants (shapes, indices) stay as they are
        half = numpy_helper.from_array(numpy_helper.to_array(weight).astype(np.float16), weight.name + "_fp16")
        graph.initializer.remove(weight)
        graph.initializer.append(half)
        # The Cast writes its float32 result under the weight's old name, so the layers that use it need no change.
        casts.append(onnx.helper.make_node("Cast", [half.name], [weight.name], to=onnx.TensorProto.FLOAT))
    # Casts first: ONNX requires every value to be made before the node that reads it.
    nodes = casts + list(graph.node)
    del graph.node[:]
    graph.node.extend(nodes)
    return model


def export(name: str) -> None:
    """Load the trained weights of one run, wrap them for the browser, and write checkpoints/<name>.onnx.
    A trained run becomes BrowserModel (pixels -> probabilities); an SVM run becomes SvmBrowserModel
    (pixels -> SVM scores)."""
    cfg = EXPERIMENTS[name]
    if cfg.svm_kernel:
        trunk = build_model(cfg)  # the frozen ImageNet ResNet-50, exactly as src/svm.py builds it
        set_head(trunk, nn.Identity())  # remove the decider: the network now outputs its 2,048 features
        features = np.load(FEATURES_DIR / f"{cfg.arch}-{cfg.image_size}-seed{cfg.seed}.npz")["x_train"]
        model = SvmBrowserModel(trunk, joblib.load(CHECKPOINT_DIR / f"{name}.joblib"), features).eval()
        output = "scores"
    else:
        network = build_model(cfg)  # the same network shape as in training, including the dropout around the new layer
        network.load_state_dict(torch.load(CHECKPOINT_DIR / f"{name}.pt", map_location="cpu"))
        model = BrowserModel(network).eval()  # eval: dropout off, batch normalisation uses its saved averages
        output = "probs"

    # A random photo: only its shape matters. The file accepts exactly one photo per call, which is all the app needs
    # (a variable batch size failed to export for ConvNeXt-Tiny: "Invalid ranges [2:1]").
    example = torch.rand(1, 3, cfg.image_size, cfg.image_size)
    program = torch.onnx.export(model, (example,), input_names=["pixels"], output_names=[output], dynamo=True,
                                verbose=False)  # its progress lines hold emoji that crash a Windows console pipe
    proto = weights_to_float16(program.model_proto)
    onnx.checker.check_model(proto)  # fails loudly if the file breaks a rule of the ONNX format
    path = CHECKPOINT_DIR / f"{name}.onnx"
    onnx.save(proto, path)
    print(f"{name}: wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")


def app_runs() -> list[str]:
    """Every run the web app can classify with: each trained run and SVM run with saved weights, except the
    pipeline checks. The combinations (experiment 7 and 8-final-ensemble) need no file: the app averages members."""
    return [name for name, cfg in EXPERIMENTS.items()
            if not name.startswith("check")
            and (CHECKPOINT_DIR / f"{name}.{'joblib' if cfg.svm_kernel else 'pt'}").exists()]


def quick_check(name: str, photos: int = 100) -> None:
    """Score the first `photos` test photos through the ONNX file and count how many guesses differ from Python's.
    Python's guesses: the saved probabilities (checkpoints/<name>-probs.npz) for a trained run, or the saved SVM on
    the cached features for an SVM run. A few changed guesses are expected from float16 weights; many mean a bug."""
    cfg = EXPERIMENTS[name]
    session = ort.InferenceSession(CHECKPOINT_DIR / f"{name}.onnx", providers=["CPUExecutionProvider"])
    samples, to_pixels = test_pixels(cfg.image_size)
    samples = samples[:photos]
    onnx_guess = np.array([
        session.run(None, {"pixels": to_pixels(Image.open(DATA_DIR / "Test" / f).convert("RGB")).unsqueeze(0).numpy()})[0][0].argmax()
        for f, _ in samples
    ])
    if cfg.svm_kernel:
        features = np.load(FEATURES_DIR / f"{cfg.arch}-{cfg.image_size}-seed{cfg.seed}.npz")["x_test"][:photos]
        python_guess = joblib.load(CHECKPOINT_DIR / f"{name}.joblib").predict(features)
    else:
        python_guess = np.load(CHECKPOINT_DIR / f"{name}-probs.npz")["test"][:photos].argmax(axis=1)
    changed = int((onnx_guess != python_guess).sum())
    print(f"{name}: {changed} of {photos} guesses differ from Python", flush=True)
    if changed > photos * 0.05:  # more than 5 in 100: not rounding, a mistake in the export
        raise SystemExit(f"{name}: the ONNX file disagrees with Python on {changed} of {photos} photos")


def test_pixels(image_size: int) -> tuple[list[Sample], transforms.Compose]:
    """The test photos as (file name, label) pairs, plus the transform that matches training's test transform minus
    Normalize (BrowserModel does that part)."""
    samples = read_labels(DATA_DIR / "test.txt")
    to_pixels = transforms.Compose([
        transforms.Resize(round(image_size * RESIZE_RATIO)),  # short side to 512, keeping the shape
        transforms.CenterCrop(image_size),  # the middle 448 x 448 square
        transforms.ToTensor(),  # pixels 0-255 -> 0-1, shape (3, 448, 448)
    ])
    return samples, to_pixels


def onnx_probs(name: str) -> np.ndarray:
    """Probabilities of one exported network for every test photo, in test.txt order (the order of the -probs.npz)."""
    session = ort.InferenceSession(CHECKPOINT_DIR / f"{name}.onnx", providers=["CPUExecutionProvider"])
    samples, to_pixels = test_pixels(EXPERIMENTS[name].image_size)
    rows, start = [], time.time()
    for file_name, _ in samples:  # one photo per call, as in the app
        pixels = to_pixels(Image.open(DATA_DIR / "Test" / file_name).convert("RGB")).unsqueeze(0)  # (1, 3, 448, 448)
        rows.append(session.run(["probs"], {"pixels": pixels.numpy()})[0])
    print(f"{name}: scored {len(samples)} test photos in {time.time() - start:.0f} s")
    return np.concatenate(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export networks to ONNX and check them against Python's scores.")
    parser.add_argument("--score", action="store_true", help="skip the export; score the existing final files")
    parser.add_argument("--all", action="store_true",
                        help="export every run the app offers (skips files already there), with a 100-photo check each")
    args = parser.parse_args()
    if args.all:
        for name in app_runs():
            if not (CHECKPOINT_DIR / f"{name}.onnx").exists():
                export(name)
            quick_check(name)
        return
    if not args.score:
        for name in ENSEMBLE.members:
            export(name)

    labels = np.array([label for _, label in read_labels(DATA_DIR / "test.txt")])
    result = {"config": {"members": list(ENSEMBLE.members), "weights": "float16, computed in float32"}}
    onnx_tables, torch_tables = [], []
    for name in ENSEMBLE.members:
        saved = np.load(CHECKPOINT_DIR / f"{name}-probs.npz")
        assert np.array_equal(saved["test_labels"], labels), "the saved probabilities follow a different photo order"
        onnx_tables.append(onnx_probs(name))
        torch_tables.append(saved["test"])
        result[name] = compare(onnx_tables[-1], torch_tables[-1], labels)
    result["8-final-ensemble"] = compare(np.mean(onnx_tables, axis=0), np.mean(torch_tables, axis=0), labels)
    for key in (*ENSEMBLE.members, "8-final-ensemble"):
        r = result[key]
        print(f"{key}: ONNX top-1 {r['onnx']['top1']:.4f} / per-class {r['onnx']['mean_per_class']:.4f}   "
              f"PyTorch {r['pytorch']['top1']:.4f} / {r['pytorch']['mean_per_class']:.4f}   "
              f"guesses changed {r['guesses_changed']}   largest probability gap {r['max_prob_gap']:.4f}")
    (RESULTS_DIR / "8-final-ensemble-onnx.json").write_text(tidy_json(result) + "\n")


def compare(onnx_table: np.ndarray, torch_table: np.ndarray, labels: np.ndarray) -> dict:
    """Both scores for the ONNX and PyTorch probabilities, plus how far apart the two tables are."""
    return {
        "onnx": summarise(onnx_table, labels, per_class=False),
        "pytorch": summarise(torch_table, labels, per_class=False),
        "guesses_changed": int((onnx_table.argmax(1) != torch_table.argmax(1)).sum()),  # photos whose top species moved
        "max_prob_gap": round(float(np.abs(onnx_table - torch_table).max()), 4),  # worst single-probability difference
    }


# Windows starts helper processes by importing this file; the guard keeps them from running main().
if __name__ == "__main__":
    main()
