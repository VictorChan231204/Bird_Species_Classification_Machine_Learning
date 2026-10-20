# Bird Species Classification

COS30082 Applied Machine Learning, Assignment 1. Victor Chan (104384986).

A CNN that names the species of a bird in a photo, out of the 200 species of Caltech-UCSD Birds 200 (CUB-200).

- **Live web app:** https://huggingface.co/spaces/VictorCKW/bird-species-classifier
- **Trained models (ONNX, all 31 runs):** https://huggingface.co/VictorCKW/bird-species-models

## Results

Final model: `7-final-convnext`, a ConvNeXt-Tiny pretrained on ImageNet-1k and fine-tuned on all 4,829 training
photos. ResNet-50 and ConvNeXt-Tiny tied on the validation photos; both were retrained on all training photos, and
the test set decided between them.

| Model, on the 1,204 test photos | Top-1 accuracy | Average per-class accuracy |
|---|---|---|
| **ConvNeXt-Tiny (`7-final-convnext`), final model** | **89.2%** | **89.0%** |
| ResNet-50 (`7-final-resnet50`), runner-up | 87.5% | 87.4% |

Every run's settings, validation and test scores, and per-epoch learning curve are in `results/<run>.json`. The
experiments and the reason for each setting are described in `src/config.py`.

## Quick start: check the final model's score (no GPU, about 5 minutes)

This scores the published final model on the test photos. It needs no training and no graphics card.

**1. Get the code and a Python 3.12 environment.**

```
git clone https://github.com/VictorChan231204/Bird_Species_Classification_Machine_Learning.git
cd Bird_Species_Classification_Machine_Learning
python -m venv .venv
```

Activate the environment. Every later command assumes it is active.

| System | Command |
|---|---|
| Windows (PowerShell) | `.venv\Scripts\Activate.ps1` |
| Windows (Command Prompt) | `.venv\Scripts\activate.bat` |
| macOS or Linux | `source .venv/bin/activate` |

**2. Install the packages.** PyTorch comes first, because the right build depends on the computer.

```
python -m pip install torch==2.14.0 torchvision==0.29.0
python -m pip install -r requirements.txt
```

That installs the processor-only build of PyTorch, which is all the quick start needs. To train on an NVIDIA GPU,
see "Reproduce the training" below.

**3. Add the dataset.** It is not in this repository. Unzip the assignment's `Train.zip` and `Test.zip`, and place
them with the two label files exactly like this (the photos sit directly inside `Train/` and `Test/`):

```
project_docs/
  Datasets/
    train.txt     4,829 lines: "<file name> <species number 0-199>"
    test.txt      1,204 lines, same format
    Train/        4,829 .jpg photos
    Test/         1,204 .jpg photos
```

**4. Score the final model.**

```
python evaluate.py
```

The script downloads `7-final-convnext.onnx` (57 MB) once into `checkpoints/`, runs all 1,204 test photos through it,
and prints:

```
7-final-convnext on 1204 test photos (ONNX, processor):
  Top-1 accuracy             89.0%
  Average per-class accuracy 88.9%
  Saved during training      89.2% / 89.0% (results/7-final-convnext.json)
```

The ONNX file stores its weights as 16-bit numbers to halve its size, so 5 of the 1,204 guesses differ from the
PyTorch run. That is why the score is 0.2 points under the one saved during training
(`python export_onnx.py --score` repeats this check against the saved PyTorch scores). Any other run can be scored the same way, for example
`python evaluate.py 5-combined`.

## Reproduce the training (NVIDIA GPU)

Training needs an NVIDIA GPU with at least 6 GB of memory. All runs were made on a laptop RTX 4050 (6 GB).

Install the GPU build of PyTorch instead of the one in step 2:

```
python -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
```

Then:

```
python run.py check              # 2-epoch pipeline check, about 1 minute
python run.py 7-final-convnext   # the final model, about 50 minutes
python run.py --help             # every run name; their settings are in src/config.py
```

Each run writes `results/<run>.json` (scores, settings, learning curve) and `checkpoints/<run>.pt` (weights).

| Experiment | Question | Runs |
|---|---|---|
| 1 | Do ImageNet weights and fine-tuning matter? | `1-random-start`, `1-frozen-new-layer`, `1-frozen-new-layer-plain`, `1-frozen-svm-linear`, `1-frozen-svm-rbf` |
| 2 | Where does a standard fine-tuning recipe land? | `2-control` |
| 3 | Learning rate schedule and optimiser | `3-adam-cosine`, `3-sgd-lr1e-03`, `3-sgd-lr3e-03`, `3-sgd-lr1e-02` |
| 4 | Does a batch of 24 beat 12? | `4-batch24-lr3e-03`, `4-batch24-lr6e-03` |
| 5 | Can it stop memorising the training photos? | `5-label-smoothing`, `5-weight-decay`, `5-dropout`, `5-strong-augment`, `5-combined` |
| 6 | Does the network design matter? | `6-resnet18-*`, `6-convnext-*` |
| 7 | Retrain the two tied networks on all photos | `7-final-convnext`, `7-final-resnet50` |

Runs ending in `-seed43` repeat a recipe with a second random seed, to measure how much scores move by chance.

## Run the web app locally

Needs Node.js (built and tested with Node 24) and an internet connection: the app downloads each model from the
model repo when it is picked.

```
cd app
npm ci
npm run dev
```

Then open the address it prints (usually http://localhost:5173). The networks run in the browser with
onnxruntime-web, on the graphics card when the browser supports WebGPU, else on the processor.

## Repository layout

| Path | Contents |
|---|---|
| `src/config.py` | Every experiment's settings, with the reason for each choice and its result |
| `src/data.py` | Reading the label files, the train/validation split, augmentation, data loaders |
| `src/models.py` | The networks: ResNet-18, ResNet-50, ConvNeXt-Tiny with a new 200-species layer |
| `src/train.py` | The training loop, learning rate schedule and scoring |
| `src/svm.py` | Frozen-network features with SVMs on top (experiment 1) |
| `src/metrics.py` | Top-1 accuracy and average per-class accuracy |
| `run.py` | Trains and scores one run by name |
| `evaluate.py` | Scores a published run on the test photos, no GPU needed |
| `results/` | One JSON file per run: settings, scores, learning curve |
| `plot.py` | The report's charts, from `results/` |
| `measure.py` | GPU memory and speed per batch size |
| `export_onnx.py` | Exports runs to ONNX and checks them against PyTorch's guesses |
| `make_app_data.py` | Writes the web app's data files from `results/` |
| `deploy_space.py` | Uploads the web app and the ONNX files to Hugging Face |
| `app/` | The web app (Vite and TypeScript) |

## Troubleshooting

| Problem | Fix |
|---|---|
| `project_docs/Datasets/test.txt is missing` | The dataset is not in place. Follow step 3 of the quick start. |
| `FileNotFoundError` for a photo in `Train/` or `Test/` | The zip unpacked into an extra folder. Move the photos up so they sit directly in `Train/` and `Test/`. |
| PowerShell will not run `Activate.ps1` | Run `Set-ExecutionPolicy -Scope Process Bypass` in that window, then activate again. |
| `run.py` trains very slowly | PyTorch is the processor-only build. Install the GPU build (see "Reproduce the training"). |
| `export_onnx.py` crashes with a `UnicodeEncodeError` on Windows | Set `PYTHONIOENCODING=utf-8` first: the exporter prints emoji. |
