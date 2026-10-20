# Bird Species Classification

COS30082 Applied Machine Learning, Assignment 1. Victor Chan (104384986).

A CNN classifier for the 200 bird species of Caltech-UCSD Birds 200 (CUB-200), with a web app that names the
species in a photo.

**Live app:** https://huggingface.co/spaces/VictorCKW/bird-species-classifier

## Final model

`8-final-convnext`: ConvNeXt-Tiny, pretrained on ImageNet-1k and fine-tuned on the 4,829 training photos.
ResNet-50 and ConvNeXt-Tiny tied on the validation photos; both were retrained on all training photos, and the test
set decided between them.

| Model on the 1,204 test photos | Top-1 accuracy | Average per-class accuracy |
|---|---|---|
| **ConvNeXt-Tiny (`8-final-convnext`), final model** | **89.2%** | **89.0%** |
| ResNet-50 (`8-final-resnet50`), runner-up | 87.5% | 87.4% |

Every run's settings, scores and per-epoch history are in `results/<name>.json`.

## The web app

`app/` is a static site (Vite + TypeScript). The networks run in the visitor's browser through onnxruntime-web, on
the graphics card when available. The app shows the final model's top 5 species. A model picker runs any of the 31
trained runs instead, baselines and SVMs included, and the page shows every run's learning curve and scores.

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Data loading, networks, training and scoring code; `src/config.py` defines every experiment |
| `run.py` | Trains and scores one experiment by name |
| `measure.py` | Measures GPU memory and speed per batch size |
| `results/` | One JSON file per run: settings, validation and test scores, learning curve |
| `export_onnx.py` | Exports the runs to ONNX for the browser and checks them against Python's scores |
| `make_app_data.py` | Writes the app's data files and copies the example photos and ONNX files into `app/` |
| `deploy_space.py` | Uploads the built app to the Hugging Face Space |
| `app/` | The web app |

## Running it

Windows, Python 3.12, an NVIDIA GPU (built on an RTX 4050 Laptop GPU, 6 GB).

```
python -m venv .venv
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
.venv/Scripts/python -m pip install -r requirements.txt
```

The dataset is not in this repository. Place the images and label files as:

```
project_docs/Datasets/Train/   project_docs/Datasets/train.txt   (4,829 images, lines "<filename> <label>")
project_docs/Datasets/Test/    project_docs/Datasets/test.txt    (1,204 images)
```

Then:

```
.venv/Scripts/python run.py check              # 2-epoch pipeline check, about 2 minutes
.venv/Scripts/python run.py 8-final-resnet50   # train one experiment (names in src/config.py)
.venv/Scripts/python export_onnx.py --all      # ONNX files for the app, each checked on 100 test photos
.venv/Scripts/python make_app_data.py          # refresh the app's data
cd app && npm install && npm run dev           # run the web app locally
```
