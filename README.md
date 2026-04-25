# Chest Radiology Deep Learning — Shape vs. Texture Bias Analysis

Investigates whether **DenseNet121** trained on CheXpert chest X-rays relies on
texture or shape features for pathology classification, using style-transfer-based
training perturbations and a reliance ratio framework adapted from
[Geirhos et al. (ICLR 2019)](https://arxiv.org/abs/1811.12231).

---

## Experiment Overview

We train **5 DenseNet121 models** on different versions of the CheXpert training
set, then evaluate all of them on the same original validation/test set to
isolate the effect of each bias.

| Model | Training data | Bias introduced |
|-------|--------------|-----------------|
| `original` | Real chest X-rays | None — baseline |
| `gb` | Gaussian-blurred X-rays | Shape (blur removes texture) |
| `ps` | Patch-shuffled X-rays | Texture (shuffle destroys global shape) |
| `ce` | Canny-edge X-rays | Shape (only structural edges remain) |
| `pr` | Patch-rotated X-rays | Texture (local rotation disrupts shape) |

After training, we run all 5 test sets through each biased model and compute
**reliance ratios** (stylized AUC ÷ original AUC). A ratio > 1 on a matching
test set confirms the bias was baked into the model weights.

---

## Setup

### 1. Clone and create the conda environment

```bash
git clone https://github.com/<your-org>/chest-radiology-deep-learning-bias-analysis.git
cd chest-radiology-deep-learning-bias-analysis
conda create -n DL_PROJECT python=3.12 -y
conda activate DL_PROJECT
pip install -e ".[dev]"
```

### 2. Add Kaggle credentials

Copy the example env file and fill in your Kaggle API key
(get it from [kaggle.com/settings](https://www.kaggle.com/settings) → API → Create New Token).

```bash
cp .env.example .env
```

Edit `.env`:
```
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

### 3. Download the CheXpert dataset

```bash
python -m src.data.download_raw_data
```

This downloads CheXpert-v1.0-small from Kaggle and places it under `src/data/1/`.
After download you should have:

```
src/data/1/
  train.csv
  valid.csv
  train/
  valid/
```

### 4. Generate Parquet manifests

Converts the raw CheXpert CSVs into Parquet manifests for all 5 experiments.
Applies U-Zero labelling policy, frontal-view filtering, and a 97/3
patient-level train/valid split (seed=42, no patient leakage).

```bash
python -m src.data.generate_manifests
```

This writes to `src/data/`:

```
train_manifest.parquet       valid_manifest.parquet      test_manifest.parquet
train_manifest_gb.parquet    test_manifest_gb.parquet
train_manifest_ps.parquet    test_manifest_ps.parquet
train_manifest_ce.parquet    test_manifest_ce.parquet
train_manifest_pr.parquet    test_manifest_pr.parquet
```

The **same patient split** is applied to every stylized variant so all
experiments share identical train/valid/test patient groups.
`valid_manifest.parquet` always contains original (unmodified) images and is
used for checkpoint selection across **all** experiments.

### 5. Generate stylized images

Applies all four style-transfer transforms to the train and test images in-place
(stylized files are saved alongside originals with a filename suffix).

```bash
python -m src.data.style_transfer.generate_stylized
```

Output suffixes written to `src/data/1/`:

| Suffix | Transform | Bias type |
|--------|-----------|-----------|
| `_gb.jpg` | Gaussian blur | Shape |
| `_ps.jpg` | Patch shuffle | Texture |
| `_ce.jpg` | Canny edge | Shape |
| `_pr.jpg` | Patch rotation | Texture |

---

## Training

All 5 models use identical hyperparameters — only the training parquet and
output directory differ per config.

```bash
# Baseline (already done if you have results/original/best_model.pth)
python -m src.train --config src/configs/train_original.yaml

# 4 biased models — paste as one block to run sequentially overnight
python -m src.train --config src/configs/train_gb.yaml; python -m src.train --config src/configs/train_ps.yaml; python -m src.train --config src/configs/train_ce.yaml; python -m src.train --config src/configs/train_pr.yaml
```

Each run saves to its `output_dir`:
- `best_model.pth` — weights from the epoch with highest val AUROC
- `training_history.parquet` — per-epoch loss, AUROC, and LR

**Training config** (same for all 5):
- Architecture: DenseNet121, ImageNet pretrained
- Loss: BCEWithLogitsLoss (multi-label)
- Optimiser: Adam (lr=1e-4, cosine LR decay)
- Epochs: 10, Batch size: 16, Image size: 224×224
- Validation: always on `valid_manifest.parquet` (original images)

---

## Evaluation

### Run the bias evaluation matrix

Runs all 4 biased models against all 5 test sets (20 forward passes total),
then computes matching and opposing reliance ratios.

```bash
python -m src.bias_eval
```

Saves to `results/bias_eval/`:
- `auc_matrix.parquet` — raw 4×5 AUROC grid
- `reliance.json` — matching/opposing reliance ratios per model
- `per_label.json` — full 14-label AUROC breakdown

### Run test evaluation on the baseline

```bash
python -m src.evaluate --config src/configs/train_original.yaml
```

Saves `results/original/test_results.json`.

### Plot training curves

```bash
# Val AUROC — all 5 models on one chart
python -m src.plot_curves

# Train vs val loss — one subplot per model
python -m src.plot_curves --loss
```

Saves to `results/training_curves.png` and `results/loss_curves.png`.

---

## Reliance Ratio Interpretation

For each biased model:

```
reliance ratio = AUC on stylized test set / AUC on original test set
```

| Ratio | Meaning |
|-------|---------|
| > 1 on matching test set | Model does **better** on its own style — bias confirmed |
| < 1 on opposing test set | Model does **worse** on the opposite style — bias confirmed |
| Both close to 1.0 | Model learned real features, style had little effect |

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_data_exploration.ipynb` | Dataset statistics, label distributions, sample images |
| `02_training_smoke_test.ipynb` | Sanity check — verify training loop on a small subset |
| `03_grad_cam_analysis.ipynb` | Grad-CAM heatmaps — visualise what each model attends to |

Run from `src/notebooks/` with the `DL_PROJECT` kernel.

---

## Project Structure

```
├── src/
│   ├── train.py                    training script
│   ├── evaluate.py                 single-model test set evaluation
│   ├── bias_eval.py                full 4×5 bias evaluation matrix
│   ├── plot_curves.py              training curve CLI
│   ├── configs/
│   │   ├── train_original.yaml
│   │   ├── train_gb.yaml
│   │   ├── train_ps.yaml
│   │   ├── train_ce.yaml
│   │   └── train_pr.yaml
│   ├── data/
│   │   ├── chexpert_dataset.py     PyTorch Dataset class
│   │   ├── download_raw_data.py    Kaggle download script
│   │   ├── generate_manifests.py   Parquet manifest generation
│   │   ├── style_transfer/
│   │   │   └── generate_stylized.py
│   │   └── *.parquet               generated manifests (not committed)
│   ├── models/
│   │   └── densenet.py             DenseNet121 classifier
│   ├── notebooks/
│   │   ├── 01_data_exploration.ipynb
│   │   ├── 02_training_smoke_test.ipynb
│   │   └── 03_grad_cam_analysis.ipynb
│   └── utils/
│       ├── plotting.py             training curve plotting functions
│       └── reliance.py             reliance ratio computation
├── results/                        model checkpoints + metrics (not committed)
│   ├── original/
│   ├── gb/
│   ├── ps/
│   ├── ce/
│   ├── pr/
│   └── bias_eval/
├── tests/
│   └── test_chexpert_dataset.py
└── pyproject.toml
```

---

## References

1. Geirhos et al. — *ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness.* ICLR 2019. [arXiv:1811.12231](https://arxiv.org/abs/1811.12231)

2. Zunaed et al. — *Learning to Generalize towards Unseen Domains via a Content-Aware Style Invariant Model for Disease Detection from Chest X-rays.* IEEE JBHI 2024. [DOI:10.1109/JBHI.2024.3372999](https://doi.org/10.1109/JBHI.2024.3372999)

3. Hernandez-Cruz et al. — *Neural Style Transfer as Data Augmentation for Improving COVID-19 Diagnosis Classification.* SN Computer Science 2021. [DOI:10.1007/s42979-021-00795-2](https://doi.org/10.1007/s42979-021-00795-2)
