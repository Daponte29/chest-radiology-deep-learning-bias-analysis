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

| Model | Training data | Bias induced |
|----|----|----|
| `original` | Real chest X-rays | None — baseline |
| `gb` | Gaussian-blurred X-rays | Texture (blur-trained; model adapts to texture-free inputs) |
| `ps` | Patch-shuffled X-rays | Texture (shuffle-trained; relies on local patch statistics) |
| `ce` | Canny-edge X-rays | Shape (edge-trained; relies on structural edge features) |
| `pr` | Patch-rotated X-rays | Shape (rotation-trained; adapts to shape-disrupted inputs) |

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
python -m data.generate_manifests
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
python -m data.style_transfer_algos.generate_stylized
```

Output suffixes written to `src/data/1/`:

| Suffix | Transform | Bias induced |
|----|----|----|
| `_gb.jpg` | Gaussian blur | Texture |
| `_ps.jpg` | Patch shuffle | Texture |
| `_ce.jpg` | Canny edge | Shape |
| `_pr.jpg` | Patch rotation | Shape |


---

## Training

Training configs live in `src/configs/`. Each YAML specifies the training parquet,
output directory, loss function, and sampler settings.

```bash
# Baseline
python -m src.train --config src/configs/train_original.yaml

# 4 biased models — paste as one block to run sequentially overnight
python -m src.train --config src/configs/train_gb.yaml
python -m src.train --config src/configs/train_ps.yaml
python -m src.train --config src/configs/train_ce.yaml
python -m src.train --config src/configs/train_pr.yaml
```

Each run saves to the `output_dir` specified in its YAML:

* `best_model.pth` — weights from the epoch with highest val AUROC
* `training_history.parquet` — per-epoch loss, AUROC, and LR

**Reference configs** (4 completed training runs) are archived under
`src/configs/archive_results_configs/config_1/` through `config_4/`, each with
its own YAML set and `results/` subfolder.


---

## Evaluation

### Run the bias evaluation matrix

Runs all 4 biased models against all 5 test sets (20 forward passes total),
then computes matching and opposing reliance ratios.

```bash
# Against a specific config's checkpoints
python -m src.bias_eval --results-dir src/configs/archive_results_configs/config_1/results
```

Saves to `<results-dir>/bias_eval/`:

* `auc_matrix.parquet` — raw 4×5 AUROC grid
* `reliance.json` — matching/opposing reliance ratios per model
* `per_label.json` — full 14-label AUROC breakdown

### Run test evaluation on the baseline

```bash
python -m src.evaluate --config src/configs/train_original.yaml
```

Saves `test_results.json` to the config's `output_dir`.

### Plotting

All plots are generated from a single unified CLI:

```bash
# Training curves (val AUROC + loss) for one config
python -m src.plot curves --results-dir src/configs/archive_results_configs/config_1/results

# Multi-config AUROC comparison + heatmaps (auto-discovers all config_* folders)
python -m src.plot compare

# Matching vs opposing reliance for all configs with bias_eval output
python -m src.plot reliance
```

All three subcommands default to `--archive src/configs/archive_results_configs`.
Add `--no-show` to suppress interactive display (e.g. for headless runs).


---

## Reliance Ratio Interpretation

For each biased model:

```
reliance ratio = AUC on stylized test set / AUC on original test set
```

| Ratio | Meaning |
|----|----|
| > 1 on matching test set | Model does **better** on its own style — bias confirmed |
| < 1 on opposing test set | Model does **worse** on the opposite style — bias confirmed |
| Both close to 1.0 | Model learned real features, style had little effect |


---

## Notebooks

| Notebook | Purpose |
|----|----|
| `01_data_exploration.ipynb` | Dataset statistics, label distributions, sample images |
| `02_training_smoke_test.ipynb` | Sanity check — verify training loop on a small subset |
| `03_grad_cam_analysis.ipynb` | Grad-CAM heatmaps — visualise what each model attends to |

Run from `notebooks/` with the `DL_PROJECT` kernel.


---

## Inference UI (Streamlit)

A self-contained web app for running inference on new chest X-rays.

```bash
streamlit run deploy/app.py
```

**Tabs:**
- **Live Demo** — pre-loaded sample X-ray with cached predictions (no upload needed)
- **Upload Your Own** — drop any chest X-ray JPEG and get results instantly

**Outputs per inference:**
- Top-3 predicted conditions with probability percentages
- Confidence bar chart across all 14 CheXpert labels
- Grad-CAM heatmap overlay showing which image regions drove the top prediction

The app is also deployed on Streamlit Community Cloud — see the repo description for the live link.


---

## Project Structure

```
├── configs/
│   └── base.yaml                       canonical hyperparameter config
├── data/
│   ├── download_raw_data.py            Kaggle download script
│   ├── generate_manifests.py           Parquet manifest generation
│   ├── style_transfer_algos/
│   │   ├── generate_stylized.py        applies all 4 transforms to frontal images
│   │   ├── texture/
│   │   │   ├── gaussian_blur.py
│   │   │   └── patch_shuffle.py
│   │   └── shape/
│   │       ├── canny_edge.py
│   │       └── patch_rotation.py
│   ├── raw/                            raw downloads (not committed)
│   ├── processed/                      preprocessed data (not committed)
│   └── external/                       third-party reference data (not committed)
├── deploy/
│   ├── app.py                          Streamlit inference UI
│   ├── assets/
│   │   └── sample_xray.jpg             demo image (patient64711)
│   ├── requirements.txt                deployment dependencies
│   └── Dockerfile                      container image for cloud deployment
├── infra/
│   ├── terraform/
│   │   ├── main.tf                     S3 bucket, ECR repo, SageMaker IAM role
│   │   └── variables.tf
│   └── cdk/
│       └── stack.py                    AWS CDK equivalent (Python)
├── monitoring/
│   ├── drift.py                        KS / chi-squared drift detection
│   ├── alerts.yaml                     alert thresholds (AUROC drop, latency)
│   └── dashboard.json                  Grafana / CloudWatch panel scaffold
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_training_smoke_test.ipynb
│   └── 03_grad_cam_analysis.ipynb
├── src/
│   ├── chexpert_dataset.py             PyTorch Dataset class
│   ├── train.py                        training script
│   ├── evaluate.py                     single-model test set evaluation
│   ├── bias_eval.py                    full 4×5 bias evaluation matrix
│   ├── plot.py                         unified plotting CLI (curves / compare / reliance)
│   ├── configs/
│   │   ├── train_original.yaml         active training configs (for new runs)
│   │   ├── train_gb.yaml
│   │   ├── train_ps.yaml
│   │   ├── train_ce.yaml
│   │   ├── train_pr.yaml
│   │   └── archive_results_configs/    completed training runs (configs + results)
│   │       ├── config_1/               Config 1 — BCE, no sampler, 11 labels
│   │       │   ├── train_original.yaml
│   │       │   ├── train_gb.yaml  ...
│   │       │   └── results/
│   │       │       ├── original/       best_model.pth, test_results.json
│   │       │       ├── gb/ ps/ ce/ pr/
│   │       │       └── bias_eval/      auc_matrix.parquet, reliance.json
│   │       ├── config_2/               Config 2 — Focal γ=1.5 + sampler, 14 labels
│   │       ├── config_3/               Config 3 — BCE + sampler, 14 labels
│   │       └── config_4/               Config 4 — Focal γ=2.0, no sampler, 14 labels
│   ├── data/
│   │   ├── 1/                          CheXpert images + CSVs (not committed — re-download with data/download_raw_data.py)
│   │   └── *.parquet                   generated manifests (not committed — regenerate with data/generate_manifests.py)
│   ├── models/
│   │   └── densenet.py                 DenseNet121 classifier
│   └── utils/
│       └── reliance.py                 reliance ratio computation
├── tests/
│   └── test_chexpert_dataset.py
├── .github/
│   └── workflows/
│       ├── ci.yml                      test + lint on every push
│       ├── train.yaml                  manual SageMaker training dispatch (workflow_dispatch)
│       └── deploy.yaml                 manual Docker build + ECR push (workflow_dispatch — requires AWS secrets)
├── results/                            generated figures (not committed)
├── Makefile                            common commands (train, evaluate, app, docker-build …)
├── .env.example                        env var template (copy to .env, never commit)
└── pyproject.toml
```


---

## References


1. Geirhos et al. — *ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness.* ICLR 2019. [arXiv:1811.12231](https://arxiv.org/abs/1811.12231)
2. Zunaed et al. — *Learning to Generalize towards Unseen Domains via a Content-Aware Style Invariant Model for Disease Detection from Chest X-rays.* IEEE JBHI 2024. [DOI:10.1109/JBHI.2024.3372999](https://doi.org/10.1109/JBHI.2024.3372999)
3. Hernandez-Cruz et al. — *Neural Style Transfer as Data Augmentation for Improving COVID-19 Diagnosis Classification.* SN Computer Science 2021. [DOI:10.1007/s42979-021-00795-2](https://doi.org/10.1007/s42979-021-00795-2)


