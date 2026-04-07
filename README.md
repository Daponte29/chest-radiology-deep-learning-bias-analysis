# CheXpert: Shape vs. Texture Bias Analysis

This project investigates the **inductive biases** of Convolutional Neural Networks (specifically **DenseNet121**) when applied to medical imaging (Chest X-Rays).

## 🔬 The Experiment: Shape vs. Texture

Standard CNNs trained on ImageNet are known to be strongly biased towards **texture** (local patterns) rather than **shape** (global structure). In medical imaging, however, pathology detection often relies on shape and structural anomalies (e.g., the size of the heart in Cardiomegaly or the edge of a lung in Pneumothorax).

### Goal
To determine if our DenseNet121 model relies more on texture or shape features for classification, and how this affects its robustness.

### Methodology
We employ **Style Transfer** to create conflicting stimuli:
1.  **Original Images**: Standard CheXpert X-rays (Baseline).
2.  **Stylized Images**: CheXpert X-rays with textures replaced by artistic styles (e.g., Van Gogh, Picasso) while preserving global shape.
3.  **Texture-less Images**: (Optional) Silhouettes or edge maps.

By testing the model on these stylized variations, we can measure the performance drop. A significant drop suggests the model was over-relying on specific textural cues that were destroyed by the style transfer.

## 🛠 Project Structure
- `src/`: Source code for models, data loading, and training.
- `src/data/`: Dataset handling (CheXpert).
- `src/models/`: DenseNet121 and Style Transfer implementations.
- `pyproject.toml`: Project dependencies and configuration.

## 📦 Installation
This project uses `pyproject.toml` for dependency management.

```bash
pip install -e .
```
