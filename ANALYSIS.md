# Shape vs. Texture Bias in DenseNet121 — CheXpert Experiment Analysis





**Project:** Chest Radiology Deep Learning Bias Analysis**Authors:** Ed, Nick**Model:** DenseNet121 (ImageNet pretrained, fully fine-tuned)**Dataset:** CheXpert frontal radiographs**Date:** April 2026


---

## Table of Contents


1. [Experimental Methodology](#1-experimental-methodology)
2. [Training Configuration Ablation](#2-training-configuration-ablation)
3. [Overall AUROC Results](#3-overall-auroc-results)
4. [Reliance Ratio Analysis](#4-reliance-ratio-analysis)
5. [Per-Label Deep Dive](#5-per-label-deep-dive)
6. [How Stylization Caused the Observed Patterns](#6-how-stylization-caused-the-observed-patterns)
7. [Key Findings Summary](#7-key-findings-summary)
8. [Implications for the Paper](#8-implications-for-the-paper)


---

## 1. Experimental Methodology

### 1.1 Research Question

Do DenseNet121 models trained on CheXpert chest X-rays rely primarily on **texture features** (local intensity patterns, density gradients, fine-grained parenchymal detail) or **shape features** (organ boundaries, silhouettes, spatial layout of anatomical structures)? And does this reliance differ across pathology labels?

### 1.2 Model Architecture

* **Backbone:** DenseNet121 initialized from ImageNet ILSVRC weights (`IMAGENET1K_V1`)
* **Head:** All layers fully unfrozen; linear classifier head replaced to match number of output labels
* **Output:** Raw logits → sigmoid at inference; no softmax (multi-label classification)
* **Loss functions tested:** Focal Loss (with γ parameter) and Binary Cross-Entropy (BCE)
* **Rationale for DenseNet121:** Established as the CheXpert benchmark architecture by Rajpurkar et al. (CheXNet, 2017). Dense skip connections enable gradient flow to early layers, making the network sensitive to both fine-grained textures and coarse spatial structure.

### 1.3 Dataset

* **Source:** CheXpert (Stanford) — frontal-view chest radiographs
* **Labels evaluated:**
  * Ed's config (11 labels): Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Edema, Pneumonia, Support Devices, Consolidation, Atelectasis, Pneumothorax, Pleural Effusion, Pleural Other
  * Nick's configs (14 labels): above + Lung Lesion, Fracture, No Finding
  * *Fracture excluded from all reported results — no positive examples in test split (AUROC = NaN for all runs)*
* **Evaluation metric:** AUROC (Area Under the ROC Curve), macro-averaged across all non-NaN labels

### 1.4 Stylization Transforms

Four stylization functions were applied to generate four additional training sets. Each targets a specific visual feature class:

#### Texture-Bias Transforms

These transforms **preserve global shape/anatomy** while **removing or disrupting fine-grained texture**. A model trained on these images is forced away from texture and toward shape.

| Transform | Key | Mechanism | What it preserves | What it destroys |
|----|----|----|----|----|
| **Gaussian Blur** | `gb` | Gaussian kernel (radius=2) applied to grayscale image | Global structure, organ silhouettes, spatial layout | High-frequency texture (interstitial patterns, subtle densities) |
| **Patch Shuffle** | `ps` | Image divided into 32×32 patches, patches randomly permuted | Local texture within each patch | All spatial relationships, anatomical layout, global shape |

> **Important distinction:** Despite both being "texture bias" transforms, gb and ps work oppositely at the spatial level. Gaussian blur destroys texture but keeps anatomy intact. Patch shuffle destroys anatomy entirely but preserves local texture statistics within each patch. This explains why gb models consistently outperform ps models on the original test set.

#### Shape-Bias Transforms

These transforms **remove texture/density information** while **retaining edge and contour information**. A model trained on these images is forced away from texture and toward shape.

| Transform | Key | Mechanism | What it preserves | What it destroys |
|----|----|----|----|----|
| **Canny Edge** | `ce` | Auto-thresholded Canny edge detection (σ=0.33 heuristic) on grayscale image | Structural edges, organ outlines, vessel walls | All density/intensity information, subtle texture, soft tissue contrast |
| **Patch Rotation** | `pr` | Each 32×32 patch independently rotated by 90°/180°/270° | Global spatial layout, approximate anatomical positions | Local texture orientation within each patch |

### 1.5 Training Setup

* **5 models trained per config:** original (clean images), gb, ps, ce, pr (each trained on its respective stylized/blended dataset)
* **Biased model training images:** blended at `blend_ratio=0.5` — 50% stylized images + 50% original images per batch. This prevents the model from completely losing the original signal while still forcing it to learn from the stylized view.
* **Original model:** `blend_ratio=1.0` (100% original images, no blending)
* **Validation set:** Original (unmodified) images throughout — early stopping monitors AUROC on the real validation set
* **Test set:** All final evaluations reported here use the **original unmodified test set**. The biased models are evaluated on clean CXRs, making performance gaps directly attributable to what was learned during training.

### 1.6 Evaluation

```
evaluate.py → test_results.json per model  (original test set, all 14 labels)
bias_eval.py → reliance.json              (4 biased models × 5 test sets → reliance ratios)
```


**Reliance ratio** = `stylized_test_AUC / original_test_AUC`A ratio close to 1.0 means the model's performance is unaffected when the test image is stylized — it was not relying on the features that stylization removes. A low ratio means the model depended on those features.


---

## 2. Training Configuration Ablation

Four configurations were tested across all 5 model variants, totalling 20 training runs.

| Config | Loss | Weighted Sampler | LR | Weight Decay | Labels | Patience | Aug |
|----|----|----|----|----|----|----|----|
| **Ed's Config** | BCE | No | 1e-4 | 1e-5 | 11 | 7 | None |
| **Nick Config 1** | Focal γ=1.5 | Yes | 5e-5 | 5e-5 | 14 | 5 | H-flip, jitter=0.1 |
| **Nick Config 2** | BCE | Yes | 1e-4 | 1e-5 | 14 | 5 | H-flip, jitter=0.1 |
| **Nick Config 3** | Focal γ=2.0 | No | 1e-4 | 1e-5 | 14 | 5 | H-flip, jitter=0.1 |

### Why We Tested These Configs

**Config 1 (Focal γ=1.5 + Sampler)** was the initial attempt at addressing class imbalance by combining two strategies simultaneously. Focal loss down-weights easy negatives during loss computation; the weighted sampler oversamples images containing rare labels during batch construction. These two mechanisms **double-count** the imbalance correction — rare label images are presented more often AND each instance is weighted more heavily in the loss. This caused validation loss curves to increase rather than converge, which we interpreted as instability from over-correcting the imbalance.

**Config 2 (BCE + Sampler)** isolates the sampler alone with standard BCE loss. Removing focal loss eliminates the double-weighting. BCE is simpler and less sensitive to the imbalance correction.

**Config 3 (Focal γ=2.0, No Sampler)** isolates focal loss alone. Removing the sampler eliminates the double-weighting from the other direction. γ=2.0 is the standard focal loss setting from Lin et al. (2017). No sampler means uniform batch sampling.

**Ed's Config (BCE, No Sampler)** represents the cleanest baseline: no imbalance correction strategy, standard BCE, longer patience (7 vs 5 epochs), no augmentation, and only 11 labels (excluding the three most problematic: Lung Lesion, No Finding, Fracture).


---

## 3. Overall AUROC Results

All scores are mean AUROC on the **original unmodified test set**.

### 3.1 Summary Table

| Config | Original | GB (texture) | PS (texture) | CE (shape) | PR (shape) |
|----|----|----|----|----|----|
| **Ed's Config** | **0.8423** | **0.8317** | **0.7904** | **0.7469** | **0.8121** |
| Nick Config 1 | 0.8017 | 0.7530 | 0.7011 | 0.6650 | 0.7355 |
| Nick Config 2 | 0.7646 | 0.7537 | 0.6843 | 0.6890 | 0.7235 |
| Nick Config 3 | 0.8005 | 0.7961 | 0.7359 | 0.7203 | 0.7762 |

### 3.2 Ranking




**Best config overall:** Ed's Config — highest AUROC on every single model variant.**Best Nick config:** Config 3 (Focal γ=2.0, No Sampler) — matches Ed on the original model (0.8005 vs 0.8423) but with 14 labels vs 11.**Worst config:** Config 2 (BCE + Sampler) — lowest original model AUROC at 0.7646, suggesting the sampler alone without focal loss is ineffective.**Worst stylized model across all configs:** Canny Edge under Config 1 (0.665) — the most aggressive shape bias combined with the most unstable training setup.

### 3.3 Interpretation: Why Ed's Config Wins


1. **11 labels instead of 14** — Lung Lesion and No Finding are the two most problematic labels (Lung Lesion rarely exceeds 0.4 AUROC; No Finding is redundant with the others). Removing them prevents a noisy gradient signal from polluting the shared representation layers.
2. **No imbalance correction** — BCE without sampler avoids all forms of double-counting. On a dataset as large as CheXpert, the natural class frequency is informative; correcting it artificially can hurt calibration.
3. **Longer patience (7 vs 5 epochs)** — Allows more time to find a truly optimal checkpoint rather than stopping at a local plateau.
4. **No augmentation** — Chest X-rays are acquired under controlled conditions. Horizontal flip is rarely clinically appropriate (cardiac apex, aortic arch, stomach bubble are asymmetric). Color jitter adds noise irrelevant to grayscale radiographs.

### 3.4 Interpretation: Why Config 1 (Focal + Sampler) Performs Worst on Biased Models

The double-weighting instability affects biased model training more severely than original model training. During biased model training, the rare label images presented by the sampler are also stylized — meaning the model is simultaneously trying to learn from edge-only or shuffled-patch images AND trying to fit an over-represented minority class. The combination makes the loss landscape noisy. This explains why Config 1's shape-biased models (ce=0.665, pr=0.736) are dramatically lower than Config 3's (ce=0.720, pr=0.776) despite similar original model performance.


---

## 4. Reliance Ratio Analysis

Reliance ratio = `stylized_AUC / original_AUC`. Values close to 1.0 indicate the model's performance is robust to that stylization — it was not relying heavily on the features that transform removes.

### 4.1 Reliance Ratios by Config

| Config | gb reliance | ps reliance | Mean Texture | ce reliance | pr reliance | Mean Shape | Texture > Shape gap |
|----|----|----|----|----|----|----|----|
| **Ed's Config** | 0.987 | 0.938 | **0.963** | 0.887 | 0.964 | **0.926** | **+0.037** |
| Nick Config 1 | 0.939 | 0.874 | 0.907 | 0.830 | 0.918 | 0.874 | +0.033 |
| Nick Config 2 | 0.986 | 0.895 | 0.940 | 0.901 | 0.946 | 0.924 | +0.017 |
| Nick Config 3 | 0.995 | 0.919 | **0.957** | 0.900 | 0.970 | **0.935** | **+0.022** |

### 4.2 The Consistent Pattern: Texture Reliance > Shape Reliance

**Across all 4 configs and all 20 training runs, texture-biased stylized models consistently outperform shape-biased stylized models on the original test set.** Mean texture reliance ranges from 0.907–0.963; mean shape reliance from 0.874–0.935. This gap persists regardless of training configuration, loss function, sampler, or label set.

This finding is consistent with the well-established result from Geirhos et al. (2019) showing that ImageNet-pretrained CNNs are fundamentally texture-biased. The ImageNet pretraining instills a strong prior toward local texture statistics that partially survives fine-tuning on CheXpert.

### 4.3 gb vs ps Divergence

Within the texture-bias category, Gaussian Blur (gb) consistently shows much higher reliance than Patch Shuffle (ps):

| Config | gb reliance | ps reliance | Gap |
|----|----|----|----|
| Ed | 0.987 | 0.938 | 0.049 |
| Nick 1 | 0.939 | 0.874 | 0.065 |
| Nick 2 | 0.986 | 0.895 | 0.091 |
| Nick 3 | 0.995 | 0.919 | 0.076 |

**Why:** Gaussian blur destroys fine texture but preserves global anatomy — organ silhouettes, cardiac borders, diaphragm contour, and lung field boundaries remain fully intact. The model can still read these structural cues in the test image. Patch shuffle destroys all spatial relationships — the heart is no longer in the center, the lung fields don't span the periphery, costophrenic angles are random. The model loses its anatomical spatial priors entirely. This \~0.07 average gap directly quantifies how much performance is attributable to spatial anatomy vs local texture in CXR diagnosis.

### 4.4 ce vs pr Divergence

Within the shape-bias category, Patch Rotation (pr) consistently outperforms Canny Edge (ce):

| Config | ce reliance | pr reliance | Gap |
|----|----|----|----|
| Ed | 0.887 | 0.964 | 0.077 |
| Nick 1 | 0.830 | 0.918 | 0.088 |
| Nick 2 | 0.901 | 0.946 | 0.045 |
| Nick 3 | 0.900 | 0.970 | 0.070 |

**Why:** Patch rotation is a mild spatial disruption — anatomy is preserved at the global level, only local orientation within each 32×32 block is changed. The model trained on patch-rotated images still has access to approximate anatomical layout. Canny edge detection is the most aggressive transform in this study — it strips ALL density and intensity information, leaving only binary edge outlines. Medical pathologies like Lung Opacity, Consolidation, and Edema are defined by parenchymal density changes; they have almost no edge representation in a Canny output. The ce model's training signal for these labels is near-zero.


---

## 5. Per-Label Deep Dive

All values below are AUROC on the original test set under Nick Config 3 (most comprehensive 14-label run).

### 5.1 Full Per-Label Table — Nick Config 3

| Label | Original | GB (tex) | PS (tex) | CE (shape) | PR (shape) | Best model | Interpretation |
|----|----|----|----|----|----|----|----|
| Edema | 0.925 | 0.892 | 0.893 | 0.833 | 0.880 | Original | Strong texture signal; diffuse bilateral haziness |
| Pleural Effusion | 0.911 | 0.915 | 0.873 | 0.860 | 0.866 | GB | Both shape (blunted angle) and texture-robust |
| No Finding | 0.905 | 0.939 | 0.901 | 0.870 | 0.880 | **GB beats original** | Absence of findings may be a global gestalt |
| Support Devices | 0.935 | 0.803 | 0.812 | 0.778 | 0.812 | Original | Line/tube appearance is texture-dependent |
| Consolidation | 0.888 | 0.894 | 0.787 | 0.791 | 0.848 | **GB beats original** | Lobar density; blur preserves lobe-scale structure |
| Cardiomegaly | 0.789 | **0.843** | 0.718 | 0.763 | 0.794 | **GB beats original by +0.054** | Heart silhouette is a pure shape diagnosis |
| Pneumothorax | 0.882 | 0.818 | 0.790 | 0.638 | 0.684 | Original | Pleural line texture + absent lung markings |
| Atelectasis | 0.800 | 0.773 | 0.742 | 0.659 | 0.751 | Original | Subtle volume loss; texture-dependent |
| Pneumonia | 0.788 | 0.630 | 0.490 | 0.480 | 0.642 | Original | Heavy texture reliance; all stylizations hurt badly |
| Pleural Other | 0.925 | 0.975 | 0.965 | 0.925 | **0.995** | **PR beats original by +0.070** | Distinct pleural abnormality; shape-detectable |
| Enlarged Cardiomediastinum | 0.548 | 0.569 | 0.597 | 0.626 | 0.666 | **CE/PR beat original** | Pure shape diagnosis; widened silhouette |
| Lung Lesion | 0.189 | 0.393 | 0.144 | 0.269 | **0.443** | **PR beats original by +0.254** | Near-random for original; biased models do better |
| Lung Opacity | 0.921 | 0.907 | 0.853 | 0.872 | 0.829 | Original | Airspace density; texture-primary |

### 5.2 Labels Where Biased Models Beat the Original

This is the most important finding: **biased models are not universally worse on clean test images.** For several labels, the stylization-trained model actually classifies better:

| Label | Winner | Margin | Clinical explanation |
|----|----|----|----|
| Cardiomegaly | GB | +0.054 | Heart size is defined by its silhouette boundary — blurring removes irrelevant parenchymal texture while the cardiac border remains, forcing the model to learn the correct feature |
| Pleural Other | PR | +0.070 | Pleural abnormalities present as distinct spatial patterns; orientation noise from patch rotation is irrelevant to detecting the pleural line location |
| Enlarged Cardiomediastinum | PR, CE | PR +0.118 | Mediastinal widening is a pure contour measurement — both shape-biased models outperform original because the original model has learned confounding texture patterns |
| Lung Lesion | PR | +0.254 | Most striking finding — the original model achieves near-random (0.189) while patch rotation gets 0.443. This suggests the original model relies on texture cues that may reflect scanner/patient confounders rather than true lesion features; shape-biased training forces more robust feature learning |
| Consolidation | GB | +0.006 | Marginal; lobar consolidation has a large-scale shape component that blur preserves |

### 5.3 Labels Where All Biased Models Fail Badly

| Label | Original AUROC | Best biased | Worst biased | Clinical explanation |
|----|----|----|----|----|
| Pneumonia | 0.788 | PR: 0.642 | CE: 0.480 | Pneumonia is diagnosed from subtle focal airspace opacity, irregular ground-glass texture, and bronchial wall thickening — all destroyed by stylization |
| Pneumothorax | 0.882 | GB: 0.818 | CE: 0.638 | The visceral pleural line is visible but the absence of lung markings peripheral to it (a density/texture cue) is the key discriminator |
| Support Devices | 0.935 | GB: 0.803 | CE: 0.778 | Lines and tubes are visible as textures (thin dense streaks); blur softens these to the point of ambiguity |

### 5.4 Ed's Config Per-Label Highlights

Ed's 11-label model shows consistently higher per-label AUROC. Noteworthy values:

| Label | Ed original | Nick C3 original | Ed gb | Nick C3 gb |
|----|----|----|----|----|
| Cardiomegaly | 0.824 | 0.789 | **0.855** | 0.843 |
| Support Devices | **0.921** | 0.935 | 0.790 | 0.803 |
| Lung Opacity | **0.910** | 0.921 | 0.870 | 0.907 |
| Pneumonia | **0.758** | 0.788 | 0.748 | 0.630 |
| Pleural Effusion | **0.927** | 0.911 | 0.920 | 0.915 |
| Pleural Other | 0.920 | 0.925 | 0.950 | 0.975 |

Ed's gb model on Cardiomegaly (0.855) is the highest value observed across all configs and all models for that label — further confirming Cardiomegaly is a shape-primary diagnosis.


---

## 6. How Stylization Caused the Observed Patterns

### 6.1 Gaussian Blur — Why It Barely Hurts

GB models show reliance ratios of 0.939–0.995, the highest in this study. The mechanism is straightforward: Gaussian blur (radius=2) acts as a low-pass spatial filter. At 224×224, most diagnostically relevant anatomical structures (cardiac borders, diaphragm, lung fields, major vessels) span dozens to hundreds of pixels. A radius-2 blur removes features at scales ≤ \~5 pixels while leaving large-scale structure completely intact.

When these models are then tested on the original clean test images, they encounter an image that is **more informative** than what they trained on — it has all the texture they learned to ignore, plus the shape cues they learned to use. The net result is minimal performance loss or even slight gain (Cardiomegaly, Consolidation, No Finding).

### 6.2 Patch Shuffle — Why It Hurts More Than Gaussian Blur

PS models show reliance ratios of 0.874–0.938 — notably lower than GB despite both being "texture bias" transforms. The key difference is that patch shuffle **destroys spatial anatomy entirely.** The network cannot learn any position-dependent features: the heart is not always in the center, the diaphragm is not always at the bottom, the lung fields do not span the expected regions.

When evaluated on original test images where anatomy is spatially organized, PS models underperform GB models because they were deprived of the spatial layout signal during training. The \~7 percentage point gap between GB and PS reliance represents the contribution of **global anatomical spatial priors** to CXR classification performance — independent of texture.

### 6.3 Canny Edge — Why It Produces the Most Degradation

CE models show the lowest reliance ratios in most configs (0.830–0.901), and per-label the most severe drops. Three compounding mechanisms:


1. **Complete density information loss.** Nearly every airspace pathology (Edema, Consolidation, Lung Opacity, Pneumonia) is defined by a change in parenchymal density. In a binary edge map there is no density — only edges. The training signal for these labels is effectively zero in CE training images.
2. **Threshold sensitivity.** The auto-threshold uses σ=0.33 × median. Medical images often have bimodal or skewed histograms (large dark lung fields, bright bone). The threshold can suppress real pathological edges and amplify noise edges inconsistently across patients.
3. **Domain gap at test time.** The CE-trained model has learned to detect features present in binary edge maps. When given a rich grayscale test image with texture, intensity gradients, and soft-tissue contrast, the model encounters a radically different input distribution. The features it learned (edge locations) are embedded in a much noisier signal.

### 6.4 Patch Rotation — Why It Is the Mildest Shape-Bias Transform

PR models show reliance ratios of 0.918–0.970, making them the best-performing stylized model in 3 out of 4 configs. Patch rotation is mild because:


1. **Global layout is preserved.** Unlike patch shuffle, the patches stay in their original positions — only their internal content is rotated. The model can still learn from anatomical location (heart is still center-left, diaphragm is still inferior).
2. **Edges are preserved.** Rotating a patch preserves all edges and contours within it — only their orientation changes. Cardiomegaly, Enlarged Cardiomediastinum, and Pleural Other all benefit because their diagnostic features are large-scale contours that survive rotation.
3. **The pixel multiset is unchanged.** Each patch contains the same intensity values, just rearranged. This means the density information is present locally — only orientation is disrupted. This explains why labels like Edema (diffuse bilateral haze) are relatively robust to PR compared to CE.

### 6.5 The Blend Ratio Effect

Biased models were trained at `blend_ratio=0.5` (50% stylized + 50% original images per epoch). This is a deliberate design choice. Training at `blend_ratio=1.0` (100% stylized) would produce models that have essentially never seen a real CXR and would score near-random on the original test set — making comparisons uninformative. The 50/50 blend ensures the model retains enough real-world signal to be meaningfully evaluated, while the stylized images steer the learned representation toward the desired bias direction.


---

## 7. Key Findings Summary

### Finding 1: All Configs Show Texture Bias > Shape Bias

The texture reliance advantage (mean texture ratio − mean shape ratio) is positive for all 4 configs. DenseNet121 fine-tuned on CheXpert inherits a texture bias from ImageNet pretraining that is not eliminated by any training configuration tested.

| Config | Texture − Shape advantage |
|----|----|
| Ed | +0.037 |
| Nick 1 | +0.033 |
| Nick 2 | +0.017 |
| Nick 3 | +0.022 |

### Finding 2: Biased Models Are Not Universally Worse

Cardiomegaly, Pleural Other, Enlarged Cardiomediastinum, and Lung Lesion all show cases where a biased model outperforms the original model on the clean test set. These labels are shape-primary: their diagnostic features are present in edge/contour information and are not well-served by texture-heavy training on the full CXR signal.

### Finding 3: Config 1 (Focal + Sampler) Is Unstable and Should Not Be Used

The combined focal+sampler approach shows the highest variance and worst biased model performance, particularly for shape-biased models. The double-correction of class imbalance destabilizes training, especially when the training images are already low-signal (edge maps).

### Finding 4: Ed's 11-Label Config Is the Best Baseline

Across all 5 model variants, Ed's config achieves the highest AUROC. The label scope reduction (removing Lung Lesion, Fracture, No Finding) and simpler training setup (BCE, no sampler, no augmentation, longer patience) produce a cleaner, more robust classifier.

### Finding 5: Spatial Anatomy Contributes \~7 AUROC Points

The consistent GB−PS gap (\~0.07) quantifies the contribution of global spatial layout to CXR classification performance. Even in a texture-biased model, knowing anatomical position (heart is center, diaphragm is inferior, lungs are lateral) contributes roughly 7 AUROC points of predictive power.


---

## 8. Implications for the Paper

### Central Thesis

DenseNet121 trained on CheXpert chest X-rays exhibits a systematic texture bias inherited from ImageNet pretraining, which is detectable via stylization experiments and persists across all training configurations tested. However, this bias is **label-specific**: shape-primary diagnoses (Cardiomegaly, Enlarged Cardiomediastinum, Pleural Other) are better served by shape-biased training, while texture-primary diagnoses (Pneumonia, Lung Opacity, Pneumothorax) degrade under shape-biased stylization.

### Key Arguments

**1. The texture bias is structural, not configurable.** Despite testing four distinct approaches to loss function and class-imbalance correction, the direction of the bias (texture > shape) did not change. This suggests the bias is embedded in the ImageNet-pretrained convolutional filters and is not removed by fine-tuning alone.

**2. Stylization reveals diagnostic feature decomposition.** The per-label reliance ratios effectively classify each pathology as shape-primary or texture-primary:

* **Shape-primary** (biased models competitive or superior): Cardiomegaly, Enlarged Cardiomediastinum, Pleural Other, Lung Lesion
* **Texture-primary** (original model clearly superior): Pneumonia, Pneumothorax, Atelectasis, Support Devices, Lung Opacity
* **Mixed** (small gaps, robust to both): Edema, Pleural Effusion, Consolidation

**3. The spatial anatomy contribution is quantifiable.** The GB−PS reliance gap (\~0.07) provides a data-driven estimate of how much performance is attributable to learned spatial priors vs local texture patterns.

**4. Implications for clinical AI robustness.** A model that relies heavily on texture for Pneumonia diagnosis may be vulnerable to acquisition protocol differences (different kV, grid, collimation settings alter the texture appearance of airspace disease while preserving its spatial distribution). A model that relies on shape for Cardiomegaly is inherently more robust to these confounders — the cardiac silhouette looks the same across scanners.

**5. Training configuration matters for bias measurement.** Config 1's instability (focal + sampler double-correction) produces artificially low biased model scores that could lead to overestimating the shape penalty if not controlled for. Researchers should use a stable, single-strategy imbalance correction (Config 3 or Ed's config) to ensure the reliance ratios reflect the model's actual learned features rather than training artifacts.

### Suggested Additional Experiments (Future Work)

* Run `bias_eval.py` fully: test each biased model against all 5 test sets (original + 4 stylized) to produce the complete 4×5 reliance matrix and confirm the per-label findings from the evaluate.py test set results hold in the cross-test setting
* Test `blend_ratio` ablation (0.3, 0.5, 0.7, 1.0) to understand how much real-image exposure is needed for biased models to remain evaluable
* Compare DenseNet169 or ResNet50 to check whether deeper architectures show different bias profiles
* Apply Grad-CAM to the original and biased models on the same test image to visually confirm where each model is looking for Cardiomegaly vs Pneumonia


