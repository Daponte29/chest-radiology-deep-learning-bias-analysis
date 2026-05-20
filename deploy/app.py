"""
Streamlit inference app — chest X-ray multi-label classifier.

Run with:
    streamlit run deploy/app.py
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import io
import sys
import types
from pathlib import Path

import matplotlib
import matplotlib.cm as mpl_cm
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

matplotlib.use("Agg")

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.chexpert_dataset import CheXpertDataset  # noqa: E402
from src.models.densenet import DenseNetClassifier      # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────

LABELS = CheXpertDataset.DEFAULT_LABELS

PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Default checkpoint committed to repo for Streamlit Cloud
_DEFAULT_CKPT = _REPO_ROOT / "src/configs/archive_results_configs/config_1/results/original/best_model.pth"

# Sample image for the demo tab
_SAMPLE_IMAGE = Path(__file__).parent / "assets" / "sample_xray.jpg"

# Auto-discover all trained checkpoints in the archive
_ARCHIVE = _REPO_ROOT / "src" / "configs" / "archive_results_configs"
CHECKPOINTS: dict[str, Path] = {
    f"Config {cfg.name.split('_')[1]} — {variant.name}": variant / "best_model.pth"
    for cfg in sorted(_ARCHIVE.glob("config_*"))
    for variant in sorted(cfg.glob("results/*/"))
    if (variant / "best_model.pth").exists()
}

# Fall back to default if archive not present (e.g. Streamlit Cloud)
if not CHECKPOINTS and _DEFAULT_CKPT.exists():
    CHECKPOINTS = {"Config 1 — original (default)": _DEFAULT_CKPT}

# ── Grad-CAM ──────────────────────────────────────────────────────────────────

def _patch_densenet_relu(model):
    def _fwd(inner_self, x):
        features = inner_self.features(x)
        out = F.relu(features, inplace=False)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return inner_self.classifier(out)
    model.backbone.forward = types.MethodType(_fwd, model.backbone)


def gradcam(model, tensor, class_idx, device):
    model.zero_grad()
    saved = {}

    def _hook(module, inp, out):
        saved["acts"] = out
        out.retain_grad()

    handle = model.backbone.features.register_forward_hook(_hook)
    try:
        out = model(tensor.to(device))
        out[0, class_idx].backward()
    finally:
        handle.remove()

    acts = saved.get("acts")
    if acts is None or acts.grad is None:
        return np.zeros((224, 224), dtype=np.float32)

    weights = acts.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * acts.detach()).sum(dim=1)).squeeze(0)
    cam = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0), size=(224, 224),
        mode="bilinear", align_corners=False,
    ).squeeze().cpu().numpy()

    lo, hi = cam.min(), cam.max()
    return ((cam - lo) / (hi - lo + 1e-8)).astype(np.float32)


def overlay(pil_img, cam, alpha=0.5):
    heatmap = mpl_cm.inferno(cam)[:, :, :3]
    img_arr = np.array(pil_img.convert("RGB").resize((224, 224))) / 255.0
    return np.clip((1 - alpha) * img_arr + alpha * heatmap, 0, 1)


# ── Model loading ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model(ckpt_bytes, device_str):
    device = torch.device(device_str)
    ckpt = torch.load(io.BytesIO(ckpt_bytes), map_location=device, weights_only=False)
    cfg     = ckpt.get("config", {})
    labels  = cfg.get("labels") or LABELS
    variant = cfg.get("model", {}).get("name", "densenet121")
    model = DenseNetClassifier(
        num_classes=len(labels), pretrained=False, variant=variant
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    _patch_densenet_relu(model)
    return model, list(labels), device


# ── Inference helper ──────────────────────────────────────────────────────────

def run_inference(pil_img, model, labels, device):
    tensor = PREPROCESS(pil_img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.sigmoid(model(tensor.to(device))).squeeze().cpu().numpy()
    return tensor, probs

# Creates the Streamlit UI Results section with predicted probabilities, bar chart, and Grad-CAM attention map.
def render_results(pil_img, tensor, probs, labels, device, model):
    top3 = np.argsort(probs)[::-1][:3]
    cols = st.columns(3)
    for col, i in zip(cols, top3):
        col.metric(labels[i], f"{probs[i]:.1%}")

    col_img, col_chart, col_cam = st.columns([1, 1.6, 1.4])

    with col_img:
        st.image(pil_img.resize((280, 280)), use_container_width=True)

    with col_chart:
        sorted_idx    = np.argsort(probs)[::-1]
        sorted_labels = [labels[i] for i in sorted_idx]
        sorted_probs  = probs[sorted_idx]

        fig, ax = plt.subplots(figsize=(5, max(3, len(labels) * 0.38)))
        colors = ["#e63946" if p > 0.5 else "#457b9d" for p in sorted_probs]
        ax.barh(range(len(sorted_labels)), sorted_probs, color=colors, edgecolor="none")
        ax.set_yticks(range(len(sorted_labels)))
        ax.set_yticklabels(sorted_labels, fontsize=8)
        ax.set_xlim(0, 1)
        ax.axvline(0.5, color="#aaa", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Predicted probability")
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_cam:
        top_idx  = int(np.argmax(probs))
        top_lbl  = labels[top_idx]
        top_prob = probs[top_idx]
        st.caption(f"Grad-CAM — top prediction: **{top_lbl}** ({top_prob:.1%})")
        with torch.enable_grad():
            cam = gradcam(model, tensor, top_idx, device)
        st.image(overlay(pil_img, cam), use_container_width=True, clamp=True)
        st.caption("Bright yellow = highest model attention · Dark = ignored")



# ── Page ──────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Chest X-Ray Classifier", layout="wide")

st.markdown("""
<style>
h1, h2, h3, h4, h5, h6,
.stCaption p,
.stMarkdown p,
.stMarkdown li,
div[data-testid="stMetricLabel"] p,
div[data-testid="stMetricValue"],
.stTabs [data-baseweb="tab"] p,
.stInfo p,
label,
[data-testid="stWidgetLabel"] p {
    color: #0a2a6e !important;
    font-weight: 700 !important;
}
.about-text, .about-text p, .about-text strong, .about-text * {
    color: rgba(49, 51, 63, 0.9) !important;
    font-weight: 400 !important;
}
.about-text strong {
    font-weight: 600 !important;
}
.research-text, .research-text p, .research-text li,
.research-text strong, .research-text * {
    color: rgba(49, 51, 63, 0.9) !important;
    font-weight: 400 !important;
}
.research-text strong { font-weight: 700 !important; }
.research-text h4 { color: #0a2a6e !important; font-weight: 700 !important; }
</style>
""", unsafe_allow_html=True)

st.title("Chest X-Ray Pathology Classifier")
st.caption(
    "Fine-tuned DenseNet121 trained on 185K chest X-rays from the CheXpert dataset — "
    "multi-label pathology detection across 14 conditions with Grad-CAM explainability. "
    "Part of a research study investigating texture vs shape bias in radiological deep learning."
)

device_str = "cuda" if torch.cuda.is_available() else "cpu"

# Sidebar
with st.sidebar:
    st.header("Model")
    if CHECKPOINTS:
        selected  = st.selectbox("Checkpoint", list(CHECKPOINTS.keys()))
        ckpt_path = CHECKPOINTS[selected]
        ckpt_bytes = ckpt_path.read_bytes()
    else:
        st.error("No checkpoint found.")
        st.stop()

    uploaded_ckpt = st.file_uploader("Or upload a .pth file", type=["pth"])
    if uploaded_ckpt:
        ckpt_bytes = uploaded_ckpt.getvalue()

    st.caption(f"Device: **{device_str}**")
    st.divider()
    st.markdown(
        '<div class="about-text">'
        "<strong>About</strong><br><br>"
        "Fine-tuned <strong>DenseNet121</strong> on the <strong>CheXpert dataset</strong> "
        "(224,316 chest X-rays, Stanford Medicine) for multi-label classification of "
        "<strong>14 pathology conditions</strong> including Pleural Effusion, Atelectasis, "
        "Cardiomegaly, Edema, and more.<br><br>"
        "Trained with <strong>BCE loss</strong>, Adam optimizer, cosine LR scheduling, "
        "and mixed-precision (AMP). Achieves <strong>0.84 mean AUROC</strong> on the "
        "CheXpert 5-label competition set — competitive with published single-model baselines.<br><br>"
        "Grad-CAM overlays highlight which image regions drove each prediction, "
        "providing model explainability critical for clinical AI."
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        '<div class="about-text">'
        "<strong>By Nicholas Daponte</strong><br>"
        '<a href="https://github.com/Daponte29/chest-radiology-deep-learning-bias-analysis" '
        'target="_blank">GitHub Repository</a>'
        "</div>",
        unsafe_allow_html=True,
    )

# Load model once and cache
with st.spinner("Loading model..."):
    model, labels, device = load_model(ckpt_bytes, device_str)

# Tabs
tab_demo, tab_upload, tab_research = st.tabs(["Live Demo", "Upload Your Own", "Research & Results"])

# ── Demo tab ──────────────────────────────────────────────────────────────────
with tab_demo:
    if not _SAMPLE_IMAGE.exists():
        st.info("No sample image found at `deploy/assets/sample_xray.jpg`. Add one to enable the demo.")
    else:
        st.caption("Pre-loaded chest X-ray from the CheXpert test set — predictions run automatically.")

        if "demo_results" not in st.session_state:
            with st.spinner("Running demo inference..."):
                pil_img = Image.open(_SAMPLE_IMAGE).convert("RGB")
                tensor, probs = run_inference(pil_img, model, labels, device)
                st.session_state["demo_results"] = (pil_img, tensor, probs)

        pil_img, tensor, probs = st.session_state["demo_results"]
        render_results(pil_img, tensor, probs, labels, device, model)

# ── Upload tab ────────────────────────────────────────────────────────────────
with tab_upload:
    img_file = st.file_uploader("Upload a chest X-ray (JPEG / PNG)", type=["jpg", "jpeg", "png"])

    if img_file:
        pil_img = Image.open(img_file).convert("RGB")
        with st.spinner("Running inference..."):
            tensor, probs = run_inference(pil_img, model, labels, device)
        render_results(pil_img, tensor, probs, labels, device, model)
    else:
        st.info("Upload a chest X-ray to get predictions.")

# ── Research tab ──────────────────────────────────────────────────────────────
with tab_research:
    st.markdown('<div class="research-text">', unsafe_allow_html=True)

    st.subheader("Research Overview")
    st.markdown(
        "This project investigates whether **DenseNet121** trained on chest X-rays relies on "
        "**texture** or **shape** features when making pathology predictions — a critical question "
        "for clinical AI robustness. Methodology is adapted from "
        "[Geirhos et al. (ICLR 2019)](https://arxiv.org/abs/1811.12231)."
    )

    st.subheader("Experimental Setup")
    st.markdown("Five DenseNet121 models were trained on different versions of CheXpert, then evaluated on the same original test set:")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
| Model | Training Data | Bias Induced |
|---|---|---|
| **Original** | Real X-rays | None — baseline |
| **Gaussian Blur** | Blurred X-rays | Texture |
| **Patch Shuffle** | Patch-shuffled X-rays | Texture |
| **Canny Edge** | Edge-only X-rays | Shape |
| **Patch Rotation** | Patch-rotated X-rays | Shape |
""")
    with col2:
        st.markdown("""
**Key insight:** If a model trained on blurred images performs *better* on blurred test images
than original test images, it learned texture shortcuts rather than clinically meaningful
anatomical features.

**Reliance Ratio** = AUC on stylized test set ÷ AUC on original test set

- Ratio > 1 on matching style → bias confirmed
- Ratio < 1 on opposing style → bias confirmed
- Both near 1.0 → model learned real features
""")

    st.subheader("Model Performance — Config 1 Baseline")
    m1, m2, m3 = st.columns(3)
    m1.metric("Mean AUROC (competition labels)", "0.84")
    m2.metric("Training Images", "185K")
    m3.metric("Architecture", "DenseNet121")
    st.caption(
        "0.84 mean AUROC on the CheXpert 5-label competition set (Atelectasis, Cardiomegaly, "
        "Consolidation, Edema, Pleural Effusion) — competitive with published leaderboard results "
        "for single-model DenseNet baselines trained on the small dataset variant."
    )

    st.subheader("Technical Stack")
    st.markdown("""
- **Framework:** PyTorch 2.0 + torchvision
- **Model:** DenseNet121 fine-tuned from ImageNet weights
- **Dataset:** CheXpert-v1.0-small (Stanford Medicine)
- **Training:** BCE loss · Adam · Cosine LR · AMP mixed precision · Early stopping
- **Explainability:** Grad-CAM (gradient-weighted class activation maps)
- **Style Transfers:** Gaussian blur · Patch shuffle · Canny edge · Patch rotation
- **Evaluation:** AUROC per label · Reliance ratio analysis
- **Deployment:** Streamlit · Docker · GitHub Actions CI
""")

    st.markdown('</div>', unsafe_allow_html=True)
