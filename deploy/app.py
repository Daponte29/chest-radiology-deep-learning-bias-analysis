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
</style>
""", unsafe_allow_html=True)

st.title("Chest X-Ray Classifier")
st.caption(
    "DenseNet121 trained on CheXpert — multi-label pathology detection with Grad-CAM attention maps. "
    "Part of a research project investigating texture vs shape bias in radiological AI."
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
        "This model was trained on the CheXpert chest X-ray dataset (185K images) "
        "to classify 14 pathology labels using a fine tuned DenseNet121 CNN model. "
        "The project studies how texture vs shape bias in training data affects diagnostic accuracy."
        "</div>",
        unsafe_allow_html=True,
    )

# Load model once and cache
with st.spinner("Loading model..."):
    model, labels, device = load_model(ckpt_bytes, device_str)

# Tabs
tab_demo, tab_upload = st.tabs(["Live Demo", "Upload Your Own"])

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
