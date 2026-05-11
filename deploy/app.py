"""
Streamlit inference app — chest X-ray multi-label classifier.

Upload a chest X-ray and get per-label predictions + Grad-CAM attention map.

Run with:
    streamlit run deploy/app.py
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # fix: libiomp5md.dll / libomp.dll conflict on Windows

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

from src.data.chexpert_dataset import CheXpertDataset  # noqa: E402
from src.models.densenet import DenseNetClassifier      # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────

LABELS = CheXpertDataset.DEFAULT_LABELS

PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Auto-discover all trained checkpoints in the archive
_ARCHIVE = _REPO_ROOT / "src" / "configs" / "archive_results_configs"
CHECKPOINTS: dict[str, Path] = {
    f"Config {cfg.name.split('_')[1]} — {variant.name}": variant / "best_model.pth"
    for cfg in sorted(_ARCHIVE.glob("config_*"))
    for variant in sorted(cfg.glob("results/*/"))
    if (variant / "best_model.pth").exists()
}

# ── Grad-CAM ──────────────────────────────────────────────────────────────────

def _patch_densenet_relu(model: DenseNetClassifier) -> None:
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
def load_model(ckpt_bytes: bytes, device_str: str):
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


# ── Page ──────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Chest X-Ray Classifier", layout="wide")
st.title("Chest X-Ray Classifier")
st.caption("Upload a chest X-ray to get multi-label pathology predictions and a Grad-CAM attention map.")

# Sidebar — model selection
with st.sidebar:
    st.header("Model")

    if CHECKPOINTS:
        selected = st.selectbox("Trained checkpoint", list(CHECKPOINTS.keys()))
        ckpt_path = CHECKPOINTS[selected]
        ckpt_bytes = ckpt_path.read_bytes()
    else:
        st.warning("No checkpoints found in archive. Upload one manually.")
        ckpt_bytes = None

    uploaded_ckpt = st.file_uploader("Or upload a .pth file", type=["pth"])
    if uploaded_ckpt:
        ckpt_bytes = uploaded_ckpt.getvalue()
        selected   = uploaded_ckpt.name

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    st.caption(f"Device: **{device_str}**")

# Main — image upload
img_file = st.file_uploader(
    "Upload a chest X-ray (JPEG / PNG)",
    type=["jpg", "jpeg", "png"],
)

if img_file and ckpt_bytes:
    pil_img = Image.open(img_file).convert("RGB")
    tensor  = PREPROCESS(pil_img).unsqueeze(0)

    with st.spinner("Running inference..."):
        model, labels, device = load_model(ckpt_bytes, device_str)
        with torch.no_grad():
            probs = torch.sigmoid(model(tensor.to(device))).squeeze().cpu().numpy()

    st.divider()

    col_img, col_chart, col_cam = st.columns([1, 1.6, 1.4])

    with col_img:
        st.subheader("Input X-ray")
        st.image(pil_img.resize((280, 280)), use_container_width=True)

    with col_chart:
        st.subheader("Label Predictions")
        sorted_idx = np.argsort(probs)[::-1]
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
        st.subheader("Grad-CAM")
        st.caption(f"Top prediction: **{top_lbl}** ({top_prob:.1%})")
        with torch.enable_grad():
            cam = gradcam(model, tensor, top_idx, device)
        st.image(overlay(pil_img, cam), use_container_width=True, clamp=True)
        st.caption("Bright yellow = highest model attention · Dark = ignored")

    # Top-3 summary below
    st.divider()
    st.subheader("Top Predictions")
    top3_idx = np.argsort(probs)[::-1][:3]
    cols = st.columns(3)
    for col, i in zip(cols, top3_idx):
        col.metric(labels[i], f"{probs[i]:.1%}")

elif img_file and not ckpt_bytes:
    st.warning("No model checkpoint available — select one from the sidebar.")
else:
    st.info("Upload a chest X-ray image to get started.")
