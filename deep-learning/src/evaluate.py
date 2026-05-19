from __future__ import annotations

import argparse
import torch

from src.config import Config
from src.data import build_loaders
from src.model import build_model
from src.loss import build_criterion
from src.metrics import compute_metrics
from src.utils import get_device, load_checkpoint


def main(cfg: Config, checkpoint: str) -> None:
    device = get_device(cfg.train.device)
    _, val_loader = build_loaders(cfg.data, cfg.train.batch_size)

    model = build_model(cfg.model).to(device)
    ckpt = load_checkpoint(checkpoint, device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    criterion = build_criterion()
    total_loss, all_preds, all_targets = 0.0, [], []

    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            preds = model(x)
            total_loss += criterion(preds, y).item()
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())

    metrics = compute_metrics(torch.cat(all_preds), torch.cat(all_targets))
    metrics["loss"] = total_loss / len(val_loader)
    print("Evaluation results:", metrics)
    # TODO: save metrics to JSON / log to W&B


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_01.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.ckpt")
    args = parser.parse_args()
    main(Config.from_yaml(args.config), args.checkpoint)
