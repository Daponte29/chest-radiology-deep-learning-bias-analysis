from __future__ import annotations

import argparse
import wandb
from pathlib import Path

from src.config import Config
from src.data import build_loaders
from src.model import build_model
from src.loss import build_criterion
from src.metrics import compute_metrics
from src.utils import set_seed, get_device, save_checkpoint


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        x, y = batch          # TODO: adjust to your batch structure
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_targets = 0.0, [], []
    for batch in loader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        preds = model(x)
        total_loss += criterion(preds, y).item()
        all_preds.append(preds.cpu())
        all_targets.append(y.cpu())
    import torch
    preds_cat = torch.cat(all_preds)
    targets_cat = torch.cat(all_targets)
    metrics = compute_metrics(preds_cat, targets_cat)
    metrics["loss"] = total_loss / len(loader)
    return metrics


def main(cfg: Config) -> None:
    set_seed(cfg.train.seed)
    device = get_device(cfg.train.device)

    wandb.init(project=cfg.experiment_name, config=cfg.__dict__)

    train_loader, val_loader = build_loaders(cfg.data, cfg.train.batch_size)
    model = build_model(cfg.model).to(device)
    criterion = build_criterion()   # TODO: pass loss name from cfg if needed
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    # TODO: add scheduler if desired

    best_val_loss = float("inf")
    for epoch in range(1, cfg.train.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        wandb.log({"train/loss": train_loss, **{f"val/{k}": v for k, v in val_metrics.items()}}, step=epoch)

        if epoch % cfg.train.log_every == 0:
            print(f"Epoch {epoch}/{cfg.train.epochs}  train_loss={train_loss:.4f}  val={val_metrics}")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint({"epoch": epoch, "model": model.state_dict(), "val": val_metrics},
                            f"{cfg.train.checkpoint_dir}/best.ckpt")

    wandb.finish()


if __name__ == "__main__":
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_01.yaml")
    args = parser.parse_args()
    main(Config.from_yaml(args.config))
