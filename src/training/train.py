"""Training entrypoint: config in, best checkpoint out.

Usage:
    python -m src.training.train --data-config configs/data.yaml \
        --model-config configs/model.yaml --train-config configs/train.yaml
"""

import argparse

import torch
import torch.nn as nn
import yaml
from torchvision import models

from src.data.dataset import build_dataloaders
from src.training.callbacks import EarlyStopping, save_checkpoint
from src.training.engine import evaluate, train_one_epoch
from src.utils.logger import get_logger

logger = get_logger(__name__)

RESNET_BUILDERS = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
}


def build_model(arch: str, num_classes: int, pretrained: bool, freeze_backbone: bool) -> nn.Module:
    weights = "DEFAULT" if pretrained else None
    model = RESNET_BUILDERS[arch](weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    params = filter(lambda p: p.requires_grad, model.parameters())
    if cfg["optimizer"] == "sgd":
        return torch.optim.SGD(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"], momentum=0.9)
    return torch.optim.Adam(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict):
    if cfg["lr_scheduler"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    if cfg["lr_scheduler"] == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(cfg["epochs"] // 3, 1))
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    args = parser.parse_args()

    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)
    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)
    with open(args.train_config) as f:
        train_cfg = yaml.safe_load(f)

    device = torch.device(train_cfg["device"] if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    loaders = build_dataloaders(
        processed_dir=data_cfg["processed_dir"],
        image_size=data_cfg["image_size"],
        batch_size=train_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
    )
    if "train" not in loaders or "val" not in loaders:
        raise RuntimeError(
            f"Expected train/ and val/ splits under {data_cfg['processed_dir']}. "
            "Run `python -m src.data.preprocess` first."
        )

    model = build_model(
        arch=model_cfg["arch"],
        num_classes=model_cfg["num_classes"],
        pretrained=model_cfg["pretrained"],
        freeze_backbone=model_cfg["freeze_backbone"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)
    early_stopping = EarlyStopping(patience=train_cfg["early_stopping_patience"], mode="max")

    best_acc = 0.0
    for epoch in range(1, train_cfg["epochs"] + 1):
        train_loss = train_one_epoch(model, loaders["train"], optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, loaders["val"], criterion, device)

        if scheduler is not None:
            scheduler.step()

        logger.info(
            "epoch=%d train_loss=%.4f val_loss=%.4f val_acc=%.4f",
            epoch, train_loss, val_loss, val_acc,
        )

        if early_stopping.step(val_acc):
            best_acc = val_acc
            path = save_checkpoint(model, train_cfg["checkpoint_dir"], "best_model.pt")
            logger.info("New best model (val_acc=%.4f) saved to %s", best_acc, path)

        if early_stopping.should_stop:
            logger.info("Early stopping triggered at epoch %d", epoch)
            break

    logger.info("Training complete. Best val_acc=%.4f", best_acc)


if __name__ == "__main__":
    main()
