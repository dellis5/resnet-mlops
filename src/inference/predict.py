"""Batch/CLI inference: load a checkpoint, classify one or more images.

Usage:
    python -m src.inference.predict --config configs/deploy.yaml image1.jpg image2.jpg
"""

import argparse

import torch
import yaml
from PIL import Image

from src.data.dataset import build_transforms
from src.training.train import build_model


def load_model(model_path: str, arch: str, num_classes: int, device: torch.device) -> torch.nn.Module:
    model = build_model(arch=arch, num_classes=num_classes, pretrained=False, freeze_backbone=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict(model: torch.nn.Module, image_paths: list[str], image_size: int, device: torch.device) -> list[int]:
    transform = build_transforms(image_size, train=False)
    batch = torch.stack([transform(Image.open(p).convert("RGB")) for p in image_paths]).to(device)
    outputs = model(batch)
    return outputs.argmax(dim=1).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/deploy.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("images", nargs="+")
    args = parser.parse_args()

    with open(args.config) as f:
        deploy_cfg = yaml.safe_load(f)
    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)

    device = torch.device(deploy_cfg["device"])
    model = load_model(deploy_cfg["model_path"], model_cfg["arch"], model_cfg["num_classes"], device)
    preds = predict(model, args.images, image_size=224, device=device)

    for path, pred in zip(args.images, preds):
        print(f"{path}: class {pred}")


if __name__ == "__main__":
    main()
