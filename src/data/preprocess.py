"""Split raw/<class_name>/*.jpg images into processed/{train,val,test}/<class_name>/."""

import argparse
import random
import shutil
from pathlib import Path

import yaml


def split_dataset(raw_dir: str, processed_dir: str, val_split: float, test_split: float, seed: int) -> None:
    random.seed(seed)
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)

    class_dirs = [d for d in raw_path.iterdir() if d.is_dir()]
    if not class_dirs:
        raise FileNotFoundError(f"No class subdirectories found under {raw_dir}")

    for class_dir in class_dirs:
        images = list(class_dir.glob("*"))
        random.shuffle(images)

        n_val = int(len(images) * val_split)
        n_test = int(len(images) * test_split)
        splits = {
            "val": images[:n_val],
            "test": images[n_val : n_val + n_test],
            "train": images[n_val + n_test :],
        }

        for split_name, files in splits.items():
            out_dir = processed_path / split_name / class_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, out_dir / f.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    split_dataset(
        raw_dir=cfg["raw_dir"],
        processed_dir=cfg["processed_dir"],
        val_split=cfg["val_split"],
        test_split=cfg["test_split"],
        seed=cfg["seed"],
    )


if __name__ == "__main__":
    main()
