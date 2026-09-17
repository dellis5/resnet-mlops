"""Reorganize a downloaded Oxford-IIIT Pet Dataset `images/` folder into the
`data/raw/<breed_name>/*.jpg` layout expected by src/data/preprocess.py.

The Oxford-IIIT Pet archive (https://www.robots.ox.ac.uk/~vgg/data/pets/) ships
all images flat in one directory, named `<Breed_Name>_<index>.jpg`
(e.g. `Abyssinian_1.jpg`, `yorkshire_terrier_12.jpg`). This script buckets
them by breed.

Usage:
    python -m src.data.prepare_oxford_pet --src data/oxford_pet_download/images --dst data/raw
"""

import argparse
import re
import shutil
from pathlib import Path

FILENAME_RE = re.compile(r"^(?P<breed>.+)_\d+\.jpg$", re.IGNORECASE)


def reorganize(src_dir: str, dst_dir: str) -> dict[str, int]:
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)

    counts: dict[str, int] = {}
    for image_path in src_path.glob("*.jpg"):
        match = FILENAME_RE.match(image_path.name)
        if not match:
            continue
        breed = match.group("breed")

        out_dir = dst_path / breed
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, out_dir / image_path.name)
        counts[breed] = counts.get(breed, 0) + 1

    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Path to extracted Oxford Pet images/ directory")
    parser.add_argument("--dst", default="data/raw", help="Output directory (class-per-folder layout)")
    args = parser.parse_args()

    counts = reorganize(args.src, args.dst)
    print(f"Organized {sum(counts.values())} images into {len(counts)} classes under {args.dst}")


if __name__ == "__main__":
    main()
