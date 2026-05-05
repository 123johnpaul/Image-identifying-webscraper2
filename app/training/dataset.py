from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from PIL import Image
import re

try:
    from torch.utils.data import Dataset
except Exception:
    class Dataset:
        pass


DEFAULT_META_CANDIDATES: Sequence[str] = (
    "styles.csv",
    "metadata.csv",
    "labels.csv",
)


def _pick_existing(paths: Sequence[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def _first_existing_column(frame: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    existing = set(frame.columns)
    for name in names:
        if name in existing:
            return name
    return None


def _derive_brand_from_name(name: str) -> str:
    tokens = [t for t in re.split(r"\s+", str(name).strip()) if t]
    if not tokens:
        return "unknown"
    t = tokens[0].strip(".,:-_/")
    if len(t) < 2:
        return "unknown"
    if t.lower() in {"men", "women", "boys", "girls", "unisex"}:
        return "unknown"
    return t


@dataclass
class Sample:
    image_path: Path
    label_name: str
    metadata: Dict[str, str]


def load_fashion_samples(
    data_dir: Path,
    max_samples: int = 0,
    require_brand: bool = False,
    seed: int = 42,
) -> List[Sample]:
    data_dir = data_dir.resolve()
    image_dir = _pick_existing(
        (
            data_dir / "images",
            data_dir / "fashion-dataset" / "images",
            data_dir / "apparel-images-dataset" / "images",
        )
    )
    if image_dir is None:
        raise FileNotFoundError(
            f"Could not find image directory in {data_dir}. Expected one of: "
            "images/, fashion-dataset/images/, apparel-images-dataset/images/."
        )

    meta_path = _pick_existing(tuple(data_dir / name for name in DEFAULT_META_CANDIDATES))
    if meta_path is None:
        raise FileNotFoundError(
            f"Could not find metadata CSV in {data_dir}. Expected one of: {DEFAULT_META_CANDIDATES}"
        )

    df = pd.read_csv(meta_path, on_bad_lines="skip")
    if df.empty:
        raise ValueError(f"Metadata file is empty: {meta_path}")

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    id_col = _first_existing_column(df, ("id", "image_id", "product_id"))
    image_col = _first_existing_column(df, ("image_path", "filename", "image", "file"))

    article_col = _first_existing_column(df, ("articleType", "product_type", "productType", "title"))
    color_col = _first_existing_column(df, ("baseColour", "color", "colour"))
    brand_col = _first_existing_column(df, ("brandName", "brand"))
    category_col = _first_existing_column(df, ("subCategory", "category", "masterCategory"))
    name_col = _first_existing_column(df, ("productDisplayName", "title", "name"))

    samples: List[Sample] = []
    for _, row in df.iterrows():
        if image_col:
            image_path = image_dir / str(row[image_col])
        elif id_col:
            image_path = image_dir / f"{row[id_col]}.jpg"
            if not image_path.exists():
                image_path = image_dir / f"{row[id_col]}.png"
        else:
            continue

        if not image_path.exists():
            continue

        article = str(row[article_col]).strip() if article_col and pd.notna(row[article_col]) else "unknown"
        color = str(row[color_col]).strip() if color_col and pd.notna(row[color_col]) else "unknown"
        category = str(row[category_col]).strip() if category_col and pd.notna(row[category_col]) else "unknown"

        if brand_col and pd.notna(row[brand_col]):
            brand = str(row[brand_col]).strip()
        else:
            raw_name = str(row[name_col]).strip() if name_col and pd.notna(row[name_col]) else ""
            brand = _derive_brand_from_name(raw_name)

        if not brand:
            brand = "unknown"

        if require_brand and brand.lower() in {"unknown", "nan", "none", "null", ""}:
            continue

        label_name = f"{category}|{article}|{color}"
        metadata = {
            "title": article,
            "brand": brand,
            "category": category,
            "color": color,
        }
        samples.append(Sample(image_path=image_path, label_name=label_name, metadata=metadata))

        if max_samples > 0 and len(samples) >= max_samples:
            break

    if not samples:
        raise ValueError("No usable samples found. Check your images and metadata file.")
    return samples


class FashionSimilarityDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Sample],
        label_to_idx: Dict[str, int],
        transform=None,
    ) -> None:
        self.samples = list(samples)
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label_idx = self.label_to_idx[sample.label_name]
        return image, label_idx
