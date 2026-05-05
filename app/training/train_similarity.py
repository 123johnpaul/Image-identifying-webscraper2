from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from app.training.dataset import load_fashion_samples
from app.training.features import extract_feature_vector


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def build_label_map(samples) -> Dict[str, int]:
    labels = sorted({s.label_name for s in samples})
    return {label: i for i, label in enumerate(labels)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("app/data"))
    parser.add_argument("--out-dir", type=Path, default=Path("app/models/similarity_v1"))
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-brand", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_fashion_samples(
        args.data_dir,
        max_samples=args.max_samples,
        require_brand=args.require_brand,
        seed=args.seed,
    )

    label_to_idx = build_label_map(samples)
    label_names = [s.label_name for s in samples]
    counts = Counter(label_names)
    min_count = min(counts.values()) if counts else 0
    stratify = label_names if min_count >= 2 else None

    train_samples, val_samples = train_test_split(
        samples, test_size=0.15, random_state=args.seed, stratify=stratify
    )

    def featurize(rows):
        feats = []
        y = []
        for sample in tqdm(rows, desc="extract features"):
            feats.append(extract_feature_vector(sample.image_path))
            y.append(label_to_idx[sample.label_name])
        return np.asarray(feats, dtype=np.float32), np.asarray(y, dtype=np.int32)

    x_train, y_train = featurize(train_samples)
    x_val, y_val = featurize(val_samples)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)

    n_components = int(min(args.embedding_dim, x_train_scaled.shape[0] - 1, x_train_scaled.shape[1]))
    n_components = max(16, n_components)
    pca = PCA(n_components=n_components, random_state=args.seed)
    train_emb = pca.fit_transform(x_train_scaled).astype(np.float32)
    val_emb = pca.transform(x_val_scaled).astype(np.float32)

    train_emb /= np.linalg.norm(train_emb, axis=1, keepdims=True) + 1e-8
    val_emb /= np.linalg.norm(val_emb, axis=1, keepdims=True) + 1e-8

    sims = val_emb @ train_emb.T
    top_idx = np.argmax(sims, axis=1)
    pred = y_train[top_idx]
    val_acc = float(np.mean(pred == y_val))

    model_path = args.out_dir / "model.joblib"
    joblib.dump(
        {
            "scaler": scaler,
            "pca": pca,
            "label_to_idx": label_to_idx,
            "embedding_dim": n_components,
            "feature_size": int(x_train.shape[1]),
        },
        model_path,
    )

    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "val_retrieval_at_1": val_acc,
                "num_samples": len(samples),
                "num_train": len(train_samples),
                "num_val": len(val_samples),
                "num_labels": len(label_to_idx),
                "embedding_dim": n_components,
                "stratified_split": bool(stratify is not None),
                "require_brand": args.require_brand,
            },
            fp,
            indent=2,
        )
    print(json.dumps({"val_retrieval_at_1": round(val_acc, 4), "model_path": str(model_path)}))


if __name__ == "__main__":
    main()
