from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from tqdm import tqdm

from app.training.dataset import load_fashion_samples
from app.training.features import extract_feature_vector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("app/data"))
    parser.add_argument("--checkpoint", type=Path, default=Path("app/models/similarity_v1/model.joblib"))
    parser.add_argument("--out-index", type=Path, default=Path("app/models/similarity_v1/index.npz"))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--require-brand", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    payload = joblib.load(args.checkpoint)
    scaler = payload["scaler"]
    pca = payload["pca"]

    samples = load_fashion_samples(
        args.data_dir,
        max_samples=args.max_samples,
        require_brand=args.require_brand,
        seed=args.seed,
    )

    feats = []
    paths = []
    metas = []
    for sample in tqdm(samples, desc="index features"):
        feats.append(extract_feature_vector(sample.image_path))
        paths.append(str(sample.image_path))
        metas.append(sample.metadata)

    x = np.asarray(feats, dtype=np.float32)
    x = scaler.transform(x)
    emb = pca.transform(x).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8

    args.out_index.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_index,
        embeddings=emb,
        image_paths=np.asarray(paths),
        metadata=np.asarray([json.dumps(m) for m in metas]),
    )
    print(f"saved index -> {args.out_index} ({len(paths)} entries)")


if __name__ == "__main__":
    main()
