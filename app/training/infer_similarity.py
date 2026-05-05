from __future__ import annotations

import argparse
import json

from app.services.local_similarity import LocalSimilarityEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to query image")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    engine = LocalSimilarityEngine()
    result = engine.infer(args.image, top_k=args.top_k)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

