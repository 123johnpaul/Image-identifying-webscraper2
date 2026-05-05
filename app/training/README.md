# Local Product Similarity Training (CPU-Friendly)

This pipeline trains a local similarity model on `app/data` using Fashion Product Images metadata.

## Expected data

- `app/data/images/*.jpg` (or png)
- `app/data/styles.csv` (preferred) or `metadata.csv`/`labels.csv`

The loader auto-detects common column names (`id`, `articleType`, `baseColour`, `brandName`, `subCategory`).

## Install training dependencies

```bash
pip install -r requirements-training.txt --break-system-packages
```

## Train

```bash
python3 -m app.training.train_similarity --data-dir app/data --max-samples 15000 --embedding-dim 128
```

Outputs:

- `app/models/similarity_v1/model.joblib`
- `app/models/similarity_v1/summary.json`

## Build nearest-neighbor index

```bash
python3 -m app.training.build_index --data-dir app/data --checkpoint app/models/similarity_v1/model.joblib
```

Output:

- `app/models/similarity_v1/index.npz`

## Test inference

```bash
python3 -m app.training.infer_similarity --image app/data/images/<some_image>.jpg --top-k 5
```

## Notes for 8GB RAM, no GPU

- Keep `max-samples` at `10000` to `15000` for the first run.
- This is an embedding model for similarity retrieval.
