from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class SimilarityNet(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int = 128) -> None:
        super().__init__()
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_features = backbone.classifier[0].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.embedding = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        emb = self.embedding(features)
        emb = nn.functional.normalize(emb, p=2, dim=1)
        logits = self.classifier(emb)
        return logits, emb


def build_model(num_classes: int, embedding_dim: int = 128) -> SimilarityNet:
    return SimilarityNet(num_classes=num_classes, embedding_dim=embedding_dim)


def save_checkpoint(path, model, label_to_idx, config):
    torch.save(
        {
            "model_state": model.state_dict(),
            "label_to_idx": label_to_idx,
            "config": config,
        },
        path,
    )


def load_checkpoint(path, model_ctor):
    payload = torch.load(path, map_location="cpu")
    model = model_ctor(
        num_classes=len(payload["label_to_idx"]),
        embedding_dim=payload["config"]["embedding_dim"],
    )
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload

