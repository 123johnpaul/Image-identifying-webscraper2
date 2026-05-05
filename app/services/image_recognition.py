from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, Optional

from app.services.color_detection import detect_dominant_color
from app.services.local_similarity import LocalSimilarityEngine


VALID_COLORS = {
    "Black", "White", "Gray", "Grey", "Red", "Blue", "Green", "Yellow",
    "Orange", "Purple", "Pink", "Brown", "Beige", "Silver", "Gold",
}


def _title_case(s: str | None) -> str | None:
    if not s:
        return None
    return str(s).strip().title()


class ImageRecognitionClient:
    def __init__(self, provider: Optional[str] = None):
        # Google removed by request. Only local_similarity + optional mock remain.
        self.provider = (provider or os.getenv("VISION_PROVIDER") or "local_similarity").lower()
        self.local_engine = LocalSimilarityEngine()

    def identify(self, image_path: str, description: Optional[str] = None) -> Dict:
        if self.provider in {"local", "local_similarity"}:
            result = self._local_identify(image_path)
            if result.get("confidence", 0.0) <= 0 and description:
                return self._mock_identify(image_path, description)
            return result

        if self.provider == "mock":
            return self._mock_identify(image_path, description)

        # Unknown provider -> safe local fallback.
        return self._local_identify(image_path)

    def _local_identify(self, image_path: str) -> Dict:
        inference = self.local_engine.infer(image_path, top_k=9)
        if not inference.get("ready"):
            return {
                "name": "unknown product",
                "category": None,
                "brand": None,
                "color": None,
                "attributes": [],
                "labels": [],
                "confidence": 0.0,
                "raw": {"provider": "local_similarity", "error": inference.get("error")},
            }

        matches = inference.get("matches", [])
        if not matches:
            return {
                "name": "unknown product",
                "category": None,
                "brand": None,
                "color": None,
                "attributes": [],
                "labels": [],
                "confidence": 0.0,
                "raw": {"provider": "local_similarity", "error": "No matches returned"},
            }

        best = matches[0]
        best_score = float(best.get("score", 0.0))
        confidence = max(0.0, min(1.0, (best_score + 1.0) / 2.0))

        considered = [m for m in matches if float(m.get("score", 0.0)) >= 0.72]
        if not considered:
            considered = matches[:4]

        title_votes = defaultdict(float)
        category_votes = defaultdict(float)
        color_votes = defaultdict(float)
        brand_votes = defaultdict(float)
        category_to_titles = defaultdict(lambda: defaultdict(float))

        for m in considered:
            score = max(float(m.get("score", 0.0)), 0.0)
            meta = m.get("metadata", {})

            title = str(meta.get("title") or "").strip()
            category = str(meta.get("category") or "").strip()
            color = _title_case(str(meta.get("color") or "").strip())
            brand = str(meta.get("brand") or "").strip()

            if category and category.lower() != "unknown":
                category_votes[category] += score
            if title and title.lower() != "unknown":
                title_votes[title] += score
                if category and category.lower() != "unknown":
                    category_to_titles[category][title] += score
            if color and color.lower() != "unknown" and color in VALID_COLORS:
                color_votes[color] += score
            if brand and brand.lower() != "unknown":
                brand_votes[brand] += score

        def top_or_none(votes):
            if not votes:
                return None, 0.0
            item = sorted(votes.items(), key=lambda x: x[1], reverse=True)[0]
            return item[0], float(item[1])

        category, category_vote = top_or_none(category_votes)

        # Ensure title comes from selected category whenever possible.
        name = None
        if category and category in category_to_titles and category_to_titles[category]:
            name, _ = top_or_none(category_to_titles[category])
        if not name:
            name, _ = top_or_none(title_votes)
        name = name or "unknown product"

        voted_color, color_vote = top_or_none(color_votes)
        brand, brand_vote = top_or_none(brand_votes)
        image_color = detect_dominant_color(image_path)

        # Strict gating to reduce random wrong predictions.
        if category_vote < 1.35:
            category = None
        if brand_vote < 1.25:
            brand = None

        color = voted_color if voted_color in VALID_COLORS and color_vote >= 1.10 else image_color
        if image_color and best_score < 0.90:
            color = image_color

        # If overall confidence low, avoid hallucinating labels.
        if best_score < 0.80:
            if category_vote < 1.60:
                category = None
            if brand_vote < 1.60:
                brand = None
            if color_vote < 1.50:
                color = image_color

        attributes = []
        if brand:
            attributes.append({"key": "brand", "value": brand, "confidence": min(0.95, confidence)})
        if color:
            attributes.append({"key": "color", "value": color, "confidence": confidence})
        if category:
            attributes.append({"key": "category", "value": category, "confidence": confidence})

        labels = [x for x in [name, brand, category, color] if x]

        return {
            "name": name,
            "category": category,
            "brand": brand,
            "color": color,
            "attributes": attributes,
            "labels": labels,
            "confidence": confidence,
            "raw": {
                "provider": "local_similarity",
                "matches": matches,
                "considered_matches": len(considered),
                "image_color": image_color,
                "voted_color": voted_color,
                "vote_strength": {
                    "category": category_vote,
                    "color": color_vote,
                    "brand": brand_vote,
                    "best_score": best_score,
                },
            },
        }

    def _mock_identify(self, image_path: str, description: Optional[str]) -> Dict:
        name = "unknown product"
        category = None
        brand = None
        color = None
        attributes = []
        labels = []
        confidence = 0.2

        if description:
            name = description.strip()
            labels = description.split()
            confidence = 0.35

        return {
            "name": name,
            "category": category,
            "brand": brand,
            "color": color,
            "attributes": attributes,
            "labels": labels,
            "confidence": confidence,
            "raw": {"provider": "mock"},
        }
