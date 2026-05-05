from __future__ import annotations

import re
from typing import Dict, List, Optional


COLOR_WORDS = {
    "black",
    "white",
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "pink",
    "brown",
    "gray",
    "grey",
    "beige",
    "gold",
    "silver",
}

CATEGORY_KEYWORDS = {
    "sneaker": "shoes",
    "shoe": "shoes",
    "boot": "shoes",
    "t-shirt": "clothing",
    "shirt": "clothing",
    "hoodie": "clothing",
    "laptop": "electronics",
    "phone": "electronics",
    "smartphone": "electronics",
    "watch": "accessories",
    "headphones": "electronics",
    "camera": "electronics",
    "backpack": "bags",
    "bag": "bags",
}


def _extract_brand_from_text(text: str) -> Optional[str]:
    match = re.search(r"\bbrand\s*[:\-]\s*([A-Za-z0-9][A-Za-z0-9\- ]{1,40})", text, re.I)
    if match:
        return match.group(1).strip()
    return None


def _extract_color_from_tokens(tokens: List[str]) -> Optional[str]:
    for t in tokens:
        t_clean = t.lower().strip(".,;:!?")
        if t_clean in COLOR_WORDS:
            return t_clean
    return None


def _extract_category_from_tokens(tokens: List[str]) -> Optional[str]:
    for t in tokens:
        t_clean = t.lower().strip(".,;:!?")
        if t_clean in CATEGORY_KEYWORDS:
            return CATEGORY_KEYWORDS[t_clean]
    return None


def extract_attributes(labels: List[str], description: Optional[str] = None) -> Dict:
    text = " ".join(labels)
    if description:
        text = f"{text} {description}"

    tokens = text.split()

    brand = _extract_brand_from_text(text)
    color = _extract_color_from_tokens(tokens)
    category = _extract_category_from_tokens(tokens)

    attributes = []
    if brand:
        attributes.append({"key": "brand", "value": brand, "confidence": 0.4})
    if color:
        attributes.append({"key": "color", "value": color, "confidence": 0.4})
    if category:
        attributes.append({"key": "category", "value": category, "confidence": 0.4})

    return {
        "brand": brand,
        "color": color,
        "category": category,
        "attributes": attributes,
    }
