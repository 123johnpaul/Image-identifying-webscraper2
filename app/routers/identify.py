from __future__ import annotations

import os
import tempfile
from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile

from app.schemas.identify import IdentifyResponse, IdentifiedAttribute, IdentifiedProduct
from app.services.attribute_extraction import extract_attributes
from app.services.image_recognition import ImageRecognitionClient
from app.services.query_builder import build_queries

router = APIRouter()


def _normalize_category(value: str) -> str:
    v = value.strip().lower().replace("_", "-")
    if v in {"tshirt", "t-shirt", "t shirt", "tee", "tees", "tshirts", "t-shirts"}:
        return "T-shirt"
    return value.strip().title()


def _filter_matches_for_context(matches, category: str):
    if not matches:
        return []

    cat = (category or "").strip().lower()
    aliases = {
        "t-shirt": {"topwear", "tshirts", "shirts", "tops", "tee"},
        "topwear": {"topwear", "tshirts", "shirts", "tops", "tee"},
        "bag": {"bags", "backpacks", "handbags", "messenger bag"},
        "bags": {"bags", "backpacks", "handbags", "messenger bag"},
        "shoe": {"shoes", "footwear", "flats", "sneakers", "sandals", "boots"},
        "shoes": {"shoes", "footwear", "flats", "sneakers", "sandals", "boots"},
        "watch": {"watches", "watch"},
        "watches": {"watches", "watch"},
    }

    target = set()
    if cat:
        target.add(cat)
        target |= aliases.get(cat, set())

    if not target:
        return matches

    filtered = []
    for m in matches:
        meta = m.get("metadata", {})
        m_cat = str(meta.get("category") or "").strip().lower()
        m_title = str(meta.get("title") or "").strip().lower()
        if any(t in m_cat for t in target) or any(t in m_title for t in target):
            filtered.append(m)

    return filtered if filtered else matches


@router.post("/identify-product", response_model=IdentifyResponse)
async def identify_product(
    image: UploadFile = File(...),
    category: str = Form(...),
    color: str = Form(...),
    product_name: Optional[str] = Form(""),
    brand: Optional[str] = Form(""),
):
    normalized_category = _normalize_category(category)
    product_type = normalized_category

    # Safely handle None values if they occur, defaulting to empty strings
    safe_name = (product_name or "").strip()
    safe_brand = (brand or "").strip()

    user_payload = {
        "name": safe_name,
        "category": normalized_category,
        "product_type": product_type,
        "brand": safe_brand,
        "color": color.strip().title(),
    }

    suffix = os.path.splitext(image.filename or "")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        contents = await image.read()
        tmp.write(contents)
        tmp_path = tmp.name

    # Filter out empty strings so the description doesn't have extra spaces
    description_parts = [
        user_payload["brand"],
        user_payload["name"],
        user_payload["category"],
        user_payload["color"]
    ]
    description = " ".join([part for part in description_parts if part]).strip()

    client = ImageRecognitionClient(provider="local_similarity")
    model_result = client.identify(tmp_path, description=description)

    extracted = extract_attributes(model_result.get("labels", []), description=description)

    final_result = {
        "name": user_payload["name"],
        "category": user_payload["category"],
        "product_type": user_payload["product_type"],
        "brand": user_payload["brand"],
        "color": user_payload["color"],
        "confidence": model_result.get("confidence"),
        "raw": model_result.get("raw", {}),
    }

    attributes = [
        IdentifiedAttribute(key="category", value=user_payload["category"], confidence=0.99),
        IdentifiedAttribute(key="product_type", value=user_payload["product_type"], confidence=0.99),
        IdentifiedAttribute(key="brand", value=user_payload["brand"], confidence=0.99),
        IdentifiedAttribute(key="color", value=user_payload["color"], confidence=0.99),
    ]

    top_matches = final_result.get("raw", {}).get("matches", [])
    top_matches = _filter_matches_for_context(top_matches, user_payload["category"])

    product = IdentifiedProduct(
        name=final_result["name"],
        category=final_result["category"],
        brand=final_result["brand"],
        color=final_result["color"],
        attributes=attributes,
        confidence=final_result["confidence"],
    )

    queries = build_queries(final_result)

    return IdentifyResponse(
        product=product,
        search_queries=queries,
        debug={
            "provider": final_result.get("raw", {}).get("provider", "local_similarity"),
            "model_ready": bool(top_matches),
            "top_matches": top_matches,
            "image_color": final_result.get("raw", {}).get("image_color"),
            "voted_color": final_result.get("raw", {}).get("voted_color"),
            "user_payload": user_payload,
            "model_hints": {
                "model_name": model_result.get("name"),
                "model_category": model_result.get("category"),
                "model_brand": model_result.get("brand"),
                "model_color": model_result.get("color"),
                "extracted": extracted,
            },
            "error": final_result.get("raw", {}).get("error"),
        },
    )
