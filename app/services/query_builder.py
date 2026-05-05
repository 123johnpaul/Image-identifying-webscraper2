from __future__ import annotations

from typing import Dict, List


def _normalize_category(value: str | None) -> str | None:
    if not value:
        return None
    v = value.strip().lower().replace("_", "-")
    if v in {"tshirt", "t-shirt", "t shirt", "tee", "tees", "tshirts", "t-shirts"}:
        return "T-shirt"
    return value.strip().title()


def _unique_parts(parts: List[str]) -> List[str]:
    seen = set()
    out = []
    for p in parts:
        key = p.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(p.strip())
    return out


def build_queries(product: Dict) -> List[str]:
    brand = (product.get("brand") or "").strip()
    name = (product.get("name") or "").strip()
    category = _normalize_category(product.get("category"))
    product_type = (product.get("product_type") or "").strip().title()
    color = (product.get("color") or "").strip().title()

    # Remove obvious duplicates between category and product_type.
    if category and product_type and category.lower() == product_type.lower():
        product_type = ""

    parts = _unique_parts([brand, name, category or "", product_type, color])

    queries = []
    base = " ".join(parts).strip()
    if base:
        queries.append(base)

    # Targeted backups.
    if name and category:
        queries.append(f"{name} {category}")
    if brand and category:
        queries.append(f"{brand} {category}")
    if product_type and category and product_type.lower() != category.lower():
        queries.append(f"{product_type} {category}")

    # Case-insensitive de-dup preserving order.
    seen = set()
    uniq = []
    for q in queries:
        k = q.lower().strip()
        if k and k not in seen:
            seen.add(k)
            uniq.append(q.strip())

    return uniq
