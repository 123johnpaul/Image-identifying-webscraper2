from pydantic import BaseModel
from typing import List, Optional


class IdentifiedAttribute(BaseModel):
    key: str
    value: str
    confidence: Optional[float] = None


class IdentifiedProduct(BaseModel):
    name: str
    category: Optional[str] = None
    brand: Optional[str] = None
    color: Optional[str] = None
    attributes: List[IdentifiedAttribute] = []
    confidence: Optional[float] = None


class IdentifyResponse(BaseModel):
    product: IdentifiedProduct
    search_queries: List[str]
    debug: Optional[dict] = None
