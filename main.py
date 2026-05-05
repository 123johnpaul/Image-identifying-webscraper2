from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any

# Import AI Image Recognition router
from app.routers.identify import router as identify_router

# Import Price Comparison utilities
from utils.price_comparator import search_all_stores, get_cheapest_product

# Load environment variables
APP_DIR = Path(__file__).resolve().parent
load_dotenv(APP_DIR / ".env")
load_dotenv(APP_DIR.parent / ".env")

# Initialize the Unified FastAPI app
app = FastAPI(title="Price Compare AI - Merged Pipeline", version="0.1.0")

# Enable CORS so the React frontend can talk to the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, this should be restricted to the frontend's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(identify_router, tags=["identify"])


# Define the schema expected by the React frontend for the compare endpoint
class ComparePayload(BaseModel):
    query: str
    product: dict | None = None

@app.get("/health")
def health():
    return {"status": "ok", "message": "Unified backend is running!"}

@app.post("/compare")
def compare_prices(payload: ComparePayload):
    if not payload.query:
        raise HTTPException(status_code=400, detail="Query string is required")
    
    # Extract the core keyword (category) if the AI provided it
    core_keyword = None
    if payload.product and payload.product.get("category"):
        core_keyword = payload.product.get("category")
    
    # Pass BOTH the query and the core keyword to the search engine
    all_results = search_all_stores(payload.query, category=core_keyword)
    
    cheapest_product = get_cheapest_product(all_results)
    
    return {
        "cheapest": cheapest_product,
        "all_results": all_results,
        "total_results_found": len(all_results)
    }
