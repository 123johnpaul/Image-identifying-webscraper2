# AI Image-Based Product Price Comparison Web Application

This repository contains the prototype web application developed during a one-week sprint to identify products from images and compare their prices across multiple online retailers in the UK.

The system seamlessly integrates an AI vision pipeline with a concurrent web scraping engine to provide real-time price comparisons.

## 🚀 Project Pipeline
1. **Image Upload**
2. **AI Product Identification**
3. **Attribute Extraction & Query Generation**
4. **Concurrent Web Data Retrieval**
5. **Product Normalization & Price Comparison**
6. **Cheapest Product Result Displayed**

---


* **`/identify-product` API Endpoint:** Created the FastAPI endpoint to receive uploaded images and structured form data (Category, Color, Brand).
* **AI Image Recognition:** Integrated computer vision logic (`ImageRecognitionClient`) to process the image and user payload.
* **Attribute Extraction:** Extracted strict attributes ensuring downstream compatibility.
* **Query Builder:** Developed the `build_queries` module to translate AI labels into robust search queries.
* **Contextual Product Filtering:** Implemented `_filter_matches_for_context` to validate matches against product categories (e.g., matching "T-shirt" to "Topwear").

* **Modular Scraper Architecture:** Built `base_scraper.py` to enforce strict product normalization and `shopify_scraper.py` to target a curated registry of 15+ UK Shopify stores concurrently.
* **Fuzzy Matching Integration:** Utilized `rapidfuzz` (`token_set_ratio` and token-level validation) to ensure AI-generated queries gracefully matched real-world e-commerce product titles.
* **Price Comparison Engine:** Developed the `search_all_stores` and `get_cheapest_product` algorithms to quickly sort and identify the best deals without blocking the main thread.
* **`/compare` API Endpoint:** Merged the scraping logic into a unified FastAPI endpoint that accepts AI queries and returns a structured array of sorted results.

### Frontend Integration
* Built using **React and Vite**.
* Features a clean, interactive UI with structured input forms to assist the AI model.
* Maps through the backend JSON responses to display visual product cards, complete with images, store names, prices in GBP (£), and direct purchasing links.

## 🛠️ Technology Stack

* Backend: Python 3.10+, FastAPI, Uvicorn

* AI Vision Model: Integrated APIs (OpenAI Vision / Google Vision / CLIP simulated context)

* Scraping & Data Retrieval: requests, rapidfuzz (for smart text matching), Public Shopify JSON Endpoints

* Frontend: React, Vite

* Concurrency: Python concurrent.futures.ThreadPoolExecutor

## 💻 Getting Started

Prerequisites
Python 3.10+

Node.js 18+

1. Environment Configuration

### Create a .env file in the root directory

```
APP_ENV=local
APP_NAME="Image Product Search Backend"
# Add any required Vision API keys here
```

2. Run the Backend

Open your terminal and navigate to the root directory:

### Create and activate virtual environment

```
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies

```
pip install -r requirements.txt
```

### Start the FastAPI server

```
uvicorn main:app --reload
```
The backend API will run at `http://127.0.0.1:8000`

3. Run the Frontend

Open a new terminal window and navigate to the web/ directory:

```
cd web
```

### Install Node dependencies

```
npm install
```

### Start the Vite development server

```
npm run dev
```

The UI will run at `http://localhost:5173`

## 💡 Usage Guide

* Open `http://localhost:5173` in your browser.

* Fill in the Category and Color fields (Product Name and Brand are optional).

* Upload an image of the product.

* Click Identify Product to allow the AI to extract features and generate search queries.

* Click Compare Prices to unleash the concurrent scraping engine across UK stores.

* Scroll through the results to find the cheapest option!

---

## 📂 System Architecture & Folder Structure

```text
price-comparison-app/
├── main.py                  # Unified FastAPI entry point (Merges Daniel & Uzochi's routers)
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (API Keys, Configs)
├── app/                     # AI & Image Recognition Module (Daniel)
│   ├── routers/
│   │   └── identify.py      # Image upload and AI parsing logic
│   ├── services/
│   └── schemas/
├── scrapers/                # Web Scraping Engine (Uzochi)
│   ├── base_scraper.py      # Abstract base class for normalization
│   └── shopify_scraper.py   # Shopify extraction and RapidFuzz logic
├── utils/                   # Comparison Logic (Uzochi)
│   └── price_comparator.py  # ThreadPool executor and cheapest-price algorithm
└── web/                     # React Frontend
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── App.jsx          # Main UI component handling Identify & Compare states
        ├── App.css          # Styling
        └── main.jsx
```
