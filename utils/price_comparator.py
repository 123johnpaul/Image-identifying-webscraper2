import concurrent.futures
from typing import Any
from scrapers.shopify_scraper import ShopifyScraper, UK_SHOPIFY_STORES

def search_all_stores(query: str, category: str = None) -> list[dict[str, Any]]:
    all_results = []
    
    def fetch_from_store(store_info):
        scraper = ShopifyScraper(store_url=store_info["url"], store_name=store_info["name"])
        # Pass the category to the scraper
        return scraper.search_products(query, category)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(UK_SHOPIFY_STORES)) as executor:
        future_to_store = {executor.submit(fetch_from_store, store): store for store in UK_SHOPIFY_STORES}
        
        for future in concurrent.futures.as_completed(future_to_store):
            try:
                store_results = future.result()
                all_results.extend(store_results)
            except Exception as e:
                store_data = future_to_store[future]
                print(f"Failed to scrape {store_data['name']}: {e}")

    # Sort results by price (lowest to highest)
    all_results.sort(key=lambda x: x['price'])
    return all_results

def get_cheapest_product(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Returns the cheapest product from a list of normalized results."""
    if not results:
        return None
    # Since search_all_stores sorts the array, the first item is the cheapest
    return results[0]
