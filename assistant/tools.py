import json
import os
import pickle
import uuid
from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError

CATALOGUE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "catalogue.json")
METADATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metadata.pkl")

# --- Pydantic Schemas for Validation ---

class LineItem(BaseModel):
    sku: str = Field(..., description="Unique product SKU code (e.g. BRK-1002)")
    quantity: int = Field(..., gt=0, description="Quantity of the item to order, must be greater than 0")

class OrderRequest(BaseModel):
    dealer_id: str = Field(..., description="Unique dealer identifier (e.g. DEALER-456)")
    line_items: List[LineItem] = Field(..., min_length=1, description="List of items to purchase")

# --- Helper Functions for Data Persistence ---

def load_catalogue_data() -> list[dict]:
    """Loads catalogue data directly from catalogue.json to ensure live stock checks."""
    if not os.path.exists(CATALOGUE_PATH):
        raise FileNotFoundError(f"Catalogue file '{CATALOGUE_PATH}' not found.")
    with open(CATALOGUE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_catalogue_data(catalogue: list[dict]):
    """Saves updated catalogue data back to catalogue.json and updates the RAG metadata cache."""
    with open(CATALOGUE_PATH, "w", encoding="utf-8") as f:
        json.dump(catalogue, f, indent=2)
    
    # Also update RAG metadata.pkl to sync in-memory cache
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "wb") as f:
            pickle.dump(catalogue, f)

# --- Agent Tools ---

def check_stock(sku: str) -> dict:
    """
    Checks the availability and details of a part using its SKU.
    Returns a dict with stock information, price, and descriptions.
    """
    try:
        catalogue = load_catalogue_data()
        sku_clean = sku.strip().upper()
        
        for item in catalogue:
            if item["sku"].upper() == sku_clean:
                return {
                    "sku": item["sku"],
                    "name": item["name"],
                    "brand": item["brand"],
                    "price_inr": item["price_inr"],
                    "stock": item["stock"],
                    "status": "Available" if item["stock"] > 0 else "Out of Stock",
                    "description": item["description"]
                }
                
        return {"error": f"SKU '{sku}' not found in the product catalogue."}
    except Exception as e:
        return {"error": f"Failed to check stock: {str(e)}"}

def find_parts_by_vehicle(make: str, model: str, year: Optional[str] = None) -> list[dict]:
    """
    Filters the catalogue for parts matching the vehicle make, model, and optionally year.
    Always includes universal parts.
    """
    try:
        # Import the search function from RAG module
        from rag import search_by_vehicle
        
        results = search_by_vehicle(make, model)
        
        # If year is specified, perform an additional filter step
        if year:
            year_clean = year.strip().lower()
            filtered_results = []
            for item in results:
                fitment = item["vehicle_fitment"].lower()
                # Include if universal or if the year matches the fitment string
                if "universal" in fitment or year_clean in fitment:
                    filtered_results.append(item)
            return filtered_results
            
        return results
    except Exception as e:
        return [{"error": f"Failed to search parts by vehicle: {str(e)}"}]

def create_order(dealer_id: str, line_items: list[dict]) -> dict:
    """
    Places an order for a list of line items on behalf of a dealer.
    Checks and updates stock levels in catalogue.json.
    Returns a validated structured JSON confirmation or a detailed error response.
    """
    try:
        # 1. Validate inputs using Pydantic
        # This guarantees structured input validation
        order_data = OrderRequest(dealer_id=dealer_id, line_items=line_items)
    except ValidationError as e:
        return {
            "status": "REJECTED",
            "reason": "Validation Error",
            "details": e.errors()
        }
        
    try:
        catalogue = load_catalogue_data()
        sku_to_item = {item["sku"].upper(): item for item in catalogue}
        
        processed_items = []
        total_amount = 0
        stock_updates = {}
        
        # 2. Verify all SKUs exist and check stock levels
        for item in order_data.line_items:
            sku_upper = item.sku.upper()
            if sku_upper not in sku_to_item:
                return {
                    "status": "REJECTED",
                    "reason": f"SKU '{item.sku}' not found in catalogue."
                }
                
            catalog_item = sku_to_item[sku_upper]
            requested_qty = item.quantity
            available_stock = catalog_item["stock"]
            
            if available_stock < requested_qty:
                return {
                    "status": "REJECTED",
                    "reason": f"Insufficient stock for SKU '{item.sku}'. Requested: {requested_qty}, Available: {available_stock}."
                }
                
            subtotal = catalog_item["price_inr"] * requested_qty
            total_amount += subtotal
            
            processed_items.append({
                "sku": catalog_item["sku"],
                "name": catalog_item["name"],
                "quantity": requested_qty,
                "price_per_unit": catalog_item["price_inr"],
                "subtotal": subtotal
            })
            
            # Record stock deduction
            stock_updates[sku_upper] = available_stock - requested_qty
            
        # 3. Apply stock deductions and save to file
        for sku_upper, new_stock in stock_updates.items():
            sku_to_item[sku_upper]["stock"] = new_stock
            
        save_catalogue_data(catalogue)
        
        # 4. Generate structured order confirmation JSON
        order_id = f"ORD-{uuid.uuid4().hex[:8].upper()}"
        return {
            "order_id": order_id,
            "dealer_id": order_data.dealer_id,
            "items": processed_items,
            "total_amount": total_amount,
            "status": "CONFIRMED",
            "message": "Order successfully placed and catalog stock updated."
        }
        
    except Exception as e:
        return {
            "status": "REJECTED",
            "reason": f"System error occurred while creating order: {str(e)}"
        }

if __name__ == "__main__":
    # Test checking stock
    print("=== Testing check_stock('BRK-1002') ===")
    print(json.dumps(check_stock("BRK-1002"), indent=2))
    
    # Test checking non-existent SKU
    print("\n=== Testing check_stock('NON-EXIST') ===")
    print(json.dumps(check_stock("NON-EXIST"), indent=2))
    
    # Test find_parts_by_vehicle
    print("\n=== Testing find_parts_by_vehicle('Bajaj', 'Pulsar 150') ===")
    parts = find_parts_by_vehicle("Bajaj", "Pulsar 150")
    print(f"Found {len(parts)} matching parts. First match:")
    if parts:
        print(json.dumps(parts[0], indent=2))
        
    # Test create_order
    print("\n=== Testing create_order for DEALER-789 (Successful) ===")
    test_line_items = [{"sku": "BRK-1002", "quantity": 2}]
    order_conf = create_order("DEALER-789", test_line_items)
    print(json.dumps(order_conf, indent=2))
    
    # Test stock depletion verify
    print("\n=== Verifying stock updated for 'BRK-1002' ===")
    print(json.dumps(check_stock("BRK-1002"), indent=2))
    
    # Test create_order failure (insufficient stock)
    print("\n=== Testing create_order failure (insufficient stock) ===")
    excess_line_items = [{"sku": "BRK-1002", "quantity": 10000}]
    fail_conf = create_order("DEALER-789", excess_line_items)
    print(json.dumps(fail_conf, indent=2))
