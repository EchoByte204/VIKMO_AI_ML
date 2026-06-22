import os
import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Define cache paths
INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faiss_index.bin")
METADATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metadata.pkl")
CATALOGUE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "catalogue.json")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Global model loader
_model = None

def get_model():
    global _model
    if _model is None:
        print(f"Loading embedding model: {MODEL_NAME}...")
        # SentenceTransformer loads locally or downloads if not cached
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def build_chunk(item: dict) -> str:
    """Builds a single text chunk for a catalogue SKU."""
    return (
        f"SKU: {item['sku']} | "
        f"Name: {item['name']} | "
        f"Brand: {item['brand']} | "
        f"Category: {item['category']} | "
        f"Fitment: {item['vehicle_fitment']} | "
        f"Price: INR {item['price_inr']} | "
        f"Stock: {item['stock']} | "
        f"Description: {item['description']}"
    )

def init_rag(force_rebuild=False):
    """Initializes the RAG index and metadata. Rebuilds if files do not exist or force_rebuild is True."""
    if not force_rebuild and os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
        print("Loading existing FAISS index and metadata from disk...")
        index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, "rb") as f:
            catalogue = pickle.load(f)
        return index, catalogue

    print("Building FAISS index and metadata from catalogue.json...")
    if not os.path.exists(CATALOGUE_PATH):
        raise FileNotFoundError(f"Required file '{CATALOGUE_PATH}' not found in current directory.")
        
    with open(CATALOGUE_PATH, "r", encoding="utf-8") as f:
        catalogue = json.load(f)
        
    chunks = [build_chunk(item) for item in catalogue]
    
    # Generate embeddings
    model = get_model()
    print(f"Generating embeddings for {len(chunks)} items...")
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    
    # L2-normalize embeddings for cosine similarity using IndexFlatIP
    print("Normalizing embeddings and indexing in FAISS...")
    faiss.normalize_L2(embeddings)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    # Save to disk
    print(f"Saving FAISS index to {INDEX_PATH} and metadata to {METADATA_PATH}...")
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(catalogue, f)
        
    return index, catalogue

# Load global index and catalogue on import
_index, _catalogue = init_rag()

def search(query: str, top_k=5) -> list[dict]:
    """
    Performs semantic search on the catalogue.
    Returns list of top_k matched catalogue item dicts.
    """
    global _index, _catalogue
    model = get_model()
    
    # Embed and normalize query vector
    query_vector = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vector)
    
    # Search index
    distances, indices = _index.search(query_vector, top_k)
    
    results = []
    for idx in indices[0]:
        if idx != -1 and idx < len(_catalogue):
            results.append(_catalogue[idx])
            
    return results

def search_by_vehicle(make: str, model_name: str) -> list[dict]:
    """
    Filters the catalogue for items matching the make and model in vehicle_fitment.
    Always includes items with 'Universal' fitment.
    """
    global _catalogue
    results = []
    
    make_clean = make.strip().lower() if make else ""
    model_clean = model_name.strip().lower() if model_name else ""
    
    for item in _catalogue:
        fitment = item["vehicle_fitment"].strip().lower()
        
        # Always include Universal fitments
        if "universal" in fitment:
            results.append(item)
            continue
            
        # Match make and model inside the fitment string
        if make_clean in fitment and model_clean in fitment:
            results.append(item)
            
    return results

def clean_str(s):
    if not isinstance(s, str):
        return s
    return s.replace("\u2014", "-").replace("\u20b9", "INR").encode("ascii", "replace").decode("ascii")

if __name__ == "__main__":
    # Test RAG search functions with diverse examples
    
    # Example 1: Original test
    print("\n=== Test 1: Searching for 'brake pads Pulsar 150' ===")
    results1 = search("brake pads Pulsar 150", top_k=2)
    for res in results1:
        print(f"  [{res['sku']}] {clean_str(res['name'])} - Stock: {res['stock']} - Price: INR {res['price_inr']}")
        
    # Example 2: Different category and vehicle (Spark Plugs for Splendor)
    print("\n=== Test 2: Searching for 'spark plugs Splendor' ===")
    results2 = search("spark plugs Splendor", top_k=2)
    for res in results2:
        print(f"  [{res['sku']}] {clean_str(res['name'])} - Stock: {res['stock']} - Price: INR {res['price_inr']}")

    # Example 3: Chain lube
    print("\n=== Test 3: Searching for 'chain lube Motul' ===")
    results3 = search("chain lube Motul", top_k=2)
    for res in results3:
        print(f"  [{res['sku']}] {clean_str(res['name'])} - Stock: {res['stock']} - Price: INR {res['price_inr']}")

    # Example 4: Vehicle filter search for make='KTM', model='Duke 390'
    print("\n=== Test 4: Vehicle filter search for make='KTM', model='Duke 390' ===")
    vehicle_results = search_by_vehicle("KTM", "Duke 390")
    print(f"Found {len(vehicle_results)} items fitting 'KTM Duke 390' (including Universal parts):")
    # Print the first 3 matches
    for res in vehicle_results[:3]:
        print(f"  [{res['sku']}] {clean_str(res['name'])} - Fitment: {clean_str(res['vehicle_fitment'])}")


