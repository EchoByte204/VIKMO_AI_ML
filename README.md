# VIKMO AI / ML Intern Assignment — Progress & Implementation Documentation

This repository contains the complete implementation of **Part A (Dealer Assistant)** and **Part B (Demand Forecasting)** for the VIKMO AI / ML Take-Home Intern Assignment. 

The project is built as a unified, high-performance automotive dealer console incorporating **Semantic Search (RAG)**, **Agentic Tool Calling**, **Multi-Key API Rotation**, **Demand Forecasting (Prophet)**, and a **Multimodal Part Scanner** in a screen-fitting, premium light-mode interface.

---

## 🚀 What Has Been Built So Far

### 1. Retrieval / RAG Engine (`rag.py`)
- **Semantic Part Indexing:** Processes the 600-product catalogue, converts each SKU's details (name, brand, category, fitment, price, and description) into a single dense document representation.
- **Dense Vector Search:** Employs the `sentence-transformers/all-MiniLM-L6-v2` model (running locally on CPU) to generate 384-dimensional normalized embeddings.
- **FAISS Vector Index:** Builds an in-memory `IndexFlatIP` (Inner Product) FAISS index for high-speed, local cosine-similarity lookups.
- **Fitment Filtering:** Implements precise metadata post-filtering for exact make and model queries (like "Bajaj Pulsar 150").

### 2. Conversational Agent & Tools (`agent.py` & `tools.py`)
- **Native Function Calling:** Configured the Gemini client with direct Python tools:
  - `check_stock(sku)`: Fetches real-time price and stock levels.
  - `find_parts_by_vehicle(make, model, year)`: Filters catalogue parts for vehicle fitments.
  - `create_order(dealer_id, line_items)`: Processes and deducts stock levels.
- **Pydantic Validation:** Employs structured schemas to validate orders (e.g. quantity boundaries and SKU formats) to guarantee database integrity.
- **Multi-Turn Memory:** Retains conversational context. If a user asks for a part without mentioning a vehicle fitment, the agent asks: *"Which vehicle (make and model) is this for?"* and links the contexts together.
- **Robustness & Rate-Limit Resilience:**
  - Implements exponential backoff retries to handle temporary Gemini API 503 overloads.
  - **Dynamic Multi-Key API Rotation:** Parses multiple keys from the environment. If a key runs out of daily quota (429 RESOURCE_EXHAUSTED), it dynamically swaps to the next key and transfers the conversation history seamlessly.

### 3. Demand Forecasting Model (`forecast.py`)
- **Prophet Pipeline:** Models weekly sales over 78 weeks across 30 target SKUs.
- **Promotional Flag Regressor:** Feeds `promo_flag` (0/1) as an extra regressor to capture spikes.
- **Leakage-Free Validation:** Splices the last 4 weeks as a held-out test window for backtesting.
- **Beating Baselines:** Prophet successfully beats baseline models:
  - **Prophet:** MAE = **5.77**, MAPE = **31.61%**
  - **Naive Baseline:** MAE = **9.36**, MAPE = **38.38%**
  - **Seasonal Naive:** MAE = **9.98**, MAPE = **65.08%**
- **Cached Predictions:** Exports forecasted values directly to `forecast_results.json`.

### 4. Interactive Console UI (`app.py`)
- **Sleek Light Glassmorphic Style:** Replaced the default Streamlit theme with a custom light-indigo radial backdrop (`#f8fafc`), clean bordered boxes, hover micro-animations, and high-contrast, readable text.
- **Compact & Screen-Fitting:** Configured page paddings and set the chat feed height to `380px` to fit the dashboard comfortably on typical screens without forcing vertical scrolling.
- **Visual Part Identifier (Multimodal):** Upload a part photo, identify the component using Gemini Vision, and automatically trigger a RAG search recommendation.
- **Quick Lookup Tool:** Sidebar-adjacent lookup component to perform quick semantic searches directly against the FAISS catalog.
- **Audit Logging Panel:** Displays real-time conversation violations logged by the agent's off-topic guardrails (saved to `guardrail_logs.json`).

---

## 🛠️ Project Structure & Files

- [app.py](file:///d:/projects/VIKMO_AI_ML_Intern_Assignment/app.py): Main Streamlit dashboard application.
- [agent.py](file:///d:/projects/VIKMO_AI_ML_Intern_Assignment/agent.py): LLM agent orchestrator, system instructions, and rotatable chat loops.
- [tools.py](file:///d:/projects/VIKMO_AI_ML_Intern_Assignment/tools.py): Native tools (`check_stock`, `find_parts_by_vehicle`, `create_order`) and Pydantic validators.
- [rag.py](file:///d:/projects/VIKMO_AI_ML_Intern_Assignment/rag.py): FAISS database generation, indexing, and embedding retrieval.
- [forecast.py](file:///d:/projects/VIKMO_AI_ML_Intern_Assignment/forecast.py): demand forecasting pipeline using Prophet vs benchmarks.
- [catalogue.json](file:///d:/projects/VIKMO_AI_ML_Intern_Assignment/catalogue.json) & [catalogue.csv](file:///d:/projects/VIKMO_AI_ML_Intern_Assignment/catalogue.csv): The raw database containing 600 SKUs.
- [sales_history.csv](file:///d:/projects/VIKMO_AI_ML_Intern_Assignment/sales_history.csv): Weekly historical units sold.
- [guardrail_logs.json](file:///d:/projects/VIKMO_AI_ML_Intern_Assignment/guardrail_logs.json): Dynamic violation logs.
- [forecast_results.json](file:///d:/projects/VIKMO_AI_ML_Intern_Assignment/forecast_results.json): Cached predictions.

---

## ⚙️ Setup & Execution

### 1. Environment Preparation
Initialize a virtual environment and install the required libraries:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables (`.env`)
Create a `.env` file in the root directory. You can specify a single API key or a comma-separated list of multiple keys to enable auto-rotation:
```ini
# Add one or more Gemini API keys separated by commas
GEMINI_API_KEY=key_one,key_two,key_three
```

### 3. Generate Database Indexes & Forecast Cache
Run the RAG indexing and Forecasting models to compile indices and results:
```bash
# Generate the FAISS binary index and metadata
python assistant/rag.py

# Train Prophet and cache forecast predictions
python forecasting/forecast.py
```

### 4. Launch the Web Console UI
Start the Streamlit dashboard:
```bash
streamlit run app.py
```
The console will open at **http://localhost:8501**.

### 5. Run the Evaluation Suite
To run the 20 automated test cases spanning Happy Paths, Ambiguous Inputs, Multi-turn Flows, Tool Invocations, Out-of-Stock/Invalid SKUs, and Out-of-Scope refusals:
```bash
python eval/run_eval.py
```
This script will execute the turns, mock-intercept the tool calls to verify their execution, run assertions on the response contents, and output a clean table of results with an overall pass/fail score.

### 6. Design & Architecture Document
For an in-depth breakdown of the RAG engine chunking choices, tool parameters, prompt guardrails, and forecasting validation metrics, please refer to [DESIGN.md](file:///d:/projects/VIKMO_AI_ML_Intern_Assignment/DESIGN.md).

---

## 💬 Example Interactions

Here are a few representative dialogues showing the assistant's behavior:

### 1. Product Search & Clarification (Ambiguous turn established)
*   **User:** "Do you have brake pads?"
*   **Assistant:** "Which vehicle (make and model) is this for?"
*   **User:** "Bajaj Pulsar 150."
*   **Assistant:** *(Calls `find_parts_by_vehicle(make="Bajaj", model="Pulsar 150")`)*
    "I found 3 parts matching Bajaj Pulsar 150:
    1.  **Brake Pad Set (Front) — Bajaj Pulsar 150**
        *   SKU: `BRK-1002` | Brand: Galfer | Price: ₹1,200 | Stock: 134 units"

### 2. Multi-turn Order Confirmation
*   **User:** "Place an order for 2 units of BRK-1002 for dealer DEALER-456."
*   **Assistant:** *(Calls `create_order(dealer_id="DEALER-456", line_items=[{"sku": "BRK-1002", "quantity": 2}])`)*
    "Order successfully placed!
    *   **Order ID:** `ORD-F10492B4`
    *   **Dealer ID:** `DEALER-456`
    *   **Total Amount:** ₹2,400.00
    *   **Status:** `CONFIRMED`"

### 3. Out-of-Scope Refusal (Guardrail active)
*   **User:** "What's the weather like in Mumbai?"
*   **Assistant:** *(Calls `log_off_topic_query(query="What's the weather like in Mumbai?", reason="off-topic weather query")`)*
    "I am an auto-parts dealer assistant for VIKMO and can only assist with catalog lookups, stock checks, and ordering. I cannot answer queries unrelated to auto parts."

---

## 🔍 Assumptions & Pointers

1.  **Grounded Catalog**: All information regarding prices and stock checks is assumed to be retrieved live from the catalog data database (`catalogue.json`) through tool calls to guarantee 100% truthfulness.
2.  **API Rate Limits**: The Gemini free tier has a limit of 15 Requests Per Minute. The automated evaluation runner is assumed to run with a 5-second turn delay to respect this constraint.
3.  **Hold-out Period**: We assumed that the final 4 weeks of sales data represent the most recent period to be held out for testing.

*   **System Architecture & Decisions**: Refer to [DESIGN.md](file:///d:/projects/VIKMO_AI_ML_Intern_Assignment/DESIGN.md) for chunking logic, Pydantic tool schemas, and forecasting modeling decisions.
*   **Evaluation Results**: Refer to [eval_results.txt](file:///d:/projects/VIKMO_AI_ML_Intern_Assignment/eval/eval_results.txt) inside `eval/` for the complete execution results table.
