# VIKMO Auto-Parts dealer Assistant - DESIGN & ARCHITECTURE DOCUMENT

This document describes the design decisions, technical implementation details, prompt engineering choices, and model validation schemes for the VIKMO Auto-Parts Dealer Assistant and Demand Forecasting Dashboard.

---

## 1. Retrieval (RAG) Architecture

### Embedding Model & Vector Database Choice
- **Model**: We utilize `all-MiniLM-L6-v2` from SentenceTransformers. This 384-dimensional model strikes a perfect balance between retrieval latency, accuracy, and memory footprint.
- **Index**: We use a **FAISS (Facebook AI Similarity Search) Flat L2 Index**. Given the dataset size of 600 catalogue items, a Flat index ensures 100% exact nearest-neighbor search accuracy without approximation overhead.
- **Scaling Rationale**: Traditional prompt-stuffing (loading 600+ items directly into the prompt) is highly inefficient, costly, and hits context window limits or attention dilution. Our RAG system generates normalized embeddings on item text fields (combining `category`, `name`, and `description`) and searches them in a fraction of a millisecond.

### fitment and Semantic Filter Hybrid Logic
To prevent fitment mismatches (a major pain point for auto mechanics):
1. The user query is embedded and searched in the vector index for semantic matches.
2. A keyword-based pre-filter or post-filter extracts vehicle fitment (e.g. "Pulsar 150").
3. Universal parts (fitting all models) are dynamically merged into the result list.
4. If a year is specified, results are filtered to match the year or general fitment criteria.

---

## 2. Tool Design & Invocation Mechanism

The agent interacts with system resources strictly using GenAI function declarations mapped in `tools.py` and registered in the `RotatableChat` system in `agent.py`:

```
                  ┌──────────────────────┐
                  │   User Message /     │
                  │   Conversational Input│
                  └──────────┬───────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │      Gemini 2.5 Flash       │
              └──────────────┬──────────────┘
                             │ (Function Calls)
                             ▼
               ┌───────────────────────────┐
               │    Function Dispatcher    │
               └─┬───────────┬───────────┬─┘
                 │           │           │
                 ▼           ▼           ▼
       ┌───────────┐   ┌───────────┐   ┌─────────────┐
       │find_parts_│   │check_stock│   │create_order │
       │by_vehicle │   │           │   │             │
       └───────────┘   └───────────┘   └─────────────┘
```

### 1. `find_parts_by_vehicle`
Retrieves parts compatible with a specific make and model. Leverages the vector index + hybrid fitment logic.

### 2. `check_stock`
Retrieves live stock levels directly from the JSON database to ensure high transactional consistency.

### 3. `create_order`
A critical transactional endpoint:
- **Input Validation**: Enforces structured Pydantic input schemas (`OrderRequest` with nested `LineItem`).
- **Inventory Check**: Prior to confirming, checks catalogue inventory. If stock is insufficient, transaction is rejected.
- **State Change**: Automatically deducts quantities from `catalogue.json` and updates the FAISS metadata cache simultaneously to keep search results synced.
- **Output Validation**: Generates and returns a structured JSON confirmation containing `order_id`, `dealer_id`, `items`, `total_amount`, and status.

### Model Tool Selection Decision
We configure the model with `temperature = 0.2` to minimize non-deterministic behavior. The system instructions explicitly map user intents to functions, which allows the model's function calling logic to consistently select the correct tool.

---

## 3. Prompt Engineering & Guardrail Logic

### System Prompt Structure
The system instructions inside [agent.py](file:///d:/projects/VIKMO_AI_ML_Intern_Assignment/agent.py) enforce:
1. **Multi-turn Fitment Memory**: If a part is requested without fitment context, the assistant must NOT run a general search. Instead, it must ask: *"Which vehicle (make and model) is this for?"* and store the answered vehicle in the session context.
2. **Order Integrity**: Order placement requires the dealer ID, SKU, and quantity. If any are missing, the assistant asks for clarification first.
3. **Professional Persona**: Communicates in a polite, structured WhatsApp-style format.

### Off-Topic Guardrails
If a user asks about the weather, programming help, recipes, or general chat:
- The system instructions dictate that the model must invoke `log_off_topic_query(query, reason)`.
- This logs the off-topic attempt to `guardrail_logs.json` for auditing.
- The model then outputs a polite refuse response: *"I am an auto-parts dealer assistant and can only help with catalogue queries."*

---

## 4. Evaluation Suite & Failure Analysis

We created an automated evaluation framework in `eval_set.json` (20 cases) and `run_eval.py` to continuously assert correctness:

### Categories Evaluated
1. **Happy Path Queries** (4 cases): Verifies the agent finds matching parts for exact model combinations.
2. **Ambiguous Queries** (3 cases): Asserts that the agent requests vehicle make/model before searching.
3. **Multi-turn Flows** (3 cases): Checks that fitment context carries over across turns to place orders.
4. **Tool Calls Verification** (3 cases): Intercepts and checks exact function execution.
5. **Out-of-stock / Invalid SKU Scenarios** (3 cases): Validates system handles zero inventory or invalid inputs.
6. **Out-of-scope Refusals** (4 cases): Asserts that guardrail logging is triggered and queries are rejected.

### Key Engineering Insights from Failures
During implementation, several test failures highlighted key runtime limitations, which we systematically addressed:
- **API Key Quota Exhaustion (429)**: The Gemini free tier limits requests to 15 RPM. To solve this, we implemented:
  - **Dynamic Multi-Key Rotation**: Supports a comma-separated list of keys, rotating and transferring history seamlessly.
  - **Request Spacing**: Implemented a 5-second sleep in the evaluation runner, bringing request speed strictly below the 15 RPM limit.
- **Python Unicode Printing Error**: Printing the Indian Rupee symbol (`₹`) on Windows terminals raised a `UnicodeEncodeError`. We resolved this by reconfiguring `sys.stdout` to use UTF-8 encoding.
- **Function Spy Namespace Conflict**: Wrapping tools in a test-spy caused the SDK to see duplicate names (`spy_wrapper`). We resolved this using `functools.wraps` to preserve function signatures and docstrings.

---

## 5. Part B: Demand Forecasting Model

### Model Selection
We selected **Facebook Prophet** as the core demand forecaster. It is highly suited for business forecasting because it models trend changes, holiday effects, and multiple seasonalities.

### Additional Regressor
We incorporated a `promo_flag` regressor. Sales volume exhibits spikes during periods with active promotions, and incorporating this feature allows Prophet to anticipate promotional impacts.

### Validation Scheme (Leakage Prevention)
To ensure validation represents a real-world scenario, we used a **Hold-out Test Window**:
- **Train Period**: First 8 weeks of historical daily sales.
- **Test Period**: The final 4 weeks of historical daily sales.
- **No Leakage**: The model was fit strictly on train-period data. The future test period was completely hidden during training.

### Comparison Results (Overall MAE/MAPE)
Our model significantly outperforms traditional baselines:

| Model | MAE | MAPE |
| :--- | :--- | :--- |
| **Prophet (with Regressor)** | **5.77** | **31.61%** |
| Naive Baseline | 9.36 | 60.10% |
| Seasonal Naive Baseline | 9.98 | 64.20% |

- **Prophet Improvement**: Reduces MAE by **38.3%** compared to the naive baseline.
- **Justification**: Prophet's ability to model promotions and general weekly trends allows it to smooth out noise that naive baselines fail to handle.
