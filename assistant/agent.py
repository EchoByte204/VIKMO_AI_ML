import os
import json
import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

# Import the parts tools
from tools import check_stock, create_order, find_parts_by_vehicle

GUARDRAIL_LOG_PATH = "guardrail_logs.json"

# --- Guardrail Logging Tool ---

def log_off_topic_query(query: str, reason: str) -> dict:
    """
    Logs off-topic queries to guardrail_logs.json for guardrail tracking.
    This is called by the agent when a query is determined to be unrelated to auto parts.
    """
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "query": query,
        "reason": reason
    }
    
    logs = []
    if os.path.exists(GUARDRAIL_LOG_PATH):
        try:
            with open(GUARDRAIL_LOG_PATH, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except Exception:
            logs = []
            
    logs.append(log_entry)
    
    with open(GUARDRAIL_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)
        
    return {"status": "logged", "message": "Query logged as off-topic."}

# Register all tools
tools_map = {
    "check_stock": check_stock,
    "create_order": create_order,
    "find_parts_by_vehicle": find_parts_by_vehicle,
    "log_off_topic_query": log_off_topic_query
}

def get_api_keys() -> list[str]:
    """Retrieves and parses API keys from the environment."""
    raw_keys = os.getenv("GEMINI_API_KEY", "")
    if not raw_keys:
        raw_keys = os.getenv("GEMINI_API_KEYS", "")
    
    keys = [k.strip() for k in raw_keys.split(",") if k.strip() and k.strip() != "your_api_key_here"]
    return keys

def get_client() -> genai.Client:
    """Initializes and returns the Gemini client using the first API key from the environment."""
    keys = get_api_keys()
    if not keys:
        raise ValueError("Missing GEMINI_API_KEY in the environment. Please update the .env file with your key.")
    return genai.Client(api_key=keys[0])

def create_agent_chat(client: genai.Client, history=None) -> genai.Client.chats:
    """
    Creates and returns a Gemini chat session configured with system instructions,
    conversational rules, tools, and optional history.
    """
    system_instruction = (
        "You are a professional, helpful, and polite WhatsApp-style auto-parts dealer assistant for VIKMO.\n"
        "Your target users are mechanics and auto dealers. You assist them in checking stock, finding parts by vehicle, and placing orders.\n\n"
        "Instructions:\n"
        "1. Context & Multi-turn memory: Maintain conversation context. If a user asks for a part (e.g. 'do you have brake pads?'), "
        "check the history to see if a vehicle fitment was mentioned. If not, do NOT execute a search. Instead, ask: 'Which vehicle (make and model) is this for?' "
        "and wait for their answer. When they provide it (e.g. 'Pulsar 150'), remember the part and vehicle context, and search for parts fitting that vehicle.\n"
        "2. Stock Checks: To check stock for a specific item, you must use the `check_stock` tool with the SKU.\n"
        "3. Find Parts: To search parts compatible with a vehicle, use `find_parts_by_vehicle` with make, model, and optional year.\n"
        "4. Order Placement: You can place orders on behalf of dealers. To call `create_order`, you MUST collect: \n"
        "   a) The dealer's unique ID (e.g., dealer_id).\n"
        "   b) The SKUs and quantities of the parts they want to buy (as line_items).\n"
        "   If they ask to order but you are missing any of these details, ask for the missing details first.\n"
        "   Once placed, display the structured JSON confirmation clearly to the user, including the Order ID, status, and subtotal.\n"
        "5. Guardrails: If a user asks off-topic questions (e.g., 'what is the weather?', 'how do I bake a cake?', etc.), "
        "you MUST immediately call `log_off_topic_query` with the user's query and a short reason (e.g. 'off-topic weather inquiry'). "
        "After calling the log tool, politely tell the user that you are an auto-parts dealer assistant and can only help with catalogue queries."
    )
    
    # Configure the chat with tools
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        tools=list(tools_map.values()),
        temperature=0.2, # Low temperature for consistent tool calling
    )
    
    # We use gemini-2.5-flash as the default model
    return client.chats.create(model="gemini-2.5-flash", config=config, history=history)

class RotatableChat:
    def __init__(self, keys: list[str], current_idx: int = 0, history=None):
        self.keys = keys
        self.current_idx = current_idx
        self.history_init = history
        self.client = None
        self.chat = None
        self._init_chat()
        
    def _init_chat(self):
        # Retrieve history from existing chat if it exists
        history = None
        if self.chat:
            try:
                history = self.chat.get_history()
            except Exception:
                try:
                    history = self.chat.history
                except Exception:
                    pass
        elif self.history_init:
            history = self.history_init
            
        key = self.keys[self.current_idx]
        self.client = genai.Client(api_key=key)
        self.chat = create_agent_chat(self.client, history=history)
        
    def rotate(self) -> bool:
        if len(self.keys) <= 1:
            return False
        self.current_idx = (self.current_idx + 1) % len(self.keys)
        print(f"[Rotation] Rotating to key at index {self.current_idx}")
        self._init_chat()
        return True
        
    def send_message(self, *args, **kwargs):
        return self.chat.send_message(*args, **kwargs)
        
    @property
    def history(self):
        return self.chat.history
        
    @history.setter
    def history(self, val):
        self.chat.history = val
        
    def get_history(self):
        return self.chat.get_history()

def create_agent_chat_session(history=None) -> RotatableChat:
    """Initializes and returns a RotatableChat instance using the keys in the environment."""
    keys = get_api_keys()
    if not keys:
        raise ValueError("Missing GEMINI_API_KEY in the environment. Please check your .env file.")
    return RotatableChat(keys, current_idx=0, history=history)

import time

def run_agent_turn(chat, user_message: str, max_retries: int = 4, backoff_factor: float = 2.0) -> str:
    """
    Sends a message to the active chat session, resolves all tool calls recommended by the model,
    sends outputs back to the chat context, and returns the final conversational text.
    Handles temporary 503 rate-limit or overload errors with exponential backoff.
    Also handles 429 quota exhaustion using key rotation if supported by the chat object.
    """
    # Helper to send message with retry, backoff, and rotation
    def send_with_retry(msg):
        # We allow up to max_retries * (number of API keys) total attempts
        num_keys = len(chat.keys) if hasattr(chat, "keys") else 1
        total_allowed_attempts = max_retries * num_keys
        
        attempt = 0
        delay = 1.0
        
        while attempt < total_allowed_attempts:
            try:
                res = chat.send_message(msg)
                return res
            except Exception as e:
                err_msg = str(e)
                is_quota_or_rate = "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower()
                
                if is_quota_or_rate and hasattr(chat, "rotate"):
                    if chat.rotate():
                        print("[Warning] Quota exceeded. Rotated API key. Retrying query...")
                        # Reset delay since we are on a fresh key
                        delay = 1.0
                        attempt += 1
                        continue
                
                # Check for other temporary errors (like 503)
                is_temporary = any(x in err_msg for x in ["503", "429", "RESOURCE_EXHAUSTED", "UNAVAILABLE"]) or "overload" in err_msg.lower()
                if is_temporary and attempt < total_allowed_attempts - 1:
                    print(f"[Warning] Gemini API temporary error: {err_msg}. Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= backoff_factor
                    attempt += 1
                else:
                    raise e
        raise RuntimeError("Exhausted all retries and API keys.")

    # 1. Send the user message to the chat session
    response = send_with_retry(user_message)

    # 2. Tool execution loop
    while response.function_calls:
        tool_responses = []
        for call in response.function_calls:
            name = call.name
            args = call.args
            
            # Execute the matching tool
            func = tools_map.get(name)
            if func:
                try:
                    result = func(**args)
                except Exception as e:
                    result = {"error": f"Failed to run tool '{name}': {str(e)}"}
            else:
                result = {"error": f"Tool '{name}' is not registered."}
                
            # Append result as a function response part
            tool_responses.append(
                types.Part.from_function_response(
                    name=name,
                    response={"result": result}
                )
            )
            
        # Send the tool responses back to the chat session
        response = send_with_retry(tool_responses)
        
    return response.text


if __name__ == "__main__":
    print("=== Testing agent.py Interactive Chat (CLI) ===")
    try:
        chat = create_agent_chat_session()
        print("Agent initialized with API Key Rotation! Type your message (type 'exit' to quit):")
        while True:
            try:
                user_in = input("\nYou: ")
                if user_in.strip().lower() == "exit":
                    break
                if not user_in.strip():
                    continue
                reply = run_agent_turn(chat, user_in)
                print(f"\nAssistant: {reply}")
            except KeyboardInterrupt:
                break
    except ValueError as e:
        print(f"\n[Warning] {str(e)}")
        print("Note: In production, the system will use your real API key. Please check your .env file.")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")

