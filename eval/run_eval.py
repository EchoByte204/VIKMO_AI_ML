import os
import sys
import json
import shutil
import time

# Reconfigure stdout to use UTF-8 on Windows to handle the Rupee symbol (₹) properly
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import sys
import os

# Add parent and assistant folders to path to import correctly from subfolders
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../assistant")))

import agent
import functools

# We will intercept tool calls.
# Keep a global list of tool calls made in the current turn.
intercepted_calls = []

def make_spy(name, original_func):
    @functools.wraps(original_func)
    def spy_wrapper(*args, **kwargs):
        intercepted_calls.append({
            "name": name,
            "args": args,
            "kwargs": kwargs
        })
        return original_func(*args, **kwargs)
    return spy_wrapper

# Monkeypatch agent.tools_map to spy on calls
original_tools = dict(agent.tools_map)
for name, func in original_tools.items():
    agent.tools_map[name] = make_spy(name, func)

def restore_catalogue():
    """Restores catalogue data from backups if they exist."""
    if os.path.exists(os.path.join(os.path.dirname(__file__), "../catalogue.json.bak")):
        shutil.copy(os.path.join(os.path.dirname(__file__), "../catalogue.json.bak"), os.path.join(os.path.dirname(__file__), "../catalogue.json"))
    if os.path.exists(os.path.join(os.path.dirname(__file__), "../assistant/metadata.pkl.bak")):
        shutil.copy(os.path.join(os.path.dirname(__file__), "../assistant/metadata.pkl.bak"), os.path.join(os.path.dirname(__file__), "../assistant/metadata.pkl"))

def backup_catalogue():
    """Creates a temporary backup of the catalog database files."""
    if os.path.exists(os.path.join(os.path.dirname(__file__), "../catalogue.json")):
        shutil.copy(os.path.join(os.path.dirname(__file__), "../catalogue.json"), os.path.join(os.path.dirname(__file__), "../catalogue.json.bak"))
    if os.path.exists(os.path.join(os.path.dirname(__file__), "../assistant/metadata.pkl")):
        shutil.copy(os.path.join(os.path.dirname(__file__), "../assistant/metadata.pkl"), os.path.join(os.path.dirname(__file__), "../assistant/metadata.pkl.bak"))

def verify_turn(case_id, turn_idx, user_input, response_text, expected_tool, tools_called, pass_criteria):
    response_lower = response_text.lower()
    
    # 1. Verify if the expected tool was called
    if expected_tool is not None:
        if expected_tool not in tools_called:
            return False, f"Expected tool '{expected_tool}' was not called. Called: {tools_called}"
    else:
        # Check that we did not make unexpected catalog searches/orders
        # Skip checking helper log_off_topic_query as it's allowed for out of scope
        non_log_tools = [t for t in tools_called if t != "log_off_topic_query"]
        if non_log_tools:
            return False, f"Expected no functional tool call, but tools were called: {non_log_tools}"

    # 2. Check semantic pass criteria / output text checks based on case_id
    if case_id == "TC-001":
        if "brk-1002" not in response_lower:
            return False, "Response did not mention Pulsar 150 brake pads SKU BRK-1002."
            
    elif case_id == "TC-002":
        if "ele-1065" not in response_lower:
            return False, "Response did not mention Classic 350 spark plugs SKU ELE-1065."
            
    elif case_id == "TC-003":
        if "tyr-1009" not in response_lower:
            return False, "Response did not mention Honda Unicorn tyres SKU TYR-1009."
            
    elif case_id == "TC-004":
        if "grp-1061" not in response_lower:
            return False, "Response did not mention KTM Duke 390 handle grips SKU GRP-1061."
            
    elif case_id in ["TC-005", "TC-006", "TC-007"]:
        keywords = ["vehicle", "model", "make", "bike", "motorcycle", "car", "which"]
        if not any(k in response_lower for k in keywords):
            return False, "Response did not ask clarifying question about vehicle model/make."
            
    elif case_id == "TC-008":
        if turn_idx == 0:
            if "pulsar" not in response_lower and "brk-1002" not in response_lower:
                return False, "Response did not list Pulsar 150 brake pads."
        elif turn_idx == 1:
            if "confirm" not in response_lower and "ord-" not in response_lower:
                return False, "Response did not confirm the order."
                
    elif case_id == "TC-009":
        if turn_idx == 0:
            if "fz" not in response_lower and "bdy-1097" not in response_lower:
                return False, "Response did not list FZ side mirrors."
        elif turn_idx == 1:
            if "stock" not in response_lower and "available" not in response_lower and "245" not in response_lower:
                return False, "Response did not return stock level information."
                
    elif case_id == "TC-010":
        if turn_idx == 0:
            if "classic 350" not in response_lower and "ele-1065" not in response_lower:
                return False, "Response did not list Classic 350 spark plugs."
        elif turn_idx == 1:
            if "confirm" not in response_lower and "ord-" not in response_lower:
                return False, "Response did not confirm the order."
                
    elif case_id == "TC-011":
        if "meteor" not in response_lower and "brk-1007" not in response_lower:
            return False, "Response did not mention Meteor or BRK-1007."
            
    elif case_id == "TC-012":
        if "apache" not in response_lower and "rtr" not in response_lower:
            return False, "Response did not list TVS Apache RTR parts."
            
    elif case_id == "TC-013":
        if "confirm" not in response_lower and "ord-" not in response_lower:
            return False, "Response did not confirm the order."
            
    elif case_id == "TC-014":
        if not any(x in response_lower for x in ["not found", "error", "invalid", "does not exist", "brk-9999"]):
            return False, "Response did not report SKU is invalid / not found."
            
    elif case_id == "TC-015":
        if not any(x in response_lower for x in ["out of stock", " 0", "zero", "no stock"]):
            return False, "Response did not report 0 stock or out of stock."
            
    elif case_id == "TC-016":
        if not any(x in response_lower for x in ["reject", "insufficient", "fail", "stock"]):
            return False, "Response did not report order rejected due to insufficient stock."
            
    elif case_id in ["TC-017", "TC-018", "TC-020", "TC-021"]:
        keywords = ["sorry", "cannot", "auto-parts", "dealer assistant", "off-topic", "catalogue", "only help with"]
        if not any(k in response_lower for k in keywords):
            return False, "Response did not politely refuse the off-topic query."
            
    return True, "Passed checks"

def main():
    print("=== VIKMO Auto-Parts Agent Evaluation Runner ===")
    
    # Backup catalog files
    backup_catalogue()
    
    try:
        with open(os.path.join(os.path.dirname(__file__), "eval_set.json"), "r", encoding="utf-8") as f:
            test_cases = json.load(f)
    except Exception as e:
        print(f"Error loading eval_set.json: {e}")
        restore_catalogue()
        return
        
    results = []
    passed_count = 0
    
    for case in test_cases:
        case_id = case["id"]
        category = case["category"]
        description = case["description"]
        turns = case["turns"]
        
        print(f"\nRunning {case_id} ({category}): {description}")
        
        # Start a fresh chat session for each test case to prevent context bleed
        try:
            chat = agent.create_agent_chat_session()
        except Exception as e:
            print(f"Failed to initialize chat session: {e}")
            results.append({
                "id": case_id,
                "pass": "FAIL",
                "tool_called": "N/A",
                "notes": f"Session init failed: {e}"
            })
            continue
            
        case_passed = True
        case_notes = []
        actual_tools_called = []
        
        for turn_idx, turn in enumerate(turns):
            user_input = turn["input"]
            expected_tool = turn.get("expected_tool_called")
            pass_criteria = turn.get("pass_criteria")
            
            # Reset intercepted calls for this turn
            global intercepted_calls
            intercepted_calls = []
            
            print(f"  Turn {turn_idx + 1} Input: '{user_input}'")
            
            try:
                # Add delay to stay within rate limits if needed
                time.sleep(5.0)
                
                # Execute agent response loop
                response_text = agent.run_agent_turn(chat, user_input)
                print(f"  Turn {turn_idx + 1} Output: '{response_text.strip()}'")
            except Exception as e:
                case_passed = False
                case_notes.append(f"Turn {turn_idx + 1} exception: {e}")
                print(f"  Turn {turn_idx + 1} Error: {e}")
                break
                
            # Intercepted tool names
            tools_called_this_turn = [call["name"] for call in intercepted_calls]
            actual_tools_called.extend(tools_called_this_turn)
            
            # Run verification assertions
            turn_passed, turn_msg = verify_turn(
                case_id=case_id,
                turn_idx=turn_idx,
                user_input=user_input,
                response_text=response_text,
                expected_tool=expected_tool,
                tools_called=tools_called_this_turn,
                pass_criteria=pass_criteria
            )
            
            if not turn_passed:
                case_passed = False
                case_notes.append(f"Turn {turn_idx+1} Failed: {turn_msg}")
            else:
                case_notes.append(f"T{turn_idx+1} OK")
                
        # Determine status
        status = "PASS" if case_passed else "FAIL"
        if case_passed:
            passed_count += 1
            
        tools_str = ", ".join(set(actual_tools_called)) if actual_tools_called else "None"
        notes_str = "; ".join(case_notes)
        
        results.append({
            "id": case_id,
            "pass": status,
            "tool_called": tools_str,
            "notes": notes_str
        })
        
        # Reset catalog to original state for the next test case
        restore_catalogue()
        
    # Clean up backups
    restore_catalogue()
    if os.path.exists(os.path.join(os.path.dirname(__file__), "../catalogue.json.bak")):
        os.remove(os.path.join(os.path.dirname(__file__), "../catalogue.json.bak"))
    if os.path.exists(os.path.join(os.path.dirname(__file__), "../assistant/metadata.pkl.bak")):
        os.remove(os.path.join(os.path.dirname(__file__), "../assistant/metadata.pkl.bak"))
        
    # Output evaluation results table
    print("\n" + "=" * 90)
    print(f"{'Case ID':<10} | {'Status':<6} | {'Tools Called':<25} | {'Notes'}")
    print("=" * 90)
    for r in results:
        print(f"{r['id']:<10} | {r['pass']:<6} | {r['tool_called']:<25} | {r['notes']}")
    print("=" * 90)
    print(f"Overall Score: {passed_count}/{len(test_cases)} passed")
    print("=" * 90 + "\n")

if __name__ == "__main__":
    main()
