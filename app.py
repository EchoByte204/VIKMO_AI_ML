import sys
import os

# Add assistant folder to search path
sys.path.append(os.path.join(os.path.dirname(__file__), "assistant"))

import streamlit as st
import json
import os
import datetime
from PIL import Image
import pandas as pd
import plotly.graph_objects as go

# Import our backend modules
from agent import create_agent_chat_session, run_agent_turn
from rag import search, search_by_vehicle
from tools import load_catalogue_data

# Set page configuration
st.set_page_config(
    page_title="VIKMO Live Control Center",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sleek Minimalist Glassmorphic Dark Tech Theme Injection ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

/* Global Font & Body Reset */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    background-color: #f8fafc !important;
    color: #0f172a;
}

/* Reduce padding of main block container to prevent scrolling */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 1.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

/* Hide Default Headers/Footers */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
div[data-testid="stHeader"] {visibility: hidden;}

/* Custom Radial Background */
.stApp {
    background: radial-gradient(circle at top right, #e0e7ff 0%, #f8fafc 60%);
    background-attachment: fixed;
}

/* Custom styled Sidebar container */
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid rgba(99, 102, 241, 0.15) !important;
}

/* Sidebar elements */
[data-testid="stSidebar"] .stMarkdown h2 {
    color: #4f46e5 !important;
}

[data-testid="stSidebar"] .stMarkdown p {
    color: #475569 !important;
}

/* Futuristic Tech Cards - Light glassmorphism on Native Bordered Containers */
[data-testid="stVerticalBlockBorder"] {
    background: rgba(255, 255, 255, 0.7) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border-radius: 16px !important;
    border: 1px solid rgba(99, 102, 241, 0.12) !important;
    padding: 20px !important;
    box-shadow: 0 10px 30px rgba(99, 102, 241, 0.04) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    color: #334155 !important;
}
[data-testid="stVerticalBlockBorder"]:hover {
    border-color: rgba(6, 182, 212, 0.4) !important;
    box-shadow: 0 10px 30px rgba(6, 182, 212, 0.15) !important;
}

/* Connection pulsing dot */
.connection-bar {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 20px;
}
.pulse-green {
    width: 10px;
    height: 10px;
    background-color: #0891b2;
    border-radius: 50%;
    box-shadow: 0 0 12px #0891b2;
    animation: pulse-cyan 2s infinite;
}
@keyframes pulse-cyan {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(8, 145, 178, 0.7); }
    70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(8, 145, 178, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(8, 145, 178, 0); }
}

/* Custom Chat scrollable feed with textured background */
.chat-wall {
    height: 380px;
    overflow-y: auto;
    background: radial-gradient(circle, rgba(99, 102, 241, 0.05) 10%, transparent 11%);
    background-size: 16px 16px;
    background-color: #ffffff;
    border: 1px solid rgba(99, 102, 241, 0.12);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: inset 0 2px 10px rgba(99, 102, 241, 0.05);
}

/* Custom speech bubbles (Indigo Gradient vs Soft Slate) */
.user-msg {
    background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%);
    color: #ffffff;
    padding: 12px 18px;
    border-radius: 16px 16px 2px 16px;
    margin-bottom: 16px;
    max-width: 75%;
    align-self: flex-end;
    margin-left: auto;
    box-shadow: 0 4px 15px rgba(79, 70, 229, 0.15);
    border-right: 3px solid #06b6d4;
    line-height: 1.5;
}

.assistant-msg {
    background: #f1f5f9;
    border: 1px solid rgba(99, 102, 241, 0.12);
    color: #0f172a;
    padding: 12px 18px;
    border-radius: 16px 16px 16px 2px;
    margin-bottom: 16px;
    max-width: 75%;
    align-self: flex-start;
    margin-right: auto;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.02);
    border-left: 3px solid #4f46e5;
    line-height: 1.5;
}

.time-label {
    font-size: 0.7rem;
    color: #64748b;
    margin-top: 6px;
}
</style>
""", unsafe_allow_html=True)

# --- Initialize session states ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your VIKMO Dealer Assistant. Ask me about part stock, vehicles, or place an order!"}
    ]

if "chat" not in st.session_state:
    try:
        st.session_state.chat = create_agent_chat_session()
        st.session_state.chat_err = None
    except Exception as e:
        st.session_state.chat = None
        st.session_state.chat_err = str(e)

# --- Sidebar Header ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #8b5cf6; margin-bottom: 0px;'>⚡ VIKMO PORTAL</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748b; font-size: 0.82rem;'>AI DECK & DIAGNOSTICS</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Mode selection styled inside sidebar
    nav_selection = st.radio(
        "Select Portal Console:",
        ["💬 Live Dealer Chatbot", "📊 Inventory & Forecasts", "🛡️ Guardrail Audit Trail"]
    )
    
    st.markdown("---")
    # Quick session cleanup
    if st.button("🧹 Clear Chat History"):
        try:
            st.session_state.chat = create_agent_chat_session()
            st.session_state.chat_err = None
        except Exception as e:
            st.session_state.chat = None
            st.session_state.chat_err = str(e)
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am your VIKMO Dealer Assistant. Ask me about part stock, vehicles, or place an order!"}
        ]
        st.rerun()

# --- Main Layout Header ---
st.markdown("<div style='font-size: 2.1rem; font-weight: 800; background: linear-gradient(90deg, #4f46e5 0%, #06b6d4 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>⚡ VIKMO LIVE CONTROL CONSOLE</div>", unsafe_allow_html=True)
st.markdown("<p style='color: #475569; font-size: 0.9rem; margin-top:-5px; margin-bottom: 25px;'>High-Performance AI Layer for Automotive Parts & Demand Forecasting</p>", unsafe_allow_html=True)

# ==================== PAGE 1: LIVE CHATBOT ====================
if nav_selection == "💬 Live Dealer Chatbot":
    
    # Split Layout
    left_col, right_col = st.columns([2, 3], gap="large")
    
    # Left Panel: Diagnostics, Neural Scan, Catalogue Look
    with left_col:
        
        # Neural scan uploader
        with st.container(border=True):
            st.markdown("<h3 style='color: #0891b2; margin-top:0px;'>📷 Visual Part Identifier</h3>", unsafe_allow_html=True)
            st.write("Drag and drop a photo of an auto component. Gemini Vision will identify it and query the RAG catalog.")
            
            uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
            
            has_client = st.session_state.chat is not None and st.session_state.chat.client is not None
            if uploaded_file and has_client:
                img = Image.open(uploaded_file)
                st.image(img, caption="Scanning Target", use_container_width=True)
                
                if st.button("⚡ Execute Neural Scan"):
                    with st.spinner("Classifying part model..."):
                        try:
                            import time
                            # Implement robust API key rotation and retries for visual model call
                            num_keys = len(st.session_state.chat.keys) if hasattr(st.session_state.chat, "keys") else 1
                            total_allowed_attempts = 4 * num_keys
                            
                            response = None
                            attempt = 0
                            delay = 1.0
                            
                            while attempt < total_allowed_attempts:
                                try:
                                    response = st.session_state.chat.client.models.generate_content(
                                        model="gemini-2.5-flash",
                                        contents=[
                                            img,
                                            "Identify this auto part from the image. Tell us its category, description, and search keyword. Be concise."
                                        ]
                                    )
                                    break
                                except Exception as e:
                                    err_msg = str(e)
                                    is_quota_or_rate = "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower()
                                    
                                    if is_quota_or_rate and hasattr(st.session_state.chat, "rotate"):
                                        if st.session_state.chat.rotate():
                                            # Reset delay since we are on a fresh key
                                            delay = 1.0
                                            attempt += 1
                                            continue
                                    
                                    is_temporary = any(x in err_msg for x in ["503", "429", "RESOURCE_EXHAUSTED", "UNAVAILABLE"]) or "overload" in err_msg.lower()
                                    if is_temporary and attempt < total_allowed_attempts - 1:
                                        time.sleep(delay)
                                        delay *= 2.0
                                        attempt += 1
                                    else:
                                        raise e
                            
                            if response is None:
                                raise RuntimeError("Exhausted all retries and API keys.")
                                
                            result_text = response.text
                            st.success(f"Scanning Results:\n{result_text}")
                            
                            # Fetch RAG matches
                            matches = search(result_text, top_k=3)
                            st.markdown("##### Catalogue Recommendations:")
                            for m in matches:
                                st.info(f"**[{m['sku']}]** {m['name']} - ₹{m['price_inr']} (Available stock: {m['stock']})")
                        except Exception as ex:
                            st.error(f"Scanner error: {str(ex)}")
            elif uploaded_file:
                st.warning("Connect a valid API key in `.env` to enable Visual Scanning.")
        
        # RAG Quick Lookup search bar
        with st.container(border=True):
            st.markdown("<h3 style='color: #4f46e5; margin-top:0px;'>🔍 Quick Catalogue Lookup</h3>", unsafe_allow_html=True)
            search_query = st.text_input("Enter part or vehicle keywords (e.g. 'Motul lube'):")
            if search_query:
                results = search(search_query, top_k=3)
                for res in results:
                    stock_color = "#10b981" if res["stock"] > 0 else "#ef4444"
                    st.markdown(f"""
                    <div style='background: rgba(99, 102, 241, 0.04); padding: 12px; border-radius: 8px; margin-bottom: 8px; border: 1px solid rgba(99, 102, 241, 0.1); color: #0f172a;'>
                        <span style='color:#4f46e5; font-weight:600;'>[{res['sku']}]</span> <b>{res['name']}</b><br/>
                        <small style='color: #475569;'>Category: {res['category']} | Price: ₹{res['price_inr']}</small><br/>
                        <small style='color: #475569;'>Stock status: <span style='color:{stock_color}; font-weight:600;'>{res['stock']} units ({'In Stock' if res['stock'] > 0 else 'Out of Stock'})</span></small>
                    </div>
                    """, unsafe_allow_html=True)
        
    with right_col:
        with st.container(border=True):
            # Chat Session Header
            st.markdown("""
            <div class="connection-bar">
                <div class="pulse-green"></div>
                <div style="font-weight: 700; font-size: 1.1rem; color: #0891b2;">LIVE CONVERSATION CONSOLE</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Render scrollable wallpaper
            chat_content = '<div class="chat-wall">'
            for msg in st.session_state.messages:
                time_label = datetime.datetime.now().strftime("%H:%M")
                if msg["role"] == "user":
                    chat_content += f'<div class="user-msg">{msg["content"]}<div class="time-label" style="text-align:right;">{time_label}</div></div>'
                else:
                    chat_content += f'<div class="assistant-msg">{msg["content"]}<div class="time-label">{time_label}</div></div>'
            chat_content += '</div>'
            
            st.markdown(chat_content, unsafe_allow_html=True)
            
            # Chat Input Area
            user_message = st.chat_input("Ask a question, query stock levels, or request an order...")
            if user_message:
                st.session_state.messages.append({"role": "user", "content": user_message})
                
                if st.session_state.chat:
                    with st.spinner("Processing Agent Response..."):
                        try:
                            reply = run_agent_turn(st.session_state.chat, user_message)
                        except Exception as err:
                            reply = f"Error communicating with agent: {str(err)}"
                else:
                    reply = f"API Client offline. Error: {st.session_state.get('chat_err', 'Unknown initialization error.')}. Please check your `.env` file."
                    
                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.rerun()

# ==================== PAGE 2: INVENTORY & FORECASTS ====================
elif nav_selection == "📊 Inventory & Forecasts":
    
    with st.container(border=True):
        st.markdown("<h3 style='color: #0891b2; margin-top:0px;'>📊 Demand Forecast Analytics Panel</h3>", unsafe_allow_html=True)
        st.write("Compare Prophet predictions containing promotional flag regressors against baseline models across 30 auto-parts SKUs.")
        
        results_file = "forecasting/forecast_results.json"
        if not os.path.exists(results_file):
            st.error("Forecasting cache `forecast_results.json` not found. Please execute the forecasting script (`python forecasting/forecast.py`) first.")
        else:
            with open(results_file, "r", encoding="utf-8") as f:
                f_data = json.load(f)
                
            overall_metrics = f_data["overall_metrics"]
            sku_results = f_data["sku_results"]
            
            # Split Layout: Select + recommendation on left, graph on right
            col_f_left, col_f_right = st.columns([1, 2], gap="large")
            
            with col_f_left:
                st.write("##### Select SKU Code")
                target_sku = st.selectbox("SKU Code:", list(sku_results.keys()))
                
                if target_sku:
                    sku_data = sku_results[target_sku]
                    
                    # Fetch details from catalogue
                    catalog = load_catalogue_data()
                    item_details = next((item for item in catalog if item["sku"] == target_sku), None)
                    
                    if item_details:
                        st.markdown(f"""
                        <div style='background: rgba(99, 102, 241, 0.04); padding: 15px; border-radius: 8px; border: 1px solid rgba(99, 102, 241, 0.1); margin-bottom: 15px; color: #0f172a;'>
                            <b>Part Name</b>: {item_details['name']}<br/>
                            <b>Brand</b>: {item_details['brand']}<br/>
                            <b>Price</b>: ₹{item_details['price_inr']} | <b>Stock</b>: {item_details['stock']} units
                        </div>
                        """, unsafe_allow_html=True)
                        
                    # Reorder calculation logic
                    st.write("##### 🤖 Smart Inventory Reorder Planner")
                    forecasted_demand = sum(sku_data["prophet_forecast"])
                    current_stock = item_details["stock"] if item_details else 0
                    safety_buffer = 15
                    suggested_reorder = max(0, int(forecasted_demand + safety_buffer - current_stock))
                    
                    if suggested_reorder > 0:
                        st.markdown(f"""
                        <div style='background: #fef2f2; padding: 15px; border-radius: 8px; border: 1px solid #fecaca; color: #991b1b;'>
                            🔴 <b>Restock Needed</b><br/>
                            Prophet predicts 4-week demand of <b>{int(forecasted_demand)} units</b>. Current stock is <b>{current_stock}</b>. Recommend purchasing <b>{suggested_reorder} units</b> immediately.
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style='background: #ecfdf5; padding: 15px; border-radius: 8px; border: 1px solid #a7f3d0; color: #065f46;'>
                            🟢 <b>Stock Level Optimal</b><br/>
                            Current stock of <b>{current_stock}</b> is sufficient to cover Prophet's 4-week forecast of <b>{int(forecasted_demand)} units</b>.
                        </div>
                        """, unsafe_allow_html=True)
                        
                    # Metric error details
                    st.write("##### Accuracy Metrics (SKU)")
                    st.markdown(f"""
                    <div style='background: rgba(99, 102, 241, 0.03); padding: 10px; border-radius: 6px; border: 1px solid rgba(99, 102, 241, 0.06); font-size: 0.85rem; color: #475569;'>
                        • Prophet SKU MAE: <b style='color: #0f172a;'>{sku_data['metrics']['prophet']['mae']:.2f}</b><br/>
                        • Naive Baseline MAE: <b style='color: #0f172a;'>{sku_data['metrics']['naive']['mae']:.2f}</b><br/>
                        • Seasonal Naive MAE: <b style='color: #0f172a;'>{sku_data['metrics']['snaive']['mae']:.2f}</b>
                    </div>
                    """, unsafe_allow_html=True)
                    
            with col_f_right:
                if target_sku:
                    # Plotly Chart
                    sku_data = sku_results[target_sku]
                    fig = go.Figure()
                    
                    # Historical Sales
                    fig.add_trace(go.Scatter(
                        x=sku_data["historical_dates"], y=sku_data["historical_sales"],
                        mode='lines+markers', name='Historical Sales',
                        line=dict(color='#8b5cf6', width=2)
                    ))
                    
                    # Test Actuals
                    fig.add_trace(go.Scatter(
                        x=sku_data["test_dates"], y=sku_data["test_sales"],
                        mode='lines+markers', name='True Sales (Held Out Window)',
                        line=dict(color='#06b6d4', width=3)
                    ))
                    
                    # Prophet Forecast
                    fig.add_trace(go.Scatter(
                        x=sku_data["test_dates"], y=sku_data["prophet_forecast"],
                        mode='lines+markers', name='Prophet Prediction',
                        line=dict(color='#a78bfa', width=3, dash='dash')
                    ))
                    
                    # Naive
                    fig.add_trace(go.Scatter(
                        x=sku_data["test_dates"], y=sku_data["naive_forecast"],
                        mode='lines', name='Naive Baseline',
                        line=dict(color='#f59e0b', width=2, dash='dot')
                    ))
                    
                    # Seasonal Naive
                    fig.add_trace(go.Scatter(
                        x=sku_data["test_dates"], y=sku_data["snaive_forecast"],
                        mode='lines', name='Seasonal Naive',
                        line=dict(color='#ef4444', width=2, dash='dot')
                    ))
                    
                    fig.update_layout(
                        title=f"Sales Forecast vs Benchmarks for {target_sku}",
                        xaxis_title="Week Commencing",
                        yaxis_title="Quantity Sold",
                        template="plotly_white",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(241, 245, 249, 0.5)",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom", y=1.02,
                            xanchor="right", x=1
                        ),
                        margin=dict(l=40, r=40, t=85, b=45)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Macro Metrics Summary
                    st.write("##### Macro Error Averages across all 30 SKUs:")
                    col_ov1, col_ov2, col_ov3 = st.columns(3)
                    col_ov1.metric("Prophet Average MAE", f"{overall_metrics['prophet']['mae']:.2f}")
                    col_ov2.metric("Naive Average MAE", f"{overall_metrics['naive']['mae']:.2f}")
                    col_ov3.metric("Seasonal Naive Avg MAE", f"{overall_metrics['snaive']['mae']:.2f}")

# ==================== PAGE 3: GUARDRAIL AUDIT TRAIL ====================
elif nav_selection == "🛡️ Guardrail Audit Trail":
    with st.container(border=True):
        st.subheader("🛡️ Conversation Guardrail Auditing logs")
        st.write("This table logs every conversation violation flagged by our prompt guardrails. "
                 "Any off-topic question will trigger the logging system to record the prompt here.")
        
        log_file = "guardrail_logs.json"
        if not os.path.exists(log_file):
            st.info("No guardrail violations recorded yet! Try asking the assistant about something completely unrelated (e.g. 'what is the weather?') to check the logging.")
        else:
            with open(log_file, "r", encoding="utf-8") as f:
                logs = json.load(f)
                
            st.write(f"Total violations logged: **{len(logs)}**")
            
            # Render log table
            log_df = pd.DataFrame(logs)
            log_df.index = log_df.index + 1
            log_df.rename(columns={
                "timestamp": "Logged Timestamp", 
                "query": "Mechanic Query Prompt", 
                "reason": "Guardrail Reason Category"
            }, inplace=True)
            st.dataframe(log_df, use_container_width=True)
