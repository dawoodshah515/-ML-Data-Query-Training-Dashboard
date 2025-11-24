"""
Streamlit Frontend for ML System.
Single-page interface with CSV upload, model training, and chatbot sections.
"""

import streamlit as st
import streamlit.components.v1 as components
import requests
import pandas as pd
import json
from typing import Dict, Any

# Import configuration
try:
    from config import API_BASE_URL
except ImportError:
    # Fallback if config.py doesn't exist
    API_BASE_URL = "http://localhost:8000"
    print("[WARNING] config.py not found, using default localhost:8000")
st.set_page_config(
    page_title="ML Data Query & Training Dashboard",
    page_icon="🤖",
    layout="wide"
)


# Pastel Bluish-White UI theme (CSS-only changes)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    :root{ --bg-soft:#f6fbff; --bg-wide: linear-gradient(180deg,#eef8ff 0%, #f7fdff 35%, #ffffff 100%); --accent-blue:#60a5fa; --accent-deep:#3b82f6; --muted-blue:#9fbffb; --card-bg:rgba(255,255,255,0.96); --card-border:rgba(59,130,246,0.09); --card-shadow:0 6px 28px rgba(59,130,246,0.08); --card-hover:0 12px 48px rgba(59,130,246,0.10); --text-primary:#0f172a; --text-secondary:#334155; --muted:#64748b; --radius:12px; --ease:cubic-bezier(0.2,0.9,0.2,1); }

    *{ box-sizing:border-box; font-family:'Poppins', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; }
    /* Make the page have a full-bleed soft bluish background while keeping cards white */
    .stApp, body, html { background: var(--bg-wide) !important; }
    .main{ background: transparent; padding:0.6rem; min-height:100vh; }
    .block-container{ max-width:1200px !important; margin:0 auto; padding:0.8rem 1rem !important; }

    h1{ font-size:1.6rem !important; font-weight:700 !important; color:var(--text-primary) !important; margin:0 0 0.5rem 0 !important; text-align:center !important; }
    h2{ font-size:1.05rem !important; font-weight:600 !important; color:var(--text-secondary) !important; margin:1rem 0 0.5rem 0 !important; }
    h3{ font-size:0.95rem !important; color:var(--text-secondary) !important; }
    p, label, li{ color:var(--text-primary) !important; font-size:0.9rem !important; line-height:1.5 !important; }
    hr{ height:1px; border:0; background:linear-gradient(90deg, transparent, rgba(15,23,42,0.06), transparent); margin:1rem 0; }

    .stButton>button{ width:100%; background:linear-gradient(90deg,var(--accent-blue),var(--accent-deep)) !important; color:white !important; border:none !important; border-radius:10px !important; padding:0.6rem 1rem !important; font-weight:600 !important; box-shadow:0 6px 18px rgba(59,130,246,0.12) !important; transition:transform 220ms var(--ease), box-shadow 220ms var(--ease), opacity 220ms var(--ease) !important; }
    .stButton>button:hover{ transform:translateY(-3px); box-shadow:var(--card-hover) !important; opacity:0.98 !important; }
    .stButton>button:active{ transform:translateY(-1px) scale(0.995); }

    .stFileUploader{ background:var(--card-bg) !important; border:1px solid var(--card-border) !important; border-radius:var(--radius) !important; padding:1.25rem !important; box-shadow:var(--card-shadow) !important; transition: transform 260ms var(--ease), box-shadow 260ms var(--ease) !important; }
    .stFileUploader:hover{ transform:translateY(-4px); box-shadow:var(--card-hover) !important; }

    /* Ensure columns wrap nicely on small screens */
    [data-testid="column"]{ display:flex; flex-direction:column; flex:1 1 auto; min-width:0; }
    /* Streamlit often wraps columns in a container; provide a safe wrap rule */
    .stColumns, .stColumns > div { flex-wrap: wrap !important; }

    /* Interactive focus & micro animations */
    .stButton>button, .stFileUploader, .streamlit-expanderHeader, .dataframe{ will-change: transform, box-shadow, opacity; }
    .stTextInput>div>div>input::placeholder, .stTextArea>div>div>textarea::placeholder{ color:rgba(100,116,139,0.6) !important; }
    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus{ transform: translateZ(0); }

    .stTextInput>div>div>input, .stTextArea>div>div>textarea{ background:white !important; border:1px solid rgba(15,23,42,0.06) !important; border-radius:10px !important; padding:0.6rem 0.9rem !important; font-size:0.9rem !important; transition: box-shadow 200ms var(--ease), border-color 200ms var(--ease) !important; }
    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus{ box-shadow:0 6px 16px rgba(59,130,246,0.08) !important; border-color:var(--accent-deep) !important; outline:none !important; }

    .stTextInput label, .stTextArea label{ color:var(--text-secondary) !important; font-weight:600 !important; margin-bottom:0.35rem !important; }

    .streamlit-expanderHeader{ background:var(--card-bg) !important; border-radius:10px !important; padding:0.9rem 1rem !important; border:1px solid var(--card-border) !important; box-shadow:var(--card-shadow) !important; transition: transform 220ms var(--ease), box-shadow 220ms var(--ease) !important; }
    .streamlit-expanderHeader:hover{ transform:translateY(-3px); box-shadow:var(--card-hover) !important; }
    .streamlit-expanderContent{ background:transparent !important; padding:1rem 0.5rem 1.25rem 0.5rem !important; }

    .dataframe{ background:white !important; border-radius:10px !important; overflow:hidden !important; box-shadow:var(--card-shadow) !important; }
    .dataframe thead tr th{ background:linear-gradient(90deg, rgba(96,165,250,0.12), rgba(59,130,246,0.10)) !important; color:var(--text-primary) !important; padding:0.6rem 0.9rem !important; font-weight:600 !important; }
    .dataframe tbody tr td{ padding:0.6rem 0.9rem !important; color:var(--text-primary) !important; border-bottom:1px solid rgba(15,23,42,0.03) !important; }
    .dataframe tbody tr:hover{ background: rgba(96,165,250,0.03) !important; }

    [data-testid="stMetricValue"]{ font-size:1.4rem !important; font-weight:700 !important; color:var(--accent-deep) !important; }
    [data-testid="stMetricLabel"]{ color:var(--muted) !important; font-weight:600 !important; }
    [data-testid="metric-container"]{ background:var(--card-bg) !important; border:1px solid var(--card-border) !important; border-radius:12px !important; padding:0.9rem !important; box-shadow:var(--card-shadow) !important; }

    [data-testid="column"]{ padding:0.35rem !important; }

    ::-webkit-scrollbar{ width:10px; height:10px; }
    ::-webkit-scrollbar-track{ background:transparent; }
    ::-webkit-scrollbar-thumb{ background: linear-gradient(180deg,var(--muted-blue, #bcdffc), var(--accent-blue)); border-radius:8px; }

    @keyframes subtleFadeUp{ from{ opacity:0; transform:translateY(8px);} to{ opacity:1; transform:translateY(0);} }
    .stApp, .block-container, .stFileUploader, .dataframe, .streamlit-expanderHeader{ animation: subtleFadeUp 420ms var(--ease) both; }

    /* Animated entrances for major UI pieces (staggered & subtle) */
    @keyframes popIn { 0%{ opacity:0; transform:scale(0.98) translateY(6px);} 60%{ opacity:1; transform:scale(1.02) translateY(-2px);} 100%{ transform:scale(1) translateY(0); } }
    @keyframes slideRight { from{ opacity:0; transform:translateX(-18px);} to{ opacity:1; transform:translateX(0);} }
    @keyframes slideLeft { from{ opacity:0; transform:translateX(18px);} to{ opacity:1; transform:translateX(0);} }
    @keyframes liftUp { from{ opacity:0; transform:translateY(14px) scale(0.995);} to{ opacity:1; transform:translateY(0) scale(1);} }

    /* Apply staggered animations to common components */
    .hero, .hero * { animation-name: popIn; animation-duration: 520ms; animation-timing-function: var(--ease); animation-fill-mode: both; }
    .stFileUploader{ animation-name: liftUp; animation-duration: 520ms; animation-delay: 80ms; }
    .streamlit-expanderHeader{ animation-name: slideRight; animation-duration: 520ms; animation-delay: 120ms; }
    .stButton>button{ animation-name: popIn; animation-duration: 420ms; animation-delay: 180ms; }
    .dataframe{ animation-name: liftUp; animation-duration: 520ms; animation-delay: 220ms; }
    [data-testid="metric-container"]{ animation-name: popIn; animation-duration: 460ms; animation-delay: 200ms; }
    [data-testid="column"]{ animation-name: slideLeft; animation-duration: 460ms; animation-delay: 140ms; }

    /* Stagger children inside container (best-effort) */
    .block-container > *:nth-child(1){ animation-delay: 40ms; }
    .block-container > *:nth-child(2){ animation-delay: 90ms; }
    .block-container > *:nth-child(3){ animation-delay: 140ms; }
    .block-container > *:nth-child(4){ animation-delay: 190ms; }
    .block-container > *:nth-child(5){ animation-delay: 240ms; }

    /* Gentle hover micro-animations */
    .stButton>button:hover{ transform: translateY(-4px) scale(1.01) !important; }
    .stFileUploader:hover{ transform: translateY(-6px) !important; }

    /* Respect reduced motion preferences */
    @media (prefers-reduced-motion: reduce){
        .stApp, .block-container, .stFileUploader, .dataframe, .streamlit-expanderHeader, .stButton>button, [data-testid="metric-container"], [data-testid="column"]{ animation:none !important; transition:none !important; }
    }

    /* Hero tweaks: compact header, subtle radial backdrop, stronger title contrast */
    .hero{ position:relative; padding:0.8rem 0 0.6rem 0; margin-bottom:0.4rem; }
    .hero::before{ content:''; position:absolute; left:50%; top:0; transform:translateX(-50%); width:1200px; height:280px; background: radial-gradient(ellipse at center, rgba(226,247,255,0.75) 0%, rgba(230,245,255,0.35) 35%, transparent 68%); filter: blur(30px); pointer-events:none; z-index:0; }
    .hero > div, .hero h1, .hero p{ position:relative; z-index:1; }

    h1{ color:var(--text-primary) !important; -webkit-text-fill-color: var(--text-primary) !important; }

    /* Slightly stronger info box (quick info) to increase contrast */
    .stInfo, .stAlert, div[role="status"] { background: rgba(227,242,255,0.9) !important; border:1px solid rgba(59,130,246,0.10) !important; color:var(--text-primary) !important; }

    /* Inputs: stronger focus outline for keyboard users */
    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus{ box-shadow:0 8px 26px rgba(59,130,246,0.12) !important; border-color:var(--accent-deep) !important; outline: 3px solid rgba(59,130,246,0.06) !important; }

    @media (max-width:1024px){ .block-container{ padding:0.75rem !important; } h1{ font-size:1.4rem !important; } .stButton>button{ padding:0.55rem 0.9rem !important; } [data-testid="column"]{ flex:1 1 48% !important; } }
    @media (max-width:768px){ .block-container{ padding:0.6rem !important; } h1{ font-size:1.2rem !important; } .stButton>button{ padding:0.5rem 0.8rem !important; font-size:0.85rem !important; } .dataframe thead tr th, .dataframe tbody tr td{ padding:0.5rem 0.6rem !important; font-size:0.78rem !important; } [data-testid="column"]{ flex:1 1 100% !important; } }
    @media (max-width:480px){ .block-container{ padding:0.5rem !important; } h1{ font-size:1.0rem !important; } .stFileUploader{ padding:1rem !important; } .stTextInput>div>div>input, .stTextArea>div>div>textarea{ padding:0.45rem 0.6rem !important; font-size:0.82rem !important; } [data-testid="column"]{ flex:1 1 100% !important; } }

    /* Small Streamlit top spacing tweak if needed */
    .css-1d391kg { padding-top: 0px !important; }

    /* Per-letter header animation (slide in from left → right) */
    /* Keep letters on the same line even if HTML has linebreaks */
    .ml-title { white-space: nowrap; display:inline-block; }
    .ml-letter { display:inline-block; opacity:0; transform:translateX(-12px) scale(0.995); animation: letterPop 420ms var(--ease) forwards; }
    .ml-space { display:inline-block; width:0.5rem; }
    @keyframes letterPop { 0%{ opacity:0; transform:translateX(-12px) scale(0.995);} 60%{ opacity:1; transform:translateX(4px) scale(1.02);} 100%{ opacity:1; transform:translateX(0) scale(1);} }
    .hero h1 { display:inline-block; }
    @media (prefers-reduced-motion: reduce){ .ml-letter{ animation:none !important; opacity:1 !important; transform:none !important; } }

    /* DEBUG outlines - remove when finished debugging */
    /* These help identify whether selectors are matching in the browser */
    .debug-outline-ml .ml-letter, .ml-letter { outline: 1px dashed rgba(59,130,246,0.28) !important; }
    .stFileUploader, input[type="file"], .upload { outline: 1px dashed rgba(96,165,250,0.18) !important; }
    .stButton>button, button, .stButton button { outline: 1px dashed rgba(59,130,246,0.18) !important; }
    .dataframe, table { outline: 1px dashed rgba(99,102,241,0.12) !important; }
    /* End debug */
    </style>
""", unsafe_allow_html=True)


# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'dataset_uploaded' not in st.session_state:
    st.session_state.dataset_uploaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'training_metrics' not in st.session_state:
    st.session_state.training_metrics = None


# Stunning Enhanced Header with soft bluish-white gradient and subtle float
components.html("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
    #custom-hero{ text-align:center; font-family:'Poppins', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; color: #0f172a; }
    #custom-hero .emoji{ font-size:2.4rem; margin-bottom:0.2rem; display:inline-block; animation: float 4s ease-in-out infinite; }
    @keyframes float{ 0%{ transform:translateY(0);} 50%{ transform:translateY(-6px);} 100%{ transform:translateY(0);} }
    /* Per-letter animation for the main title */
    .title-letter{ display:inline-block; opacity:0; transform:translateX(-12px) scale(0.995); animation: letterPop 420ms cubic-bezier(0.2,0.9,0.2,1) forwards; }
    .title-space{ display:inline-block; width:0.42rem; }
    @keyframes letterPop{ 0%{ opacity:0; transform:translateX(-12px) scale(0.995);} 60%{ opacity:1; transform:translateX(4px) scale(1.02);} 100%{ opacity:1; transform:translateX(0) scale(1);} }
    @media (prefers-reduced-motion: reduce){ .title-letter{ animation:none !important; opacity:1 !important; transform:none !important; } }
    #custom-hero h1{ font-size:1.9rem; font-weight:800; margin:0; letter-spacing:-0.2px; line-height:1.05; filter: drop-shadow(0 8px 22px rgba(15,23,42,0.06)); color:#0f172a; }
    #custom-hero p{ color: rgba(15,23,42,0.78); font-size:1rem; margin-top:0.6rem; font-weight:500; }
</style>
<div id="custom-hero">
    <div class="emoji">🤖</div>

    <!-- Main title now animates per-letter -->
    <h1>
        <span class="title-letter" style="animation-delay:0ms">M</span>
        <span class="title-letter" style="animation-delay:40ms">L</span>
        <span class="title-space"> </span>
        <span class="title-letter" style="animation-delay:80ms">D</span>
        <span class="title-letter" style="animation-delay:120ms">a</span>
        <span class="title-letter" style="animation-delay:160ms">t</span>
        <span class="title-letter" style="animation-delay:200ms">a</span>
        <span class="title-space"> </span>
        <span class="title-letter" style="animation-delay:240ms">Q</span>
        <span class="title-letter" style="animation-delay:280ms">u</span>
        <span class="title-letter" style="animation-delay:320ms">e</span>
        <span class="title-letter" style="animation-delay:360ms">r</span>
        <span class="title-letter" style="animation-delay:400ms">y</span>
        <span class="title-space"> </span>
        <span class="title-letter" style="animation-delay:440ms">&</span>
        <span class="title-space"> </span>
        <span class="title-letter" style="animation-delay:480ms">T</span>
        <span class="title-letter" style="animation-delay:520ms">r</span>
        <span class="title-letter" style="animation-delay:560ms">a</span>
        <span class="title-letter" style="animation-delay:600ms">i</span>
        <span class="title-letter" style="animation-delay:640ms">n</span>
        <span class="title-letter" style="animation-delay:680ms">i</span>
        <span class="title-letter" style="animation-delay:720ms">n</span>
        <span class="title-letter" style="animation-delay:760ms">g</span>
        <span class="title-space"> </span>
        <span class="title-letter" style="animation-delay:800ms">D</span>
        <span class="title-letter" style="animation-delay:840ms">a</span>
        <span class="title-letter" style="animation-delay:880ms">s</span>
        <span class="title-letter" style="animation-delay:920ms">h</span>
        <span class="title-letter" style="animation-delay:960ms">b</span>
        <span class="title-letter" style="animation-delay:1000ms">o</span>
        <span class="title-letter" style="animation-delay:1040ms">a</span>
        <span class="title-letter" style="animation-delay:1080ms">r</span>
        <span class="title-letter" style="animation-delay:1120ms">d</span>
    </h1>
    <p>⚡ Upload CSV → Save → Train Model → Ask Anything About Your Data</p>
</div>
""", height=220)
st.markdown("**Upload CSV • Train Models • Ask Questions**")
st.markdown("---")


# Section 1: Upload CSV
st.header("📁 Section 1: Upload CSV Dataset")

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file to store in the database and train a model"
    )

with col2:
    st.markdown("### Quick Info")
    st.info("✓ Duplicate detection enabled\n\n✓ Auto-hash verification\n\n✓ Instant storage")

if uploaded_file is not None:
    try:
        # Read file for preview
        df_preview = pd.read_csv(uploaded_file)
        
        # Show preview
        with st.expander("📊 Preview Uploaded Data", expanded=True):
            st.write(f"**Shape:** {df_preview.shape[0]} rows × {df_preview.shape[1]} columns")
            st.write(f"**Columns:** {', '.join(df_preview.columns.tolist())}")
            st.dataframe(df_preview.head(4), use_container_width=True)
        
        # Upload button
        if st.button("🚀 Upload to Database", key="upload_btn"):
            with st.spinner("Uploading dataset..."):
                # Reset file pointer
                uploaded_file.seek(0)
                
                # Send to API
                files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}
                response = requests.post(f"{API_BASE_URL}/upload", files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if result['is_duplicate']:
                        st.warning("⚠️ **Dataset already exists.**\n\nThis dataset has been uploaded before.")
                    else:
                        st.success(f"✅ **Dataset stored successfully!**\n\nDataset ID: {result['dataset_id']}")
                        st.session_state.dataset_uploaded = True
                        
                        # Display metadata
                        if result.get('metadata'):
                            meta = result['metadata']
                            col_a, col_b, col_c = st.columns(3)
                            col_a.metric("Rows", meta['rows'])
                            col_b.metric("Columns", meta['columns'])
                            col_c.metric("Dataset ID", meta['dataset_id'])
                else:
                    st.error(f"❌ Upload failed: {response.json().get('detail', 'Unknown error')}")
    
    except Exception as e:
        error_msg = str(e)
        if "Connection refused" in error_msg and "localhost" in API_BASE_URL:
            st.error(f"❌ **Connection Failed**\n\nThe app is trying to connect to `{API_BASE_URL}`, but the connection was refused.\n\n**Why is this happening?**\nYou are likely running this app on the Cloud, but it's trying to connect to your *local* computer (localhost), which it cannot reach.\n\n**How to fix:**\n1. Deploy your backend to a cloud provider (like Render or Railway).\n2. Go to your Streamlit App Settings -> Secrets.\n3. Add `API_BASE_URL = 'https://your-backend-url.com'`.")
        else:
            st.error(f"❌ Error reading file: {error_msg}")

st.markdown("---")


# Section 2: Train Model
st.header("🧠 Section 2: Train Machine Learning Model")

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("🎯 Train Model", key="train_btn"):
        with st.spinner("Training model... This may take a few moments."):
            try:
                response = requests.post(f"{API_BASE_URL}/train")
                
                if response.status_code == 200:
                    result = response.json()
                    st.session_state.model_trained = True
                    st.session_state.training_metrics = result['metrics']
                    st.success("✅ **Model trained successfully!**")
                else:
                    error_detail = response.json().get('detail', 'Unknown error')
                    st.error(f"❌ Training failed: {error_detail}")
            
            except requests.exceptions.RequestException as e:
                error_msg = str(e)
                if "Connection refused" in error_msg and "localhost" in API_BASE_URL:
                    st.error(f"❌ **Connection Failed**\n\nThe app is trying to connect to `{API_BASE_URL}`, but the connection was refused.\n\n**Why is this happening?**\nYou are likely running this app on the Cloud, but it's trying to connect to your *local* computer (localhost), which it cannot reach.\n\n**How to fix:**\n1. Deploy your backend to a cloud provider (like Render or Railway).\n2. Go to your Streamlit App Settings -> Secrets.\n3. Add `API_BASE_URL = 'https://your-backend-url.com'`.")
                else:
                    st.error(f"❌ Connection error: {error_msg}\n\nMake sure the Backend server is running.")

with col2:
    st.info("**Training Info**\n\n✓ Auto-detects target column\n\n✓ Handles missing values\n\n✓ Saves model as model.pkl")

# Display training metrics if available
if st.session_state.training_metrics:
    st.subheader("📊 Training Results")
    
    metrics = st.session_state.training_metrics
    
    # Create metrics display
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Samples", metrics.get('n_samples', 'N/A'))
    col2.metric("Features", metrics.get('n_features', 'N/A'))
    col3.metric("Test Size", metrics.get('test_size', 'N/A'))
    
    if metrics.get('model_type') == 'classification':
        col4.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
    else:
        col4.metric("R² Score", f"{metrics.get('r2_score', 0):.4f}")
    
    # Detailed metrics
    with st.expander("📈 Detailed Metrics", expanded=True):
        st.json(metrics)

st.markdown("---")


# Section 3: Chatbot
st.header("💬 Section 3: AI Chatbot")

st.markdown("""
Ask questions about your data! I can help you with:
- **Summary statistics**: "What is the average of column X?"
- **Data insights**: "How many rows are in the dataset?"
- **Predictions**: "Predict the target for [values]"
- **General info**: "What is this dataset about?"
""")

# Example questions
with st.expander("💡 Example Questions"):
    st.markdown("""
    - What is the mean of all numeric columns?
    - How many rows and columns are in the dataset?
    - What are the column names?
    - Show me the first few rows
    - What is the maximum value of [column_name]?
    - Describe the dataset
    """)

# Chat input
question = st.text_input(
    "Ask a question:",
    placeholder="e.g., What is the average value of column X?",
    key="chat_input"
)

if st.button("📤 Send", key="chat_send") and question:
    with st.spinner("Thinking..."):
        try:
            response = requests.post(
                f"{API_BASE_URL}/chat",
                json={"question": question}
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result['answer']
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'question': question,
                    'answer': answer
                })
            else:
                st.error(f"❌ Error: {response.json().get('detail', 'Unknown error')}")
        
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Connection error: {str(e)}")

# Display chat history
if st.session_state.chat_history:
    st.subheader("💭 Chat History")
    
    for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
        with st.container():
            st.markdown(f"**🧑 You:** {chat['question']}")
            st.markdown(f"**🤖 Bot:** {chat['answer']}")
            st.markdown("---")

# Clear chat button
if st.session_state.chat_history:
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>All Rights Reserved @ Dawood Shah </p>

</div>
""", unsafe_allow_html=True)
