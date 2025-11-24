"""
Configuration file for API endpoints.
"""

import os
import streamlit as st

# 1. Try to get from Streamlit Secrets (Best for Cloud)
# 2. Try to get from Environment Variables (Best for Docker/Render)
# 3. Fallback to Localhost (Best for Local Development)

try:
    # Check if running on Streamlit Cloud and if secret exists
    if "API_BASE_URL" in st.secrets:
        API_BASE_URL = st.secrets["API_BASE_URL"]
    else:
        API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
except FileNotFoundError:
    # st.secrets fails locally if .streamlit/secrets.toml doesn't exist
    API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
except Exception:
    API_BASE_URL = "http://localhost:8000"

# Remove trailing slash if present
if API_BASE_URL.endswith("/"):
    API_BASE_URL = API_BASE_URL[:-1]

print(f"[CONFIG] Using API URL: {API_BASE_URL}")
