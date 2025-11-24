"""
Configuration file for API endpoints.
"""

import os

# Check if running in production (Streamlit Cloud) or locally
IS_PRODUCTION = os.getenv("STREAMLIT_RUNTIME_ENV") == "cloud"

# API Base URL
if IS_PRODUCTION:
    # TODO: Replace with your deployed backend URL (e.g., Railway, Render, Heroku)
    API_BASE_URL = "https://your-backend-url.railway.app"
else:
    # Local development
    API_BASE_URL = "http://localhost:8000"

print(f"[CONFIG] Running in {'PRODUCTION' if IS_PRODUCTION else 'DEVELOPMENT'} mode")
print(f"[CONFIG] API Base URL: {API_BASE_URL}")
