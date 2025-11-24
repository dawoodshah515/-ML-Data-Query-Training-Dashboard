# Deployment Guide

## Current Issue
The Streamlit frontend is deployed on Streamlit Cloud (or similar), but the FastAPI backend is running locally on your Windows machine. They cannot communicate because:
- Frontend is on cloud (Linux environment)
- Backend is on localhost (your Windows machine)

## Solutions

### Option 1: Run Everything Locally (Development) ✅ CURRENTLY ACTIVE

Both servers are now running on your local machine:

**Backend (FastAPI):**
```bash
cd backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
- Running on: http://localhost:8000
- Status: ✅ RUNNING

**Frontend (Streamlit):**
```bash
cd frontend
streamlit run app.py
```
- Running on: http://localhost:8502
- Status: ✅ RUNNING

**Access the app:** Open http://localhost:8502 in your browser

---

### Option 2: Deploy Backend to Cloud (Production)

To use your deployed Streamlit app, you need to deploy the FastAPI backend to a cloud service:

#### Recommended Services:
1. **Railway** (easiest, free tier)
2. **Render** (free tier available)
3. **Heroku** (requires credit card)
4. **Google Cloud Run**
5. **AWS Lambda**

#### Steps for Railway Deployment:

1. **Create `Procfile` in backend folder:**
```
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

2. **Create `runtime.txt` in backend folder:**
```
python-3.13.0
```

3. **Push to GitHub** (already done ✅)

4. **Deploy on Railway:**
   - Go to https://railway.app
   - Sign in with GitHub
   - Create new project → Deploy from GitHub
   - Select your repository
   - Select `backend` folder as root directory
   - Railway will auto-detect and deploy

5. **Get your backend URL:**
   - Railway will give you a URL like: `https://your-app.railway.app`

6. **Update `frontend/config.py`:**
```python
if IS_PRODUCTION:
    API_BASE_URL = "https://your-app.railway.app"  # Replace with your Railway URL
```

7. **Push changes to GitHub:**
```bash
git add .
git commit -m "Update production API URL"
git push origin main
```

8. **Streamlit Cloud will auto-redeploy** with the new backend URL

---

### Option 3: Deploy Both to Same Service

Deploy both frontend and backend to the same cloud service (more complex but possible).

---

## Quick Test

Test if backend is accessible:
```bash
curl http://localhost:8000/health
```

Should return: `{"status":"healthy"}`

---

## Environment Variables (Alternative)

Instead of hardcoding URLs, you can use environment variables:

**In Streamlit Cloud:**
1. Go to App Settings → Secrets
2. Add:
```toml
API_BASE_URL = "https://your-backend-url.railway.app"
```

**In `frontend/config.py`:**
```python
import os
import streamlit as st

# Try to get from Streamlit secrets first, then environment, then default
try:
    API_BASE_URL = st.secrets.get("API_BASE_URL", os.getenv("API_BASE_URL", "http://localhost:8000"))
except:
    API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
```

---

## Current Status

✅ **Backend:** Running locally on http://localhost:8000
✅ **Frontend:** Running locally on http://localhost:8502

**Next Steps:**
1. Open http://localhost:8502 in your browser
2. Upload a CSV file
3. Click "Upload to Database" - should work now!

OR

1. Deploy backend to Railway/Render
2. Update config.py with production URL
3. Use your Streamlit Cloud deployment
