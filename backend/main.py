"""
FastAPI Backend for CSV Upload System.
Provides endpoint for CSV upload with duplicate detection.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import pandas as pd
import io
import hashlib

# Import local modules
import database
import utils


# Initialize FastAPI app
app = FastAPI(
    title="CSV Upload API",
    description="FastAPI backend for CSV upload and storage",
    version="2.0.0"
)

# Configure CORS to allow Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    database.init_db()
    print("[INFO] FastAPI server started successfully")


# Pydantic models for request/response
class UploadResponse(BaseModel):
    success: bool
    message: str
    dataset_id: Optional[int] = None
    is_duplicate: bool
    metadata: Optional[Dict[str, Any]] = None


class TrainResponse(BaseModel):
    success: bool
    message: str
    metrics: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    success: bool
    answer: str


def compute_file_hash(file_content: bytes) -> str:
    """Compute MD5 hash of file content."""
    return hashlib.md5(file_content).hexdigest()


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint to verify API is running."""
    return {
        "message": "ML System API is running",
        "endpoints": ["/upload", "/datasets", "/train", "/chat", "/health"]
    }


# Upload CSV endpoint
@app.post("/upload", response_model=UploadResponse)
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file, check for duplicates, and store in database.
    
    Args:
        file: Uploaded CSV file
        
    Returns:
        UploadResponse with success status and metadata
    """
    try:
        # Read file content
        content = await file.read()
        
        # Compute hash for duplicate detection
        file_hash = compute_file_hash(content)
        
        # Check if duplicate exists
        is_duplicate = database.check_duplicate(file_hash)
        
        if is_duplicate:
            return UploadResponse(
                success=True,
                message="Dataset already exists.",
                is_duplicate=True
            )
        
        # Parse CSV
        try:
            df = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")
        
        # Store dataset
        dataset_id = database.store_dataset(file.filename, file_hash, df)
        
        # Get metadata
        metadata = {
            'filename': file.filename,
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': df.columns.tolist(),
            'dataset_id': dataset_id
        }
        
        return UploadResponse(
            success=True,
            message="Dataset stored successfully",
            dataset_id=dataset_id,
            is_duplicate=False,
            metadata=metadata
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


# Get all datasets endpoint
@app.get("/datasets")
async def get_datasets():
    """
    Get metadata for all stored datasets.
    
    Returns:
        List of dataset metadata
    """
    try:
        datasets = database.get_all_datasets()
        return {"datasets": datasets}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve datasets: {str(e)}")


# Train model endpoint
@app.post("/train", response_model=TrainResponse)
async def train_model():
    """
    Train a machine learning model on the latest dataset.
    
    Returns:
        TrainResponse with success status and training metrics
    """
    try:
        # Get latest dataset from database
        df = database.get_latest_data()
        
        if df is None:
            raise HTTPException(status_code=400, detail="No dataset found. Please upload data first.")
        
        # Train model
        result = utils.train_model(df)
        
        if not result['success']:
            raise HTTPException(status_code=400, detail=result['message'])
        
        return TrainResponse(
            success=True,
            message=result['message'],
            metrics=result['metrics']
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


# Chatbot endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Answer questions about the dataset using AI chatbot.
    
    Args:
        request: ChatRequest containing the user's question
        
    Returns:
        ChatResponse with the answer
    """
    try:
        # Get latest dataset from database
        df = database.get_latest_data()
        
        if df is None:
            raise HTTPException(status_code=400, detail="No dataset found. Please upload data first.")
        
        # Load trained model (if available)
        model_data = utils.load_model()
        
        # Answer question
        answer = utils.answer_question(request.question, df, model_data)
        
        return ChatResponse(
            success=True,
            answer=answer
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
