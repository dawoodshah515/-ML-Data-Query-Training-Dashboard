"""
Utility functions for ML model training, data processing, and chatbot logic.
"""

import hashlib
import pickle
import re
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report
from sklearn.preprocessing import LabelEncoder


MODEL_PATH = "model.pkl"
ENCODER_PATH = "label_encoder.pkl"


def compute_file_hash(file_content: bytes) -> str:
    """
    Compute MD5 hash of file content.
    
    Args:
        file_content: Raw bytes of the file
        
    Returns:
        MD5 hash as hex string
    """
    return hashlib.md5(file_content).hexdigest()


def detect_target_column(df: pd.DataFrame) -> Optional[str]:
    """
    Auto-detect the target column for ML training.
    Uses the last column by default.
    
    Args:
        df: Pandas DataFrame
        
    Returns:
        Name of the target column, or None if detection fails
    """
    if len(df.columns) < 2:
        return None  # Need at least 2 columns (1 feature + 1 target)
    
    return df.columns[-1]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data by handling missing values.
    
    Args:
        df: Pandas DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    df_cleaned = df.copy()
    
    for column in df_cleaned.columns:
        if df_cleaned[column].isnull().sum() > 0:
            # Handle numeric columns
            if pd.api.types.is_numeric_dtype(df_cleaned[column]):
                df_cleaned[column] = df_cleaned[column].fillna(df_cleaned[column].mean())
            # Handle categorical columns
            else:
                mode_value = df_cleaned[column].mode()[0] if not df_cleaned[column].mode().empty else "Unknown"
                df_cleaned[column] = df_cleaned[column].fillna(mode_value)
    
    return df_cleaned


def train_model(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Train a machine learning model on the provided dataset.
    
    Args:
        df: Pandas DataFrame with features and target
        
    Returns:
        Dictionary containing training results and metrics
    """
    # Detect target column
    target_col = detect_target_column(df)
    
    if target_col is None:
        return {
            'success': False,
            'message': 'Could not detect target column. Need at least 2 columns.',
            'metrics': {}
        }
    
    # Clean data
    df_cleaned = clean_data(df)
    
    # Separate features and target
    X = df_cleaned.drop(columns=[target_col])
    y = df_cleaned[target_col]
    
    # Handle categorical features
    label_encoders = {}
    for column in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column].astype(str))
        label_encoders[column] = le
    
    # Determine if classification or regression
    is_classification = False
    target_encoder = None
    
    if pd.api.types.is_numeric_dtype(y):
        # Check if it's discrete (classification) or continuous (regression)
        unique_ratio = len(y.unique()) / len(y)
        if unique_ratio < 0.05 or len(y.unique()) < 20:  # Heuristic for classification
            is_classification = True
    else:
        # Categorical target - definitely classification
        is_classification = True
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y.astype(str))
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train appropriate model
    if is_classification:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        metrics = {
            'model_type': 'classification',
            'accuracy': float(accuracy),
            'target_column': target_col,
            'n_samples': len(df),
            'n_features': X.shape[1],
            'test_size': len(X_test)
        }
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'model_type': 'regression',
            'mse': float(mse),
            'rmse': float(np.sqrt(mse)),
            'r2_score': float(r2),
            'target_column': target_col,
            'n_samples': len(df),
            'n_features': X.shape[1],
            'test_size': len(X_test)
        }
    
    # Save model and encoders
    model_data = {
        'model': model,
        'target_column': target_col,
        'feature_columns': X.columns.tolist(),
        'label_encoders': label_encoders,
        'target_encoder': target_encoder,
        'is_classification': is_classification
    }
    
    save_model(model_data)
    
    return {
        'success': True,
        'message': 'Model trained successfully',
        'metrics': metrics
    }


def save_model(model_data: Dict[str, Any]) -> None:
    """
    Save the trained model to disk.
    
    Args:
        model_data: Dictionary containing model and metadata
    """
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"[SUCCESS] Model saved to {MODEL_PATH}")


def load_model() -> Optional[Dict[str, Any]]:
    """
    Load the trained model from disk.
    
    Returns:
        Dictionary containing model and metadata, or None if not found
    """
    try:
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        return None


def answer_question(question: str, df: pd.DataFrame, model_data: Optional[Dict[str, Any]] = None) -> str:
    """
    Answer user questions using data analysis or ML predictions.
    
    Args:
        question: User's question
        df: Dataset DataFrame
        model_data: Loaded model data (optional)
        
    Returns:
        Answer string
    """
    question_lower = question.lower()
    
    # Route to appropriate handler
    if any(word in question_lower for word in ['predict', 'forecast', 'estimate', 'classification', 'what would be']):
        return handle_prediction_question(question, df, model_data)
    elif any(word in question_lower for word in ['mean', 'average', 'sum', 'count', 'max', 'min', 'median', 'std']):
        return handle_summary_question(question, df)
    elif any(word in question_lower for word in ['dataset', 'about', 'columns', 'rows', 'shape', 'info', 'describe']):
        return handle_metadata_question(question, df)
    elif any(word in question_lower for word in ['show', 'display', 'first', 'last', 'head', 'tail', 'sample']):
        return handle_display_question(question, df)
    else:
        return "I need more information. I can help you with:\n- Dataset summary (mean, count, max, min, etc.)\n- Dataset information (columns, rows, shape)\n- Predictions using the trained model\n- Displaying sample data"


def handle_prediction_question(question: str, df: pd.DataFrame, model_data: Optional[Dict[str, Any]]) -> str:
    """Handle prediction-related questions."""
    if model_data is None:
        return "No trained model available. Please train a model first by uploading data."
    
    return "To make a prediction, please provide specific feature values. For example: 'Predict for [feature1=value1, feature2=value2, ...]'"


def handle_summary_question(question: str, df: pd.DataFrame) -> str:
    """Handle statistical summary questions."""
    question_lower = question.lower()
    
    # Try to extract column name from question
    column_name = None
    for col in df.columns:
        if col.lower() in question_lower:
            column_name = col
            break
    
    try:
        if 'mean' in question_lower or 'average' in question_lower:
            if column_name and pd.api.types.is_numeric_dtype(df[column_name]):
                return f"The mean of '{column_name}' is {df[column_name].mean():.2f}"
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                result = "Mean values:\n"
                for col in numeric_cols[:5]:  # Limit to first 5
                    result += f"- {col}: {df[col].mean():.2f}\n"
                return result
        
        elif 'count' in question_lower or 'how many' in question_lower:
            if 'row' in question_lower:
                return f"The dataset contains {len(df)} rows."
            elif 'column' in question_lower:
                return f"The dataset contains {len(df.columns)} columns."
            else:
                return f"The dataset contains {len(df)} rows and {len(df.columns)} columns."
        
        elif 'max' in question_lower or 'maximum' in question_lower:
            if column_name and pd.api.types.is_numeric_dtype(df[column_name]):
                return f"The maximum value of '{column_name}' is {df[column_name].max():.2f}"
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                result = "Maximum values:\n"
                for col in numeric_cols[:5]:
                    result += f"- {col}: {df[col].max():.2f}\n"
                return result
        
        elif 'min' in question_lower or 'minimum' in question_lower:
            if column_name and pd.api.types.is_numeric_dtype(df[column_name]):
                return f"The minimum value of '{column_name}' is {df[column_name].min():.2f}"
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                result = "Minimum values:\n"
                for col in numeric_cols[:5]:
                    result += f"- {col}: {df[col].min():.2f}\n"
                return result
        
        elif 'sum' in question_lower:
            if column_name and pd.api.types.is_numeric_dtype(df[column_name]):
                return f"The sum of '{column_name}' is {df[column_name].sum():.2f}"
            else:
                return "Please specify which column you want to sum."
        
        elif 'median' in question_lower:
            if column_name and pd.api.types.is_numeric_dtype(df[column_name]):
                return f"The median of '{column_name}' is {df[column_name].median():.2f}"
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                result = "Median values:\n"
                for col in numeric_cols[:5]:
                    result += f"- {col}: {df[col].median():.2f}\n"
                return result
        
        else:
            return "I can calculate mean, median, sum, count, max, or min. Please specify what you'd like to know."
    
    except Exception as e:
        return f"Error processing your question: {str(e)}"


def handle_metadata_question(question: str, df: pd.DataFrame) -> str:
    """Handle dataset metadata questions."""
    question_lower = question.lower()
    
    if 'column' in question_lower:
        columns = df.columns.tolist()
        return f"The dataset has {len(columns)} columns:\n" + ", ".join(columns)
    
    elif 'row' in question_lower:
        return f"The dataset contains {len(df)} rows."
    
    elif 'shape' in question_lower:
        return f"The dataset shape is {df.shape[0]} rows × {df.shape[1]} columns."
    
    elif 'about' in question_lower or 'describe' in question_lower or 'info' in question_lower:
        info = f"Dataset Information:\n"
        info += f"- Rows: {len(df)}\n"
        info += f"- Columns: {len(df.columns)}\n"
        info += f"- Column names: {', '.join(df.columns.tolist())}\n"
        info += f"- Numeric columns: {len(df.select_dtypes(include=[np.number]).columns)}\n"
        info += f"- Categorical columns: {len(df.select_dtypes(include=['object']).columns)}\n"
        return info
    
    else:
        return f"Dataset has {len(df)} rows and {len(df.columns)} columns: {', '.join(df.columns.tolist())}"


def handle_display_question(question: str, df: pd.DataFrame) -> str:
    """Handle data display questions."""
    question_lower = question.lower()
    
    if 'first' in question_lower or 'head' in question_lower:
        n = 5  # Default
        return f"First {n} rows:\n{df.head(n).to_string()}"
    
    elif 'last' in question_lower or 'tail' in question_lower:
        n = 5
        return f"Last {n} rows:\n{df.tail(n).to_string()}"
    
    elif 'sample' in question_lower:
        n = 3
        return f"Random sample of {n} rows:\n{df.sample(min(n, len(df))).to_string()}"
    
    else:
        return f"First 5 rows:\n{df.head().to_string()}"
