"""
Database module for SQLite operations.
Handles dataset storage, retrieval, and duplicate detection.
"""

import sqlite3
import json
import io
from datetime import datetime
from typing import Optional, Dict, List, Any
import pandas as pd


DATABASE_PATH = "data.db"


def get_connection():
    """Create and return a database connection."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row  # Enable column access by name
    return conn


def init_db():
    """Initialize the database with required tables."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Create dataset_info table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dataset_info (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            file_hash TEXT UNIQUE NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            row_count INTEGER,
            column_count INTEGER,
            columns_json TEXT,
            data_json TEXT
        )
    """)
    
    conn.commit()
    conn.close()
    print("[SUCCESS] Database initialized successfully")


def check_duplicate(file_hash: str) -> bool:
    """
    Check if a dataset with the given hash already exists.
    
    Args:
        file_hash: MD5 hash of the dataset
        
    Returns:
        True if duplicate exists, False otherwise
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT id FROM dataset_info WHERE file_hash = ?", (file_hash,))
    result = cursor.fetchone()
    
    conn.close()
    return result is not None


def store_dataset(filename: str, file_hash: str, df: pd.DataFrame) -> int:
    """
    Store dataset and its metadata in the database.
    
    Args:
        filename: Original filename of the uploaded CSV
        file_hash: MD5 hash of the dataset
        df: Pandas DataFrame containing the data
        
    Returns:
        ID of the stored dataset
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Prepare data for storage
    row_count, column_count = df.shape
    columns_list = df.columns.tolist()
    
    # Convert DataFrame to JSON for storage
    data_json = df.to_json(orient='records')
    columns_json = json.dumps(columns_list)
    
    cursor.execute("""
        INSERT INTO dataset_info (filename, file_hash, row_count, column_count, columns_json, data_json)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (filename, file_hash, row_count, column_count, columns_json, data_json))
    
    dataset_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    print(f"[SUCCESS] Dataset stored successfully with ID: {dataset_id}")
    return dataset_id


def get_latest_data() -> Optional[pd.DataFrame]:
    """
    Retrieve the most recently uploaded dataset.
    
    Returns:
        Pandas DataFrame with the latest data, or None if no data exists
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT data_json 
        FROM dataset_info 
        ORDER BY timestamp DESC 
        LIMIT 1
    """)
    
    result = cursor.fetchone()
    conn.close()
    
    if result is None:
        return None
    
    # Convert JSON back to DataFrame
    data_json = result['data_json']
    df = pd.read_json(io.StringIO(data_json), orient='records')
    
    return df


def get_latest_metadata() -> Optional[Dict[str, Any]]:
    """
    Retrieve metadata of the most recently uploaded dataset.
    
    Returns:
        Dictionary containing dataset metadata
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT filename, file_hash, timestamp, row_count, column_count, columns_json
        FROM dataset_info 
        ORDER BY timestamp DESC 
        LIMIT 1
    """)
    
    result = cursor.fetchone()
    conn.close()
    
    if result is None:
        return None
    
    return {
        'filename': result['filename'],
        'file_hash': result['file_hash'],
        'timestamp': result['timestamp'],
        'row_count': result['row_count'],
        'column_count': result['column_count'],
        'columns': json.loads(result['columns_json'])
    }


def get_all_datasets() -> List[Dict[str, Any]]:
    """
    Retrieve metadata for all stored datasets.
    
    Returns:
        List of dictionaries containing dataset metadata
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, filename, file_hash, timestamp, row_count, column_count
        FROM dataset_info 
        ORDER BY timestamp DESC
    """)
    
    results = cursor.fetchall()
    conn.close()
    
    datasets = []
    for row in results:
        datasets.append({
            'id': row['id'],
            'filename': row['filename'],
            'file_hash': row['file_hash'],
            'timestamp': row['timestamp'],
            'row_count': row['row_count'],
            'column_count': row['column_count']
        })
    
    return datasets


if __name__ == "__main__":
    # Initialize database when running this module directly
    init_db()
