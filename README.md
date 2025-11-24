# Project-11-ML-Data-Query-Training-Dashboard

A comprehensive ML system with FastAPI backend and Streamlit frontend for data upload, model training, and intelligent querying.

## Features

- **CSV File Upload**: Upload CSV files with automatic duplicate detection
- **Data Storage**: SQLite database for efficient data management
- **Model Training**: Train machine learning models on uploaded data
- **Chatbot Interface**: Query data and make predictions through an interactive chatbot
- **RESTful API**: FastAPI backend with comprehensive endpoints

## Tech Stack

- **Backend**: FastAPI, SQLite
- **Frontend**: Streamlit
- **Machine Learning**: scikit-learn, pandas, numpy
- **Database**: SQLite

## Project Structure

```
project/
├── backend/          # FastAPI backend
├── frontend/         # Streamlit frontend
├── sample_data.csv   # Sample dataset
└── README.md         # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/dawoodshah515/-ML-Data-Query-Training-Dashboard.git
cd -ML-Data-Query-Training-Dashboard
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the backend:
```bash
cd backend
uvicorn main:app --reload
```

4. Run the frontend (in a new terminal):
```bash
cd frontend
streamlit run app.py
```

## Usage

1. Upload CSV files through the Streamlit interface
2. Train models on your uploaded data
3. Use the chatbot to query data and make predictions

## License

MIT License
