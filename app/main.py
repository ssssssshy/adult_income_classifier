from typing import Dict, List, Optional, Union
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, conlist, Field
import numpy as np
import joblib
import os
import pandas as pd
import shap
from typing import List, Optional
import requests
from io import StringIO

# Initialize FastAPI app with metadata
app = FastAPI(
    title="Adult Income Classifier",
    description="""
    API for predicting adult income using machine learning models with SHAP interpretation.
    
    Features:
    - Multiple model support (LightGBM, XGBoost, Neural Network)
    - Model interpretation using SHAP
    - Data loading from external sources
    - Comprehensive prediction results
    - Interactive web interface
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

class InputData(BaseModel):
    """Input data model for prediction endpoint."""
    features: conlist(float, min_length=14, max_length=14) = Field(
        ...,
        description="List of 14 features in the following order: age, workclass, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country, income"
    )

    class Config:
        schema_extra = {
            "example": {
                "features": [39, 1, 2, 13, 1, 2, 1, 1, 1, 2174, 0, 40, 1, 0]
            }
        }

class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    prediction: int = Field(..., description="Predicted class (0: <=50K, 1: >50K)")
    probability: float = Field(..., description="Probability of positive class (>50K)")
    feature_importance: Dict[str, float] = Field(..., description="SHAP feature importance values")

    class Config:
        schema_extra = {
            "example": {
                "prediction": 1,
                "probability": 0.85,
                "feature_importance": {
                    "age": 0.15,
                    "education": 0.25,
                    "hours-per-week": 0.20
                }
            }
        }

class DataLoadResponse(BaseModel):
    """Response model for data loading endpoint."""
    message: str = Field(..., description="Status message")
    rows: int = Field(..., description="Number of rows in loaded data")
    columns: List[str] = Field(..., description="List of column names")

# Load model
MODEL_PATH: str = "models/lightgbm_model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"LightGBM model not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# Feature names for interpretation
FEATURE_NAMES: List[str] = [
    "age", "workclass", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain",
    "capital-loss", "hours-per-week", "native-country", "income"
]

@app.get("/", response_class=HTMLResponse, tags=["Web Interface"])
async def root(request: Request):
    """Serve the main web interface."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(data: InputData) -> PredictionResponse:
    """
    Make prediction using the LightGBM model and return SHAP values.
    
    Args:
        data: InputData object containing feature values
        
    Returns:
        PredictionResponse object containing:
        - prediction: Binary classification result
        - probability: Probability of positive class
        - feature_importance: SHAP values for feature importance
        
    Raises:
        HTTPException: If prediction fails
    """
    try:
        # Convert input to numpy array
        X: np.ndarray = np.array(data.features).reshape(1, -1)
        
        # Make prediction
        pred_proba: np.ndarray = model.predict_proba(X)[0]
        pred_label: int = int(model.predict(X)[0])
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values: np.ndarray = explainer.shap_values(X)[0]
        
        # Create feature importance dictionary
        feature_importance: Dict[str, float] = dict(zip(FEATURE_NAMES, abs(shap_values)))
        
        return PredictionResponse(
            prediction=pred_label,
            probability=float(pred_proba[1]),
            feature_importance=feature_importance
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/load_data", response_model=DataLoadResponse, tags=["Data"])
async def load_data(source: str = "default") -> DataLoadResponse:
    """
    Load data from different sources (CSV from URL or local file).
    
    Args:
        source: Data source URL or "default" for local file
        
    Returns:
        DataLoadResponse object containing:
        - message: Status message
        - rows: Number of rows in loaded data
        - columns: List of column names
        
    Raises:
        HTTPException: If data loading fails
    """
    try:
        if source == "default":
            # Load from local data directory
            data_path: str = "data/adult.csv"
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found at {data_path}")
            df: pd.DataFrame = pd.read_csv(data_path)
        elif source.startswith("http"):
            # Load from URL
            response = requests.get(source)
            response.raise_for_status()
            df = pd.read_csv(StringIO(response.text))
        else:
            raise ValueError("Invalid data source")
            
        return DataLoadResponse(
            message="Data loaded successfully",
            rows=len(df),
            columns=list(df.columns)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 