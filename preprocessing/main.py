import logging
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import uvicorn
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Preprocessing Service",
              description="Preprocesses data before sending to inference service",
              version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, this should be more restrictive
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inference service URL
INFERENCE_URL = os.getenv("INFERENCE_URL", "http://0.0.0.0:8002/predict")

# Define data models
class FeatureData(BaseModel):
    features: List[float]
    metadata: Optional[Dict[str, Any]] = None

class PreprocessedData(BaseModel):
    features: List[float]
    metadata: Optional[Dict[str, Any]] = None
    preprocessing_info: Dict[str, Any]

@app.get("/")
def read_root():
    return {"message": "Preprocessing Service is running"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/preprocess")
async def preprocess_data(data: FeatureData):
    """
    Preprocesses the input data before sending to the inference service
    """
    logger.info(f"Received data for preprocessing: {data}")
    
    try:
        # Get the features as numpy array
        features = np.array(data.features)
        
        # Perform preprocessing operations
        # 1. Check for missing values
        if np.isnan(features).any():
            logger.warning("Missing values detected in features")
            # Replace NaN with mean of the feature
            features = np.nan_to_num(features, nan=np.nanmean(features))
        
        # 2. Standardize the features (mean=0, std=1)
        mean = np.mean(features)
        std = np.std(features)
        if std > 0:
            features = (features - mean) / std
        
        # 3. Clip extreme values
        features = np.clip(features, -5, 5)
        
        # Create preprocessed data object
        preprocessed_data = PreprocessedData(
            features=features.tolist(),
            metadata=data.metadata,
            preprocessing_info={
                "mean": float(mean),
                "std": float(std),
                "replaced_missing": bool(np.isnan(np.array(data.features)).any())
            }
        )
        
        # Forward preprocessed data to inference service
        try:
            response = requests.post(INFERENCE_URL, json=preprocessed_data.dict())
            
            if response.status_code != 200:
                logger.error(f"Inference service error: {response.text}")
                raise HTTPException(status_code=response.status_code, 
                                  detail=f"Inference service error: {response.text}")
            
            inference_result = response.json()
            logger.info(f"Inference completed successfully")
            
            return inference_result
            
        except requests.RequestException as e:
            logger.error(f"Error connecting to inference service: {str(e)}")
            # In case of connection error, still return the preprocessed data
            return preprocessed_data.dict()
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {str(e)}")

if __name__ == "__main__":
    # Run the application
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
