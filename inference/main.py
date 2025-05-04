import logging
import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import uvicorn
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Inference Service",
              description="Makes predictions using a pre-trained ML model",
              version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, this should be more restrictive
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Post-processing service URL
POSTPROCESSING_URL = os.getenv("POSTPROCESSING_URL", "http://0.0.0.0:8003/postprocess")

# Define data models
class PreprocessedData(BaseModel):
    features: List[float]
    metadata: Optional[Dict[str, Any]] = None
    preprocessing_info: Optional[Dict[str, Any]] = None

class PredictionResponse(BaseModel):
    prediction: List[float]
    prediction_probabilities: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None
    preprocessing_info: Optional[Dict[str, Any]] = None

# Load the ML model
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")

try:
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        logger.warning(f"Model file {MODEL_PATH} not found. Will attempt to generate it.")
        # Import and run the model generator
        import model_generator
        model_generator.generate_model(MODEL_PATH)
        logger.info(f"Model generated and saved to {MODEL_PATH}")
    
    # Load the model
    model = joblib.load(MODEL_PATH)
    logger.info(f"Successfully loaded model from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

@app.get("/")
def read_root():
    return {"message": "Inference Service is running"}

@app.get("/health")
def health_check():
    if model is not None:
        return {"status": "healthy", "model_loaded": True}
    return {"status": "unhealthy", "model_loaded": False}

@app.post("/predict")
async def predict(data: PreprocessedData):
    """
    Makes predictions using the loaded ML model
    """
    logger.info(f"Received data for prediction")
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert features to numpy array
        features = np.array(data.features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)
        
        # Get prediction probabilities if the model supports it
        prediction_probabilities = None
        if hasattr(model, 'predict_proba'):
            try:
                prediction_probabilities = model.predict_proba(features).tolist()[0]
            except Exception as e:
                logger.warning(f"Could not get prediction probabilities: {str(e)}")
        
        # Create prediction response
        prediction_response = PredictionResponse(
            prediction=prediction.tolist(),
            prediction_probabilities=prediction_probabilities,
            metadata=data.metadata,
            preprocessing_info=data.preprocessing_info
        )
        
        # Forward to post-processing service
        try:
            response = requests.post(POSTPROCESSING_URL, json=prediction_response.dict())
            
            if response.status_code != 200:
                logger.error(f"Post-processing service error: {response.text}")
                # Even if post-processing fails, return the prediction
                return prediction_response.dict()
            
            postprocessed_result = response.json()
            logger.info(f"Post-processing completed successfully")
            
            return postprocessed_result
            
        except requests.RequestException as e:
            logger.error(f"Error connecting to post-processing service: {str(e)}")
            # Return prediction without post-processing
            return prediction_response.dict()
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    # Run the application
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)
