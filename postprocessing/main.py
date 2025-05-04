import logging
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Postprocessing Service",
              description="Processes prediction results and returns final output",
              version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, this should be more restrictive
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define data models
class PredictionData(BaseModel):
    prediction: List[float]
    prediction_probabilities: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None
    preprocessing_info: Optional[Dict[str, Any]] = None

class ProcessedResultData(BaseModel):
    prediction: List[float]
    prediction_probabilities: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None
    preprocessing_info: Optional[Dict[str, Any]] = None
    postprocessing_info: Dict[str, Any]
    timestamp: str

@app.get("/")
def read_root():
    return {"message": "Postprocessing Service is running"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/postprocess")
async def postprocess_prediction(data: PredictionData):
    """
    Processes prediction results and applies business logic
    """
    logger.info(f"Received prediction for postprocessing")
    
    try:
        # Get the prediction
        prediction = np.array(data.prediction)
        
        # Apply post-processing operations
        # 1. Apply any business rules or transformations
        # For example, ensure predictions are in a valid range for your application
        processed_prediction = np.clip(prediction, 0, 100)  # Clip between 0 and 100
        
        # 2. Round to specified decimal places if needed
        processed_prediction = np.round(processed_prediction, 2)
        
        # 3. Add confidence metrics if available
        confidence = None
        if data.prediction_probabilities:
            confidence = np.max(data.prediction_probabilities)
        
        # Create processed result object
        current_time = datetime.now().isoformat()
        processed_result = ProcessedResultData(
            prediction=processed_prediction.tolist(),
            prediction_probabilities=data.prediction_probabilities,
            metadata=data.metadata,
            preprocessing_info=data.preprocessing_info,
            postprocessing_info={
                "confidence": confidence,
                "modified": bool(np.any(processed_prediction != prediction)),
                "original_range": {
                    "min": float(np.min(prediction)),
                    "max": float(np.max(prediction))
                }
            },
            timestamp=current_time
        )
        
        logger.info(f"Postprocessing completed successfully")
        return processed_result.dict()
        
    except Exception as e:
        logger.error(f"Error during postprocessing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Postprocessing error: {str(e)}")

if __name__ == "__main__":
    # Run the application
    uvicorn.run("main:app", host="0.0.0.0", port=8003, reload=True)
