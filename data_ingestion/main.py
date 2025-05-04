import logging
import os
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
app = FastAPI(title="Data Ingestion Service",
              description="Receives raw data and forwards it to the preprocessing service",
              version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, this should be more restrictive
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Preprocessing service URL
PREPROCESSING_URL = os.getenv("PREPROCESSING_URL", "http://0.0.0.0:8001/preprocess")

# Define data models
class FeatureData(BaseModel):
    features: List[float]
    metadata: Optional[Dict[str, Any]] = None

@app.get("/")
def read_root():
    return {"message": "Data Ingestion Service is running"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/ingest")
async def ingest_data(data: FeatureData):
    """
    Ingests data and forwards it to the preprocessing service
    """
    logger.info(f"Received data for ingestion: {data}")
    
    try:
        # Forward the data to the preprocessing service
        response = requests.post(PREPROCESSING_URL, json=data.dict())
        
        if response.status_code != 200:
            logger.error(f"Preprocessing service error: {response.text}")
            raise HTTPException(status_code=response.status_code, 
                               detail=f"Preprocessing service error: {response.text}")
        
        preprocessed_data = response.json()
        logger.info(f"Data successfully ingested and preprocessed")
        
        return {
            "status": "success",
            "message": "Data ingested and preprocessed successfully",
            "data": preprocessed_data
        }
        
    except requests.RequestException as e:
        logger.error(f"Error connecting to preprocessing service: {str(e)}")
        raise HTTPException(status_code=503, 
                           detail=f"Error connecting to preprocessing service: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    # Run the application
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
