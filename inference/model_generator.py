import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_model(model_path="model.pkl"):
    """
    Generates a simple RandomForest model and saves it as a pickle file
    
    Args:
        model_path (str): Path where to save the model
    """
    logger.info("Generating a new ML model...")
    
    # Generate synthetic data for training
    np.random.seed(42)
    X = np.random.rand(100, 4)  # 100 samples, 4 features
    y = 3*X[:, 0] - 2*X[:, 1] + 0.5*X[:, 2] + np.random.normal(0, 0.1, 100)  # Linear relationship with noise
    
    # Train a Random Forest model
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)
    
    # Save the model
    joblib.dump(model, model_path)
    logger.info(f"Model successfully generated and saved to {model_path}")
    
    return model

if __name__ == "__main__":
    generate_model()
