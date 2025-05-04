import os
import logging
import subprocess
import signal
import sys
import time
from flask import Flask, request, jsonify, render_template, Response
import requests
import json

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("SESSION_SECRET", "ml-inference-system-secret")

# Service endpoints
SERVICE_INFO = {
    "data_ingestion": {
        "url": "http://0.0.0.0:8000",
        "health_endpoint": "/health",
        "description": "Receives data and forwards it to preprocessing"
    },
    "preprocessing": {
        "url": "http://0.0.0.0:8001",
        "health_endpoint": "/health",
        "description": "Prepares data for inference"
    },
    "inference": {
        "url": "http://0.0.0.0:8002",
        "health_endpoint": "/health",
        "description": "Makes predictions using the ML model"
    },
    "postprocessing": {
        "url": "http://0.0.0.0:8003",
        "health_endpoint": "/health",
        "description": "Processes and formats prediction results"
    }
}

# Microservice processes
service_processes = {}

def start_services():
    """Start all microservices as background processes"""
    try:
        # Data Ingestion Service
        logger.info("Starting Data Ingestion Service...")
        service_processes["data_ingestion"] = subprocess.Popen(
            ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"],
            cwd="data_ingestion"
        )
        
        # Preprocessing Service
        logger.info("Starting Preprocessing Service...")
        service_processes["preprocessing"] = subprocess.Popen(
            ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"],
            cwd="preprocessing"
        )
        
        # Inference Service
        logger.info("Starting Inference Service...")
        service_processes["inference"] = subprocess.Popen(
            ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"],
            cwd="inference"
        )
        
        # Postprocessing Service
        logger.info("Starting Postprocessing Service...")
        service_processes["postprocessing"] = subprocess.Popen(
            ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"],
            cwd="postprocessing"
        )
        
        # Give services time to start
        logger.info("Waiting for services to start...")
        time.sleep(5)
        
    except Exception as e:
        logger.error(f"Error starting services: {str(e)}")

def stop_services():
    """Stop all microservices"""
    for service_name, process in service_processes.items():
        logger.info(f"Stopping {service_name} service...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning(f"{service_name} service did not terminate gracefully, killing...")
            process.kill()

def check_service_health(service_name):
    """Check the health of a microservice"""
    service = SERVICE_INFO.get(service_name)
    if not service:
        return {"status": "unknown", "error": "Service not found"}
    
    try:
        health_url = f"{service['url']}{service['health_endpoint']}"
        response = requests.get(health_url, timeout=2)
        
        if response.status_code == 200:
            return {"status": "healthy", "details": response.json()}
        else:
            return {"status": "unhealthy", "error": f"Received status code {response.status_code}"}
            
    except requests.RequestException as e:
        return {"status": "unhealthy", "error": str(e)}

@app.route("/")
def get_dashboard():
    """Render the dashboard page"""
    return render_template("index.html")

@app.route("/api/services")
def get_services():
    """Get information about all services"""
    services_status = {}
    
    for service_name in SERVICE_INFO:
        services_status[service_name] = {
            "info": SERVICE_INFO[service_name],
            "health": check_service_health(service_name)
        }
    
    return jsonify(services_status)

@app.route("/api/test-inference", methods=["POST"])
def test_inference():
    """Test the inference pipeline with sample data"""
    try:
        # Try to get features from request
        data = request.get_json()
        features = data if data else None
    except:
        features = None
        
    if features is None or not isinstance(features, list):
        # Generate random features if none provided
        import random
        features = [random.uniform(0, 1) for _ in range(4)]
    
    data = {
        "features": features,
        "metadata": {
            "source": "dashboard",
            "timestamp": time.time()
        }
    }
    
    try:
        response = requests.post(f"{SERVICE_INFO['data_ingestion']['url']}/ingest", json=data, timeout=5)
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({"error": f"Error from ingestion service: {response.text}"}), response.status_code
            
    except requests.RequestException as e:
        return jsonify({"error": f"Error connecting to ingestion service: {str(e)}"}), 503

def signal_handler(sig, frame):
    """Handle Ctrl+C to gracefully shut down services"""
    logger.info("Shutting down services...")
    stop_services()
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    # Start the microservices
    start_services()
    
    # Run the application on port 5000
    app.run(host="0.0.0.0", port=5000, debug=True)
