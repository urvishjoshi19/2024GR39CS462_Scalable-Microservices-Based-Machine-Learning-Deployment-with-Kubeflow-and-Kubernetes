import requests
import json
import random
import time
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Client for ML Inference Microservices')
parser.add_argument('--ingest-url', type=str, default='http://localhost:8000/ingest',
                    help='URL for the data ingestion service (default: http://localhost:8000/ingest)')
parser.add_argument('--num-requests', type=int, default=5,
                    help='Number of requests to send (default: 5)')
parser.add_argument('--delay', type=float, default=1.0,
                    help='Delay between requests in seconds (default: 1.0)')
args = parser.parse_args()

def generate_sample_data():
    """
    Generate random sample data for inference
    """
    # Generate 4 random features
    features = [random.uniform(0, 1) for _ in range(4)]
    
    # Add some metadata
    metadata = {
        "source": "client_demo",
        "client_timestamp": time.time(),
        "client_id": f"demo-client-{random.randint(1000, 9999)}"
    }
    
    return {
        "features": features,
        "metadata": metadata
    }

def send_inference_request(data, url=args.ingest_url):
    """
    Send data to the inference pipeline through the ingestion service
    """
    try:
        logger.info(f"Sending request to {url}")
        response = requests.post(url, json=data)
        
        if response.status_code == 200:
            logger.info("Request successful")
            return response.json()
        else:
            logger.error(f"Request failed with status code {response.status_code}: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error sending request: {str(e)}")
        return None

def main():
    logger.info(f"Starting client demo, sending {args.num_requests} requests to {args.ingest_url}")
    
    for i in range(args.num_requests):
        logger.info(f"Request {i+1}/{args.num_requests}")
        
        # Generate sample data
        data = generate_sample_data()
        logger.info(f"Generated sample data: {json.dumps(data, indent=2)}")
        
        # Send request
        result = send_inference_request(data)
        
        if result:
            logger.info(f"Received result: {json.dumps(result, indent=2)}")
        
        # Wait before sending the next request
        if i < args.num_requests - 1:
            time.sleep(args.delay)
    
    logger.info("Client demo completed")

if __name__ == "__main__":
    main()
