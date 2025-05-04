FROM python:3.9-slim

WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY . .

# Expose port
EXPOSE 8002

# Generate model (if not already created)
# RUN python model_generator.py

# Start the service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002", "--reload"]