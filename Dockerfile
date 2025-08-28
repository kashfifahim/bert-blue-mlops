FROM tensorflow/tensorflow:2.11.0-gpu

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY config.py .

# Copy models
COPY models/ models/

# Expose port for API
EXPOSE 8501

# Create an API endpoint for inference
COPY api.py .
CMD ["python", "api.py"]