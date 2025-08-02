# Use lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src

# Copy model artifacts (ensure these are created before docker build)
COPY model.joblib .
COPY unquant_params.joblib .
COPY quant_params.joblib .

# Run prediction (for all 3 model versions)
CMD ["python", "src/predict.py"]
