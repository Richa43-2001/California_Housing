FROM python:3.9-slim

WORKDIR /app

# Copy dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src

# Copy model artifacts (assumes you're building after training & quantization)
COPY model.joblib .
COPY quant_params.joblib .

# Default command
CMD ["python", "src/predict.py"]
