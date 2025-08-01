# ML OPS Major Assignment
Problem Statement : 
Build a complete MLOps pipeline focused on Linear Regression only, integrating training, testing,
quantization, Dockerization, and CI/CD — all managed within a single main branch.

Dataset description :
• Source: sklearn.datasets.fetch_california_housing
• Target: Predict housing prices using features like median income, average rooms, population, etc.

Model
• Algorithm: LinearRegression from scikit-learn

Evaluation Metrics:

• R² Score

• Mean Squared Error (MSE)

Workflow Summary:

1. Training
• Loads California dataset
• Trains Linear Regression model
• Saves model using joblib

2. Testing
• Unit tests check:
• Dataset loading
• Model type and training success
• R² score threshold

3. Quantization
• Extracts coefficients and intercept
• Quantizes to uint8
• Performs manual dequantized inference
• Saves both raw and quantized parameters

4. Prediction
• Loads model
• Runs prediction on sample test data

5. Dockerization
• Dockerfile installs dependencies and runs prediction script

6. CI/CD
• GitHub Actions pipeline includes:
• Unit Testing
• Model Training & Quantization
• Docker Image Build & Test Run

Comparison Table:

Version	R² Score	MSE Loss	Notes

Unquantized	~0.60	~0.53	Raw model from training

Quantized	~0.58	Slight ↑	Approx. due to rounding

