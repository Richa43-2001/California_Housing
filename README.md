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

PS C:\Richa\Assignment\MLOPS\Major_Assignment\California_Housing> python src\train.py       
R2 Score: 0.606232685199805
MSE Loss: 0.5243209861846072
PS C:\Richa\Assignment\MLOPS\Major_Assignment\California_Housing> python tests\test_train.py
PS C:\Richa\Assignment\MLOPS\Major_Assignment\California_Housing> python src\quantize.py    
Loading trained model...
Saved unquantized parameters to 'unquant_params.joblib'
Saved quantized parameters to 'quant_params.joblib'

📊 Model Comparison
----------------------------------------
Model           R2 Score   MSE
----------------------------------------
Unquantized     0.6062     0.5243
Quantized       -30595.9322 40741.3542
----------------------------------------
PS C:\Richa\Assignment\MLOPS\Major_Assignment\California_Housing> python src\predict.py
Original Model Predictions: [4.13164983 3.97660644 3.67657094 3.2415985  2.41358744]
Quantized Model Predictions: [-194.81067352 -196.71077726 -192.11569879 -198.90952408 -198.53438303]
