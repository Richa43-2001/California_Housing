import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
import os

# Load dataset
X, _ = fetch_california_housing(return_X_y=True)
X_sample = X[:5]

print("\n🔍 Sample input shape:", X_sample.shape)

# === Trained Model Predictions ===
if os.path.exists("model.joblib"):
    model = joblib.load("model.joblib")
    preds_trained = model.predict(X_sample)
    print("\n✅ Trained Model Predictions:")
    print(preds_trained)
else:
    print("\n❌ Trained model (model.joblib) not found.")

# === Unquantized Params Predictions ===
if os.path.exists("unquant_params.joblib"):
    unquant = joblib.load("unquant_params.joblib")
    coef_unq = unquant["coef"]
    intercept_unq = unquant["intercept"]
    preds_unquant = X_sample @ coef_unq + intercept_unq
    print("\n✅ Unquantized Params Predictions:")
    print(preds_unquant)
else:
    print("\n❌ unquant_params.joblib not found.")

# === Quantized Params Predictions ===
if os.path.exists("quant_params.joblib"):
    quant = joblib.load("quant_params.joblib")
    coef_q = quant["coef"]
    intercept_q = quant["intercept"]
    preds_quant = X_sample @ coef_q + intercept_q
    print("\n✅ Quantized Params Predictions:")
    print(preds_quant)
else:
    print("\n❌ quant_params.joblib not found.")
