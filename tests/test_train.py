import joblib
import os
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error

def test_debug():
    print("✅ test_debug ran!")
    assert True

def test_model_artifacts_exist():
    assert os.path.exists("model.joblib"), "model.joblib missing"
    assert os.path.exists("unquant_params.joblib"), "unquant_params.joblib missing"
    assert os.path.exists("quant_params.joblib"), "quant_params.joblib missing"

def test_model_prediction_similarity():
    X, y = fetch_california_housing(return_X_y=True)

    model = joblib.load("model.joblib")
    y_pred_model = model.predict(X)

    unq = joblib.load("unquant_params.joblib")
    y_pred_unquant = X @ unq["coef"] + unq["intercept"]

    q = joblib.load("quant_params.joblib")
    y_pred_quant = X @ q["coef"] + q["intercept"]

    mse_diff_u = mean_squared_error(y_pred_model, y_pred_unquant)
    mse_diff_q = mean_squared_error(y_pred_model, y_pred_quant)

    print(f"🔍 MSE diff (unquantized): {mse_diff_u}")
    print(f"🔍 MSE diff (quantized): {mse_diff_q}")

    assert mse_diff_u < 1e-6, "Unquantized params mismatch"
    assert mse_diff_q < 0.01, "Quantized params mismatch"
