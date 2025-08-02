import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score

def test_model_exists():
    assert os.path.exists("model.joblib")

def test_model_is_linear_regression():
    model = joblib.load("model.joblib")
    assert isinstance(model, LinearRegression)

def test_model_has_coefficients():
    model = joblib.load("model.joblib")
    assert hasattr(model, "coef_")

def test_r2_score_above_threshold():
    model = joblib.load("model.joblib")
    X, y = fetch_california_housing(return_X_y=True)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    assert r2 > 0.5
