import joblib
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing

def test_dataset_loaded():
    data = fetch_california_housing()
    assert data.data.shape[0] > 0

def test_model_type():
    model = joblib.load("model.joblib")
    assert isinstance(model, LinearRegression)

def test_model_trained():
    model = joblib.load("model.joblib")
    assert hasattr(model, "coef_")

def test_r2_score():
    data = fetch_california_housing()
    X, y = data.data, data.target
    model = joblib.load("model.joblib")
    y_pred = model.predict(X)
    from sklearn.metrics import r2_score
    assert r2_score(y, y_pred) > 0.5
