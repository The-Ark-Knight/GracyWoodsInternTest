import pytest
import json
from app import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_home_page(client):
    """Test the home page."""
    response = client.get('/')
    assert response.status_code == 200
    assert b'<!DOCTYPE html>' in response.data  # Assuming your index.html contains this tag

def test_predict_success(client):
    """Test the predict endpoint with valid input."""
    test_data = {
        "age": 30,
        "workclass": "Private",
        "fnlwgt": 200000.0,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Tech-support",
        "relationship": "Not-in-family",
        "race": "White",
        "capital-gain": 0.0,
        "capital-loss": 0.0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    response = client.post('/predict', data=json.dumps(test_data), content_type='application/json')
    assert response.status_code == 200
    response_json = json.loads(response.data)
    assert 'prediction' in response_json
    assert isinstance(response_json['prediction'], (int, float))  # Assuming your model predicts numeric values

def test_predict_missing_field(client):
    """Test the predict endpoint with missing fields."""
    test_data = {
        "age": 30,
        "workclass": "Private",
        "fnlwgt": 200000.0,
        # Missing 'education'
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Tech-support",
        "relationship": "Not-in-family",
        "race": "White",
        "capital-gain": 0.0,
        "capital-loss": 0.0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    response = client.post('/predict', data=json.dumps(test_data), content_type='application/json')
    assert response.status_code == 400  # Assuming your app returns a 400 error for bad input
    response_json = json.loads(response.data)
    assert 'error' in response_json

def test_predict_invalid_data(client):
    """Test the predict endpoint with invalid data."""
    test_data = {
        "age": "thirty",
        "workclass": "Private",
        "fnlwgt": "two hundred thousand",
        "education": "Bachelors",
        "education-num": "thirteen",
        "marital-status": "Never-married",
        "occupation": "Tech-support",
        "relationship": "Not-in-family",
        "race": "White",
        "capital-gain": "none",
        "capital-loss": "none",
        "hours-per-week": "forty",
        "native-country": "United-States"
    }
    response = client.post('/predict', data=json.dumps(test_data), content_type='application/json')
    assert response.status_code == 400  # Assuming your app returns a 400 error for invalid data
    response_json = json.loads(response.data)
    assert 'error' in response_json
