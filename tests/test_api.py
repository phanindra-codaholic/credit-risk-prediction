import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.database import init_db

@pytest.fixture(scope="session", autouse=True)
def setup_database():
    """Ensure database tables and schema are created before tests execute."""
    init_db()

@pytest.fixture
def client():
    """Fixture providing TestClient with active lifespan context."""
    with TestClient(app) as test_client:
        yield test_client


def test_read_root(client):
    """Verify that root endpoint returns HTTP 200 and greeting."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert data["status"] == "running"


def test_health_check(client):
    """Verify that health check reports healthy status with DB and model loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model"] == "loaded"
    assert data["database"] == "connected"


def test_predict_low_risk_applicant(client):
    """Test valid applicant with high income, low loan amount, and prime credit grade."""
    payload = {
        "person_age": 35,
        "person_income": 95000.0,
        "person_home_ownership": "MORTGAGE",
        "person_emp_length": 8.0,
        "loan_intent": "HOMEIMPROVEMENT",
        "loan_grade": "A",
        "loan_amnt": 5000.0,
        "loan_int_rate": 7.5,
        "loan_percent_income": 0.05,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 10
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] == "Low Risk"
    assert "probability" in data
    assert 0.0 <= data["probability"] <= 1.0
    assert data["prediction_id"] is not None


def test_predict_high_risk_applicant(client):
    """Test applicant with very low income, massive loan, high rate, and previous default."""
    payload = {
        "person_age": 22,
        "person_income": 15000.0,
        "person_home_ownership": "RENT",
        "person_emp_length": 0.5,
        "loan_intent": "DEBTCONSOLIDATION",
        "loan_grade": "F",
        "loan_amnt": 25000.0,
        "loan_int_rate": 20.5,
        "loan_percent_income": 0.85,
        "cb_person_default_on_file": "Y",
        "cb_person_cred_hist_length": 2
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] == "High Risk"
    assert data["probability"] > 0.5


def test_predict_invalid_data_validation(client):
    """Test that pydantic catches invalid data types, negative values, and illegal categories."""
    invalid_payload = {
        "person_age": 12,  # Invalid: below minimum age of 18
        "person_income": -5000.0,  # Invalid: negative income
        "person_home_ownership": "INVALID_STATUS",  # Invalid category
        "loan_intent": "VACATION",  # Invalid intent category
        "loan_grade": "Z",  # Invalid grade
        "loan_amnt": -100.0,  # Invalid amount
        "cb_person_default_on_file": "MAYBE",  # Invalid default
        "cb_person_cred_hist_length": -1  # Invalid negative length
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422  # Unprocessable Entity
    data = response.json()
    assert "detail" in data


def test_get_predictions_history(client):
    """Verify that predictions stored in DB can be retrieved via GET /predictions."""
    response = client.get("/predictions?limit=5&offset=0")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    first_record = data[0]
    assert "id" in first_record
    assert "prediction" in first_record
    assert "probability" in first_record


def test_get_prediction_by_id(client):
    """Verify that a specific prediction can be fetched by ID."""
    list_resp = client.get("/predictions?limit=1")
    assert list_resp.status_code == 200
    records = list_resp.json()
    if records:
        pred_id = records[0]["id"]
        response = client.get(f"/predictions/{pred_id}")
        assert response.status_code == 200
        record = response.json()
        assert record["id"] == pred_id


def test_get_prediction_not_found(client):
    """Verify that querying a non-existent ID returns HTTP 404."""
    response = client.get("/predictions/99999999")
    assert response.status_code == 404
    assert "detail" in response.json()


def test_get_model_metrics(client):
    """Verify that model evaluation metrics endpoint returns trained model performance."""
    response = client.get("/model/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "best_model" in data
    assert data["best_model"] == "Random Forest"
    assert "all_model_evaluations" in data
