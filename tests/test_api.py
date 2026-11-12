import pytest
from fastapi.testclient import TestClient

from app.database import init_db
from app.main import app


@pytest.fixture(scope="session", autouse=True)
def setup_database():
    init_db()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_read_root(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert "version" in data


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model"] == "loaded"
    assert data["database"] == "connected"


def test_predict_low_risk_applicant(client):
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
        "cb_person_cred_hist_length": 10,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "Low Risk"
    assert 0.0 <= data["probability"] <= 1.0
    assert data["prediction_id"] is not None


def test_predict_high_risk_applicant(client):
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
        "cb_person_cred_hist_length": 2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "High Risk"
    assert data["probability"] > 0.5


def test_predict_invalid_data_validation(client):
    invalid_payload = {
        "person_age": 12,
        "person_income": -5000.0,
        "person_home_ownership": "INVALID_STATUS",
        "loan_intent": "VACATION",
        "loan_grade": "Z",
        "loan_amnt": -100.0,
        "cb_person_default_on_file": "MAYBE",
        "cb_person_cred_hist_length": -1,
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
    assert "detail" in response.json()


def test_get_predictions_history(client):
    response = client.get("/predictions?limit=5&offset=0")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "id" in data[0]
    assert "prediction" in data[0]
    assert "probability" in data[0]


def test_get_prediction_by_id(client):
    records = client.get("/predictions?limit=1").json()
    if records:
        pred_id = records[0]["id"]
        response = client.get(f"/predictions/{pred_id}")
        assert response.status_code == 200
        assert response.json()["id"] == pred_id


def test_get_prediction_not_found(client):
    response = client.get("/predictions/99999999")
    assert response.status_code == 404
    assert "detail" in response.json()


def test_get_model_metrics(client):
    response = client.get("/model/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["best_model"] == "Random Forest"
    assert "all_model_evaluations" in data


def test_filter_predictions_sql(client):
    resp_high = client.get("/predictions?prediction=High%20Risk&limit=5")
    assert resp_high.status_code == 200
    for record in resp_high.json():
        assert record["prediction"] == "High Risk"

    resp_grade = client.get("/predictions?loan_grade=A&limit=5")
    assert resp_grade.status_code == 200
    for record in resp_grade.json():
        assert record["loan_grade"] == "A"


def test_get_predictions_stats_sql(client):
    response = client.get("/predictions/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["total_predictions"] >= 2
    assert data["high_risk_count"] + data["low_risk_count"] == data["total_predictions"]
    assert 0.0 <= data["avg_probability"] <= 1.0
    assert isinstance(data["by_loan_grade"], list)
    assert isinstance(data["by_loan_intent"], list)


def test_update_and_delete_prediction_sql(client):
    payload = {
        "person_age": 28,
        "person_income": 60000.0,
        "person_home_ownership": "RENT",
        "person_emp_length": 3.0,
        "loan_intent": "EDUCATION",
        "loan_grade": "B",
        "loan_amnt": 7000.0,
        "loan_int_rate": 10.5,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 4,
    }
    create_resp = client.post("/predict", json=payload)
    assert create_resp.status_code == 200
    pred_id = create_resp.json()["prediction_id"]
    assert pred_id is not None

    patch_resp = client.patch(
        f"/predictions/{pred_id}",
        json={"review_status": "APPROVED", "reviewer_notes": "Income verified."},
    )
    assert patch_resp.status_code == 200
    updated = patch_resp.json()
    assert updated["review_status"] == "APPROVED"
    assert updated["reviewer_notes"] == "Income verified."

    assert client.delete(f"/predictions/{pred_id}").status_code == 200
    assert client.get(f"/predictions/{pred_id}").status_code == 404
    assert client.patch("/predictions/99999999", json={"review_status": "REJECTED"}).status_code == 404
    assert client.delete("/predictions/99999999").status_code == 404
