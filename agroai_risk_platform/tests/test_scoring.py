from fastapi.testclient import TestClient

from app.main_api import app

client = TestClient(app)


def test_scoring_predict():
    payload = {
        "age": 35,
        "monthly_income": 6000,
        "loan_amount": 15000,
        "loan_term_months": 36,
        "past_due": "no",
        "employment_type": "employed",
        "has_mortgage": "no",
        "num_prev_loans": 1,
        "credit_utilization": 0.3,
    }
    resp = client.post("/scoring/predict", json=payload)
    assert resp.status_code == 200
    score = resp.json()["credit_risk_score"]
    assert 0 <= score <= 1
