from fastapi.testclient import TestClient

from app.main_api import app

client = TestClient(app)


def test_combined_endpoint():
    credit = {
        "age": 40,
        "monthly_income": 7000,
        "loan_amount": 12000,
        "loan_term_months": 24,
        "past_due": "no",
        "employment_type": "employed",
        "has_mortgage": "yes",
        "num_prev_loans": 2,
        "credit_utilization": 0.4,
    }
    fraud = {
        "amount": 300,
        "avg_amount_client": 250,
        "tx_hour": 10,
        "tx_count_last_24h": 3,
        "country": "GB",
        "device_trust_score": 0.7,
    }
    resp = client.post("/risk/combined", json={"credit_data": credit, "fraud_data": fraud})
    assert resp.status_code == 200
    data = resp.json()
    assert "credit_risk_score" in data and "fraud_risk_score" in data and "final_decision" in data
