from fastapi.testclient import TestClient

from app.main_api import app

client = TestClient(app)


def sample_fraud_payload():
    return {
        "amount": 220,
        "avg_amount_client": 180,
        "tx_hour": 23,
        "tx_count_last_24h": 5,
        "country": "US",
        "device_trust_score": 0.6,
    }


def test_fraud_predict():
    resp = client.post("/fraud/predict", json=sample_fraud_payload())
    assert resp.status_code == 200
    score = resp.json()["fraud_score"]
    assert 0 <= score <= 1


def test_fraud_explain():
    resp = client.post("/fraud/explain", json=sample_fraud_payload())
    assert resp.status_code == 200
    data = resp.json()
    assert "fraud_score" in data and "explanation" in data
