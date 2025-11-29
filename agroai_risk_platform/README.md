# AgroAI Risk Platform

Unified FastAPI service combining credit scoring and fraud detection with a combined risk decision.

## Features
- `/scoring/predict` — credit default probability using the AgroAIScoring model.
- Fraud module (hybrid): synthetic pro-grade dataset (20k rows), feature engineering, supervised + anomaly blend:
  - `/fraud/predict` — returns `combined_fraud_score`, `supervised_score`, `anomaly_score`.
  - `/fraud/explain` — same scores + rule-based explanation.
  - `/fraud/retrain` — regenerates data, retrains, saves artifacts.
- `/risk/combined` — merges scoring + combined fraud score with business rules to approve / manual_review / decline.

## Repository structure
```
app/
  main_api.py           # FastAPI entrypoint
  shared/               # shared config and utils
  scoring/              # credit scoring router
  fraud/                # fraud router + auto-trained artifacts
model/                  # credit scoring artifacts (model.pkl, preprocessor.pkl)
model_training/         # credit scoring training script + helpers
data/                   # credit scoring dataset (synthetic)
tests/                  # endpoint smoke tests
requirements.txt
```

## Install
```bash
cd agroai_risk_platform
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the API
```bash
uvicorn app.main_api:app --reload
```
Swagger UI: http://localhost:8000/docs  
Redoc: http://localhost:8000/redoc

## Example requests
Credit scoring:
```bash
curl -X POST http://localhost:8000/scoring/predict \
  -H "Content-Type: application/json" \
  -d '{"age":35,"monthly_income":6000,"loan_amount":15000,"loan_term_months":36,"past_due":"no","employment_type":"employed","has_mortgage":"no","num_prev_loans":1,"credit_utilization":0.3}'
```

Fraud detection:
```bash
curl -X POST http://localhost:8000/fraud/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "T1",
    "customer_id": "C10",
    "amount": 220,
    "currency": "USD",
    "timestamp": "2023-02-01T23:15:00",
    "country": "US",
    "merchant_id": "M50",
    "merchant_category": "electronics",
    "channel": "ONLINE",
    "device_id": "D10",
    "is_new_country": 0,
    "is_new_device": 0,
    "same_day_transactions_count": 5,
    "average_customer_spend": 180
  }'
```
Explanation:
```bash
curl -X POST http://localhost:8000/fraud/explain \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "T2",
    "customer_id": "C22",
    "amount": 950,
    "currency": "USD",
    "timestamp": "2023-02-02T01:10:00",
    "country": "RU",
    "merchant_id": "M99",
    "merchant_category": "luxury",
    "channel": "ONLINE",
    "device_id": "D77",
    "is_new_country": 1,
    "is_new_device": 1,
    "same_day_transactions_count": 12,
    "average_customer_spend": 120
  }'
```

Combined risk:
```bash
curl -X POST http://localhost:8000/risk/combined \
  -H "Content-Type: application/json" \
  -d '{
    "credit_data": {"age":35,"monthly_income":6000,"loan_amount":15000,"loan_term_months":36,"past_due":"no","employment_type":"employed","has_mortgage":"no","num_prev_loans":1,"credit_utilization":0.3},
    "fraud_data": {
      "transaction_id": "T3",
      "customer_id": "C33",
      "amount": 300,
      "currency": "USD",
      "timestamp": "2023-02-02T10:00:00",
      "country": "GB",
      "merchant_id": "M150",
      "merchant_category": "grocery",
      "channel": "POS",
      "device_id": "D5",
      "is_new_country": 0,
      "is_new_device": 0,
      "same_day_transactions_count": 3,
      "average_customer_spend": 250
    }
  }'
```

## Decision rules (combined)
- if `fraud_score > 0.8` → decline
- else if `credit_risk_score > 0.7` → decline
- else if `0.4 < fraud_score ≤ 0.8` and `credit_risk_score > 0.5` → manual_review
- otherwise → approve

## Models
- Credit scoring artifacts (`model/model.pkl`, `model/preprocessor.pkl`) are copied from AgroAIScoring.
- Fraud module: hybrid supervised + anomaly (GradientBoosting + IsolationForest). Full preprocessing (OHE + scaling + risk features) saved to `app/fraud/fraud_preprocessor.pkl`; models + anomaly stats saved to `app/fraud/fraud_model.pkl`. If missing, they are auto-trained on a 20k synthetic dataset with realistic fraud patterns.

## Retraining
- Credit scoring: from the project root run `python -m model_training.train_model`. Artifacts will be regenerated in `model/` and the dataset is stored in `data/credit_data.csv`.
- Fraud: run `python app/fraud/train_fraud_model.py` (or call `/fraud/retrain`). Artifacts will be regenerated.

## Hybrid fraud model
- Dataset: 20k synthetic transactions with realistic signals (amount spikes vs. customer avg, new country/device, velocity, high-risk merchants, night activity).
- Features: ratio-to-avg, velocity, weekend/night flags, country/merchant/device risk scores, one-hot for categorical fields.
- Models: GradientBoosting classifier + IsolationForest anomaly detector. Combined score: `0.6 * supervised + 0.4 * anomaly`.
- Explanations: rule-based reasons (amount x above avg, new country + high amount + night, velocity spikes, new device, high-risk merchant/country).

## Known limitations
- Synthetic data; real-world calibration would require production data and monitoring.
- Country/merchant risk maps are static at train time; unseen entities default to global rate.
- Thresholds for rules and combined scoring are illustrative and may need tuning for real deployments.

## Tests
```bash
pytest tests
```
