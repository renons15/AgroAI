import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from app.scoring.scoring_api import router as scoring_router, CreditInput, predict_score as predict_credit
from app.fraud.fraud_api import router as fraud_router, FraudRequest, _compute_scores
from app.shared import config

app = FastAPI(title="AgroAI Risk Platform", version="1.0.0")

app.include_router(scoring_router)
app.include_router(fraud_router)


@app.post("/risk/combined")
def combined_risk(credit_data: CreditInput, fraud_data: FraudRequest):
    credit_risk_score = predict_credit(credit_data)
    fraud_scores = _compute_scores(pd.DataFrame([fraud_data.dict()]))
    fraud_risk_score = fraud_scores["combined_fraud_score"]

    if fraud_risk_score > config.FRAUD_DECLINE_THRESHOLD:
        final_decision = "decline"
    elif credit_risk_score > config.SCORING_DECLINE_THRESHOLD:
        final_decision = "decline"
    elif config.FRAUD_MANUAL_THRESHOLD < fraud_risk_score <= config.FRAUD_DECLINE_THRESHOLD and credit_risk_score > config.SCORING_MANUAL_THRESHOLD:
        final_decision = "manual_review"
    else:
        final_decision = "approve"

    return {
        "credit_risk_score": credit_risk_score,
        "fraud_risk_score": fraud_risk_score,
        "final_decision": final_decision,
    }


@app.get("/", response_class=HTMLResponse)
def root():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>AgroAI Risk Platform</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg: #0f172a;
                --card: #111827;
                --text: #e5e7eb;
                --muted: #9ca3af;
                --accent: #22c55e;
                --accent2: #3b82f6;
                --border: #1f2937;
            }
            * { box-sizing: border-box; }
            body {
                margin: 0;
                font-family: 'Inter', system-ui, -apple-system, sans-serif;
                background: linear-gradient(135deg, #0b1224 0%, #0f172a 50%, #0b1325 100%);
                color: var(--text);
                display: flex;
                align-items: center;
                justify-content: center;
                min-height: 100vh;
                padding: 32px 16px;
            }
            .container {
                width: 100%;
                max-width: 1100px;
                background: var(--card);
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 32px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.35);
            }
            .header {
                display: flex;
                flex-direction: column;
                gap: 8px;
                margin-bottom: 28px;
            }
            .title {
                font-size: 32px;
                font-weight: 700;
                letter-spacing: -0.02em;
            }
            .subtitle {
                font-size: 18px;
                color: var(--muted);
            }
            .description {
                margin-top: 4px;
                color: var(--muted);
                line-height: 1.6;
            }
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
                gap: 16px;
                margin-top: 12px;
            }
            .card {
                background: #0d1729;
                border: 1px solid var(--border);
                padding: 20px;
                border-radius: 12px;
                display: flex;
                flex-direction: column;
                gap: 10px;
            }
            .card h3 {
                margin: 0;
                font-size: 18px;
            }
            .card p {
                margin: 0;
                color: var(--muted);
                line-height: 1.5;
            }
            .btn {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
                padding: 10px 14px;
                border-radius: 10px;
                border: 1px solid transparent;
                font-weight: 600;
                text-decoration: none;
                color: white;
                transition: all 0.15s ease;
                margin-top: 4px;
            }
            .btn-primary {
                background: linear-gradient(135deg, #22c55e, #16a34a);
            }
            .btn-primary:hover {
                transform: translateY(-1px);
                box-shadow: 0 10px 24px rgba(34, 197, 94, 0.35);
            }
            .btn-secondary {
                background: #0b1224;
                border: 1px solid var(--border);
                color: var(--text);
            }
            .btn-secondary:hover {
                border-color: #1f2937;
                transform: translateY(-1px);
            }
            .footer {
                margin-top: 24px;
                padding-top: 16px;
                border-top: 1px solid var(--border);
                color: var(--muted);
                font-size: 13px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                flex-wrap: wrap;
                gap: 8px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="title">AgroAI Risk Platform</div>
                <div class="subtitle">Unified Credit Scoring & Fraud Detection</div>
                <div class="description">
                    A single API that blends production-style credit scoring and hybrid fraud detection with explainability and combined risk decisions.
                </div>
            </div>
            <div class="grid">
                <div class="card">
                    <h3>Credit Scoring</h3>
                    <p>Estimate probability of default with a trained model and preprocessing pipeline. Supports approvals, manual review, and declines.</p>
                    <a class="btn btn-secondary" href="/docs#operations-default-scoring_predict">View scoring endpoint</a>
                </div>
                <div class="card">
                    <h3>Fraud Detection</h3>
                    <p>Hybrid supervised + anomaly detector on engineered transaction features with rule-based explanations.</p>
                    <a class="btn btn-secondary" href="/docs#operations-default-fraud_predict">View fraud endpoint</a>
                </div>
            </div>
            <div style="margin-top: 24px;">
                <a class="btn btn-primary" href="/docs">Open API Docs (Swagger)</a>
            </div>
            <div class="footer">
                <span>Powered by FastAPI &amp; AgroAI</span>
                <span>/scoring/predict · /fraud/predict · /fraud/explain · /fraud/retrain · /risk/combined</span>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content
