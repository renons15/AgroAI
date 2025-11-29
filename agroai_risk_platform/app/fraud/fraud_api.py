from datetime import datetime
from typing import Dict, List

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.shared import config, utils
from app.fraud.train_fraud_model import compute_features, load_artifacts, train_and_save

router = APIRouter(prefix="/fraud", tags=["fraud"])


class FraudRequest(BaseModel):
    transaction_id: str
    customer_id: str
    amount: float = Field(..., gt=0)
    currency: str
    timestamp: datetime
    country: str
    merchant_id: str
    merchant_category: str
    channel: str = Field(..., pattern="^(POS|ATM|ONLINE|MOBILE)$")
    device_id: str
    is_new_country: int = Field(..., ge=0, le=1)
    is_new_device: int = Field(..., ge=0, le=1)
    same_day_transactions_count: int = Field(..., ge=0)
    average_customer_spend: float = Field(..., gt=0)


def _load_runtime_artifacts():
    if not (config.FRAUD_MODEL_PATH.exists() and config.FRAUD_PREPROCESSOR_PATH.exists()):
        utils.logger.info("Fraud artifacts missing; training now.")
        train_and_save()
    return load_artifacts()


PREPROCESSOR_ARTIFACTS, MODEL_ARTIFACTS = _load_runtime_artifacts()


def _compute_scores(feats: pd.DataFrame) -> Dict[str, float]:
    X = PREPROCESSOR_ARTIFACTS["preprocessor"].transform(feats)
    supervised_model = MODEL_ARTIFACTS["supervised_model"]
    anomaly_model = MODEL_ARTIFACTS["anomaly_model"]
    sup_score = float(supervised_model.predict_proba(X)[0][1])
    raw_anomaly = anomaly_model.decision_function(X)[0]
    min_s, max_s = MODEL_ARTIFACTS["anomaly_min"], MODEL_ARTIFACTS["anomaly_max"]
    if max_s - min_s <= 1e-6:
        anomaly_score = 0.0
    else:
        anomaly_score = (max_s - raw_anomaly) / (max_s - min_s)
        anomaly_score = max(0.0, min(1.0, anomaly_score))
    combined = 0.6 * sup_score + 0.4 * anomaly_score
    return {
        "combined_fraud_score": combined,
        "supervised_score": sup_score,
        "anomaly_score": anomaly_score,
    }


def _explain(row: pd.Series, scores: Dict[str, float]) -> List[str]:
    reasons: List[str] = []
    ratio = row["amount"] / max(row["average_customer_spend"], 1e-6)
    if ratio > 2.0:
        reasons.append(f"Amount is {ratio:.1f}x above customer average")
    if row.get("is_new_country", 0) and ratio > 1.5 and row.get("is_night", False):
        reasons.append("New country + large amount + nighttime")
    if row.get("same_day_transactions_count", 0) > 8:
        reasons.append("Too many transactions in the last hours")
    if row.get("is_new_device", 0):
        reasons.append("Device has never been used before")
    if row.get("merchant_risk_score", 0) > 0.05:
        reasons.append("Merchant category unusual/high risk")
    if row.get("country_risk_score", 0) > 0.03:
        reasons.append("Country risk score is high")
    if row.get("is_new_country", 0) and not row.get("is_new_device", 0) and row.get("country_risk_score", 0) > 0.02:
        reasons.append("New country detected")
    if not reasons:
        reasons.append("Pattern appears typical")
    return reasons


@router.post("/predict")
def predict(payload: FraudRequest) -> Dict[str, float]:
    try:
        df = pd.DataFrame([payload.dict()])
        feats = compute_features(df.copy(), PREPROCESSOR_ARTIFACTS["risk_maps"])
        scores = _compute_scores(feats)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return scores


@router.post("/explain")
def explain(payload: FraudRequest) -> Dict[str, object]:
    try:
        df = pd.DataFrame([payload.dict()])
        feats = compute_features(df.copy(), PREPROCESSOR_ARTIFACTS["risk_maps"])
        scores = _compute_scores(feats)
        reasons = _explain(feats.iloc[0], scores)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        **scores,
        "explanation": reasons,
    }


@router.post("/retrain")
def retrain() -> Dict[str, str]:
    train_and_save()
    global PREPROCESSOR_ARTIFACTS, MODEL_ARTIFACTS
    PREPROCESSOR_ARTIFACTS, MODEL_ARTIFACTS = load_artifacts()
    return {"status": "retrained"}
