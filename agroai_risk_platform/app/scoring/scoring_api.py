from typing import Dict

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.shared import config, utils

router = APIRouter(prefix="/scoring", tags=["scoring"])


class CreditInput(BaseModel):
    age: int = Field(..., ge=18, le=90)
    monthly_income: float = Field(..., gt=0)
    loan_amount: float = Field(..., gt=0)
    loan_term_months: int = Field(..., gt=0)
    past_due: str = Field(..., pattern="^(yes|no)$")
    employment_type: str = Field(..., pattern="^(employed|self-employed|unemployed)$")
    has_mortgage: str = Field(..., pattern="^(yes|no)$")
    num_prev_loans: int = Field(..., ge=0)
    credit_utilization: float = Field(..., ge=0, le=1)


def _load_artifacts():
    try:
        model = utils.load_artifact(config.SCORING_MODEL_PATH)
        preprocessor = utils.load_artifact(config.SCORING_PREPROCESSOR_PATH)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return model, preprocessor


MODEL, PREPROCESSOR = _load_artifacts()


def predict_score(payload: CreditInput) -> float:
    df = pd.DataFrame([payload.dict()])
    proba = float(MODEL.predict_proba(df)[0][1])
    return proba


@router.post("/predict")
def predict(payload: CreditInput) -> Dict[str, float]:
    """
    Predict credit default probability.
    """
    score = predict_score(payload)
    return {"credit_risk_score": score}
