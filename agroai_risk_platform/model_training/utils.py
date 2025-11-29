"""Utility helpers for credit scoring demo."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

# Feature schema used across training and inference
NUMERIC_FEATURES: List[str] = [
    "age",
    "monthly_income",
    "loan_amount",
    "loan_term_months",
    "num_prev_loans",
    "credit_utilization",
]

CATEGORICAL_FEATURES: List[str] = [
    "past_due",
    "employment_type",
    "has_mortgage",
]


def risk_category(prob: float) -> str:
    """Map raw probability into a discrete risk bucket."""
    if prob < 0.35:
        return "Low"
    if prob < 0.65:
        return "Medium"
    return "High"


def recommended_action(risk: str) -> str:
    """Recommended decision based on risk bucket."""
    if risk == "Low":
        return "Approve automatically"
    if risk == "Medium":
        return "Send to manual review"
    return "Reject / Decline"


def aggregate_feature_importances(
    model, preprocessor, numeric_features: Iterable[str], categorical_features: Iterable[str]
) -> Dict[str, float]:
    """Aggregate one-hot encoded importances back to original feature names."""
    classifier = getattr(model, "named_steps", {}).get("classifier", model)
    importances = getattr(classifier, "feature_importances_", None)
    if importances is None:
        return {}

    ohe = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    num_names = list(numeric_features)
    cat_ohe_names = list(ohe.get_feature_names_out(categorical_features))
    feature_names = num_names + cat_ohe_names

    values = pd.Series(importances, index=feature_names)
    aggregated: Dict[str, float] = {}
    for num in num_names:
        aggregated[num] = float(values.get(num, 0.0))

    for cat in categorical_features:
        cat_cols = [c for c in cat_ohe_names if c.startswith(f"{cat}_")]
        aggregated[cat] = float(values[cat_cols].sum()) if cat_cols else 0.0

    # Sort by importance descending
    return dict(sorted(aggregated.items(), key=lambda kv: kv[1], reverse=True))


def generate_reason_codes(row: pd.Series, prob: float) -> List[str]:
    """Simple human-readable reasons based on input features and risk."""
    reasons: List[str] = []

    loan_to_income = row["loan_amount"] / max(row["monthly_income"], 1)
    if loan_to_income > 1.2:
        reasons.append("Loan amount is high relative to monthly income")
    elif loan_to_income < 0.5 and prob < 0.2:
        reasons.append("Loan amount is modest relative to income")

    if row.get("credit_utilization", 0) > 0.75:
        reasons.append("High credit utilization suggests heavier debt load")
    elif row.get("credit_utilization", 0) < 0.35 and prob < 0.2:
        reasons.append("Low credit utilization shows conservative borrowing")

    if row.get("past_due") == "yes":
        reasons.append("History of past due payments increases risk")
    else:
        reasons.append("No past due history reported")

    employment = row.get("employment_type")
    if employment == "unemployed":
        reasons.append("Unemployment status increases uncertainty of repayment")
    elif employment == "self-employed":
        reasons.append("Self-employed income can be variable")
    elif prob < 0.2:
        reasons.append("Stable employment supports repayment ability")

    if row.get("num_prev_loans", 0) > 5:
        reasons.append("Many previous loans indicate heavier obligations")
    elif row.get("num_prev_loans", 0) <= 1 and prob < 0.2:
        reasons.append("Limited prior borrowing reduces current obligations")

    if row.get("age", 0) < 23:
        reasons.append("Young applicant may have limited credit history")
    elif row.get("age", 0) > 55 and prob < 0.2:
        reasons.append("Longer life experience aligns with lower observed risk")

    if row.get("has_mortgage") == "yes":
        reasons.append("Existing mortgage reduces available cash flow")

    # Deduplicate while preserving order
    seen = set()
    unique_reasons = []
    for reason in reasons:
        if reason not in seen:
            unique_reasons.append(reason)
            seen.add(reason)

    # Keep explanations concise
    return unique_reasons[:4] if unique_reasons else ["Model assessment based on provided financial profile"]


def build_text_explanation(prob: float, risk: str, reasons: List[str]) -> str:
    """Compose a human-friendly paragraph using the calculated outputs."""
    percentage = f"{prob * 100:.1f}%"
    joined_reasons = "; ".join(reasons)
    recommendation = recommended_action(risk)
    return (
        f"The client is classified as {risk} risk (p = {percentage}). "
        f"Key factors: {joined_reasons}. Recommended action: {recommendation}."
    )
