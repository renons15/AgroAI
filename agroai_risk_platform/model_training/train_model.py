"""Train credit risk model, evaluate it, and persist artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from model_training import utils

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "credit_data.csv"
MODEL_PATH = BASE_DIR / "model" / "model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "model" / "preprocessor.pkl"


def generate_synthetic_data(n_samples: int = 3500, random_state: int = 42) -> pd.DataFrame:
    """Create a synthetic credit dataset with realistic signal."""
    rng = np.random.default_rng(random_state)

    age = rng.integers(21, 70, size=n_samples)
    monthly_income = np.clip(rng.normal(6000, 2000, size=n_samples), 1500, 18000).astype(int)
    loan_term_months = rng.choice([12, 24, 36, 48, 60], size=n_samples, p=[0.15, 0.2, 0.25, 0.2, 0.2])
    loan_amount = np.clip(monthly_income * rng.uniform(0.3, 1.8, size=n_samples), 500, 40000).astype(int)
    past_due = rng.choice(["yes", "no"], size=n_samples, p=[0.22, 0.78])
    employment_type = rng.choice(["employed", "self-employed", "unemployed"], size=n_samples, p=[0.65, 0.2, 0.15])
    has_mortgage = rng.choice(["yes", "no"], size=n_samples, p=[0.35, 0.65])
    num_prev_loans = np.clip(rng.poisson(2.0, size=n_samples), 0, 12)
    credit_utilization = np.clip(rng.beta(2, 5, size=n_samples) + rng.normal(0, 0.05, size=n_samples), 0.01, 0.98)

    # Build an underlying risk score combining intuitive signals
    loan_to_income = loan_amount / np.maximum(monthly_income, 1)
    risk_score = (
    -4.0
    + 0.8 * loan_to_income
    + 1.2 * (credit_utilization - 0.4)
    + 1.0 * (past_due == "yes").astype(float)
    + 1.2 * (employment_type == "unemployed").astype(float)
    + 0.3 * (employment_type == "self-employed").astype(float)
    + 0.4 * (has_mortgage == "yes").astype(float)
    + 0.2 * (num_prev_loans > 5)
    + 0.3 * (age < 25)
    - 0.5 * (age > 55)
)

    probability_default = 1 / (1 + np.exp(-risk_score))
    y = rng.binomial(1, np.clip(probability_default, 0.01, 0.99))

    df = pd.DataFrame(
        {
            "age": age,
            "monthly_income": monthly_income,
            "loan_amount": loan_amount,
            "loan_term_months": loan_term_months,
            "past_due": past_due,
            "employment_type": employment_type,
            "has_mortgage": has_mortgage,
            "num_prev_loans": num_prev_loans,
            "credit_utilization": credit_utilization.round(3),
            "default": y,
        }
    )
    return df


def load_or_create_dataset(path: Path) -> pd.DataFrame:
    """Load dataset from disk or generate a fresh one."""
    if path.exists():
        print(f"Loading existing dataset at {path}")
        return pd.read_csv(path)

    print("Dataset not found. Generating synthetic data...")
    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_data()
    df.to_csv(path, index=False)
    print(f"Synthetic dataset saved to {path}")
    return df


def build_preprocessor() -> ColumnTransformer:
    """Build preprocessing pipeline for numeric and categorical features."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, utils.NUMERIC_FEATURES),
            ("cat", categorical_transformer, utils.CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[float, float, np.ndarray]:
    """Compute evaluation metrics for reporting."""
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)
    cm = confusion_matrix(y_test, preds)
    return acc, auc, cm


def train():
    df = load_or_create_dataset(DATA_PATH)

    X = df[utils.NUMERIC_FEATURES + utils.CATEGORICAL_FEATURES]
    y = df["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()
    classifier = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    print("Training model...")
    model.fit(X_train, y_train)

    acc, auc, cm = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {acc:.3f}")
    print(f"ROC-AUC: {auc:.3f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    # Show top feature importances
    importances = utils.aggregate_feature_importances(
        model, preprocessor, utils.NUMERIC_FEATURES, utils.CATEGORICAL_FEATURES
    )
    if importances:
        print("Top feature importances:")
        for name, score in list(importances.items())[:6]:
            print(f" - {name}: {score:.4f}")

    # Persist artifacts
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print(f"Saved model pipeline to {MODEL_PATH}")
    print(f"Saved preprocessor to {PREPROCESSOR_PATH}")


if __name__ == "__main__":
    train()
