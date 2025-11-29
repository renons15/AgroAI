import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.shared import config, utils

RNG = np.random.default_rng(42)


def generate_synthetic_transactions(n: int = 20000) -> pd.DataFrame:
    num_customers = 5000
    customers = [f"C{idx}" for idx in range(num_customers)]
    countries = ["US", "GB", "DE", "FR", "ES", "IT", "RU", "CN", "IN", "NG", "BR", "ZA"]
    risky_countries = {"RU", "CN", "NG", "ZA"}
    currencies = ["USD", "EUR", "GBP"]
    merchant_cats = ["grocery", "electronics", "travel", "gaming", "luxury", "services", "fuel", "jewelry"]
    risky_merchants = {"gaming", "luxury", "jewelry"}
    channels = ["POS", "ATM", "ONLINE", "MOBILE"]

    customer_profile = {}
    for cid in customers:
        base_spend = float(RNG.lognormal(mean=4.0, sigma=0.5))
        home_country = RNG.choice(countries, p=[0.15, 0.12, 0.1, 0.1, 0.08, 0.08, 0.08, 0.06, 0.06, 0.05, 0.06, 0.06])
        device = f"D{RNG.integers(1, 8000)}"
        customer_profile[cid] = {
            "avg": base_spend,
            "home_country": home_country,
            "device": device,
        }

    rows = []
    start = datetime(2023, 1, 1)
    for i in range(n):
        cid = RNG.choice(customers)
        profile = customer_profile[cid]
        avg_spend = profile["avg"] * float(RNG.normal(1.0, 0.15))
        amount = max(1.0, float(RNG.lognormal(mean=np.log(avg_spend + 1), sigma=0.6)))
        ts = start + timedelta(minutes=int(RNG.integers(0, 60 * 24 * 120)))
        hour = ts.hour
        day_of_week = ts.weekday()
        is_weekend = int(day_of_week >= 5)
        country = profile["home_country"] if RNG.random() > 0.07 else RNG.choice(countries)
        currency = RNG.choice(currencies, p=[0.6, 0.35, 0.05])
        merchant_id = f"M{RNG.integers(1, 4000)}"
        merchant_category = RNG.choice(merchant_cats, p=[0.24, 0.15, 0.14, 0.08, 0.08, 0.16, 0.1, 0.05])
        channel = RNG.choice(channels, p=[0.35, 0.15, 0.35, 0.15])
        device_id = profile["device"] if RNG.random() > 0.1 else f"D{RNG.integers(1, 10000)}"
        same_day_transactions_count = int(max(0, RNG.poisson(3)))
        if RNG.random() < 0.05:
            same_day_transactions_count += int(RNG.integers(5, 15))

        is_new_country = int(country != profile["home_country"])
        is_new_device = int(device_id != profile["device"])
        deviation_from_avg = amount / max(avg_spend, 1e-3)

        risk_score = -5.0
        if deviation_from_avg > 2.0:
            risk_score += 2.0
        if is_new_country and deviation_from_avg > 1.5:
            risk_score += 2.2
        if hour < 6 or hour > 22:
            risk_score += 1.2
        if merchant_category in risky_merchants:
            risk_score += 1.1
        if same_day_transactions_count > 8:
            risk_score += 1.3
        if is_new_device and channel in {"ONLINE", "MOBILE"}:
            risk_score += 0.8
        if country in risky_countries:
            risk_score += 0.9
        if is_new_device and is_new_country:
            risk_score += 0.7
        if amount > 3 * avg_spend and channel == "ONLINE":
            risk_score += 1.0
        if is_weekend and hour > 20 and channel == "ONLINE":
            risk_score += 0.6

        prob = 1 / (1 + np.exp(-risk_score))
        label = int(RNG.random() < prob)

        rows.append(
            {
                "transaction_id": f"T{i}",
                "customer_id": cid,
                "amount": round(amount, 2),
                "currency": currency,
                "timestamp": ts,
                "hour": hour,
                "day_of_week": day_of_week,
                "country": country,
                "merchant_id": merchant_id,
                "merchant_category": merchant_category,
                "channel": channel,
                "device_id": device_id,
                "is_new_country": is_new_country,
                "is_new_device": is_new_device,
                "same_day_transactions_count": same_day_transactions_count,
                "average_customer_spend": round(avg_spend, 2),
                "deviation_from_avg": deviation_from_avg,
                "fraud": label,
            }
        )
    df = pd.DataFrame(rows)
    # target fraud rate ~1.5-2%
    current_rate = df["fraud"].mean()
    utils.logger.info("Generated dataset with fraud rate %.3f", current_rate)
    return df


def build_risk_maps(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    global_rate = df["fraud"].mean()
    def _rate_map(series: pd.Series) -> Dict[str, float]:
        rates = df.groupby(series)["fraud"].mean().to_dict()
        return {k: float(v) for k, v in rates.items()}

    return {
        "country": _rate_map(df["country"]),
        "merchant_category": _rate_map(df["merchant_category"]),
        "device_id": _rate_map(df["device_id"]),
        "global_rate": float(global_rate),
    }


def compute_features(df: pd.DataFrame, risk_maps: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_night"] = ((df["hour"] < 6) | (df["hour"] > 22)).astype(int)
    df["deviation_from_avg"] = df["amount"] / df["average_customer_spend"].clip(lower=1e-3)
    df["velocity_score"] = np.log1p(df["same_day_transactions_count"])
    df["country_risk_score"] = df["country"].map(risk_maps.get("country", {})).fillna(risk_maps.get("global_rate", 0.01))
    df["merchant_risk_score"] = df["merchant_category"].map(risk_maps.get("merchant_category", {})).fillna(
        risk_maps.get("global_rate", 0.01)
    )
    df["device_risk_score"] = df["device_id"].map(risk_maps.get("device_id", {})).fillna(risk_maps.get("global_rate", 0.01))
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_features = [
        "amount",
        "hour",
        "day_of_week",
        "same_day_transactions_count",
        "average_customer_spend",
        "deviation_from_avg",
        "country_risk_score",
        "merchant_risk_score",
        "device_risk_score",
        "is_new_country",
        "is_new_device",
        "is_weekend",
        "is_night",
        "velocity_score",
    ]
    categorical_features = ["currency", "country", "merchant_id", "merchant_category", "channel", "device_id"]

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        sparse_threshold=0.0,  # force dense for downstream models
    )
    preprocessor.feature_names_in_ = numeric_features + categorical_features  # for reference
    return preprocessor


def train_models(df: pd.DataFrame) -> Tuple[dict, dict]:
    risk_maps = build_risk_maps(df)
    df_feats = compute_features(df.copy(), risk_maps)

    preprocessor = build_preprocessor()
    X = df_feats[
        [
            "amount",
            "hour",
            "day_of_week",
            "same_day_transactions_count",
            "average_customer_spend",
            "deviation_from_avg",
            "country_risk_score",
            "merchant_risk_score",
            "device_risk_score",
            "is_new_country",
            "is_new_device",
            "is_weekend",
            "is_night",
            "velocity_score",
            "currency",
            "country",
            "merchant_id",
            "merchant_category",
            "channel",
            "device_id",
        ]
    ]
    y = df_feats["fraud"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    supervised_model = GradientBoostingClassifier(random_state=42)
    supervised_model.fit(X_train_proc, y_train)

    iso = IsolationForest(
        n_estimators=150,
        contamination=0.02,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_train_proc)

    sup_proba = supervised_model.predict_proba(X_test_proc)[:, 1]
    sup_pred = (sup_proba > 0.5).astype(int)
    acc = accuracy_score(y_test, sup_pred)
    auc = roc_auc_score(y_test, sup_proba)
    cm = confusion_matrix(y_test, sup_pred)
    utils.logger.info("Supervised model accuracy %.3f AUC %.3f", acc, auc)
    utils.logger.info("Confusion matrix:\n%s", cm)

    anomaly_scores = iso.decision_function(X_train_proc)
    anomaly_min, anomaly_max = float(anomaly_scores.min()), float(anomaly_scores.max())

    preprocessor_artifacts = {
        "preprocessor": preprocessor,
        "risk_maps": risk_maps,
    }
    model_artifacts = {
        "supervised_model": supervised_model,
        "anomaly_model": iso,
        "anomaly_min": anomaly_min,
        "anomaly_max": anomaly_max,
    }
    return preprocessor_artifacts, model_artifacts


def save_artifacts(preprocessor_artifacts: dict, model_artifacts: dict) -> None:
    utils.ensure_dir(config.FRAUD_MODEL_PATH)
    utils.ensure_dir(config.FRAUD_PREPROCESSOR_PATH)
    joblib.dump(preprocessor_artifacts, config.FRAUD_PREPROCESSOR_PATH)
    joblib.dump(model_artifacts, config.FRAUD_MODEL_PATH)
    utils.logger.info("Saved fraud artifacts to %s and %s", config.FRAUD_MODEL_PATH, config.FRAUD_PREPROCESSOR_PATH)


def train_and_save() -> None:
    df = generate_synthetic_transactions()
    preprocessor_artifacts, model_artifacts = train_models(df)
    save_artifacts(preprocessor_artifacts, model_artifacts)


def load_artifacts() -> Tuple[dict, dict]:
    pre = utils.load_artifact(config.FRAUD_PREPROCESSOR_PATH)
    model = utils.load_artifact(config.FRAUD_MODEL_PATH)
    return pre, model


def ensure_artifacts() -> Tuple[dict, dict]:
    if not (config.FRAUD_MODEL_PATH.exists() and config.FRAUD_PREPROCESSOR_PATH.exists()):
        train_and_save()
    return load_artifacts()


if __name__ == "__main__":
    train_and_save()
