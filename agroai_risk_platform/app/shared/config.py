from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

# Scoring artifacts (copied from AgroAIScoring)
SCORING_MODEL_PATH = BASE_DIR / "model" / "model.pkl"
SCORING_PREPROCESSOR_PATH = BASE_DIR / "model" / "preprocessor.pkl"

# Fraud artifacts
FRAUD_MODEL_PATH = BASE_DIR / "app" / "fraud" / "fraud_model.pkl"
FRAUD_PREPROCESSOR_PATH = BASE_DIR / "app" / "fraud" / "fraud_preprocessor.pkl"

# Risk thresholds
FRAUD_DECLINE_THRESHOLD = 0.8
SCORING_DECLINE_THRESHOLD = 0.7
FRAUD_MANUAL_THRESHOLD = 0.4
SCORING_MANUAL_THRESHOLD = 0.5
