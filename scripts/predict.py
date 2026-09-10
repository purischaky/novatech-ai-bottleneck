"""
Batch inference orchestrator for the Intake Baseline Model.

Loads trained artifacts and generates SLA breach probabilities and
estimated resolution times for incoming support tickets.

Usage:
    python scripts/predict.py --input data/new_tickets.csv --output outputs/predictions.csv
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import Optional

import joblib
import numpy as np
import pandas as pd

# Add repository root to system path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class IntakePredictor:
    """Production predictor wrapper encapsulating model loading and scoring."""
    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.clf_model = None
        self.reg_model = None
        self.preprocessor = None
        self.required_features = []
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Validate paths and deserialize fitted estimators."""
        clf_path = self.model_dir / "clf_baseline.joblib"
        reg_path = self.model_dir / "reg_baseline.joblib"
        prep_path = self.model_dir / "preprocessor.joblib"

        if not clf_path.exists() or not prep_path.exists():
            raise FileNotFoundError(
                f"Missing required model artifacts in {self.model_dir}. "
                "Ensure 'clf_baseline.joblib' and 'preprocessor.joblib' exist."
            )

        self.clf_model = joblib.load(clf_path)
        self.preprocessor = joblib.load(prep_path)
        
        # Optional regressor loading
        if reg_path.exists():
            self.reg_model = joblib.load(reg_path)

        # Extract required raw column names directly from ColumnTransformer
        self.required_features = []
        for _, _, cols in self.preprocessor.transformers:
            if isinstance(cols, list):
                self.required_features.extend(cols)
        
        logger.info("Loaded models. Dynamic input schema: %s", self.required_features)

    def predict(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Validate inputs and enrich dataframe with model predictions.
        """
        missing = set(self.required_features) - set(df_input.columns)
        if missing:
            raise ValueError(f"Input data missing required columns: {missing}")

        X_raw = df_input[self.required_features]
        X_proc = self.preprocessor.transform(X_raw)

        # Generate classification predictions
        df_out = df_input.copy()
        df_out["predicted_breach"] = self.clf_model.predict(X_proc)
        df_out["breach_probability"] = np.round(self.clf_model.predict_proba(X_proc)[:, 1], 4)

        # Generate continuous regression estimates if regressor is present
        if self.reg_model is not None:
            df_out["predicted_resolution_hours"] = np.round(self.reg_model.predict(X_proc), 2)

        return df_out

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for batch inference."""
    parser = argparse.ArgumentParser(description="Run batch inference on support tickets.")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="outputs/models/intake_baseline",
        help="Path to folder holding model artifacts."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/support_tickets.csv",
        help="Path to input CSV containing tickets to score."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/predictions.csv",
        help="Destination path for output predictions CSV."
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("STARTING BATCH INFERENCE SERVICE")
    logger.info("=" * 60)

    predictor = IntakePredictor(model_dir=Path(args.model_dir))

    logger.info("Reading input records from: %s", args.input)
    df_new = pd.read_csv(args.input)

    logger.info("Generating predictions for %d records...", len(df_new))
    df_scored = predictor.predict(df_new)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_scored.to_csv(output_path, index=False)

    breach_rate = df_scored["predicted_breach"].mean() * 100
    logger.info("Predictions saved successfully to: %s", output_path)
    logger.info("Total Records: %d | Predicted Breaches: %d (%.1f%%)",
                len(df_scored), df_scored["predicted_breach"].sum(), breach_rate)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()