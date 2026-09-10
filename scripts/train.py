"""
Production Training Orchestrator for the Intake Baseline Model (Random Forest).

Ingests tickets, isolates real-time intake features, executes model training,
and persists versioned model artifacts and evaluation metrics.

Usage:
    python scripts/train.py --data data/support_tickets.csv --output outputs/models/intake_baseline
"""

import argparse
import json
import logging
from pathlib import Path
import sys
import joblib

# Resolve repository root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_pipeline import SupportDataPipeline
from src.ml_baseline import MLBaselineModel

#Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """Parse command line flags for training orchestration."""
    parser = argparse.ArgumentParser(description="Train and persist the Intake Baseline Model.")
    parser.add_argument(
        "--data",
        type=str,
        default="data/support_tickets.csv",
        help="Path to source CSV ticket dataset."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/models/intake_baseline",
        help="Destination directory for serialized model artifacts."
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of records allocated to the test set."
    )
    parser.add_argument(
        "--temporal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply chronological split to prevent time-series data leakage."
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of decision trees for the Random Forest estimators."
    )
    return parser.parse_args()


def main() -> None:
    """Execute end-to-end data pipeline, model fitting, and artifact export."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Starting intake baseline model training")
    logger.info("=" * 60)

    # Initialize Pipeline (Intake attributes available at ticket creation)
    pipeline = SupportDataPipeline(
        filepath=args.data,
        num_cols=["sla_hours"],
        cat_cols=["category", "priority", "channel"]
    )

    # Load and Clean
    df_raw = pipeline.load()
    df_clean = pipeline.clean(df_raw)

    # Partition Train/Test
    splits = pipeline.split(df_clean, test_size=args.test_size, temporal=args.temporal)

    # Build Transformer
    preprocessor = pipeline.build_preprocessor()

    # Fit Estimators
    model = MLBaselineModel(n_estimators=args.n_estimators, random_state=42)
    model.fit(preprocessor, splits)

    # Evaluate Performance
    logger.info("Evaluating fitted models on test partition...")
    results = model.evaluate(splits)

    # Persist Artifacts and Metadata
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(output_dir))

    # Explicitly save clean feature names for XAI and inference services
    raw_feature_names = model.preprocessor.get_feature_names_out()
    cleaned_feature_names = [name.split("__")[-1] for name in raw_feature_names]
    joblib.dump(cleaned_feature_names, output_dir / "feature_names.joblib")
    logger.info("Saved clean feature names list to: %s", output_dir / "feature_names.joblib")

    # Save metrics manifest
    metrics_payload = {
        "classification": results.classification,
        "regression": results.regression,
        "hyperparameters": {
            "n_estimators": args.n_estimators,
            "test_size": args.test_size,
            "temporal_split": args.temporal
        }
    }
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=4)
    logger.info("Saved evaluation summary to: %s", metrics_path)

    logger.info("=" * 60)
    logger.info("Training pipeline completed")
    logger.info("Artifacts available in: %s", output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()