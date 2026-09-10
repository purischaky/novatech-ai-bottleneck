"""
Scikit-Learn Baseline Model for predicting SLA and resolution time.

Train two RandomForest models:
- Classification: predicts whether a ticket will violate the SLA (sla_breach)
- Regression: predicts resolution time (resolution_hours)

Usage:
    python src/ml_baseline.py
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_pipeline import DataSplit, SupportDataPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Data class: model results
@dataclass
class ModelResults:
    """Inmutable contrainer for model metrics."""
    classification: Dict[str, float]
    regression: Dict[str, float]

# Principal class: MLBaselineModel
class MLBaselineModel:
    """
    Baseline model that trains RandomForest for classification and regression.

    Internal state:
        clf_model: Classifiction model
        reg_model: Regression model
        preprocessor: Feature transformer
        feature_names_: Feature names

    Example:
        model = MLBaselineModel(n_estimators=200, random_state=42)
        model.fit(preprocessor, splits)
        results = model.evaluate(splits)
        model.save("outputs/models/")
    """
    
    # Definition of feature scenarios
    # Scenario A: (T=0)
    INTAKE_NUM_COLS = ['sla_hours']
    INTAKE_CAT_COLS = ['category', 'priority', 'channel']
    
    # Scenario B: All features
    OMNISCIENT_NUM_COLS = ['first_response_hours', 'waiting_hours', 'queue_age_hours', 'sla_hours', 'resolution_hours']
    OMNISCIENT_CAT_COLS = ['category', 'priority', 'channel', 'assigned_team', 'current_stage']

    # File name
    CLF_FILENAME = "clf_baseline.joblib"
    REG_FILENAME = "reg_baseline.joblib"
    PREPROC_FILENAME = "preprocessor.joblib"
    FEATURE_NAMES_FILENAME = "feature_names.joblib"

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.clf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs,
            class_weight='balanced',
        )
        self.reg_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.preprocessor = None
        self.feature_names_: Optional[List[str]] = None

    # Train both models
    def fit(self, preprocessor, splits: DataSplit) -> 'MLBaselineModel':
        """
        Train both models (classification and regression).

        Args:
            preprocessor: Already built ColumnTransformer (not yet tuned)
            splits: DataSplit with training data

        Returns:
            self (for method chaining)
        """
        logger.info("🔧 Adjusting the preprocessor in train...")
        X_train_proc = preprocessor.fit_transform(splits.X_train)
        self.preprocessor = preprocessor

        # Use get_feature_names_out() from ColumnnTransformer
        self.feature_names_ = list(preprocessor.get_feature_names_out())
        logger.info(f"📊 Generated features: {len(self.feature_names_)}")

        # Training classification
        logger.info("🌲 Training classification model (sla_breach)...")
        self.clf_model.fit(X_train_proc, splits.y_breach_train)

        # Train Regression
        logger.info("🌲 Training regression model (resolution_hours)...")
        self.reg_model.fit(X_train_proc, splits.y_resolution_train)

        logger.info("✅ Training complete")
        return self

    # Evaluate in a test
    def evaluate(self, splits: DataSplit) -> ModelResults:
        """Evaluate both models on the test set."""
        self._validate_is_fitted()

        logger.info("📈 Evaluating models on test set...")
        X_test_proc = self.preprocessor.transform(splits.X_test)

        # Classification metrics
        y_pred_clf = self.clf_model.predict(X_test_proc)
        y_pred_proba = self.clf_model.predict_proba(X_test_proc)[:, 1]

        clf_metrics = {
            'accuracy': accuracy_score(splits.y_breach_test, y_pred_clf),
            'precision': precision_score(splits.y_breach_test, y_pred_clf),
            'recall': recall_score(splits.y_breach_test, y_pred_clf),
            'f1': f1_score(splits.y_breach_test, y_pred_clf),
            'roc_auc': roc_auc_score(splits.y_breach_test, y_pred_proba),
        }

        # Regression metrics
        y_pred_reg = self.reg_model.predict(X_test_proc)
        reg_metrics = {
            'mae': mean_absolute_error(splits.y_resolution_test, y_pred_reg),
            'rmse': np.sqrt(mean_squared_error(splits.y_resolution_test, y_pred_reg)),
            'r2': r2_score(splits.y_resolution_test, y_pred_reg),
        }

        logger.info("📊 Classification Metrics:")
        for k, v in clf_metrics.items():
            logger.info(f"   {k:>10}: {v:.4f}")

        logger.info("📊 Regression Metrics:")
        for k, v in reg_metrics.items():
            logger.info(f"   {k:>10}: {v:.4f}")

        return ModelResults(classification=clf_metrics, regression=reg_metrics)

    # Feature impotance
    def feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """
        Returns the top N most importance features of the classifiction model.

        Raises:
            RuntimeError: if the model has not been trained or loaded correctly
        """
        self._validate_is_fitted()

        if self.feature_names_ is None:
            raise RuntimeError(
                "feature_names_ is not available. "
                "Train the model with fit() or load it with load()."
            )

        if len(self.feature_names_) != len(self.clf_model.feature_importances_):
            raise RuntimeError(
                f"Inconsistency: {len(self.feature_names_)} nname vs "
                f"{len(self.clf_model.feature_importances_)} importances"
            )

        importances = self.clf_model.feature_importances_
        df = pd.DataFrame({
            'feature': self.feature_names_,
            'importance': importances,
        }).sort_values('importance', ascending=False)

        return df.head(top_n)

    # Save models
    def save(self, output_dir: str) -> None:
        """
        Saves the trained models and feture names to disk.

        Raises:
            RuntimeError: if the model has not been trained
        """
        self._validate_is_fitted()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.clf_model, output_path / self.CLF_FILENAME)
        joblib.dump(self.reg_model, output_path / self.REG_FILENAME)
        joblib.dump(self.preprocessor, output_path / self.PREPROC_FILENAME)
        # Persist feature_names_ as artefact
        joblib.dump(self.feature_names_, output_path / self.FEATURE_NAMES_FILENAME)

        logger.info(f"💾 Models saved in {output_path}/")
        logger.info(f"   - {self.CLF_FILENAME}")
        logger.info(f"   - {self.REG_FILENAME}")
        logger.info(f"   - {self.PREPROC_FILENAME}")
        logger.info(f"   - {self.FEATURE_NAMES_FILENAME}")

    # Load models
    def load(self, model_dir: str) -> 'MLBaselineModel':
        """
        Loads previously saved models, including feature_names_.

        Raises:
            FileNotFoundError: if any required file is missing
        """
        model_path = Path(model_dir)

        # Fail-fast validation: verify that all files exist 
        required_files = [
            self.CLF_FILENAME,
            self.REG_FILENAME,
            self.PREPROC_FILENAME,
            self.FEATURE_NAMES_FILENAME,
        ]
        missing = [f for f in required_files if not (model_path / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing files in {model_path}: {missing}. "
                f"Did you train the model first with fit() and save()?"
            )

        self.clf_model = joblib.load(model_path / self.CLF_FILENAME)
        self.reg_model = joblib.load(model_path / self.REG_FILENAME)
        self.preprocessor = joblib.load(model_path / self.PREPROC_FILENAME)
        # Load feature_names_ to maintain a consistent state
        self.feature_names_ = joblib.load(model_path / self.FEATURE_NAMES_FILENAME)

        logger.info(f"📂 Models loaded from {model_path}/")
        logger.info(f"   Available features: {len(self.feature_names_)}")
        return self

    # Status validation
    def _validate_is_fitted(self) -> None:
        """
        Verifies that the model has been trained or loaded correctly.
        Fails quickly with a clear message if the states is invalid.
        """
        if self.preprocessor is None:
            raise RuntimeError(
                "The model has not been trained or loaded. "
                "Call fit() or load()."
            )

# Main execution block
def generate_report(results: ModelResults, feature_imp: pd.DataFrame, output_path: str) -> None:
    """Generate a Markdown report with the results."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, 'w', encoding='utf-8') as f:
        f.write("# 📊 Baseline Model Report\n\n")
        f.write(f"**Model** RandomForest (n_estimators=200)\n")
        f.write(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        f.write("## 🎯 Clasification: SLA Breach\n\n")
        f.write("| Metric | Value |\n")
        f.write("|---------|-------|\n")
        for k, v in results.classification.items():
            f.write(f"| {k} | {v:.4f} |\n")

        f.write("\n## 📈 Regression: Resolution Hours\n\n")
        f.write("| Metric | Value |\n")
        f.write("|---------|-------|\n")
        for k, v in results.regression.items():
            f.write(f"| {k} | {v:.4f} |\n")

        f.write("\n## 🏆 Top 10 Features (Clasification)\n\n")
        f.write("| Feature | Importance |\n")
        f.write("|---------|-------------|\n")
        for _, row in feature_imp.iterrows():
            f.write(f"| {row['feature']} | {row['importance']:.4f} |\n")

    logger.info(f"📄 Report saved in: {output}")


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("Comparing training: Intake vs omniscient")
    logger.info("=" * 70)

    pipeline = SupportDataPipeline("data/support_tickets.csv")
    df_raw = pipeline.load()
    df_clean = pipeline.clean(df_raw)
    splits = pipeline.split(df_clean, test_size=0.2)

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("ENTRENAMIENTO COMPARATIVO: INTAKE vs OMNISCIENT")
    logger.info("=" * 70)

    pipeline = SupportDataPipeline("data/support_tickets.csv")
    df_raw = pipeline.load()
    df_clean = pipeline.clean(df_raw)
    splits = pipeline.split(df_clean, test_size=0.2)

    # Scenario 1: Model Intake (T=0) 
    logger.info("\n🔮 Training Intake Model (T=0, Without Leakage)")
    pipeline_intake = SupportDataPipeline(
        "data/support_tickets.csv",
        num_cols=MLBaselineModel.INTAKE_NUM_COLS,
        cat_cols=MLBaselineModel.INTAKE_CAT_COLS
    )
    # Redo the split with the restricted columns
    splits_intake = pipeline_intake.split(df_clean, test_size=0.2)
    preprocessor_intake = pipeline_intake.build_preprocessor()
    
    model_intake = MLBaselineModel(n_estimators=200, random_state=42)
    model_intake.fit(preprocessor_intake, splits_intake)
    results_intake = model_intake.evaluate(splits_intake)
    model_intake.save("outputs/models/intake_baseline")

    # Scenario 2: Omniscient model (Post-Mortem)
    logger.info("\n🔬 Training Omnicient Model (Post-Mortem, With Leakage)")
    pipeline_omni = SupportDataPipeline(
        "data/support_tickets.csv",
        num_cols=MLBaselineModel.OMNISCIENT_NUM_COLS,
        cat_cols=MLBaselineModel.OMNISCIENT_CAT_COLS
    )
    splits_omni = pipeline_omni.split(df_clean, test_size=0.2)
    preprocessor_omni = pipeline_omni.build_preprocessor()
    
    model_omni = MLBaselineModel(n_estimators=200, random_state=42)
    model_omni.fit(preprocessor_omni, splits_omni)
    results_omni = model_omni.evaluate(splits_omni)
    model_omni.save("outputs/models/omniscient_baseline")

    # Report
    logger.info("\n" + "=" * 70)
    logger.info("📊 Performance (ROC-AUC Classification)")
    logger.info("=" * 70)
    logger.info(f"   Intake Model (T=0)     : {results_intake.classification['roc_auc']:.4f}")
    logger.info(f"   Omniscient Model (T=End): {results_omni.classification['roc_auc']:.4f}")
    logger.info("=" * 70)
    logger.info("✅ Training Complete. Review outputs/reports/")