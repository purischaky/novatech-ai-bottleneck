"""
Explainable AI (XAI) Module using SHAP.

Generates interpretability visualizations to audit model behavior:
- SHAP Beeswarm Summary Plot: Global feature importances across instances.
- SHAP Waterfall Plot: Local instance breakdown for high-risk predictions.

Usage:
    python src/explainability.py
"""

import logging
from pathlib import Path
import sys
from typing import List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend to avoid display errors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

# Add repository root to system path for modular imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_pipeline import SupportDataPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Artifact loading & metadata derivation
def load_model_artifacts(model_dir: Path) -> Tuple[object, object, List[str]]:
    """
    Load trained model components and extract clean feature labels.

    Args:
        model_dir: Directory where artifacts are saved.

    Returns:
        clf_model: Fitted classification model.
        preprocessor: Fitted ColumnTransformer.
        feature_names: Cleaned feature names without transformer prefixes.
    """
    logger.info("📂 Loading model artifacts from %s...", model_dir)

    clf_path = model_dir / "clf_baseline.joblib"
    prep_path = model_dir / "preprocessor.joblib"
    feature_names_path = model_dir / "feature_names.joblib"

    if not clf_path.exists():
        raise FileNotFoundError(f"❌ Classifier not found at: {clf_path}")
    if not prep_path.exists():
        raise FileNotFoundError(f"❌ Preprocessor not found at: {prep_path}")

    clf_model = joblib.load(clf_path)
    preprocessor = joblib.load(prep_path)

    if feature_names_path.exists():
        feature_names = joblib.load(feature_names_path)
        logger.info("✅ Feature names loaded from %s", feature_names_path.name)
    else:
        logger.warning("⚠️ %s not found. Extracting directly from preprocessor...", feature_names_path.name)
        raw_names = list(preprocessor.get_feature_names_out())
        feature_names = [name.split("__")[-1] for name in raw_names]
        logger.info("✅ Extracted %d cleaned feature names", len(feature_names))

    return clf_model, preprocessor, feature_names

# Data preparation for background explainer
def prepare_shap_data(
    preprocessor: object,
    splits: object,
    sample_size: int = 1000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate transformed background and test matrices for TreeSHAP.

    Args:
        preprocessor: Fitted ColumnTransformer.
        splits: DataSplit instance containing train and test sets.
        sample_size: Number of background records for TreeExplainer.

    Returns:
        X_background: Scaled and encoded background sample matrix.
        X_test_proc: Scaled and encoded test set matrix.
        y_test: Binary test target array.
    """
    logger.info("📊 Preparing background and evaluation samples...")

    n_samples = min(sample_size, len(splits.X_train))
    background_sample = splits.X_train.sample(n=n_samples, random_state=42)

    X_background = preprocessor.transform(background_sample)
    X_test_proc = preprocessor.transform(splits.X_test)
    y_test = splits.y_breach_test.to_numpy()

    logger.info("✅ Background samples: %d | Test samples: %d", len(X_background), len(X_test_proc))
    return X_background, X_test_proc, y_test

# Global interpretability: Summary Beeswarm plot
def generate_summary_plot(
    explainer: shap.TreeExplainer,
    X_background_proc: np.ndarray,
    feature_names: List[str],
    output_path: Path
) -> np.ndarray:
    """
    Compute SHAP values and output a beeswarm distribution plot.

    Args:
        explainer: Fitted shap.TreeExplainer instance.
        X_background_proc: 2D array of preprocessed background observations.
        feature_names: Human-readable names matching transformed column dimensions.
        output_path: Destination PNG path.

    Returns:
        shap_values: Array of computed SHAP attribution values for class 1.
    """
    logger.info("📈 Computing background SHAP values and generating Summary Plot...")

    raw_shap_values = explainer.shap_values(X_background_proc)

    # Standardize output format across scikit-learn / shap versions
    if isinstance(raw_shap_values, list):
        shap_values = raw_shap_values[1]  # Positive class (SLA breach = 1)
    elif isinstance(raw_shap_values, np.ndarray) and raw_shap_values.ndim == 3:
        shap_values = raw_shap_values[:, :, 1]
    else:
        shap_values = raw_shap_values

    # Validation guard for dimension alignment
    if shap_values.shape[1] != X_background_proc.shape[1]:
        raise ValueError(
            f"Shape mismatch: SHAP values column count ({shap_values.shape[1]}) does not "
            f"match feature matrix column count ({X_background_proc.shape[1]})."
        )

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_background_proc,
        feature_names=feature_names,
        max_display=15,
        plot_size=(10, 6),
        show=False
    )
    plt.title("SHAP Summary: Feature Impact on SLA Breach Probability", fontsize=13, pad=15)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("✅ Summary Plot saved to: %s", output_path)
    return shap_values

# Local interpretability: WATERFALL PLOT
def generate_waterfall_plot(
    explainer: shap.TreeExplainer,
    clf_model: object,
    X_test_proc: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    output_path: Path
) -> None:
    """
    Generate an individual decision decomposition waterfall plot for a high-risk ticket.

    Args:
        explainer: Fitted shap.TreeExplainer instance.
        clf_model: Trained classifier for inference.
        X_test_proc: 2D array of preprocessed test observations.
        y_test: Ground truth labels.
        feature_names: Cleaned feature list.
        output_path: Destination PNG path.
    """
    logger.info("🔍 Computing local instance attribution (Waterfall Plot)...")

    test_probs = clf_model.predict_proba(X_test_proc)[:, 1]
    high_risk_candidates = np.where((test_probs > 0.8) & (y_test == 1))[0]

    if len(high_risk_candidates) > 0:
        idx = int(high_risk_candidates[0])
    else:
        idx = int(np.argmax(test_probs))
        logger.warning("⚠️ No high-confidence breach found (>80%%). Using maximum probability index.")

    logger.info("   Target sample index: %d", idx)
    logger.info("   Predicted breach probability: %.2f%%", test_probs[idx] * 100)
    logger.info("   Actual ground truth: %s", "Breach" if y_test[idx] == 1 else "No Breach")

    # Generate single-instance explanation using the modern Explanation API
    single_instance = X_test_proc[idx : idx + 1]
    explanation_obj = explainer(single_instance)[0]

    # Slice for class 1 if explanation output contains multi-class axis (n_features, 2)
    if len(explanation_obj.shape) == 2 and explanation_obj.shape[1] == 2:
        explanation_slice = explanation_obj[:, 1]
    else:
        explanation_slice = explanation_obj

    # Attach human-readable feature names directly to the explanation slice
    explanation_slice.feature_names = feature_names

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation_slice, max_display=10, show=False)
    plt.title(
        f"Local Explanation: Test Record #{idx} (P(Breach) = {test_probs[idx]:.1%})",
        fontsize=12,
        pad=15
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("✅ Waterfall Plot saved to: %s", output_path)

# Feature importance ranking
def compute_feature_importance(shap_values: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    """
    Compute mean absolute SHAP value ranking across all instances.
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

# Main execution pipeline
def main():
    logger.info("=" * 70)
    logger.info("INITIATING EXPLAINABILITY ANALYSIS (SHAP)")
    logger.info("=" * 70)

    model_dir = Path("outputs/models/intake_baseline")
    if not model_dir.exists():
        # Fallback to general model outputs folder
        fallback_dir = Path("outputs/models")
        if (fallback_dir / "clf_baseline.joblib").exists():
            model_dir = fallback_dir
        else:
            logger.error("❌ Model directory not found: %s", model_dir)
            logger.error("Please run 'python src/ml_baseline.py' first to train and persist the model.")
            return

    clf_model, preprocessor, feature_names = load_model_artifacts(model_dir)

    pipeline = SupportDataPipeline(
        "data/support_tickets.csv",
        num_cols=["sla_hours"],
        cat_cols=["category", "priority", "channel"]
    )
    df_clean = pipeline.clean(pipeline.load())
    splits = pipeline.split(df_clean, test_size=0.2, temporal=True)

    X_background, X_test_proc, y_test = prepare_shap_data(preprocessor, splits, sample_size=1000)

    logger.info("🧠 Initializing TreeExplainer...")
    explainer = shap.TreeExplainer(clf_model)

    reports_dir = Path("outputs/reports")
    summary_path = reports_dir / "shap_summary_plot.png"
    waterfall_path = reports_dir / "shap_waterfall_plot.png"

    shap_values = generate_summary_plot(explainer, X_background, feature_names, summary_path)
    generate_waterfall_plot(explainer, clf_model, X_test_proc, y_test, feature_names, waterfall_path)

    # Log summary of top driving features
    importance_df = compute_feature_importance(shap_values, feature_names)
    logger.info("\n" + "=" * 70)
    logger.info("📝 SHAP GLOBAL IMPORTANCE SUMMARY (Top 5)")
    logger.info("=" * 70)
    for i, row in importance_df.head(5).iterrows():
        logger.info("   %d. %-24s | Mean |SHAP|: %.4f", i + 1, row["feature"], row["mean_abs_shap"])

    logger.info("=" * 70)
    logger.info("✅ EXPLAINABILITY ANALYSIS COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()