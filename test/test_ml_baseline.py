"""
Unit tests for the Scikit-Learn baseline model module.

Run with:
    pytest tests/test_ml_baseline.py -v
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_pipeline import DataSplit
from src.ml_baseline import MLBaselineModel, ModelResults

# Fixtures
@pytest.fixture
def small_data_split() -> DataSplit:
    """Creates a deterministic synthetic DataSplit with balanced labels."""
    rng = np.random.default_rng(42)
    n = 100

    X = pd.DataFrame({
        "sla_hours": rng.choice([8, 24, 48, 72], n),
        "category": rng.choice(["Bug Report", "Billing", "Security"], n),
        "priority": rng.choice(["Low", "Medium", "High", "Critical"], n),
        "channel": rng.choice(["Email", "Chat", "Phone"], n),
    })

    # Alternating labels ensure binary representation in train and test
    y_breach = pd.Series([0, 1] * (n // 2), name="sla_breach")
    y_resolution = pd.Series(rng.uniform(5, 100, n), name="resolution_hours")

    split_idx = int(n * 0.8)
    return DataSplit(
        X_train=X.iloc[:split_idx].reset_index(drop=True),
        X_test=X.iloc[split_idx:].reset_index(drop=True),
        y_breach_train=y_breach.iloc[:split_idx].reset_index(drop=True),
        y_breach_test=y_breach.iloc[split_idx:].reset_index(drop=True),
        y_resolution_train=y_resolution.iloc[:split_idx].reset_index(drop=True),
        y_resolution_test=y_resolution.iloc[split_idx:].reset_index(drop=True),
    )

@pytest.fixture
def simple_preprocessor() -> ColumnTransformer:
    """Minimal preprocessor handling numeric scaling and one-hot encoding."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["sla_hours"]),
            (
                "cat",
                OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                ["category", "priority", "channel"],
            ),
        ],
        remainder="drop",
    )

@pytest.fixture
def trained_model(small_data_split: DataSplit, simple_preprocessor: ColumnTransformer) -> MLBaselineModel:
    """Instantiated and trained model ready for evaluation and persistence tests."""
    model = MLBaselineModel(n_estimators=10, random_state=42)
    model.fit(simple_preprocessor, small_data_split)
    return model

# UNIT TESTS
def test_model_initialization():
    """Model attributes must mirror constructor arguments and start unfitted."""
    model = MLBaselineModel(n_estimators=50, max_depth=5, random_state=123)

    assert model.n_estimators == 50
    assert model.max_depth == 5
    assert model.random_state == 123
    assert model.preprocessor is None
    assert model.feature_names_ is None

def test_fit_trains_both_models(trained_model: MLBaselineModel):
    """fit() must fit the preprocessor and both scikit-learn estimators."""
    assert trained_model.preprocessor is not None
    assert trained_model.feature_names_ is not None
    assert len(trained_model.feature_names_) > 0

    assert hasattr(trained_model.clf_model, "estimators_")
    assert hasattr(trained_model.reg_model, "estimators_")

def test_evaluate_returns_valid_metrics(trained_model: MLBaselineModel, small_data_split: DataSplit):
    """evaluate() must return bounded metrics and valid finite numbers."""
    results = trained_model.evaluate(small_data_split)

    assert isinstance(results, ModelResults)
    for metric_name, value in results.classification.items():
        assert 0.0 <= value <= 1.0, f"{metric_name} out of bounds: {value}"

    for metric_name, value in results.regression.items():
        assert np.isfinite(value), f"{metric_name} is not finite: {value}"

def test_save_and_load_preserves_state(trained_model, tmp_path):
    """save() and load() must restore models."""
    trained_model.save(str(tmp_path))
    
    # Verify that files eist
    assert (tmp_path / "clf_baseline.joblib").exists()
    assert (tmp_path / "reg_baseline.joblib").exists()
    assert (tmp_path / "preprocessor.joblib").exists()
    assert (tmp_path / "feature_names.joblib").exists()

    # Load new model
    loaded_model = MLBaselineModel().load(str(tmp_path))

    # Verify consistency
    assert loaded_model.feature_names_ == trained_model.feature_names_
    assert loaded_model.preprocessor is not None
    assert hasattr(loaded_model.clf_model, 'estimators_')
    assert hasattr(loaded_model.reg_model, 'estimators_')

def test_feature_importance_fails_before_fit():
    """feature_importance() must fail if invoked on an unfitted model."""
    model = MLBaselineModel()

    with pytest.raises(RuntimeError):
        model.feature_importance()

def test_feature_importance_returns_sorted_dataframe(trained_model: MLBaselineModel):
    """feature_importance() must return a sorted pandas DataFrame."""
    df = trained_model.feature_importance(top_n=5)

    assert isinstance(df, pd.DataFrame)
    assert len(df) <= 5
    assert list(df.columns) == ["feature", "importance"]

    importances = df["importance"].to_numpy()
    assert np.all(importances[:-1] >= importances[1:]), "Features are not sorted in descending order"