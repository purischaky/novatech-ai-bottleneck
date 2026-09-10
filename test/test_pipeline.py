"""
Unit test for the support data pipeline.

Run with:
    pytest test/ -v
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

# Add root folder to sys.path as fallback
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_pipeline import SupportDataPipeline, DataSplit

# FIXTURES
@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Small synthetic dataset including temporal column and all required fields."""
    return pd.DataFrame({
        'ticket_id': [f'TKT-{i:03d}' for i in range(10)],
        'created_at': pd.date_range(start='2025-01-01', periods=10, freq='D'),
        'category': ['Technical Issue', 'Billing', 'Security', 'Integration/API', 'Product Question'] * 2,
        'priority': ['High', 'Low', 'Medium', 'Critical', 'High'] * 2,
        'channel': ['Email', 'Chat', 'Web Form', 'Phone', 'Email'] * 2,
        'assigned_team': ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 2', 'Tier 1'] * 2,
        'current_stage': ['Resolved', 'Open', 'In Progress', 'Resolved', 'Triage'] * 2,
        'first_response_hours': [1.5, 2.3, 0.8, 5.2, 3.1, 2.0, 1.1, 4.5, 2.8, 3.3],
        'waiting_hours': [10.0, 20.0, 5.0, 30.0, 15.0, 12.0, 8.0, 25.0, 18.0, 22.0],
        'queue_age_hours': [10.0, 20.0, 5.0, 30.0, 15.0, 12.0, 8.0, 25.0, 18.0, 22.0],
        'sla_hours': [24, 48, 8, 24, 48, 24, 48, 8, 24, 48],
        'sla_breach': [1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
        'resolution_hours': [12.5, 30.0, 5.2, 45.0, 20.0, 18.5, 25.0, 6.0, 35.0, 40.0],
    })

@pytest.fixture
def pipeline(sample_df: pd.DataFrame, tmp_path: Path) -> SupportDataPipeline:
    """Pipeline instance configured with a temporary test CSV."""
    csv_path = tmp_path / "test_tickets.csv"
    sample_df.to_csv(csv_path, index=False)
    return SupportDataPipeline(str(csv_path))

# UNIT TESTS
def test_filepath_validation():
    """Fails fast if the file does not exist or has invalid extension."""
    with pytest.raises(FileNotFoundError):
        SupportDataPipeline("data/non_existent_file.csv")

    with pytest.raises(ValueError, match="CSV"):
        SupportDataPipeline(__file__)  # Passing .py file not allowed

def test_load_returns_valid_dataframe(pipeline: SupportDataPipeline):
    """load() must return a DataFrame with expected shape."""
    df = pipeline.load()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10
    assert 'sla_breach' in df.columns

def test_clean_converts_breach_to_int(pipeline: SupportDataPipeline):
    """clean() must coerce boolean/float targets into integer values."""
    df = pipeline.load()
    df['sla_breach'] = df['sla_breach'].astype(float)
    df_clean = pipeline.clean(df)
    
    assert np.issubdtype(df_clean['sla_breach'].dtype, np.integer)
    assert set(df_clean['sla_breach'].unique()).issubset({0, 1})

def test_clean_drops_nulls(pipeline: SupportDataPipeline):
    """clean() must drop records with null values in critical columns."""
    df = pipeline.load()
    df.loc[0, 'first_response_hours'] = np.nan
    df_clean = pipeline.clean(df)
    
    assert len(df_clean) == 9
    assert not df_clean['first_response_hours'].isna().any()

def test_clean_raises_on_invalid_target(pipeline: SupportDataPipeline):
    """clean() must raise ValueError if target contains invalid non-binary values."""
    df = pipeline.load()
    # Convertir a object primero para que pandas acepte el string
    df['sla_breach'] = df['sla_breach'].astype(object)
    df.loc[0, 'sla_breach'] = 'invalid_text'
    with pytest.raises(ValueError):
        pipeline.clean(df)

def test_clean_does_not_modify_original(pipeline: SupportDataPipeline):
    """Verifies that clean() does not modify the original DataFrame."""
    df = pipeline.load()
    original_len = len(df)
    df.loc[0, 'first_response_hours'] = np.nan
    
    _ = pipeline.clean(df)
    
    assert len(df) == original_len
    assert df['first_response_hours'].isna().sum() == 1

def test_split_returns_expected_data_split(pipeline: SupportDataPipeline):
    """split() must return a DataSplit with consistent dimensions."""
    df = pipeline.load()
    df_clean = pipeline.clean(df)
    test_size = 0.3
    
    splits = pipeline.split(df_clean, test_size=test_size, random_state=42)
    
    assert isinstance(splits, DataSplit)
    assert len(splits.X_train) + len(splits.X_test) == len(df_clean)
    
    # Tolerance of ±1 to avoid fragility with small data
    expected_test_size = int(len(df_clean) * test_size)
    assert abs(len(splits.X_test) - expected_test_size) <= 1
    
    assert len(splits.y_breach_train) == len(splits.X_train)
    assert len(splits.y_resolution_test) == len(splits.X_test)

def test_stratification_preserves_class_ratio(pipeline: SupportDataPipeline):
    """Verifies that stratify maintains the class proportion across splits."""
    df = pipeline.load()
    df_clean = pipeline.clean(df)
    splits = pipeline.split(df_clean, test_size=0.3)
    
    train_ratio = splits.y_breach_train.mean()
    test_ratio = splits.y_breach_test.mean()
    full_ratio = df_clean['sla_breach'].mean()
    
    assert abs(train_ratio - full_ratio) < 0.10
    assert abs(test_ratio - full_ratio) < 0.10

def test_preprocessor_fit_transform(pipeline: SupportDataPipeline):
    """Preprocessor must fit and transform features into a valid numeric matrix."""
    df = pipeline.load()
    df_clean = pipeline.clean(df)
    splits = pipeline.split(df_clean, test_size=0.2)
    
    preprocessor = pipeline.build_preprocessor()
    assert isinstance(preprocessor, ColumnTransformer)

    X_train_transformed = preprocessor.fit_transform(splits.X_train)
    X_test_transformed = preprocessor.transform(splits.X_test)
    
    assert isinstance(X_train_transformed, np.ndarray)
    assert X_train_transformed.shape[0] == len(splits.X_train)
    assert X_test_transformed.shape[1] == X_train_transformed.shape[1]
    assert not np.isnan(X_train_transformed).any()

def test_full_pipeline_integration(pipeline: SupportDataPipeline):
    """End-to-end test: load → clean → preprocess → split."""
    df_raw = pipeline.load()
    df_clean = pipeline.clean(df_raw)
    preprocessor = pipeline.build_preprocessor()
    splits = pipeline.split(df_clean, test_size=0.2)
    
    X_train_proc = preprocessor.fit_transform(splits.X_train)
    X_test_proc = preprocessor.transform(splits.X_test)
    
    assert X_train_proc.shape[0] == len(splits.X_train)
    assert X_test_proc.shape[0] == len(splits.X_test)
    assert not np.isnan(X_train_proc).any()
    assert not np.isnan(X_test_proc).any()