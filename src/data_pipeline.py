"""
Data pipeline for technical support bottleneck analysis.

This module handles:
1. Loading the CSV of tickets
2. Validating that it contains the expected columns
3. Cleaning null values
4. Dividing into training and test sets
5. Building the preprocessor (scaling + encoding)
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Data class: immutable structure for splits
@dataclass
class DataSplit:
    """
    Name container fot the 6 arrays resulting from the split.
    Prevent sorting errors (ej. confusing y_breach with y_resolution).
    
    Usage:
        splits = pipeline.split(df)
        print(splits.X_train.shape)
    """
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_breach_train: pd.Series
    y_breach_test: pd.Series
    y_resolution_train: pd.Series
    y_resolution_test: pd.Series

# Main class: SupportDataPipeline
class SupportDataPipeline:
    """
    Pipeline for loading, cleaning, and splitting technical support data.
    
    Atributes:
        filepath: Path to the CSV file
        cat_cols: Categorical columns to encode
        num_cols: Numerical columns to scale
        target_breach: Binary target column (SLA breach)
        target_resolution: Continuous target column (resolution times)
    
    Example:
        pipeline = SupportDataPipeline("data/support_tickets.csv")
        df = pipeline.load()
        df_clean = pipeline.clean(df)
        preprocessor = pipeline.build_preprocessor()
        splits = pipeline.split(df_clean)
    """
    
    # Default columns
    DEFAULT_CAT_COLS = ['category', 'priority', 'channel', 'assigned_team', 'current_stage']
    DEFAULT_NUM_COLS = ['first_response_hours', 'waiting_hours', 'queue_age_hours', 'sla_hours']
    TARGET_BREACH = 'sla_breach'
    TARGET_RESOLUTION = 'resolution_hours'
    
    def __init__(
        self,
        filepath: str,
        cat_cols: Optional[List[str]] = None,
        num_cols: Optional[List[str]] = None,
    ):
        """
        Inicializes the pipeline.
        
        Args:
            filepath: Path to the CSV file of tickets
            cat_cols: List of categorical columns (optional)
            num_cols: List of numeric columns (optional)
        """
        self.filepath = Path(filepath)
        self.cat_cols = cat_cols or self.DEFAULT_CAT_COLS
        self.num_cols = num_cols or self.DEFAULT_NUM_COLS
        self._validate_filepath()
    
    def _validate_filepath(self) -> None:
        """Validates that the file exists and is CSV (fail-fast)."""
        if not self.filepath.exists():
            raise FileNotFoundError(f"❌ File not found: {self.filepath}")
        if self.filepath.suffix.lower() != '.csv':
            raise ValueError(f"❌ Expected CSV, received: {self.filepath.suffix}")
    
    # Load data
    def load(self) -> pd.DataFrame:
        """
        Loads the CSV file without modification.
        
        Returns:
            Raw DataFrame with all records
        
        Raises:
            FileNotFoundError: if the file does not exist
            ValueError: if required columns are missing
        """
        logger.info(f"📂 Loading data from {self.filepath}")
        df = pd.read_csv(self.filepath)
        
        # Validate required columns
        required_cols = self.cat_cols + self.num_cols + [self.TARGET_BREACH, self.TARGET_RESOLUTION]
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"❌ Missing columns in the dataset: {missing}")
        
        logger.info(f"✅ Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
        return df
    
    # Clean data
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the dataset:
        - Removes rows with nulls in critical columns
        - Converts sla_breach to an integer (0/1)
        
        Args:
            df: Raw DataFrame
        
        Returns:
            Clean DataFrame (does not modify the original)
        """
        logger.info("🧹 Starting cleaning...")
        df = df.copy()  
        initial_rows = len(df)
        
        # Critical columns
        critical_cols = self.cat_cols + self.num_cols + [self.TARGET_BREACH, self.TARGET_RESOLUTION]
        df = df.dropna(subset=critical_cols)
        
        dropped = initial_rows - len(df)
        if dropped > 0:
            pct = dropped / initial_rows * 100
            logger.warning(f"⚠️ Dropped {dropped:,} rows with nulls ({pct:.2f}%)")
        
        # Validate and convert sla_breach to integer
        unique_values = df[self.TARGET_BREACH].dropna().unique()
        if not set(unique_values).issubset({0, 1, 0.0, 1.0, True, False}):
            raise ValueError(
                f"❌ '{self.TARGET_BREACH}' must be binary (0/1). "
                f"Values found: {unique_values}"
            )
        df[self.TARGET_BREACH] = df[self.TARGET_BREACH].astype(int)
        
        logger.info(f"✅ Cleaning completed: {len(df):,} rows")
        return df
    
    # Build preprocessor
    def build_preprocessor(self) -> ColumnTransformer:
        """
        Builds the sklearn preprocessor:
        - Numeric: StandardScaler (average=0, std=1)
        - Categorical: OneHotEncoder (drop='first' to avoid multicollinearity)
        
        Returns:
            ColumnTransformer ready for fit_transform
        """
        logger.info("🔧 Building preprocesor...")
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.num_cols),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), self.cat_cols),
            ],
            remainder='drop'  # Drop unlisted columns (eg. ticket_id)
        )
        return preprocessor
    
    # Train/test split
    def split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
        temporal: bool = True,
    ) -> DataSplit:
        """
        Split the dataset into train/test with stratification or temporal cut.
        """
        df = df.copy()

        # Asegurar tipo datetime y orden cronológico
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'])
            df = df.sort_values('created_at').reset_index(drop=True)

        if temporal:
            # Corte cronológico
            split_idx = int(len(df) * (1 - test_size))

            train_df = df.iloc[:split_idx]
            test_df = df.iloc[split_idx:]

            X_train = train_df[self.num_cols + self.cat_cols]
            X_test = test_df[self.num_cols + self.cat_cols]
            y_b_train = train_df[self.TARGET_BREACH]
            y_b_test = test_df[self.TARGET_BREACH]
            y_r_train = train_df[self.TARGET_RESOLUTION]
            y_r_test = test_df[self.TARGET_RESOLUTION]

            train_start = df['created_at'].iloc[0].date()
            train_end = df['created_at'].iloc[split_idx - 1].date()
            test_start = df['created_at'].iloc[split_idx].date()
            test_end = df['created_at'].iloc[-1].date()

            logger.info(
                f"📅 Split temporal: train={train_start} → {train_end}, test={test_start} → {test_end}"
            )
        else:
            # Split aleatorio estratificado
            X = df[self.num_cols + self.cat_cols]
            y_breach = df[self.TARGET_BREACH]
            y_resolution = df[self.TARGET_RESOLUTION]

            X_train, X_test, y_b_train, y_b_test, y_r_train, y_r_test = train_test_split(
                X, y_breach, y_resolution,
                test_size=test_size,
                random_state=random_state,
                stratify=y_breach,
            )
            logger.info(f"🎲 Random split: train={len(X_train):,}, test={len(X_test):,}")

        return DataSplit(
            X_train=X_train, X_test=X_test,
            y_breach_train=y_b_train, y_breach_test=y_b_test,
            y_resolution_train=y_r_train, y_resolution_test=y_r_test,
        )

# Quick test block (only runs if you run this file)
if __name__ == "__main__":
    # Run full test pipeline
    pipeline = SupportDataPipeline("data/support_tickets.csv")
    
    df_raw = pipeline.load()
    df_clean = pipeline.clean(df_raw)
    preprocessor = pipeline.build_preprocessor()
    splits = pipeline.split(df_clean)
    
    print("\n" + "="*60)
    print("🎉 Pipeline functioning correctly")
    print("="*60)
    print(f"X_train shape: {splits.X_train.shape}")
    print(f"X_test shape:  {splits.X_test.shape}")
    print(f"\nDistribution of sla_breach in train:")
    print(splits.y_breach_train.value_counts(normalize=True))