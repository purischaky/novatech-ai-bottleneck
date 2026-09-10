"""
Detects data leakage by benchmarking full vs. leak-free operational feature subsets.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_pipeline import SupportDataPipeline


def run_leakage_diagnostics(data_path: str = "data/support_tickets.csv") -> None:
    pipeline = SupportDataPipeline(data_path)
    df = pipeline.clean(pipeline.load())

    # 1. Deterministic Rule Check
    exact_resolution_rule = (df["resolution_hours"] > df["sla_hours"]) == df["sla_breach"]
    match_queue = ((df["queue_age_hours"] > df["sla_hours"]) == df["sla_breach"]).mean()
    match_waiting = ((df["waiting_hours"] > df["sla_hours"]) == df["sla_breach"]).mean()

    print("=" * 70)
    print("DETECTION OF TARGET & FEATURE LEAKAGE")
    print("=" * 70)
    print(f"Exact Target Rule (resolution_hours > sla_hours): {exact_resolution_rule.mean():.2%}")
    print(f"Proxy Rule (queue_age_hours > sla_hours):        {match_queue:.2%}")
    print(f"Proxy Rule (waiting_hours > sla_hours):          {match_waiting:.2%}")

    # 2. Define Evaluation Scenarios
    # Real-time intake features (what an agent sees at t=0)
    intake_cat = ["category", "priority", "channel", "assigned_team"]
    lifecycle_cat = intake_cat + ["current_stage"]

    scenarios = {
        "1. Omniscient (Leaky Lifecycle)": {
            "num": ["first_response_hours", "waiting_hours", "queue_age_hours", "sla_hours"],
            "cat": lifecycle_cat,
        },
        "2. Mid-Workflow (No Queue Metrics)": {
            "num": ["first_response_hours", "sla_hours"],
            "cat": lifecycle_cat,
        },
        "3. True Ticket Intake (Zero Leakage)": {
            "num": ["sla_hours"],
            "cat": intake_cat,
        },
    }

    y = df["sla_breach"].astype(int)

    for label, config in scenarios.items():
        X = df[config["num"] + config["cat"]]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), config["num"]),
                ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), config["cat"]),
            ],
            remainder="drop",
        )

        X_train_proc = preprocessor.fit_transform(X_train)
        X_test_proc = preprocessor.transform(X_test)

        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X_train_proc, y_train)
        y_prob = clf.predict_proba(X_test_proc)[:, 1]
        auc = roc_auc_score(y_test, y_prob)

        print(f"\nScenario: {label}")
        print(f"  -> Features: {len(config['num'])} numeric, {len(config['cat'])} categorical")
        print(f"  -> ROC-AUC:  {auc:.4f}")


if __name__ == "__main__":
    run_leakage_diagnostics()