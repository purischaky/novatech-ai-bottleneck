"""
Multitask Deep Learning Model using PyTorch.

Corrections applied:
1. Robust handling of unseen categories
2. Uncertainty weighting to balance losses (Kendall et al. 2018)
3. Evaluation at each epoch with early stopping
4. Removal of redundant linear layer for numerical features
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score,
    recall_score, roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_pipeline import SupportDataPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

EPOCHS = 50
BATCH_SIZE = 256
LEARNING_RATE = 0.001
HIDDEN_DIM = 64
DROPOUT = 0.2
EARLY_STOP_PATIENCE = 10

# Network definition
class MultitaskNet(nn.Module):
    """
    Multi-task neural network with:
    - Embeddings with a reserved slot fo unknown categories
    - Uncertainty weighting (Kendall et al. 2018)
    - No redundant linear layer for numerical features
    """

    def __init__(
        self,
        num_features: int,
        cat_cardinalities: List[int],  # +1 for unknown
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = DROPOUT,
    ):
        super(MultitaskNet, self).__init__()

        # Embeddings
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(
                num_embeddings=n_cats,  # len(encoder) + 1
                embedding_dim=min(8, (n_cats - 1) // 2 + 2)
            )
            for n_cats in cat_cardinalities
        ])
        self.cat_embedding_dim = sum(emb.embedding_dim for emb in self.cat_embeddings)

        # Shared Representation Backbone
        input_dim = num_features + self.cat_embedding_dim
        self.shared_body = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Output Heads
        self.clf_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # automatic learning
        self.log_var_clf = nn.Parameter(torch.zeros(1))
        self.log_var_reg = nn.Parameter(torch.zeros(1))

    def forward(
        self, x_num: torch.Tensor, x_cat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Direct concatenation
        cat_embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        x_cat_processed = torch.cat(cat_embedded, dim=1)
        x_combined = torch.cat([x_num, x_cat_processed], dim=1)

        shared_repr = self.shared_body(x_combined)
        clf_logits = self.clf_head(shared_repr).squeeze(-1)
        reg_outputs = self.reg_head(shared_repr).squeeze(-1)

        return clf_logits, reg_outputs

    def compute_loss(
        self,
        clf_logits: torch.Tensor,
        reg_outputs: torch.Tensor,
        y_breach: torch.Tensor,
        y_res: torch.Tensor,
        clf_criterion: nn.Module,
        reg_criterion: nn.Module,
    ) -> torch.Tensor:
        loss_clf = clf_criterion(clf_logits, y_breach)
        loss_reg = reg_criterion(reg_outputs, y_res)

        # Learned precisions
        precision_clf = torch.exp(-self.log_var_clf)
        precision_reg = torch.exp(-self.log_var_reg)

        # Total loss with adjustment for uncertainties
        loss = (
            precision_clf * loss_clf
            + precision_reg * loss_reg
            + self.log_var_clf
            + self.log_var_reg
        )
        return loss

# Data preparation
def prepare_pytorch_data(splits, cat_cols: List[str]):
    num_cols = [c for c in splits.X_train.columns if c not in cat_cols]

    # Normalize Continuous Inputs
    num_mean = splits.X_train[num_cols].mean().values.astype(np.float32)
    num_std = (splits.X_train[num_cols].std() + 1e-8).values.astype(np.float32)

    X_train_num = ((splits.X_train[num_cols].values - num_mean) / num_std).astype(np.float32)
    X_test_num = ((splits.X_test[num_cols].values - num_mean) / num_std).astype(np.float32)

    # B) Encode Categoricals with Fallback Index
    cat_cardinalities = []
    X_train_cat_list = []
    X_test_cat_list = []
    cat_encoders = {}

    for col in cat_cols:
        unique_vals = splits.X_train[col].astype(str).unique()
        encoder = {val: i for i, val in enumerate(unique_vals)}
        cat_encoders[col] = encoder
        cat_cardinalities.append(len(encoder) + 1)

        train_encoded = splits.X_train[col].astype(str).map(encoder)
        train_encoded = train_encoded.fillna(len(encoder)).values.astype(np.int64)
        X_train_cat_list.append(train_encoded)

        test_encoded = splits.X_test[col].astype(str).map(
            lambda x: encoder.get(str(x), len(encoder))
        )
        test_encoded = test_encoded.fillna(len(encoder)).values.astype(np.int64)
        X_test_cat_list.append(test_encoded)

    X_train_cat = np.column_stack(X_train_cat_list)
    X_test_cat = np.column_stack(X_test_cat_list)

    # Targets
    y_train_breach = splits.y_breach_train.values.astype(np.float32)
    y_test_breach = splits.y_breach_test.values.astype(np.float32)

    # Normalize y_resolution to balance loss scale
    y_res_mean = float(splits.y_resolution_train.mean())
    y_res_std = float(splits.y_resolution_train.std() + 1e-8)
    y_train_res = ((splits.y_resolution_train.values - y_res_mean) / y_res_std).astype(np.float32)
    y_test_res = ((splits.y_resolution_test.values - y_res_mean) / y_res_std).astype(np.float32)

    # DataLoaders
    train_dataset = TensorDataset(
        torch.tensor(X_train_num), torch.tensor(X_train_cat),
        torch.tensor(y_train_breach), torch.tensor(y_train_res),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test_num), torch.tensor(X_test_cat),
        torch.tensor(y_test_breach), torch.tensor(y_test_res),
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    metadata = {
        'num_mean': num_mean, 'num_std': num_std,
        'y_res_mean': y_res_mean, 'y_res_std': y_res_std,
        'cat_encoders': cat_encoders,
    }
    return train_loader, test_loader, cat_cardinalities, metadata

# Training with Early Stopping
def train_model(model, train_loader, test_loader, epochs=EPOCHS):
    """
    Evaluation at each epoch + early stopping.
    """
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    clf_criterion = nn.BCEWithLogitsLoss()
    reg_criterion = nn.MSELoss()

    best_test_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0

    logger.info(f"🚀 Training on {DEVICE} for maximun {epochs} epochs")
    logger.info(f"   Early stopping patience: {EARLY_STOP_PATIENCE}")

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_num, batch_cat, batch_y_breach, batch_y_res in train_loader:
            batch_num = batch_num.to(DEVICE)
            batch_cat = batch_cat.to(DEVICE)
            batch_y_breach = batch_y_breach.to(DEVICE)
            batch_y_res = batch_y_res.to(DEVICE)

            optimizer.zero_grad()
            clf_logits, reg_outputs = model(batch_num, batch_cat)
            loss = model.compute_loss(
                clf_logits, reg_outputs, batch_y_breach, batch_y_res,
                clf_criterion, reg_criterion
            )
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Evaluate at the end of each epoch
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_num, batch_cat, batch_y_breach, batch_y_res in test_loader:
                batch_num = batch_num.to(DEVICE)
                batch_cat = batch_cat.to(DEVICE)
                batch_y_breach = batch_y_breach.to(DEVICE)
                batch_y_res = batch_y_res.to(DEVICE)

                clf_logits, reg_outputs = model(batch_num, batch_cat)
                loss = model.compute_loss(
                    clf_logits, reg_outputs, batch_y_breach, batch_y_res,
                    clf_criterion, reg_criterion
                )
                test_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)

        # Log with learned uncertainties
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"Train: {avg_train_loss:.4f} | Test: {avg_test_loss:.4f} | "
                f"σ²_clf: {model.log_var_clf.item():.3f} | σ²_reg: {model.log_var_reg.item():.3f}"
            )

        # Early stopping
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            logger.info(f"⏹️ Early stopping in epoch {epoch+1} (no improvement by {EARLY_STOP_PATIENCE} epochs)")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"✅ Best restored model (test loss: {best_test_loss:.4f})")

    return model

# Evaluation (For interpretable metrics)
def evaluate_model(model, test_loader, y_res_mean, y_res_std):
    """Evaluates the model and denormalizes the regression for interpretable metrics"""
    model.to(DEVICE)
    model.eval()

    all_clf_logits, all_reg_outputs = [], []
    all_y_breach, all_y_res = [], []

    with torch.no_grad():
        for batch_num, batch_cat, batch_y_breach, batch_y_res in test_loader:
            batch_num = batch_num.to(DEVICE)
            batch_cat = batch_cat.to(DEVICE)
            clf_logits, reg_outputs = model(batch_num, batch_cat)
            all_clf_logits.append(clf_logits.cpu().numpy())
            all_reg_outputs.append(reg_outputs.cpu().numpy())
            all_y_breach.append(batch_y_breach.numpy())
            all_y_res.append(batch_y_res.numpy())

    clf_logits = np.concatenate(all_clf_logits)
    reg_outputs_norm = np.concatenate(all_reg_outputs)
    y_breach = np.concatenate(all_y_breach)
    y_res_norm = np.concatenate(all_y_res)

    # Denormalize regression for metrics in real hours
    reg_outputs = reg_outputs_norm * y_res_std + y_res_mean
    y_res = y_res_norm * y_res_std + y_res_mean

    clf_probs = 1 / (1 + np.exp(-clf_logits))
    clf_preds = (clf_probs >= 0.5).astype(int)

    clf_metrics = {
        'accuracy': accuracy_score(y_breach, clf_preds),
        'precision': precision_score(y_breach, clf_preds, zero_division=0),
        'recall': recall_score(y_breach, clf_preds, zero_division=0),
        'f1': f1_score(y_breach, clf_preds, zero_division=0),
        'roc_auc': roc_auc_score(y_breach, clf_probs),
    }

    reg_metrics = {
        'mae': mean_absolute_error(y_res, reg_outputs),
        'rmse': np.sqrt(mean_squared_error(y_res, reg_outputs)),
        'r2': r2_score(y_res, reg_outputs),
    }

    return {'classification': clf_metrics, 'regression': reg_metrics}

# Load and Save
def save_model(model, output_dir, cat_cardinalities, metadata):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), output_path / "dl_multitask.pth")
    full_metadata = {
        **metadata,
        'cat_cardinalities': cat_cardinalities,
        'hidden_dim': HIDDEN_DIM,
        'dropout': DROPOUT,
    }
    joblib.dump(full_metadata, output_path / "dl_metadata.joblib")
    logger.info(f"💾 Modelo guardado en {output_path}/")

# Entry Point
if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("Training multitask neural network")
    logger.info("=" * 70)

    pipeline = SupportDataPipeline(
        "data/support_tickets.csv",
        num_cols=['sla_hours'],
        cat_cols=['category', 'priority', 'channel'],
    )
    df_raw = pipeline.load()
    df_clean = pipeline.clean(df_raw)
    splits = pipeline.split(df_clean, test_size=0.2)

    train_loader, test_loader, cat_cardinalities, metadata = prepare_pytorch_data(
        splits, pipeline.cat_cols
    )

    num_features = len(metadata['num_mean'])
    logger.info(f"📊 Numeric features: {num_features}")
    logger.info(f"📊 Categorical features: {len(pipeline.cat_cols)} (Cardinalities with unknown: {cat_cardinalities})")

    model = MultitaskNet(
        num_features=num_features,
        cat_cardinalities=cat_cardinalities,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    )

    logger.info(f"\n🧠 Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    trained_model = train_model(model, train_loader, test_loader, epochs=EPOCHS)

    logger.info("\n📈 Evaluating model on test set...")
    results = evaluate_model(
        trained_model, test_loader,
        metadata['y_res_mean'], metadata['y_res_std']
    )

    logger.info("📊 Classification metrics (SLA Breach):")
    for k, v in results['classification'].items():
        logger.info(f"   {k:>10}: {v:.4f}")

    logger.info("📊 Regression Metrics (Resolution Hours):")
    for k, v in results['regression'].items():
        logger.info(f"   {k:>10}: {v:.4f}")

    save_model(trained_model, "outputs/models/dl_multitask", cat_cardinalities, metadata)

    # Guardar reporte
    report_path = Path("outputs/reports/dl_multitask_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 🧠 Deep Learning Multitask Report \n\n")
        f.write(f"**Model:** MultitaskNet with uncertainty weighting\n")
        f.write(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        f.write("## 🎯 Classificationn\n")
        f.write("| Metric | Value |\n|---------|-------|\n")
        for k, v in results['classification'].items():
            f.write(f"| {k} | {v:.4f} |\n")

        f.write("\n## 📈 Regression\n\n")
        f.write("| Metric | Value |\n|---------|-------|\n")
        for k, v in results['regression'].items():
            f.write(f"| {k} | {v:.4f} |\n")

    logger.info(f"📄 Report save in: {report_path}")
    logger.info("\n" + "=" * 70)
    logger.info("✅ Training complete")
    logger.info("=" * 70)