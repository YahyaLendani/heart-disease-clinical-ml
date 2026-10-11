"""Shared pytest fixtures for the test suite."""

import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import HeartMLP


@pytest.fixture
def heart_dataset():
    """Load the cleaned Heart Disease dataset."""
    csv_path = PROJECT_ROOT / "cleaned_heart_disease.csv"
    if not csv_path.exists():
        pytest.skip("cleaned_heart_disease.csv not found. Run clean_data.py first.")

    df = pd.read_csv(csv_path)
    if "target" in df.columns:
        df["target"] = pd.to_numeric(df["target"], errors="coerce").fillna(0).astype(int)
    else:
        pytest.skip("Dataset missing 'target' column.")

    drop_cols = [c for c in ["diagnosis", "disease_code", "target"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df["target"]
    return X, y


@pytest.fixture
def mimic_dataset():
    """Load the MIMIC hospital mortality dataset with patient identifiers."""
    csv_path = PROJECT_ROOT / "mimic_hospital_mortality.csv"
    if not csv_path.exists():
        pytest.skip("mimic_hospital_mortality.csv not found. Run mimic_prepare_dataset.py first.")

    df = pd.read_csv(csv_path)
    y = pd.to_numeric(df["hospital_expire_flag"], errors="coerce").fillna(0).astype(int)
    subject_ids = df["subject_id"].values if "subject_id" in df.columns else None
    hadm_ids = df["hadm_id"].values if "hadm_id" in df.columns else None
    X = df.drop(columns=["hospital_expire_flag", "subject_id", "hadm_id"], errors="ignore")
    return X, y, subject_ids, hadm_ids


@pytest.fixture
def dummy_model():
    """Instantiate a HeartMLP with default dimensions for quick tests."""
    model = HeartMLP(input_dim=13, hidden_dims=(64, 32), dropout=0.25)
    return model


@pytest.fixture
def dummy_tensors():
    """Create dummy input and target tensors for model tests."""
    torch.manual_seed(42)
    batch_size = 16
    input_dim = 13
    X = torch.randn(batch_size, input_dim)
    y = torch.randint(0, 2, (batch_size,), dtype=torch.float32)
    return X, y
