import argparse
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from project_paths import HEART_CSV, RAW_DATA_DIR, resolve_data_path

COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "diagnosis",
    "disease_code",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean heart-disease dataset.")
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="Optional input file path. If omitted, uses heart.csv if available, otherwise cleve.mod.",
    )
    return parser


def load_raw_data(base_dir: Path, input_arg: str) -> tuple[pd.DataFrame, Path]:
    """Load the raw heart data; ``base_dir`` is the folder holding heart.csv / cleve.mod."""
    if input_arg:
        source_path = resolve_data_path(input_arg)
    else:
        heart_csv = base_dir / "heart.csv"
        cleve_mod = base_dir / "cleve.mod"
        source_path = heart_csv if heart_csv.exists() else cleve_mod

    if not source_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {source_path}")

    if source_path.suffix.lower() == ".csv":
        df = pd.read_csv(source_path)
        df.columns = [str(col).strip().lower() for col in df.columns]
    else:
        df = pd.read_csv(
            source_path,
            sep=r"\s+",
            comment="%",
            header=None,
            names=COLUMNS,
            engine="python",
        )

    return df, source_path


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    object_cols = cleaned.select_dtypes(include=["object", "string"]).columns.tolist()
    for col in object_cols:
        cleaned[col] = cleaned[col].astype(str).str.strip()

    cleaned = cleaned.replace("?", pd.NA)

    numeric_candidate_cols = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "target",
    ]
    for col in numeric_candidate_cols:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    # NOTE: Imputation is deliberately NOT done here.
    # Missing values are preserved so that imputation can be performed
    # inside sklearn Pipelines (per-fold) to prevent data leakage.
    # Use build_preprocessing_pipeline() in training scripts.

    if "target" not in cleaned.columns and "diagnosis" in cleaned.columns:
        cleaned["target"] = (cleaned["diagnosis"].astype(str).str.lower() == "sick").astype(int)

    if "target" in cleaned.columns:
        cleaned["target"] = pd.to_numeric(cleaned["target"], errors="coerce")
        cleaned["target"] = cleaned["target"].fillna(0).astype(int)
        # Clinical sanity check: in canonical heart disease data, exercise angina (exang)
        # and ST depression (oldpeak) correlate positively with disease.
        # Kaggle heart.csv has an inverted target (1 = healthy, 0 = sick).
        if "exang" in cleaned.columns:
            exang_num = pd.to_numeric(cleaned["exang"], errors="coerce")
            corr = exang_num.corr(cleaned["target"])
            if pd.notna(corr) and corr < -0.1:
                print(f"[*] Inverted target detected (exang correlation = {corr:.3f} < -0.1).")
                print("    Correcting target polarity: 1 = heart disease, 0 = healthy.")
                cleaned["target"] = 1 - cleaned["target"]

    return cleaned


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    output_path = HEART_CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df, source_path = load_raw_data(RAW_DATA_DIR, args.input)

    cleaned_df = clean_dataset(df)
    cleaned_df.to_csv(output_path, index=False)

    print(f"Source dataset: {source_path.name}")
    print(f"Input rows: {len(df)}")
    print(f"Cleaned rows: {len(cleaned_df)}")
    print(f"Missing values after cleaning: {int(cleaned_df.isna().sum().sum())}")
    print(f"Saved: {output_path}")


def build_preprocessing_pipeline(
    X: pd.DataFrame,
    scale_numeric: bool = True,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Build a leakage-free preprocessing pipeline.

    The returned ColumnTransformer includes SimpleImputer so that
    imputation statistics (median / most-frequent) are computed only
    on the data it is *fit* on (i.e. training data within each fold).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (used only to detect column dtypes).
    scale_numeric : bool
        If True, apply StandardScaler after imputation for numeric features.
        Set False for tree-based models that don't need scaling.

    Returns
    -------
    tuple[ColumnTransformer, list[str], list[str]]
        (preprocessor, numeric_features, categorical_features)
    """
    categorical_features = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_features = X.select_dtypes(exclude=["object", "string"]).columns.tolist()

    if scale_numeric:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
    else:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor, numeric_features, categorical_features


if __name__ == "__main__":
    main()
