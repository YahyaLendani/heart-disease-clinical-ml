"""Download and assemble the full 4-center international UCI Heart Disease dataset.

The 4 clinical centers are:
1. Cleveland Clinic Foundation, USA (cleveland)
2. Hungarian Institute of Cardiology, Budapest (hungarian)
3. University Hospital, Zurich, Switzerland (switzerland)
4. V.A. Medical Center, Long Beach, CA, USA (va_long_beach)

Total raw cohort size: 920 patients across 4 international centers.
Attributes: 14 standard clinical features + target + center/site origin.
"""

import urllib.request
from pathlib import Path

import pandas as pd

from project_paths import HEART_COMBINED_CSV, RAW_DATA_DIR

DATASET_URLS = {
    "cleveland": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    "hungarian": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data",
    "switzerland": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data",
    "va_long_beach": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data",
}

FEATURE_COLUMNS = [
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
    "num",
]


def download_raw_files() -> dict[str, Path]:
    local_paths = {}
    for site, url in DATASET_URLS.items():
        fname = f"processed.{site}.data" if site != "cleveland" else "processed.cleveland.data"
        if site == "va_long_beach":
            fname = "processed.va.data"
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        local_path = RAW_DATA_DIR / fname
        if not local_path.exists():
            print(f"Downloading {site} from {url}...")
            urllib.request.urlretrieve(url, local_path)
            print(f"  -> Saved {fname} ({local_path.stat().st_size} bytes)")
        else:
            print(f"Using local cached {fname}")
        local_paths[site] = local_path
    return local_paths


def build_combined_dataset() -> pd.DataFrame:
    local_paths = download_raw_files()
    dfs = []

    for site, path in local_paths.items():
        df = pd.read_csv(
            path,
            header=None,
            names=FEATURE_COLUMNS,
            na_values=["?", "-9.0", "-9"],
            skipinitialspace=True,
        )
        df["site"] = site
        print(f"  [{site}] loaded {len(df)} rows")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # Convert all candidate numeric features
    numeric_cols = [c for c in FEATURE_COLUMNS if c != "site"]
    for col in numeric_cols:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    # Binarize target: 0 = healthy (no disease), 1-4 = presence of heart disease
    combined["target"] = (combined["num"] > 0).astype(int)
    combined = combined.drop(columns=["num"])

    # Drop exact duplicates
    initial_count = len(combined)
    combined = combined.drop_duplicates(subset=[c for c in combined.columns if c != "site"]).reset_index(drop=True)
    dropped = initial_count - len(combined)
    if dropped > 0:
        print(f"Dropped {dropped} cross-site duplicate rows.")

    return combined


def main(argv: list[str] | None = None) -> None:
    print("=" * 60)
    print("Building 4-Center International Heart Disease Dataset (UCI)")
    print("=" * 60)

    combined_df = build_combined_dataset()

    output_csv = HEART_COMBINED_CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_csv, index=False)

    print("\nDataset generation complete!")
    print(f"Output file: {output_csv}")
    print(f"Total patients: {len(combined_df)}")
    print(f"Healthy patients (target=0): {(combined_df['target'] == 0).sum()} ({((combined_df['target'] == 0).mean()*100):.1f}%)")
    print(f"Heart disease patients (target=1): {(combined_df['target'] == 1).sum()} ({((combined_df['target'] == 1).mean()*100):.1f}%)")
    print("\nPatient distribution by clinical center:")
    for site, count in combined_df["site"].value_counts().items():
        print(f"  - {site:15s}: {count:3d} patients")
    print("\nMissing values per feature across all 918 patients:")
    missing = combined_df.isna().sum()
    for col, count in missing[missing > 0].items():
        pct = (count / len(combined_df)) * 100
        print(f"  - {col:10s}: {count:3d} missing ({pct:.1f}%)")


if __name__ == "__main__":
    main()
