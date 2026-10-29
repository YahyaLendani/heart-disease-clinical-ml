"""Central place for project directories and data-file lookup.

Data lives in ``data/raw`` (downloaded source files) and ``data/processed``
(cleaned CSVs the models train on). ``resolve_data_path`` keeps old
command lines working: a bare name such as ``cleaned_heart_disease.csv`` is
looked up in the working directory, the project root and both data folders.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

HEART_CSV = PROCESSED_DATA_DIR / "cleaned_heart_disease.csv"
HEART_COMBINED_CSV = PROCESSED_DATA_DIR / "cleaned_heart_disease_combined.csv"
MIMIC_CSV = PROCESSED_DATA_DIR / "mimic_hospital_mortality.csv"

# Primary heart benchmark: the 918-patient 4-center cohort. Its test split is
# three times larger than Cleveland's alone, which roughly halves CI widths.
# Cleveland-only runs remain available with --dataset cleaned_heart_disease.csv.
PRIMARY_HEART_DATASET = HEART_COMBINED_CSV.name
MIMIC_DISCHARGE_CSV = PROCESSED_DATA_DIR / "mimic_hospital_mortality_discharge.csv"


def resolve_data_path(value: str | Path) -> Path:
    """Return the best existing match for a data file, or the processed-data default.

    Search order: absolute path, working directory, project root,
    ``data/processed``, ``data/raw``. If nothing exists, the
    ``data/processed`` location is returned so writers have a sensible target.
    """
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [
        Path.cwd() / path,
        PROJECT_ROOT / path,
        PROCESSED_DATA_DIR / path,
        RAW_DATA_DIR / path,
        PROCESSED_DATA_DIR / path.name,
        RAW_DATA_DIR / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return PROCESSED_DATA_DIR / path.name


def resolve_output_dir(value: str | Path) -> Path:
    """Resolve an output directory relative to the project root and create it."""
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path
