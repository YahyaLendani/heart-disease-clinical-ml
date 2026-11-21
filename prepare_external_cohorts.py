"""Fetch and clean two independent external clinical cardiovascular cohorts.

1. **UCI ID 412 — Z-Alizadeh Sani CAD Cohort (Shaheed Rajaei Cardiovascular Center, Tehran, N=303):**
   Angiography-confirmed coronary artery disease (>= 50% stenosis), providing a 5th
   international cardiology center (zero patient overlap with UCI Cleveland, Budapest,
   Zurich, or VA Long Beach) to augment Stage-2 Cardiology Workup training.
2. **Framingham Heart Study Cohort (Massachusetts, USA, N=4,172 clean rows):**
   Prospective longitudinal 10-year coronary heart disease (TenYearCHD, 15.2% incidence)
   cohort used as a gold-standard external prospective validation & ``calibrate-site``
   transfer benchmark for Stage-1 Primary Care Screening.

Uses Python stdlib ``zipfile`` + ``xml.etree.ElementTree`` to read the UCI ``.xlsx``
archive with zero extra dependencies.
"""

from __future__ import annotations

import argparse
import io
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import PROCESSED_DATA_DIR

UCI_ALIZADEH_URL = "https://archive.ics.uci.edu/static/public/412/z+alizadeh+sani.zip"
FRAMINGHAM_URL = (
    "https://raw.githubusercontent.com/GauravPadawe/Framingham-Heart-Study/master/framingham.csv"
)


def _read_xlsx_stdlib(raw_bytes: bytes) -> pd.DataFrame:
    """Parse first worksheet of an .xlsx file using Python stdlib only."""
    zf = zipfile.ZipFile(io.BytesIO(raw_bytes))
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    sst: list[str] = []
    if "xl/sharedStrings.xml" in zf.namelist():
        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
        for si in root.findall("s:si", ns):
            sst.append("".join(t.text or "" for t in si.findall(".//s:t", ns)))
    sheet = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
    rows: list[list[str | None]] = []
    for row in sheet.findall(".//s:sheetData/s:row", ns):
        vals: dict[int, str] = {}
        for c in row.findall("s:c", ns):
            ref = "".join(ch for ch in c.attrib["r"] if ch.isalpha())
            col_idx = 0
            for ch in ref:
                col_idx = col_idx * 26 + (ord(ch) - ord("A") + 1)
            t = c.attrib.get("t")
            v_el = c.find("s:v", ns)
            if v_el is None or v_el.text is None:
                continue
            val = sst[int(v_el.text)] if t == "s" else v_el.text
            vals[col_idx - 1] = val
        if vals:
            max_c = max(vals) + 1
            rows.append([vals.get(i) for i in range(max_c)])
    return pd.DataFrame(rows[1:], columns=rows[0])


def fetch_tehran_alizadeh(output_path: Path | None = None) -> Path:
    """Download and map UCI ID 412 Z-Alizadeh Sani (Tehran, N=303) to Stage-2 schema."""
    out = output_path or (PROCESSED_DATA_DIR / "cleaned_tehran_cad_303.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(UCI_ALIZADEH_URL, headers={"User-Agent": "python-clinical-ml"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        zf = zipfile.ZipFile(io.BytesIO(resp.read()))
        raw = _read_xlsx_stdlib(zf.read("Z-Alizadeh sani dataset.xlsx"))

    age = pd.to_numeric(raw["Age"], errors="coerce")
    sex = (raw["Sex"].astype(str).str.strip().str.lower() == "male").astype(int)
    typ = pd.to_numeric(raw["Typical Chest Pain"], errors="coerce").fillna(0).astype(int)
    atyp = (raw["Atypical"].astype(str).str.strip().str.upper() == "Y").astype(int)
    nonang = (raw["Nonanginal"].astype(str).str.strip().str.upper() == "Y").astype(int)
    cp = np.where(typ == 1, 1, np.where(atyp == 1, 2, np.where(nonang == 1, 3, 4)))
    trestbps = pd.to_numeric(raw["BP"], errors="coerce").clip(70.0, 240.0)
    # Friedewald total cholesterol estimate: LDL + HDL + TG / 5
    ldl = pd.to_numeric(raw["LDL"], errors="coerce")
    hdl = pd.to_numeric(raw["HDL"], errors="coerce")
    tg = pd.to_numeric(raw["TG"], errors="coerce")
    chol = (ldl + hdl + tg / 5.0).clip(100.0, 500.0)
    fbs = (pd.to_numeric(raw["FBS"], errors="coerce") > 120.0).astype(int)
    lvh = (raw["LVH"].astype(str).str.strip().str.upper() == "Y").astype(int)
    stt = (
        (pd.to_numeric(raw["St Depression"], errors="coerce").fillna(0) > 0)
        | (pd.to_numeric(raw["St Elevation"], errors="coerce").fillna(0) > 0)
        | (pd.to_numeric(raw["Tinversion"], errors="coerce").fillna(0) > 0)
    ).astype(int)
    restecg = np.where(lvh == 1, 2, np.where(stt == 1, 1, 0))
    exang = (raw["Exertional CP"].astype(str).str.strip().str.upper() == "Y").astype(int)
    target = (raw["Cath"].astype(str).str.strip().str.lower() == "cad").astype(int)

    cleaned = pd.DataFrame({
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": np.nan,
        "exang": exang,
        "oldpeak": np.nan,
        "slope": np.nan,
        "ca": np.nan,
        "thal": np.nan,
        "target": target,
        "center": "tehran_alizadeh",
    })
    cleaned.to_csv(out, index=False)
    print(f"Saved Tehran Z-Alizadeh Sani CAD cohort -> {out} (N={len(cleaned)}, CAD={int(target.sum())})")
    return out


def fetch_framingham_prospective(output_path: Path | None = None) -> Path:
    """Download and clean the 4,240-patient Framingham Heart Study prospective cohort."""
    out = output_path or (PROCESSED_DATA_DIR / "cleaned_framingham_4k.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    fram = pd.read_csv(FRAMINGHAM_URL)
    fram = fram.dropna(subset=["age", "male", "sysBP", "diaBP", "BMI", "totChol", "TenYearCHD"]).copy()
    fram["glucose"] = fram["glucose"].fillna(fram["glucose"].median())
    fram["currentSmoker"] = fram["currentSmoker"].fillna(0).astype(int)

    sbp = fram["sysBP"].clip(70.0, 245.0).values
    dbp = np.minimum(fram["diaBP"].clip(45.0, 155.0).values, sbp - 10.0)
    sex = fram["male"].astype(int).values
    height = np.where(sex == 1, 175.0, 162.0)
    bmi = fram["BMI"].clip(14.0, 60.0).values
    weight = bmi * ((height / 100.0) ** 2)
    chol_tier = np.where(fram["totChol"].values < 200.0, 0, np.where(fram["totChol"].values < 240.0, 1, 2))
    gluc_tier = np.where(fram["glucose"].values < 100.0, 0, np.where(fram["glucose"].values < 126.0, 1, 2))
    pp = sbp - dbp
    map_bp = dbp + pp / 3.0

    cleaned = pd.DataFrame({
        "age_years": fram["age"].values.astype(float),
        "sex": sex,
        "height": height,
        "weight": weight,
        "bmi": bmi,
        "ap_hi": sbp,
        "ap_lo": dbp,
        "pulse_pressure": pp,
        "mean_arterial_pressure": map_bp,
        "cholesterol": chol_tier,
        "gluc": gluc_tier,
        "smoke": fram["currentSmoker"].values.astype(int),
        "alco": np.zeros(len(fram), dtype=int),
        "active": np.ones(len(fram), dtype=int),
        "target": fram["TenYearCHD"].astype(int).values,
    })
    cleaned.to_csv(out, index=False)
    print(
        f"Saved Framingham Prospective Cohort -> {out} "
        f"(N={len(cleaned)}, 10yr CHD={cleaned['target'].mean()*100:.1f}%)"
    )
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch external Tehran CAD (303) & Framingham (4.2k) cohorts.")
    parser.add_argument("--skip-tehran", action="store_true")
    parser.add_argument("--skip-framingham", action="store_true")
    args = parser.parse_args(argv)
    if not args.skip_tehran:
        fetch_tehran_alizadeh()
    if not args.skip_framingham:
        fetch_framingham_prospective()


if __name__ == "__main__":
    main()
