# Hospital IT Security & GDPR (RODO) Architecture Brief
## CardioGraph CDS — On-Premise Silent-Mode Shadow Deployment

This 1-page technical brief is prepared for **Hospital IT Security**, **Data Protection Officers (IOD / DPO)**, and **Bioethics Committees (*Komisja Bioetyczna*)**.

---

### 1. Deployment Topology (100% Air-Gapped On-Premise)
- **Zero External Network Calls:** The Docker container (`Dockerfile` / `docker-compose.yml`) executes completely offline inside the hospital VLAN. No patient data, telemetry, or model queries ever leave the hospital perimeter.
- **Hardware Footprint:** Runs natively on CPU (**9.5–9.8 ms per patient**, ~500 MB RAM, no GPU required).
- **Non-Root Container Execution:** Runs as unprivileged user `cdsuser` inside `python:3.11-slim`.

### 2. Data Minimization & GDPR / RODO Compliance
- **Zero PII Required:** Only pseudonymized encounter tokens (e.g., `ENC-2026-00412`) and structured physiological measurements (age, sex, vitals, routine lab tiers or ECG/stress summaries) enter the service. Names, PESEL numbers, addresses, and dates of birth are never sent to the API.
- **One-Way Audit Hashing:** Before writing to the append-only audit log (`cds_audit.jsonl`), `patient_pseudo_id` is truncated-SHA-256 hashed (`hashlib.sha256(...).hexdigest()[:16]`). No raw identifier is persisted.
- **Silent-Mode Enforcement (`CDS_SHADOW_MODE=true`):** Shadow endpoints (`POST /v1/shadow/stage1`, `POST /v1/shadow/stage2`) suppress all risk probabilities and triage labels from the HTTP response (`{"status": "logged_silently"}`), guaranteeing zero accidental display in clinical workflows.

### 3. Supply-Chain & Deserialization Security
- **Zero `pickle` Deserialization:** All production model bundles (`clinical_cds_bundle.pt`, `clinical_workup_bundle.pt`) store neural weights as pure PyTorch tensors loaded strictly with `torch.load(..., weights_only=True)` and tree boosters as pure UTF-8 JSON. Arbitrary code execution via pickle is eliminated by construction.
- **Cryptographic Bundle Pinning:** Startup enforces SHA-256 verification via `CDS_EXPECTED_SHA256` and `CDS_WORKUP_EXPECTED_SHA256`, halting immediately if any model artifact is modified.
- **API Security Controls:** Optional Bearer / `X-API-Key` authentication (`CDS_API_KEY`), strict CORS allow-list (`CDS_CORS_ORIGINS`), and defensive HTTP headers (`X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`).

### 4. Integration Options (Read-Only)
1. **Nightly Pseudonymized CSV Batch *(Recommended for Pilot)*:** Hospital data warehouse exports a daily pseudonymized CSV; a local script submits batches to `POST /v1/predict/batch` or `/v1/shadow/stage1`.
2. **Read-Only HL7 FHIR R4 Mirror:** Standard LOINC-coded FHIR R4 `Bundle` resources posted to `/v1/predict/fhir` or `/v1/workup/predict/fhir`.

### 5. Verification Evidence
- **127 automated unit, leakage, concurrency, and security tests** passing in CI across Python 3.10–3.13 (`90%+` code coverage, `ruff` + `mypy` clean).
- **Source & Model Cards:** `https://github.com/YahyaLendani/heart-disease-clinical-ml` (Release `v2.0.0` with signed `SHA256SUMS.txt`).
