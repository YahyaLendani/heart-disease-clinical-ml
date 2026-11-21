# Silent-Mode Observational Shadow Validation Study Concept
## Local Recalibration & Conformal Triage for Cardiovascular Risk Prediction (`CardioGraph CDS`)

**Study Type:** Prospective / Retrospective Non-Interventional Observational Shadow Study (Zero Patient Risk)  
**Reporting Standards:** TRIPOD+AI & DECIDE-AI  
**Technical Lead:** Yahya Lendani (BSc Computer Science, Warsaw — `yahyalendani.me` / `github.com/YahyaLendani/heart-disease-clinical-ml`)  
**Clinical Role Sought:** Principal Investigator (Cardiology) & Co-Investigators (*Senior / Corresponding Authorship*)

---

### 1. Clinical Problem & Rationale
Machine-learning cardiovascular risk scores routinely fail silently when transferred to a new hospital due to prevalence shift and local lab differences. In our multi-center evaluation across 4 international hospitals ($N = 918$) and the prospective Framingham Heart Study ($N = 4{,}172$), zero-shot 90% conformal coverage dropped to **66.8%–80.9%** and calibration error (ECE) spiked to **0.423**.

We developed **CardioGraph CDS**, an open-source, offline 2-stage clinical decision-support prototype combining a **Graph-Augmented Clinical Transformer** (PLR periodic embeddings + UniMP masked neighbor attention) with gradient-boosted trees, sex-stratified **Mondrian Conformal Triage** (`LOW_RISK`, `HIGH_RISK`, or fail-safe `DEFER_TO_CLINICIAN`), Out-of-Distribution (OOD) detection, and a 2-second local hospital recalibration step (`calibrate-site`) that reduced Framingham ECE by **98.6%** (`0.4233 → 0.0061`) and restored 90% coverage (`66.8% → 90.9%`).

### 2. Study Objective
Evaluate whether a locked, offline 2-stage cardiovascular risk model—after 1-time local site recalibration on the hospital's own cohort—maintains reliable discrimination, sub-5% calibration error, ≥90% empirical conformal coverage, and demographic parity across biological sex in real-world Polish cardiology care.

### 3. Non-Interventional Silent-Mode Design (Zero Patient Risk)
- **Complete Clinical Blinding:** The container runs in `CDS_SHADOW_MODE=true` (`/v1/shadow/stage1` and `/v1/shadow/stage2`). Predictions are logged silently to a de-identified local file and **never displayed to treating clinicians**. Standard of care is 100% unchanged.
- **Sample Size & Timeline:** $N = 500\text{--}1{,}000$ consecutive adult outpatient or cardiology-workup encounters (retrospective extract or 3–6 month prospective shadow run).
  - *Phase A (Local Calibration Slice, first ~200 adjudicated encounters):* Refit Temperature $T^*$, Platt scaling, and Mondrian quantiles locally (`heart-pipeline calibrate-site`); lock bundle SHA-256.
  - *Phase B (Locked Evaluation Slice, remaining 300–800 encounters):* Score silently and join with blinded clinical reference standard (coronary angiography $\ge 50\%$ stenosis or cardiologist-adjudicated diagnosis).

### 4. Pre-Specified Endpoints
1. **Calibration:** Expected Calibration Error (ECE) and Brier Score before vs. after local `calibrate-site` recalibration.
2. **Uncertainty & Safety:** Empirical coverage of 90% and 85% sex-stratified Mondrian conformal prediction sets, deferral rate, and OOD alert rate.
3. **Discrimination & Clinical Utility:** Holdout ROC-AUC, PR-AUC, and Decision Curve Analysis Net Benefit vs. SCORE2 / Framingham guideline equations.
4. **Fairness:** Sex-stratified sensitivity, specificity, and false-negative rate (FNR) gap.

### 5. Hospital IT, Privacy & Burden
- **Zero Cloud / Zero Data Egress:** Runs 100% on-premise inside a non-root Docker container on a single CPU (`<10 ms/patient`, no GPU required).
- **Zero PII Stored:** Pseudonymized encounter IDs are SHA-256-hashed before audit logging.
- **Zero Engineering Burden for the Clinical Team:** Technical partner handles Docker deployment, calibration, statistical analysis, and drafting the Methods/Results sections; Clinical PI oversees protocol, *Komisja Bioetyczna* submission, clinical adjudication, and manuscript leadership.
