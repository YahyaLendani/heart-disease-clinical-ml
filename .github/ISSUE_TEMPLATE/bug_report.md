---
name: Bug Report / Reproducibility Issue
about: Report a bug, numerical discrepancy, or pipeline execution failure
title: "[BUG] "
labels: ["bug", "triage"]
assignees: ""
---

## Bug Description
<!-- A clear and concise description of the bug, exception, or unexpected metric behavior. -->

## Command Executed
<!-- Provide the exact CLI command that triggered the issue. -->
```bash
python run_pipeline.py --device cpu --xgb-device cpu
```

## Expected Behavior vs. Actual Behavior
- **Expected:** 
- **Actual (Traceback / Metric Output):**
```text
Paste full error traceback or console log here
```

## Environment Metadata
- **OS:** [e.g., Windows 11 / Ubuntu 22.04 / macOS Sonoma]
- **Python Version:** [e.g., Python 3.11.8]
- **PyTorch Version:** [e.g., 2.2.1+cu121 or CPU]
- **XGBoost Version:** [e.g., 2.0.3]
- **Hardware (`--device`):** [e.g., CPU / NVIDIA RTX GPU]
- **Dataset Used (`--dataset`):** [e.g., `cleaned_heart_disease.csv` or `mimic_hospital_mortality.csv`]

## Additional Context
<!-- Attach any relevant logs from `artifacts/` or pytest outputs. -->

