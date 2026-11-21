FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CDS_DEVICE=cpu \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2 \
    PORT=7860

WORKDIR /app

# Install minimal inference-only CPU runtime (excludes matplotlib/seaborn/shap to stay lean on 512MB cloud tiers)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
      "numpy>=1.26.0" \
      "scipy>=1.11.0" \
      "pandas>=2.1.0" \
      "scikit-learn>=1.3.0" \
      "xgboost>=2.0.0" \
      "fastapi>=0.111.0" \
      "uvicorn>=0.30.0" \
      "pydantic>=2.7.0"

COPY models/ ./models/
COPY clinical_cds/ ./clinical_cds/
COPY docs/thesis_defense_slides.html ./docs/thesis_defense_slides.html
COPY project_paths.py prepare_large_cvd_dataset.py ./

# Run as dedicated non-root UID 1000 user (compatible with Render, HF Spaces & hospital Docker)
RUN useradd --create-home --uid 1000 --shell /bin/false cdsuser && \
    mkdir -p /app/artifacts && \
    chown -R cdsuser:cdsuser /app

USER cdsuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"7860\")}/v1/health')"

# Auto-download SHA-256-verified bundles on first boot if not bind-mounted locally
CMD ["sh", "-c", "if [ ! -f /app/artifacts/clinical_cds_bundle.pt ] || [ ! -f /app/artifacts/clinical_workup_bundle.pt ]; then python -m clinical_cds.download_bundles; fi && exec uvicorn clinical_cds.api:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
