FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install CPU PyTorch and runtime dependencies
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml requirements.txt ./
RUN pip install -r requirements.txt && \
    pip install "fastapi>=0.111.0" "uvicorn>=0.30.0" "pydantic>=2.7.0"

COPY models/ ./models/
COPY clinical_cds/ ./clinical_cds/
COPY project_paths.py prepare_large_cvd_dataset.py ./
COPY artifacts/clinical_cds_bundle.pt ./artifacts/clinical_cds_bundle.pt

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v1/health')"

CMD ["uvicorn", "clinical_cds.api:app", "--host", "0.0.0.0", "--port", "8000"]
