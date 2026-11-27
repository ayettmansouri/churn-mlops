FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 1) Pin scikit-learn / pandas / numpy to the versions used for training (read from the model metadata)
COPY requirements-api.txt .
COPY models/model_metadata.json /tmp/model_metadata.json
RUN python -c "import json; v = json.load(open('/tmp/model_metadata.json'))['library_versions']; print('\n'.join(f'{k}=={v[k]}' for k in ('scikit-learn', 'pandas', 'numpy')))" > /tmp/pins.txt \
    && pip install -r requirements-api.txt -r /tmp/pins.txt

# 2) Application code and the trained model
COPY api/ api/
COPY models/model.joblib models/model_metadata.json models/background_sample.csv models/

# 3) Run as a non-root user; prediction logs go to /app/logs (mount a volume to keep them)
RUN useradd --create-home appuser && mkdir -p /app/logs && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
