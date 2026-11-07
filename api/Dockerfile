# Model API image for Google Cloud Run (Task 7).
# The model is BUILT IN here: Cloud Run has no disk we could mount it from.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2 MODEL_PATH=/srv/model/champion.keras PORT=8080

WORKDIR /srv
RUN useradd --create-home --uid 10001 appuser

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY visionguard/ visionguard/
COPY server.py .
COPY champion.keras model/champion.keras

USER appuser
EXPOSE 8080
# Cloud Run tells the container which port to use through $PORT.
CMD exec gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 120 server:app
