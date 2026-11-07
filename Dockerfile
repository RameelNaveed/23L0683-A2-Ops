# VisionGuard web app (Task 6.2)
# Order matters for the layer cache: things that rarely change come first.
FROM python:3.11-slim

# No .pyc files, logs flushed straight away, no pip cache inside the image.
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /srv

# A normal user, so a bug in the app cannot act as root inside the container.
RUN useradd --create-home --uid 10001 appuser

# 1) Dependencies first. This layer is only rebuilt when requirements-serve.txt changes.
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# 2) Source code last. Editing the code only rebuilds these small layers.
COPY visionguard/ visionguard/
COPY app/ app/

USER appuser
EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=4).status == 200 else 1)"

# gunicorn is a production WSGI server; one worker keeps one copy of the model in RAM.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "120", "app.wsgi:app"]
