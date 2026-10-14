FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
# CPU-only PyTorch keeps the image small; remove the first pip line for GPU builds (use a CUDA base image).
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt
COPY . .
RUN useradd -m appuser && mkdir -p instance weights && chown -R appuser /app
USER appuser
VOLUME ["/app/instance", "/app/weights"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/healthz')" || exit 1
CMD ["gunicorn", "-c", "gunicorn.conf.py", "wsgi:app"]
