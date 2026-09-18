FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install build deps (gcc for any wheels that need compiling) + CBC solver
# (PuLP's default CBC backend) + curl for HEALTHCHECK.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      gcc \
      coinor-cbc \
      coinor-libcbc-dev \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better layer caching on rebuilds)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY app /app/app

# Frontend SPA (served at / when present; see app/main.py)
COPY frontend /app/frontend

# Non-root user
RUN useradd -m -u 1000 gridwise \
 && chown -R gridwise:gridwise /app
USER gridwise

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD curl -fsS http://127.0.0.1:${PORT:-8000}/health || exit 1

ENV PORT=8000 \
    HOST=0.0.0.0 \
    LOG_LEVEL=INFO \
    LLM_PROVIDER=gemini

# Render sets $PORT automatically. Use shell form so the variable expands.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]