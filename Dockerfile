FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl git && \
    rm -rf /var/lib/apt/lists/* && \
    pip install uv --no-cache-dir

COPY pyproject.toml .
COPY server/ ./server/
COPY openenv.yaml .
COPY README.md .

RUN uv pip install --system \
    "openenv-core[core]>=0.2.2" \
    "openai>=1.0.0" \
    "uvicorn[standard]>=0.29.0" \
    "fastapi>=0.110.0" \
    "gradio>=4.0.0"

ENV ENABLE_WEB_INTERFACE=true
ENV PYTHONPATH=/app

EXPOSE 8004

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8004/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8004"]