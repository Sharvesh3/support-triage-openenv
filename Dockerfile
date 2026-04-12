# Dockerfile — Support Triage Pro
#
# Pin strategy:
#   openenv-core>=0.2.2   needs websockets>=15.0.1
#   gradio>=5.25.0,<6.0.0 needs websockets<16
#   → websockets 15.x satisfies both
#
# Port: 7860 (HF Spaces requirement)

FROM python:3.11-slim

WORKDIR /app

# System deps + uv
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl git && \
    rm -rf /var/lib/apt/lists/* && \
    pip install uv --no-cache-dir

# Copy project (pair with .dockerignore to exclude .venv/__pycache__/.git)
COPY . .

# Install pinned dependencies
RUN uv pip install --system \
    "openenv-core>=0.2.2" \
    "openai>=1.0.0" \
    "uvicorn[standard]>=0.29.0" \
    "fastapi>=0.110.0" \
    "gradio>=5.25.0,<6.0.0" \
    "httpx>=0.27.0"

# Runtime environment
ENV ENABLE_WEB_INTERFACE=true
ENV PYTHONPATH=/app
ENV PORT=7860

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Direct uvicorn — no main.py wrapper
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]