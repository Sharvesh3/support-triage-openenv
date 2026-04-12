# Dockerfile — Support Triage Pro
#
# Uses COPY . . to copy the full project in one step.
# Pair with .dockerignore to exclude .venv, __pycache__, .git, *.egg-info
#
# Port: 7860 (required by HF Spaces; driven by ENV PORT=7860)

FROM python:3.11-slim

WORKDIR /app

# System deps + uv
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl git && \
    rm -rf /var/lib/apt/lists/* && \
    pip install uv --no-cache-dir

# Copy the full project — .dockerignore filters out .venv, __pycache__, .git
COPY . .

# Install Python dependencies
RUN uv pip install --system \
    "openenv-core[core]>=0.2.2" \
    "openai>=1.0.0" \
    "uvicorn[standard]>=0.29.0" \
    "fastapi>=0.110.0" \
    "gradio>=4.0.0"

# Runtime environment
ENV ENABLE_WEB_INTERFACE=true
ENV PYTHONPATH=/app
ENV PORT=7860

# HF Spaces requirement
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# main.py reads PORT from environment
CMD ["python", "main.py"]