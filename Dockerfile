# Dockerfile — Support Triage Pro
#
# Dependency resolution:
#   openenv-core>=0.2.2   requires websockets>=15.0.1
#   gradio is NOT installed (ENABLE_WEB_INTERFACE=false — never imported)
#   websockets==15.x satisfies openenv-core with no conflict
#
# Port: 7860 (HF Spaces hard requirement)

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl git && \
    rm -rf /var/lib/apt/lists/* && \
    pip install uv --no-cache-dir

COPY . .

# Install all deps in one uv call so the resolver sees every constraint at once.
# Gradio is intentionally excluded — ENABLE_WEB_INTERFACE=false means it is
# never imported, eliminating the mount_gradio_app(theme=...) TypeError entirely.
RUN uv pip install --system \
    "openenv-core[core]>=0.2.2" \
    "openai>=1.0.0" \
    "uvicorn[standard]>=0.29.0" \
    "fastapi>=0.110.0" \
    "httpx>=0.27.0" \
    "websockets>=15.0.1,<16.0"

ENV ENABLE_WEB_INTERFACE=false
ENV PYTHONPATH=/app
ENV PORT=7860

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]