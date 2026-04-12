from __future__ import annotations
import os
import sys

# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from typing import List
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from openenv.core.env_server.http_server import create_app

try:
    from .models import TriageAction, TriageObservation
    from .logic import SupportTriageEnvironment
except ImportError:
    from models import TriageAction, TriageObservation
    from logic import SupportTriageEnvironment

# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------
AVAILABLE_TASKS: List[str] = [
    "auth_lockout",
    "db_timeout",
    "cascade_failure",
]

# ---------------------------------------------------------------------------
# 1. Create the OpenEnv sub-app
# ---------------------------------------------------------------------------
_env_app = create_app(
    SupportTriageEnvironment,
    TriageAction,
    TriageObservation,
    max_concurrent_envs=4,
)

# ---------------------------------------------------------------------------
# 2. Master app instance
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Support Triage Pro",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# 3. Define "Validator" routes FIRST (High Priority)
# ---------------------------------------------------------------------------
@app.get("/", tags=["Meta"])
async def root() -> JSONResponse:
    """Root liveness — HF Space iframe and Meta grader ping."""
    return JSONResponse(content={"status": "ok"})

@app.get("/health", tags=["Meta"])
async def health() -> JSONResponse:
    """Docker HEALTHCHECK and HF Space monitor."""
    return JSONResponse(content={"status": "healthy"})

@app.get("/tasks", tags=["Environment Info"])
async def list_tasks() -> JSONResponse:
    """Meta Phase 2 grader task enumeration."""
    return JSONResponse(content=AVAILABLE_TASKS)

# ---------------------------------------------------------------------------
# 4. Mount the OpenEnv app at the ROOT last (Catch-all)
# This handles /reset, /step, /state, and critically, /ws WITHOUT rewriting.
# ---------------------------------------------------------------------------
app.mount("/", _env_app)


def main():
    """Standardized entry point for the OpenEnv validator."""
    import uvicorn
    # Note: we use the string "server.app:app" to support hot-reloading if needed
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()