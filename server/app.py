"""
server/app.py — FastAPI 'Master App' for Support Triage Pro.

Route priority (defined BEFORE mount so they win over Gradio):
  GET  /          → 200 {"status": "ok"}
  GET  /health    → 200 {"status": "healthy"}
  GET  /tasks     → ["auth_lockout", "db_timeout", "cascade_failure"]
  POST /reset     → served by mounted env_app
  POST /step      → served by mounted env_app
  GET  /state     → served by mounted env_app
  GET  /schema    → served by mounted env_app
  WS   /ws        → served by mounted env_app
  GET  /web       → Gradio playground (when ENABLE_WEB_INTERFACE=true)
"""

from __future__ import annotations

import os
from typing import List

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.web_interface import create_web_interface_app

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
# Build the OpenEnv sub-app
# ---------------------------------------------------------------------------

_enable_web = os.environ.get("ENABLE_WEB_INTERFACE", "true").lower() in ("1", "true", "yes")

if _enable_web:
    _env_app = create_web_interface_app(
        SupportTriageEnvironment,
        TriageAction,
        TriageObservation,
        env_name="triage_env",
        max_concurrent_envs=4,
    )
else:
    _env_app = create_app(
        SupportTriageEnvironment,
        TriageAction,
        TriageObservation,
        max_concurrent_envs=4,
    )

# ---------------------------------------------------------------------------
# Master app  — our routes are registered FIRST so they beat Gradio redirects
# ---------------------------------------------------------------------------

app = FastAPI(title="Support Triage Pro", version="1.0.0")


@app.get("/", tags=["Meta"])
async def root() -> JSONResponse:
    """Liveness root — Meta grader pings this."""
    return JSONResponse({"status": "ok"})


@app.get("/health", tags=["Meta"])
async def health() -> JSONResponse:
    """Health check — Docker HEALTHCHECK and HF Space monitor."""
    return JSONResponse({"status": "healthy"})


@app.get("/tasks", tags=["Environment Info"])
async def list_tasks() -> JSONResponse:
    """
    Return available task names.
    Required format: ["auth_lockout", "db_timeout", "cascade_failure"]
    """
    return JSONResponse(content=AVAILABLE_TASKS)


# Mount the full OpenEnv app last — it catches everything we haven't claimed.
app.mount("/", _env_app)