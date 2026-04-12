"""
server/app.py — FastAPI application for Support Triage Pro.

Auto-created endpoints (from openenv-core):
  POST /reset    — start a new episode  (body: {"task": "auth_lockout|db_timeout|cascade_failure"})
  POST /step     — execute an action
  GET  /state    — current episode state
  GET  /schema   — action / observation JSON schemas
  GET  /health   — liveness check (returns {"status": "healthy"})
  WS   /ws       — WebSocket persistent session
  GET  /web      — Gradio web playground (when ENABLE_WEB_INTERFACE=true)

Manually added endpoints:
  GET  /tasks    — returns the list of available task names (required by Phase 2 grader)
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
# Task registry — single source of truth for task names
# ---------------------------------------------------------------------------

AVAILABLE_TASKS: List[str] = [
    "auth_lockout",
    "db_timeout",
    "cascade_failure",
]

# ---------------------------------------------------------------------------
# App creation
# ---------------------------------------------------------------------------

_enable_web = os.environ.get("ENABLE_WEB_INTERFACE", "true").lower() in ("1", "true", "yes")

if _enable_web:
    app = create_web_interface_app(
        SupportTriageEnvironment,
        TriageAction,
        TriageObservation,
        env_name="triage_env",
        max_concurrent_envs=4,
    )
else:
    app = create_app(
        SupportTriageEnvironment,
        TriageAction,
        TriageObservation,
        max_concurrent_envs=4,
    )

# ---------------------------------------------------------------------------
# /tasks endpoint — Phase 2 grader requirement
#
# Must return a JSON list of task name strings. The grader uses this to
# enumerate tasks, run each one, and verify scores fall in (0.0, 1.0).
# ---------------------------------------------------------------------------


@app.get(
    "/tasks",
    summary="List available tasks",
    description=(
        "Returns the list of task names this environment supports. "
        "Each task name can be passed as the 'task' parameter in /reset."
    ),
    tags=["Environment Info"],
)
async def list_tasks() -> JSONResponse:
    """
    Return the available task names.

    Response format (required by Meta grader):
        ["auth_lockout", "db_timeout", "cascade_failure"]
    """
    return JSONResponse(content=AVAILABLE_TASKS)


# NOTE: /health is already registered by openenv-core's create_app /
# create_web_interface_app. It returns {"status": "healthy"}.
# Do NOT re-register it here — duplicate route registration raises an error.

# ---------------------------------------------------------------------------
# Development entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the server locally for development. Uses PORT env var (default 7860)."""
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()