"""
FastAPI application for the Support Triage Environment.

Endpoints (auto-created by openenv-core):
  POST /reset    – start a new episode  (body: {"task": "easy|medium|hard"})
  POST /step     – take an action
  GET  /state    – current episode state
  GET  /schema   – action / observation JSON schemas
  WS   /ws       – WebSocket persistent session
  GET  /web      – Gradio web playground (ENABLE_WEB_INTERFACE=true)
"""

from __future__ import annotations

import os

from openenv.core.env_server.web_interface import create_web_interface_app
from openenv.core.env_server.http_server import create_app

try:
    from .models import TriageAction, TriageObservation
    from .logic import SupportTriageEnvironment
except ImportError:
    from models import TriageAction, TriageObservation
    from logic import SupportTriageEnvironment


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


def start_server():
    """Helper to run the server locally for development."""
    import uvicorn
    # This matches the port used in your logic.py and inference.py
    uvicorn.run(app, host="0.0.0.0", port=8004)

if __name__ == "__main__":
    start_server()