"""
main.py — Entry point for Support Triage Pro server.

Reads PORT from the environment (default 7860 for Hugging Face Spaces).
Used by:
  - Hugging Face Space runtime (sets PORT=7860)
  - Docker container (sets PORT via -e PORT=7860)
  - Local development (falls back to 7860)
"""

from __future__ import annotations

import os

import uvicorn
from server.app import app


def main() -> None:
    """Start the uvicorn server on the configured port."""
    # HF Spaces injects PORT=7860; locally defaults to 7860.
    # Never hardcode 8004 — that would break the Space deployment.
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting Support Triage Pro on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()