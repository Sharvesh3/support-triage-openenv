# Support Triage Environment

An OpenEnv benchmark where an LLM agent handles IT support tickets.

## Tasks

| Task   | Description                                   | Max Reward |
|--------|-----------------------------------------------|-----------|
| easy   | Classify a login ticket (Auth / High)          | 0.7        |
| medium | Resolve Error 992 using `search_kb` tool       | 0.8        |
| hard   | Triage 3 tickets in correct priority order     | 0.8        |

## Quick start

```python
from openenv.core.generic_client import GenericEnvClient, GenericAction

async with GenericEnvClient("http://localhost:8004") as env:
    obs = await env.reset(task="medium")
    obs = await env.step(GenericAction(tool="search_kb", tool_input="992"))
    obs = await env.step(GenericAction(resolution="Restart background service"))
```