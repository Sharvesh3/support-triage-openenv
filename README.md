# 🚀 Support Triage Pro: An OpenEnv Benchmark

[![OpenEnv](https://img.shields.io/badge/Framework-OpenEnv-blueviolet)](https://github.com/meta-llama/openenv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

**Support Triage Pro** is a high-fidelity reinforcement learning environment designed to evaluate the decision-making capabilities of Large Language Models in technical support workflows.

An OpenEnv benchmark where an LLM agent handles IT support tickets. Most benchmarks measure an LLM's ability to chat. Support Triage Pro measures its ability to act. This environment simulates a real-world IT Support Triage desk, testing an agent's capability to use tools, reason through urgency, and adhere to strict organizational schemas.

## 🎮 The Benchmark Tasks

The environment consists of three discrete tasks designed to challenge different frontier model capabilities:

| Level | Task | Capability Tested | Success Criteria |
| :--- | :--- | :--- | :--- |
| **Easy** | **Ticket Classification** | Schema Adherence | Correctly identify `Category` and `Priority`. |
| **Medium** | **KB Investigation** | **Tool-Use (ReAct)** | Call `search_kb` and apply the fix to the resolution. |
| **Hard** | **Priority Sorting** | Logic & Urgency | Sort 3 concurrent tickets: Outage > Billing > Feature. |


## 🛠️ Technical Architecture

### **Environment Design**
- **Core:** Built on `openenv.core.env_server`, containerized via **Docker**.
- **Reward Shaping:** Implements a non-sparse reward signal. Agents receive partial credit for correct tool usage (+0.1) and major credit for task completion (+0.7).
- **Mathematical Stability:** All rewards and final scores are clamped to the strictly open interval **(0.001, 0.999)** to satisfy RL validator constraints and ensure deterministic grading.

### **Inference Pipeline**
The `inference.py` script utilizes the OpenAI SDK to interact with frontier models (e.g., Qwen 2.5 72B) via the Hugging Face Inference Router. It produces structured telemetry for automated evaluation:
- `[START]` - Episode initialization and task selection.
- `[STEP]` - Real-time action/reward logging.
- `[END]` - Final success normalization and step-count reporting.

## Quick start

```python
from openenv.core.generic_client import GenericEnvClient, GenericAction

async with GenericEnvClient("http://localhost:8004") as env:
    obs = await env.reset(task="medium")
    obs = await env.step(GenericAction(tool="search_kb", tool_input="992"))
    obs = await env.step(GenericAction(resolution="Restart background service"))
```
