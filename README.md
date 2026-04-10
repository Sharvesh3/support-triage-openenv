# 🚀 Support Triage Pro: An OpenEnv Agentic Benchmark

# 🚀 Support Triage Pro: An OpenEnv Agentic Benchmark

[![OpenEnv](https://img.shields.io/badge/Framework-OpenEnv-blueviolet)](https://github.com/meta-llama/openenv)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/sharvesh-33/support-triage-pro)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


**Support Triage Pro** is a high-fidelity reinforcement learning environment designed to evaluate the decision-making and tool-use capabilities of Large Language Models (LLMs). While most benchmarks measure an LLM's conversational skills, this project measures its ability to **act** within a complex, stateful IT support workflow.

## 🎯 Project Overview
This environment simulates a real-world IT Support Triage desk. It forces agents to move beyond simple text generation and perform functional tasks: navigating a knowledge base, interpreting technical error codes, and prioritizing business impact over routine requests.

### 🎮 Benchmark Tasks
The environment evaluates agents across three discrete difficulty levels:

| Level | Task | Capability Tested | Success Criteria |
| :--- | :--- | :--- | :--- |
| **Easy** | **Ticket Triage** | Schema Adherence | Correctly classify category and priority. |
| **Medium** | **KB Resolver** | **Tool-Use (ReAct)** | Query `search_kb` and apply the correct fix. |
| **Hard** | **Priority Pulse** | Logic & Urgency | Sort 3 concurrent tickets by business impact. |

---

## 🛠️ Technical Architecture

### **Environment Design**
- **Core:** Built on the `openenv.core.env_server` specification and containerized via **Docker**.
- **Reward Shaping:** Implements a dense, non-sparse reward signal. Agents receive credit for correct tool discovery (+0.1) and major credit for functional task resolution (+0.7).
- **Mathematical Stability:** All rewards and final scores are clamped to the strictly open interval **(0.001, 0.999)** to ensure deterministic grading and stability for Reinforcement Learning (RL) evaluators.

### **Telemetry & Logging**
The environment produces structured, real-time telemetry suitable for automated evaluation pipelines:
- `[START]` - Episode initialization and task selection.
- `[STEP]` - Real-time action validation and reward signals.
- `[END]` - Final success normalization and performance metrics.

---

## 📦 Getting Started

### **1. Installation**

git clone [https://github.com/Sharvesh3/support-triage-openenv.git](https://github.com/Sharvesh3/support-triage-openenv.git)
cd support-triage-openenv
pip install -r requirements.txt

### **2. Launching the Environment**

Run the server locally to begin testing agents:
python server/app.py --port 8004

### **3. Running an Agent (Example)**

from openenv.core.generic_client import GenericEnvClient, GenericAction

async with GenericEnvClient("http://localhost:8004") as env:
    # Initialize a specific task
    obs = await env.reset(task="medium")
    
    # Step 1: Use the Knowledge Base tool
    res = await env.step(GenericAction(tool="search_kb", tool_input="992"))
    
    # Step 2: Submit final resolution based on tool feedback
    final = await env.step(GenericAction(resolution="Restart background service"))
    print(f"Task Completed. Score: {final.reward}")

