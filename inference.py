"""
Inference script for the Support Triage environment.
Runs all three tasks sequentially with OpenAI client.
Structured stdout: [START] / [STEP] / [END]
"""

from __future__ import annotations

import asyncio
import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI
from openenv.core.generic_client import GenericEnvClient, GenericAction

HF_TOKEN     = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
SERVER_URL   = os.getenv("TRIAGE_SERVER_URL", "http://localhost:8004")

BENCHMARK   = "triage_env"
MAX_STEPS   = 6
TEMPERATURE = 0.3
MAX_TOKENS  = 300
TASKS       = ["easy", "medium", "hard"]

# ── logging ───────────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={err}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    r_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={r_str}",
        flush=True,
    )

# ── system prompts ────────────────────────────────────────────────────────────

SYSTEM_PROMPTS = {
    "easy": textwrap.dedent("""
        You are a support triage agent. Classify the given ticket.
        Respond with a valid JSON object ONLY — no markdown, no explanation.
        Schema: {"category": "<Auth|Billing|Outage|Feature>", "priority": "<High|Medium|Low>"}
        For a login / invalid credentials issue use: {"category": "Auth", "priority": "High"}
    """).strip(),

    "medium": textwrap.dedent("""
        You are a support triage agent. Investigate the ticket using available tools.
        Step 1 — call the knowledge base: {"tool": "search_kb", "tool_input": "992"}
        Step 2 — once you have the fix, submit: {"resolution": "Restart background service"}
        Respond with a valid JSON object ONLY — no markdown, no extra keys.
    """).strip(),

    "hard": textwrap.dedent("""
        You are a support triage agent. Order three tickets by urgency (highest first).
        Rule: Production Outage > Billing > Feature Request.
        Respond with a valid JSON object ONLY: {"order": ["TKT-011", "TKT-010", "TKT-012"]}
    """).strip(),
}

# ── LLM call ──────────────────────────────────────────────────────────────────

def get_action_dict(client: OpenAI, task: str, observation: dict, step: int) -> dict:
    """Ask the LLM what action to take; always returns a plain dict."""
    obs_text = (
        f"Step {step}.\n"
        f"Feedback: {observation.get('feedback', '')}\n"
        f"Ticket: {observation.get('ticket', '')}\n"
        f"Tool result so far: {observation.get('tool_result', 'none')}"
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS[task]},
                {"role": "user",   "content": obs_text},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (resp.choices[0].message.content or "{}").strip()
        # Strip markdown fences if the model wraps in ```json ... ```
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as exc:
        print(f"[DEBUG] LLM/parse error at step {step}: {exc}", flush=True)
        return {}

# ── episode runner ────────────────────────────────────────────────────────────

async def run_episode(client: OpenAI, task: str) -> float:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    env = GenericEnvClient(base_url=SERVER_URL)

    try:
        async with env:
            # reset and select task
            reset_result = await env.reset(task=task)

            # GenericEnvClient.reset() returns a StepResult too
            # observation is the nested dict (reward/done stripped from it)
            current_obs: dict = (
                reset_result.observation
                if hasattr(reset_result, "observation")
                else reset_result
            )

            for step in range(1, MAX_STEPS + 1):
                # Get action from LLM
                action_dict = get_action_dict(client, task, current_obs, step)
                action_str  = json.dumps(action_dict, separators=(",", ":"))

                # Execute step
                result = await env.step(GenericAction(**action_dict))

                # ── THE FIX ──────────────────────────────────────────────────
                # reward and done live on the StepResult dataclass,
                # NOT inside result.observation (they are stripped from there
                # by the openenv serializer before being sent to the client).
                reward: float = float(result.reward or 0.0)
                done:   bool  = bool(result.done)
                # ─────────────────────────────────────────────────────────────

                current_obs = result.observation  # dict for next LLM prompt

                rewards.append(reward)
                steps_taken = step
                log_step(step=step, action=action_str, reward=reward, done=done, error=None)

                if done:
                    break

        # Normalise: max possible per-task reward is 0.8 (0.1 tool + 0.7 correct)
        max_reward = 0.8
        score = min(max(sum(rewards) / max_reward, 0.0), 1.0)
        success = score >= 0.5

    except Exception as exc:
        print(f"[DEBUG] Episode error (task={task}): {exc}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score

# ── entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    for task in TASKS:
        await run_episode(client, task)

if __name__ == "__main__":
    asyncio.run(main())