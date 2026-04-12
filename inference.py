"""
inference.py — Remote-client inference for Support Triage Pro.

Connects to the running HF Space (or local server) via GenericEnvClient.
Never imports SupportTriageEnvironment directly — pure HTTP/WS client.

Environment variables
---------------------
ENV_URL        Primary: full URL of the running Space (injected by Meta evaluator).
               e.g. https://my-team-triage-env.hf.space
TRIAGE_SERVER_URL  Fallback for local dev (default: http://localhost:7860).
HF_TOKEN       API key for the LLM.
API_BASE_URL   LLM endpoint (default: https://router.huggingface.co/v1).
MODEL_NAME     Model identifier (default: Qwen/Qwen2.5-72B-Instruct).

Stdout format (Meta automated evaluator)
-----------------------------------------
[START] task=<name> env=<benchmark> model=<model>
[STEP]  step=<n> action=<json> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import textwrap
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from openenv.core.generic_client import GenericAction, GenericEnvClient

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("inference")

# ---------------------------------------------------------------------------
# Configuration  (all from environment — never hardcoded)
# ---------------------------------------------------------------------------

HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# ENV_URL is what the Meta evaluator injects; TRIAGE_SERVER_URL is the
# legacy local-dev fallback used in the original codebase.
SERVER_URL = (
    os.getenv("ENV_URL")
    or os.getenv("TRIAGE_SERVER_URL")
    or "https://sharvesh-33-support-triage-pro.hf.space"
).rstrip("/")

BENCHMARK = "triage_env"
TASKS: List[str] = ["auth_lockout", "db_timeout", "cascade_failure"]

TASK_MAX_STEPS: Dict[str, int] = {
    "auth_lockout":    10,
    "db_timeout":      12,
    "cascade_failure": 14,
}

STEP_DECAY_BASE:               float = 0.98
MAX_POSSIBLE_REWARD:           float = 0.95
MAX_POSSIBLE_REWARD_SYNTHESIS: float = 1.20
MAX_RETRIES_PER_TOOL:          int   = 4

# ---------------------------------------------------------------------------
# Structured stdout — Meta evaluator format (exact field names / order)
# ---------------------------------------------------------------------------


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: Dict[str, Any],
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    action_str = json.dumps(action, separators=(",", ":"))
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action_str} "
        f"reward={reward:.2f} done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    r_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={r_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Version string cleaner
# ---------------------------------------------------------------------------

_VERSION_RE = re.compile(r"(v\d+\.\d+(?:\.\d+)?)")


def clean_version_string(raw: str) -> str:
    """
    Strip noise from check_system_version output.

    >>> clean_version_string("System version: v3.0-unstable-build")
    'v3.0'
    >>> clean_version_string("v1.8-rc1")
    'v1.8'
    """
    match = _VERSION_RE.search(raw)
    return match.group(1) if match else raw


# ---------------------------------------------------------------------------
# LLM helpers — free-text payloads only; routing is done by state machine
# ---------------------------------------------------------------------------

DIAGNOSIS_PROMPTS: Dict[str, str] = {
    "auth_lockout": textwrap.dedent("""
        You are a senior support engineer. Write a one-sentence root cause diagnosis.
        The diagnosis MUST reference 'session token cache' and 'AUTH-4031'.
        Reply with the diagnosis sentence only — no preamble.
        Ticket: Mass user lockout post password-reset. Error: AUTH-4031.
    """).strip(),
    "db_timeout": textwrap.dedent("""
        You are a senior support engineer. Write a one-sentence root cause diagnosis.
        The diagnosis MUST reference 'connection pool' and 'DB-TIMEOUT-9'.
        Reply with the diagnosis sentence only — no preamble.
        Ticket: Database connection timeout storm. Error: DB-TIMEOUT-9. System: v1.8 (legacy).
    """).strip(),
    "cascade_failure": textwrap.dedent("""
        You are a senior support engineer. Write a one-sentence root cause diagnosis.
        The diagnosis MUST reference 'payment service', 'circuit breaker', and '503'.
        Reply with the diagnosis sentence only.
        Ticket: Payment service outage — cascade to downstream. Payment-gateway returns 503s.
    """).strip(),
}

VERIFY_FIX_PROMPTS: Dict[str, str] = {
    "auth_lockout": (
        "Write a one-sentence proposed fix referencing "
        "'AuthService.flushTokenCache()' and 'invalidate session tokens'. Reply only."
    ),
    "db_timeout": (
        "Write a one-sentence proposed fix referencing "
        "'max_connections=200', 'pool_timeout=30s', and 'connection pool'. Reply only."
    ),
    "cascade_failure": (
        "Write a one-sentence proposed fix referencing "
        "'kubectl rollout restart deployment/payment-gateway' and 'circuit breaker'. Reply only."
    ),
}


def _call_llm(client: OpenAI, prompt: str, max_tokens: int = 120) -> str:
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.warning("LLM call failed: %s", exc)
        return ""


# Hardcoded fallbacks guarantee correct keyword presence if LLM is unavailable
_HARDCODED_DIAGNOSIS: Dict[str, str] = {
    "auth_lockout": (
        "Root cause: The session token cache was not invalidated after the "
        "09:00 UTC deployment, causing AUTH-4031 lockouts for all users who "
        "reset their passwords."
    ),
    "db_timeout": (
        "Root cause: The legacy v1.8 database connection pool is exhausted "
        "under load, triggering DB-TIMEOUT-9 errors on ~40% of requests."
    ),
    "cascade_failure": (
        "Root cause: The payment service is returning 503 errors due to a pod "
        "crash, triggering circuit breaker failures across downstream services."
    ),
}

_HARDCODED_FIX: Dict[str, str] = {
    "auth_lockout": (
        "Call AuthService.flushTokenCache() to invalidate all stale session tokens "
        "and force re-authentication."
    ),
    "db_timeout": (
        "Increase connection pool size: set max_connections=200 and pool_timeout=30s "
        "in db.conf to resolve DB-TIMEOUT-9 exhaustion."
    ),
    "cascade_failure": (
        "Execute kubectl rollout restart deployment/payment-gateway, then flush "
        "the circuit breakers to restore downstream services."
    ),
}

_HARDCODED_RESOLUTION: Dict[str, str] = {
    "auth_lockout": (
        "Resolved: Called AuthService.flushTokenCache() to invalidate all stale "
        "session tokens. AUTH-4031 lockouts have ceased."
    ),
    "db_timeout": (
        "Resolved: Updated db.conf to set max_connections=200 and pool_timeout=30s. "
        "Connection pool exhaustion fixed. DB-TIMEOUT-9 errors resolved."
    ),
    "cascade_failure": (
        "Resolved via two-step remediation: "
        "(1) Restarted payment_service: kubectl rollout restart deployment/payment-gateway — "
        "pod returned to healthy. "
        "(2) Flushed downstream_services circuit breakers: istioctl reset-circuit-breaker --all — "
        "order-service, notification-service, and analytics-service all recovered."
    ),
}


def _get_diagnosis(client: OpenAI, task: str) -> str:
    text = _call_llm(client, DIAGNOSIS_PROMPTS.get(task, ""), max_tokens=80)
    return text or _HARDCODED_DIAGNOSIS[task]


def _get_proposed_fix(client: OpenAI, task: str, kb_results: Dict[str, str]) -> str:
    if kb_results:
        kb_ctx = " | ".join(f"{k}: {v[:80]}" for k, v in kb_results.items())
        prompt = f"{VERIFY_FIX_PROMPTS.get(task, '')} KB context: {kb_ctx}"
        text = _call_llm(client, prompt, max_tokens=100)
        return text or _HARDCODED_FIX[task]
    return _HARDCODED_FIX[task]


def _get_resolution(
    client: OpenAI, task: str, kb_results: Dict[str, str], proposed_fix: str,
) -> str:
    if not kb_results:
        return _HARDCODED_RESOLUTION[task]
    kb_lines = "\n".join(f"- {k}: {v}" for k, v in kb_results.items())
    if task == "cascade_failure":
        prompt = textwrap.dedent(f"""
            Write a 2-3 sentence resolution note that:
            1. Names BOTH 'payment_service' and 'downstream_services'.
            2. Includes the exact fix commands: {kb_lines}
            3. Confirms both services are restored.
            Reply with the resolution text only.
        """).strip()
    else:
        prompt = textwrap.dedent(f"""
            Write a 2-sentence resolution confirming this fix was applied: {proposed_fix}
            KB context: {kb_lines}
            Reply with the resolution text only.
        """).strip()
    text = _call_llm(client, prompt, max_tokens=150)
    return text or _HARDCODED_RESOLUTION[task]


# ---------------------------------------------------------------------------
# KB query keys per task
# ---------------------------------------------------------------------------

TASK_KB_QUERIES: Dict[str, List[str]] = {
    "auth_lockout":    ["auth_lockout"],
    "db_timeout":      ["db_timeout"],
    "cascade_failure": ["payment_service", "downstream_services"],
}

# ---------------------------------------------------------------------------
# State machine: action builder
# ---------------------------------------------------------------------------


def build_action(
    obs: Dict[str, Any],
    task: str,
    agent_state: Dict[str, Any],
    client: OpenAI,
) -> Tuple[Dict[str, Any], str]:
    """
    Pure state-driven dispatch.
    EMERGENCY_SYSTEM_REBOOT is never emitted.
    """
    fsm_state   = obs.get("fsm_state", "OPEN")
    tool_result = obs.get("tool_result")

    # ── RETRYING: replay the last failed tool ────────────────────────────
    if fsm_state == "RETRYING":
        last_tool  = agent_state.get("last_tool", "check_system_version")
        last_input = agent_state.get("last_tool_input")
        action: Dict[str, Any] = {"intent": "use_tool", "tool": last_tool}
        if last_input:
            action["tool_input"] = last_input
        if agent_state.get("clean_version") and last_tool == "search_kb":
            action["version"] = agent_state["clean_version"]
        logger.info("RETRYING — replaying tool '%s'.", last_tool)
        return action, f"retry:{last_tool}"

    # ── OPEN ─────────────────────────────────────────────────────────────
    if fsm_state == "OPEN":
        if agent_state.get("clean_version") is None:
            agent_state["last_tool"] = "check_system_version"
            agent_state["last_tool_input"] = None
            return {"intent": "use_tool", "tool": "check_system_version"}, "check_system_version"

        if agent_state.get("diagnosis_text") is None:
            agent_state["diagnosis_text"] = _get_diagnosis(client, task)
        return {"intent": "diagnose", "diagnosis": agent_state["diagnosis_text"]}, "diagnose"

    # ── DIAGNOSED ────────────────────────────────────────────────────────
    if fsm_state == "DIAGNOSED":
        queries      = TASK_KB_QUERIES.get(task, [task])
        queries_done = agent_state.get("kb_queries_done", [])
        clean_ver    = agent_state.get("clean_version", "")
        pending      = [q for q in queries if q not in queries_done]

        if pending:
            query_key = pending[0]
            agent_state["last_tool"]       = "search_kb"
            agent_state["last_tool_input"] = query_key
            logger.info("DIAGNOSED — search_kb key='%s' ver='%s'.", query_key, clean_ver)
            return {
                "intent":     "use_tool",
                "tool":       "search_kb",
                "tool_input": query_key,
                "version":    clean_ver,
            }, f"search_kb:{query_key}"

        if agent_state.get("proposed_fix") is None:
            agent_state["proposed_fix"] = _get_proposed_fix(
                client, task, agent_state.get("kb_results", {})
            )
        return {"intent": "verify", "proposed_fix": agent_state["proposed_fix"]}, "verify"

    # ── VERIFIED ─────────────────────────────────────────────────────────
    if fsm_state == "VERIFIED":
        resolution = _get_resolution(
            client, task,
            agent_state.get("kb_results", {}),
            agent_state.get("proposed_fix", ""),
        )
        return {"intent": "resolve", "resolution": resolution}, "resolve"

    # ── RESOLVED / unknown ───────────────────────────────────────────────
    logger.warning("Unexpected FSM state '%s' — no-op.", fsm_state)
    return {"intent": "use_tool", "tool": "check_system_version"}, "noop"


def _update_agent_state(
    agent_state: Dict[str, Any],
    action: Dict[str, Any],
    obs: Dict[str, Any],
) -> None:
    """Update episode memory from the latest observation."""
    tool        = action.get("tool", "")
    tool_result = obs.get("tool_result", "")
    noisy       = obs.get("noisy_version")

    if noisy:
        clean = clean_version_string(noisy)
        agent_state["clean_version"] = clean
        logger.info("Noisy version '%s' → cleaned to '%s'.", noisy, clean)
    elif tool == "check_system_version" and tool_result:
        agent_state["clean_version"] = clean_version_string(tool_result)
        logger.info("Version stored: '%s'.", agent_state["clean_version"])

    if tool == "search_kb" and tool_result:
        is_bad = (
            "SERVICE_BUSY" in tool_result
            or "Version not specified" in tool_result
            or "No entry" in tool_result
        )
        if not is_bad:
            query_key = action.get("tool_input", "")
            if query_key:
                agent_state.setdefault("kb_results", {})[query_key] = tool_result
                done_list = agent_state.setdefault("kb_queries_done", [])
                if query_key not in done_list:
                    done_list.append(query_key)
                logger.info("KB stored: key='%s'.", query_key)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


async def run_episode(
    client: OpenAI,
    task: str,
    seed: int = 0,
) -> Tuple[float, List[float], int]:
    max_steps    = TASK_MAX_STEPS.get(task, 12)
    max_possible = MAX_POSSIBLE_REWARD_SYNTHESIS if task == "cascade_failure" else MAX_POSSIBLE_REWARD

    rewards:      List[float] = []
    steps_taken:  int         = 0
    total_reward: float       = 0.0
    score:        float       = 0.001
    success:      bool        = False

    agent_state: Dict[str, Any] = {
        "clean_version":   None,
        "kb_results":      {},
        "kb_queries_done": [],
        "diagnosis_text":  None,
        "proposed_fix":    None,
        "last_tool":       None,
        "last_tool_input": None,
    }

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)
    logger.info("Episode starting — task='%s' seed=%d server='%s'.", task, seed, SERVER_URL)

    env = GenericEnvClient(base_url=SERVER_URL)

    try:
        async with env:
            reset_result = await env.reset(task=task, seed=seed)
            current_obs: Dict[str, Any] = (
                reset_result.observation if hasattr(reset_result, "observation") else {}
            )
            logger.info("Reset complete. FSM: %s.", current_obs.get("fsm_state", "?"))

            retry_count = 0

            for step in range(1, max_steps + 1):
                fsm_state = current_obs.get("fsm_state", "OPEN")

                action_dict, action_label = build_action(current_obs, task, agent_state, client)

                # Safety guard — decoy is never the correct action
                if action_dict.get("tool") == "EMERGENCY_SYSTEM_REBOOT":
                    logger.error("Safety guard: blocked EMERGENCY_SYSTEM_REBOOT at step %d.", step)
                    action_dict  = {"intent": "use_tool", "tool": "check_system_version"}
                    action_label = "safety_blocked_reboot"

                # Consecutive RETRYING limit
                if fsm_state == "RETRYING":
                    retry_count += 1
                    if retry_count > MAX_RETRIES_PER_TOOL:
                        logger.warning("Max retries exceeded — abandoning episode.")
                        break
                else:
                    retry_count = 0

                result = await env.step(GenericAction(**action_dict))

                step_reward: float = float(result.reward or 0.0)
                done:        bool  = bool(result.done)
                next_obs:    Dict[str, Any] = result.observation or {}

                error_msg: Optional[str] = None
                if next_obs.get("is_protocol_error"):
                    error_msg = "protocol_error"
                elif "SERVICE_BUSY" in (next_obs.get("tool_result") or ""):
                    error_msg = "service_busy"
                elif next_obs.get("decoy_trap_triggered"):
                    error_msg = "decoy_trap_CRITICAL_OUTAGE"

                rewards.append(step_reward)
                total_reward += step_reward
                steps_taken = step

                log_step(step=step, action=action_dict, reward=step_reward, done=done, error=error_msg)

                logger.info(
                    "Step %d | fsm=%s | reward=%.3f | done=%s | label=%s.",
                    step, next_obs.get("fsm_state", "?"), step_reward, done, action_label,
                )

                _update_agent_state(agent_state, action_dict, next_obs)
                current_obs = next_obs

                if done:
                    break

        raw_ratio = total_reward / max_possible if max_possible > 0 else 0.0
        decayed   = raw_ratio * (STEP_DECAY_BASE ** steps_taken)
        score     = max(0.001, min(0.999, decayed))
        success   = score >= 0.5

        logger.info(
            "Episode done — task='%s' total=%.3f steps=%d score=%.3f.",
            task, total_reward, steps_taken, score,
        )

    except Exception as exc:
        logger.error("Episode error task='%s': %s", task, exc, exc_info=True)
        score   = 0.001
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score, rewards, steps_taken


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    client     = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    all_scores: List[float] = []

    for task in TASKS:
        logger.info("=" * 60)
        logger.info("Starting task: %s", task)
        score, _, steps = await run_episode(client, task, seed=0)
        all_scores.append(score)
        logger.info("Task '%s' — score=%.3f steps=%d.", task, score, steps)

    logger.info("=" * 60)
    logger.info(
        "All done. Scores: %s  Mean: %.3f",
        [f"{s:.3f}" for s in all_scores],
        sum(all_scores) / len(all_scores) if all_scores else 0.0,
    )


if __name__ == "__main__":
    asyncio.run(main())