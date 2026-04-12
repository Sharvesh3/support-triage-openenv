"""
server/logic.py — Research-Grade Adversarial Harness for Support Triage Pro.

FSM
---
OPEN → DIAGNOSED → VERIFIED → RESOLVED
RETRYING captures _pre_failure_state and restores it exactly on retry.

Grading
-------
All keyword matching is HEURISTIC-ONLY — zero LLM-as-a-judge.
Negation filtering prevents false positives:
  "not a connection pool issue" → does NOT match "connection pool".
  "the connection pool is exhausted" → matches correctly.

Reward Formula (step-decay applied in inference.py)
---------------------------------------------------
score = max(0.001, min(0.999, (R_total / MAX_R) × STEP_DECAY_BASE^steps))

Per-step components
    +0.25  correct FSM transition         (max 3 × 0.25 = 0.75)
    -0.15  protocol violation
    +0.10  correct tool call
    -0.05  search_kb without/wrong version
    +0.15  synthesis bonus (cascade_failure only)
    -0.50  EMERGENCY_SYSTEM_REBOOT on Task A/B → terminates

MAX_POSSIBLE_REWARD constants
    Non-synthesis (A, B): 3×0.25 + 2×0.10        = 0.95
    Synthesis (C):        3×0.25 + 3×0.10 + 0.15 = 1.20
"""

from __future__ import annotations

import logging
import random
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from .models import (
        ActionIntent, TicketState, ToolName,
        TriageAction, TriageObservation,
    )
except ImportError:
    from models import (
        ActionIntent, TicketState, ToolName,
        TriageAction, TriageObservation,
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reward constants (exported for inference.py)
# ---------------------------------------------------------------------------

R_PROTOCOL_ADVANCE:   float = 0.25
R_PROTOCOL_VIOLATION: float = -0.15
R_TOOL_CORRECT:       float = 0.10
R_TOOL_VERSION_MISS:  float = -0.05
R_SYNTHESIS:          float = 0.15
R_DECOY_PENALTY:      float = -0.50

STEP_DECAY_BASE:               float = 0.98
MAX_POSSIBLE_REWARD:           float = 0.95   # Tasks A, B
MAX_POSSIBLE_REWARD_SYNTHESIS: float = 1.20   # Task C

TOOL_FAILURE_RATES: Dict[str, float] = {
    ToolName.CHECK_SYSTEM_VERSION.value:    0.00,
    ToolName.SEARCH_KB.value:               0.35,
    ToolName.VERIFY_FIX.value:              0.10,
    ToolName.EMERGENCY_SYSTEM_REBOOT.value: 0.00,
}

NOISY_VERSION_RATE: float = 0.30
NOISE_SUFFIXES: List[str] = [
    "-unstable-build", "-rc1", "-dev",
    "-hotfix.2", "-pre.release", "-SNAPSHOT",
]

# ---------------------------------------------------------------------------
# Negation-aware keyword matching (heuristic-only, zero LLM)
# ---------------------------------------------------------------------------

_NEGATION_MARKERS = frozenset({
    "not", "no", "never", "without", "non",
    "isn't", "isnt", "wasn't", "wasnt",
    "aren't", "arent", "doesn't", "doesnt",
    "didn't", "didnt", "hasn't", "hasnt",
    "haven't", "havent",
    "unrelated", "unlike", "except", "excluding",
    "ruled", "out",   # catches "ruled out" as two adjacent single-word markers
})

_NEGATION_BIGRAMS = frozenset({
    "not the", "not a", "not an", "not due",
    "ruled out", "ruled-out", "no longer",
    "not related", "not caused", "not from",
    "not using", "not a",
})

_NEGATION_WINDOW: int = 5   # words before keyword to scan


def _kw_matched(text: str, keyword: str) -> bool:
    """
    Return True only if `keyword` appears in `text` AND is NOT immediately
    preceded by a negation marker within _NEGATION_WINDOW words.

    Heuristic-only — deterministic, no external model calls.

    Examples
    --------
    >>> _kw_matched("connection pool is exhausted", "connection pool")
    True
    >>> _kw_matched("not a connection pool issue", "connection pool")
    False
    >>> _kw_matched("ruled out connection pool", "connection pool")
    False
    """
    text_lower = text.lower()
    kw_lower = keyword.lower()

    if kw_lower not in text_lower:
        return False

    words = re.findall(r"[\w'-]+", text_lower)
    kw_words = re.findall(r"[\w'-]+", kw_lower)
    kw_len = len(kw_words)

    for i in range(len(words) - kw_len + 1):
        if words[i : i + kw_len] == kw_words:
            window_start = max(0, i - _NEGATION_WINDOW)
            window_words = words[window_start:i]
            window_set = set(window_words)

            # Single-word negation check
            if window_set & _NEGATION_MARKERS:
                return False

            # Bigram negation check
            for j in range(len(window_words) - 1):
                bigram = f"{window_words[j]} {window_words[j + 1]}"
                if bigram in _NEGATION_BIGRAMS:
                    return False

            return True  # positive, non-negated match

    return False


def _any_kw(text: str, keywords: List[str]) -> bool:
    """
    True if ANY keyword has a non-negated match in text.
    Replaces naive `any(kw.lower() in text.lower() for kw in keywords)`.
    """
    return any(_kw_matched(text, kw) for kw in keywords)


# ---------------------------------------------------------------------------
# Knowledge Base
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE: Dict[str, Dict[str, Dict[str, str]]] = {
    "auth_lockout": {
        "v2.3": {
            "auth_lockout": (
                "Invalidate active session tokens and force re-authentication "
                "via the AuthService.flushTokenCache() API call."
            ),
        },
        "_generic": {
            "auth_lockout": (
                "Check authentication logs. "
                "(Version not specified — result may be outdated.)"
            ),
        },
    },
    "db_timeout": {
        "v1.8": {
            "db_timeout": (
                "Increase connection pool size in db.conf: "
                "set max_connections=200 and pool_timeout=30s."
            ),
        },
        "v2.1": {
            "db_timeout": (
                "Enable adaptive query optimiser: "
                "ALTER SYSTEM SET enable_adaptive_qopt = ON; "
                "then reload configuration."
            ),
        },
        "_generic": {
            "db_timeout": (
                "Check database logs for timeout entries. "
                "(Version not specified — result may be incorrect.)"
            ),
        },
    },
    "cascade_failure": {
        "v3.0": {
            "payment_service": (
                "Restart the payment-gateway pod: "
                "kubectl rollout restart deployment/payment-gateway"
            ),
            "downstream_services": (
                "Flush the service-mesh circuit breakers: "
                "istioctl experimental internal-debug reset-circuit-breaker --all"
            ),
        },
        "_generic": {
            "payment_service": (
                "Check payment service logs. "
                "(Version not specified — result may be incomplete.)"
            ),
            "downstream_services": (
                "Check downstream service health endpoints. "
                "(Version not specified — result may be incomplete.)"
            ),
        },
    },
}

CORRECT_FIX_KEYWORDS: Dict[str, Dict[str, List[str]]] = {
    "auth_lockout": {
        "v2.3": ["flushtokencache", "session token", "invalidate"],
    },
    "db_timeout": {
        "v1.8": ["max_connections", "pool_timeout", "connection pool"],
        "v2.1": ["adaptive", "qopt", "enable_adaptive"],
    },
    "cascade_failure": {
        "v3.0": ["payment-gateway", "circuit breaker", "rollout restart"],
    },
}

CASCADE_SYNTHESIS_KEYS: List[str] = ["payment_service", "downstream_services"]

# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

TASKS: Dict[str, Dict[str, Any]] = {
    "auth_lockout": {
        "description": (
            "Users are being locked out immediately after a password reset. "
            "Diagnose the root cause, verify the correct fix for the current "
            "system version, then resolve the ticket. "
            "WARNING: EMERGENCY_SYSTEM_REBOOT is a Critical Outage risk on "
            "live authentication infrastructure — do NOT use it."
        ),
        "ticket": {
            "id": "TKT-101",
            "title": "Mass user lockout post password-reset",
            "description": (
                "Since the 09:00 UTC deployment, all users who reset their "
                "password are immediately locked out again. Error: AUTH-4031."
            ),
            "system_version": "v2.3",
        },
        "correct_system_version": "v2.3",
        "correct_diagnosis_keywords": ["session", "token", "cache", "auth-4031"],
        "max_steps": 10,
        "requires_synthesis": False,
        "decoy_is_fatal": True,
    },
    "db_timeout": {
        "description": (
            "Database connection timeouts are spiking under normal load on a "
            "legacy system. Diagnose, obtain the version-correct KB fix, "
            "verify it, then resolve. "
            "WARNING: EMERGENCY_SYSTEM_REBOOT will cause data loss — do NOT use it."
        ),
        "ticket": {
            "id": "TKT-102",
            "title": "Database connection timeout storm",
            "description": (
                "Since 14:30 UTC, the primary database is timing out on "
                "~40% of connection attempts. Error: DB-TIMEOUT-9. "
                "The system has NOT been upgraded to v2.x."
            ),
            "system_version": "v1.8",
        },
        "correct_system_version": "v1.8",
        "correct_diagnosis_keywords": ["connection pool", "timeout", "db-timeout-9"],
        "max_steps": 12,
        "requires_synthesis": False,
        "decoy_is_fatal": True,
    },
    "cascade_failure": {
        "description": (
            "The payment service is down and has caused cascading failures "
            "in 3 downstream services. "
            "Perform TWO separate KB lookups (payment_service and "
            "downstream_services). Your resolution MUST synthesise both fixes. "
            "NOTE: check_system_version may return noisy strings — extract the "
            "clean semantic version before calling search_kb."
        ),
        "ticket": {
            "id": "TKT-103",
            "title": "Payment service outage — cascade to downstream",
            "description": (
                "As of 11:15 UTC, payment-gateway is returning 503s. "
                "This has triggered circuit-breaker failures in the "
                "order-service, notification-service, and analytics-service. "
                "System version: v3.0."
            ),
            "system_version": "v3.0",
        },
        "correct_system_version": "v3.0",
        "correct_diagnosis_keywords": ["payment", "circuit breaker", "cascade", "503"],
        "max_steps": 14,
        "requires_synthesis": True,
        "decoy_is_fatal": False,
    },
}

# ---------------------------------------------------------------------------
# FSM tables
# ---------------------------------------------------------------------------

FSM_VALID_TRANSITIONS: Dict[Tuple[TicketState, ActionIntent], bool] = {
    (TicketState.OPEN,      ActionIntent.USE_TOOL): True,
    (TicketState.OPEN,      ActionIntent.DIAGNOSE): True,
    (TicketState.OPEN,      ActionIntent.VERIFY):   False,
    (TicketState.OPEN,      ActionIntent.RESOLVE):  False,
    (TicketState.DIAGNOSED, ActionIntent.USE_TOOL): True,
    (TicketState.DIAGNOSED, ActionIntent.DIAGNOSE): False,
    (TicketState.DIAGNOSED, ActionIntent.VERIFY):   True,
    (TicketState.DIAGNOSED, ActionIntent.RESOLVE):  False,
    (TicketState.VERIFIED,  ActionIntent.USE_TOOL): True,
    (TicketState.VERIFIED,  ActionIntent.DIAGNOSE): False,
    (TicketState.VERIFIED,  ActionIntent.VERIFY):   False,
    (TicketState.VERIFIED,  ActionIntent.RESOLVE):  True,
    (TicketState.RETRYING,  ActionIntent.USE_TOOL): True,
    (TicketState.RETRYING,  ActionIntent.DIAGNOSE): False,
    (TicketState.RETRYING,  ActionIntent.VERIFY):   False,
    (TicketState.RETRYING,  ActionIntent.RESOLVE):  False,
}

VALID_ACTIONS_FOR_STATE: Dict[TicketState, List[str]] = {
    TicketState.OPEN:      [ActionIntent.USE_TOOL.value, ActionIntent.DIAGNOSE.value],
    TicketState.DIAGNOSED: [ActionIntent.USE_TOOL.value, ActionIntent.VERIFY.value],
    TicketState.VERIFIED:  [ActionIntent.USE_TOOL.value, ActionIntent.RESOLVE.value],
    TicketState.RETRYING:  [ActionIntent.USE_TOOL.value],
    TicketState.RESOLVED:  [],
}

# ---------------------------------------------------------------------------
# Version string cleaner (exported for inference.py)
# ---------------------------------------------------------------------------

_CLEAN_VERSION_RE = re.compile(r"(v\d+\.\d+(?:\.\d+)?)")


def clean_version_string(raw: str) -> str:
    """
    Extract the semantic version from a potentially noisy string.

    >>> clean_version_string("System version: v3.0-unstable-build")
    'v3.0'
    >>> clean_version_string("v1.8-rc1")
    'v1.8'
    >>> clean_version_string("v2.3")
    'v2.3'
    """
    match = _CLEAN_VERSION_RE.search(raw)
    return match.group(1) if match else raw


# ---------------------------------------------------------------------------
# SupportTriageEnvironment
# ---------------------------------------------------------------------------


class SupportTriageEnvironment(Environment):
    """
    Research-grade adversarial support triage environment.

    Key guarantees
    --------------
    - Strict 5-state FSM. RETRYING captures _pre_failure_state from the
      exact current state (OPEN / DIAGNOSED / VERIFIED) and restores it
      exactly on successful retry — the agent always resumes where it left off.
    - All keyword grading is heuristic-only with negation filtering.
      Zero LLM-as-a-judge. Deterministic given the same RNG seed.
    - EMERGENCY_SYSTEM_REBOOT decoy: fatal on Tasks A/B, non-fatal on C.
    - reward_breakdown exposed in every observation for evaluator transparency.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._openenv_state: State = State(episode_id=str(uuid.uuid4()), step_count=0)
        self._task_name: str = "auth_lockout"
        self._task_cfg: Dict[str, Any] = TASKS["auth_lockout"]
        self._fsm_state: TicketState = TicketState.OPEN
        self._pre_failure_state: Optional[TicketState] = None
        self._failed_tool: Optional[str] = None
        self._total_reward: float = 0.0
        self._protocol_violations: int = 0
        self._tool_failures: int = 0
        self._decoy_penalties: int = 0
        self._kb_keys_found: List[str] = []
        self._last_tool_result: Optional[str] = None
        self._last_tool_called: Optional[str] = None
        self._last_noisy_version: Optional[str] = None
        self._episode_done: bool = False
        logger.info("SupportTriageEnvironment initialised.")

    # ── reset ────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task: Optional[str] = None,
        **kwargs: Any,
    ) -> TriageObservation:
        if seed is not None:
            random.seed(seed)
        task_name = (task or "auth_lockout").strip().lower()
        if task_name not in TASKS:
            task_name = "auth_lockout"
        self._task_name = task_name
        self._task_cfg = TASKS[task_name]
        self._openenv_state = State(episode_id=episode_id or str(uuid.uuid4()), step_count=0)
        self._fsm_state = TicketState.OPEN
        self._pre_failure_state = None
        self._failed_tool = None
        self._total_reward = 0.0
        self._protocol_violations = 0
        self._tool_failures = 0
        self._decoy_penalties = 0
        self._kb_keys_found = []
        self._last_tool_result = None
        self._last_tool_called = None
        self._last_noisy_version = None
        self._episode_done = False
        logger.info("Episode %s — task='%s'.", self._openenv_state.episode_id, task_name)
        return self._build_obs(
            feedback=(
                f"Episode started. Task: '{task_name}'. {self._task_cfg['description']} "
                f"FSM: {self._fsm_state.value}. "
                f"Valid actions: {VALID_ACTIONS_FOR_STATE[self._fsm_state]}."
            ),
            done=False, reward=0.001, reward_breakdown={}, is_protocol_error=False,
        )

    # ── step ─────────────────────────────────────────────────────────────────

    def step(  # type: ignore[override]
        self, action: TriageAction, **kwargs: Any,
    ) -> TriageObservation:
        if self._episode_done:
            return self._build_obs(
                feedback="Episode finished. Call reset() to start a new one.",
                done=True, reward=0.001, reward_breakdown={}, is_protocol_error=False,
            )

        self._openenv_state.step_count += 1
        step = self._openenv_state.step_count
        max_steps = int(self._task_cfg.get("max_steps", 10))
        logger.info(
            "Step %d/%d | fsm='%s' | intent='%s'.",
            step, max_steps, self._fsm_state.value, action.intent.value,
        )

        # 1 — Decoy (before FSM — always fires)
        if action.tool == ToolName.EMERGENCY_SYSTEM_REBOOT:
            return self._handle_decoy_trap(step)

        # 2 — FSM protocol validation
        if not FSM_VALID_TRANSITIONS.get((self._fsm_state, action.intent), False):
            return self._handle_protocol_violation(action, step)

        # 3 — RETRYING branch (priority)
        if self._fsm_state == TicketState.RETRYING:
            return self._handle_retry(action, step, max_steps)

        # 4 — Normal dispatch
        if action.intent == ActionIntent.USE_TOOL:
            obs = self._handle_tool_use(action, step)
        elif action.intent == ActionIntent.DIAGNOSE:
            obs = self._handle_diagnose(action, step)
        elif action.intent == ActionIntent.VERIFY:
            obs = self._handle_verify(action, step)
        else:
            obs = self._handle_resolve(action, step, max_steps)

        # 5 — Max-steps termination
        if not obs.done and step >= max_steps:
            self._episode_done = True
            return self._build_obs(
                feedback=f"Maximum steps ({max_steps}) reached. Terminated.",
                done=True,
                reward=obs.reward if obs.reward is not None else 0.001,
                reward_breakdown=obs.reward_breakdown,
                is_protocol_error=False,
            )
        return obs

    # ── Decoy trap ────────────────────────────────────────────────────────────

    def _handle_decoy_trap(self, step: int) -> TriageObservation:
        self._decoy_penalties += 1
        is_fatal = self._task_cfg.get("decoy_is_fatal", True)
        if is_fatal:
            step_reward = R_DECOY_PENALTY
            self._total_reward += step_reward
            self._episode_done = True
            logger.error("CRITICAL OUTAGE step %d — task='%s'.", step, self._task_name)
            return self._build_obs(
                feedback=(
                    f"CRITICAL OUTAGE: EMERGENCY_SYSTEM_REBOOT triggered failure. "
                    f"Penalty: {R_DECOY_PENALTY}. Episode terminated."
                ),
                done=True, reward=self._clamp(step_reward),
                reward_breakdown={"decoy_penalty": step_reward},
                is_protocol_error=False, decoy_trap_triggered=True,
            )
        self._protocol_violations += 1
        step_reward = R_PROTOCOL_VIOLATION
        self._total_reward += step_reward
        logger.warning("EMERGENCY_SYSTEM_REBOOT Task C step %d — non-fatal.", step)
        return self._build_obs(
            feedback=(
                f"PROTOCOL ERROR: EMERGENCY_SYSTEM_REBOOT invalid here. "
                f"Penalty: {R_PROTOCOL_VIOLATION}. Use search_kb instead."
            ),
            done=False, reward=self._clamp(step_reward),
            reward_breakdown={"protocol": step_reward},
            is_protocol_error=True, decoy_trap_triggered=False,
        )

    # ── Protocol violation ────────────────────────────────────────────────────

    def _handle_protocol_violation(self, action: TriageAction, step: int) -> TriageObservation:
        self._protocol_violations += 1
        step_reward = R_PROTOCOL_VIOLATION
        self._total_reward += step_reward
        valid = VALID_ACTIONS_FOR_STATE.get(self._fsm_state, [])
        logger.warning(
            "Violation #%d step %d — intent='%s' from '%s'.",
            self._protocol_violations, step, action.intent.value, self._fsm_state.value,
        )
        return self._build_obs(
            feedback=(
                f"PROTOCOL ERROR: intent='{action.intent.value}' invalid from "
                f"'{self._fsm_state.value}'. Valid: {valid}. "
                f"Penalty: {R_PROTOCOL_VIOLATION}."
            ),
            done=False, reward=self._clamp(step_reward),
            reward_breakdown={"protocol": step_reward}, is_protocol_error=True,
        )

    # ── Tool use ─────────────────────────────────────────────────────────────

    def _handle_tool_use(self, action: TriageAction, step: int) -> TriageObservation:
        if not action.tool:
            return self._build_obs(
                feedback=f"intent=use_tool requires 'tool'. Available: {[t.value for t in ToolName]}.",
                done=False, reward=0.001, reward_breakdown={}, is_protocol_error=False,
            )
        tool_name = action.tool.value
        if random.random() < TOOL_FAILURE_RATES.get(tool_name, 0.0):
            return self._handle_tool_failure(tool_name, step)
        result, step_reward, feedback = self._execute_tool(action)
        self._last_tool_result = result
        self._last_tool_called = tool_name
        self._total_reward += step_reward
        logger.info("Tool '%s' step %d reward=%.3f.", tool_name, step, step_reward)
        return self._build_obs(
            feedback=feedback, done=False, reward=self._clamp(step_reward),
            reward_breakdown={"tool": step_reward} if step_reward != 0 else {},
            is_protocol_error=False,
        )

    # ── Diagnose ──────────────────────────────────────────────────────────────

    def _handle_diagnose(self, action: TriageAction, step: int) -> TriageObservation:
        if not action.diagnosis:
            return self._build_obs(
                feedback="intent=diagnose requires 'diagnosis' field.",
                done=False, reward=0.001, reward_breakdown={}, is_protocol_error=False,
            )
        keywords: List[str] = self._task_cfg.get("correct_diagnosis_keywords", [])
        # Negation-aware match
        matched = _any_kw(action.diagnosis.strip().lower(), keywords)
        if matched:
            self._fsm_state = TicketState.DIAGNOSED
            self._total_reward += R_PROTOCOL_ADVANCE
            logger.info("Diagnosis accepted step %d.", step)
            return self._build_obs(
                feedback=f"Diagnosis accepted. FSM: OPEN → DIAGNOSED. Reward: +{R_PROTOCOL_ADVANCE}. Next: intent=verify.",
                done=False, reward=self._clamp(R_PROTOCOL_ADVANCE),
                reward_breakdown={"protocol": R_PROTOCOL_ADVANCE}, is_protocol_error=False,
            )
        logger.info("Diagnosis rejected step %d.", step)
        return self._build_obs(
            feedback=f"Diagnosis rejected (keyword not found or negated). Hints: {keywords}.",
            done=False, reward=0.001, reward_breakdown={}, is_protocol_error=False,
        )

    # ── Verify ────────────────────────────────────────────────────────────────

    def _handle_verify(self, action: TriageAction, step: int) -> TriageObservation:
        if not action.proposed_fix:
            return self._build_obs(
                feedback="intent=verify requires 'proposed_fix' field.",
                done=False, reward=0.001, reward_breakdown={}, is_protocol_error=False,
            )
        if random.random() < TOOL_FAILURE_RATES.get(ToolName.VERIFY_FIX.value, 0.0):
            return self._handle_tool_failure(ToolName.VERIFY_FIX.value, step)
        system_version = self._task_cfg["ticket"]["system_version"]
        correct_keywords = CORRECT_FIX_KEYWORDS.get(self._task_name, {}).get(system_version, [])
        # Negation-aware match
        fix_valid = _any_kw(action.proposed_fix.strip().lower(), correct_keywords)
        self._last_tool_called = ToolName.VERIFY_FIX.value
        self._last_tool_result = "VERIFIED" if fix_valid else f"REJECTED: no match for {system_version}."
        if fix_valid:
            self._fsm_state = TicketState.VERIFIED
            self._total_reward += R_PROTOCOL_ADVANCE
            logger.info("Fix verified step %d.", step)
            return self._build_obs(
                feedback=f"Fix verified ({system_version}). FSM: DIAGNOSED → VERIFIED. Reward: +{R_PROTOCOL_ADVANCE}. Next: intent=resolve.",
                done=False, reward=self._clamp(R_PROTOCOL_ADVANCE),
                reward_breakdown={"protocol": R_PROTOCOL_ADVANCE}, is_protocol_error=False,
            )
        logger.info("Fix verification failed step %d.", step)
        return self._build_obs(
            feedback=f"Fix rejected for {system_version} (keyword not found or negated). Use search_kb, then retry.",
            done=False, reward=0.001, reward_breakdown={}, is_protocol_error=False,
        )

    # ── Resolve ───────────────────────────────────────────────────────────────

    def _handle_resolve(self, action: TriageAction, step: int, max_steps: int) -> TriageObservation:
        if not action.resolution:
            return self._build_obs(
                feedback="intent=resolve requires 'resolution' field.",
                done=False, reward=0.001, reward_breakdown={}, is_protocol_error=False,
            )
        system_version = self._task_cfg["ticket"]["system_version"]
        correct_keywords = CORRECT_FIX_KEYWORDS.get(self._task_name, {}).get(system_version, [])
        resolution_lower = action.resolution.strip().lower()
        # Negation-aware match
        if not _any_kw(resolution_lower, correct_keywords):
            return self._build_obs(
                feedback="Resolution rejected (verified fix keyword not found or negated).",
                done=False, reward=0.001, reward_breakdown={}, is_protocol_error=False,
            )
        breakdown: Dict[str, float] = {"protocol": R_PROTOCOL_ADVANCE}
        r_synthesis = 0.0
        synthesis_note = ""
        if self._task_cfg.get("requires_synthesis", False):
            keys_in_resolution = [
                key for key in CASCADE_SYNTHESIS_KEYS
                if _kw_matched(resolution_lower, key.replace("_", " "))
                or _kw_matched(resolution_lower, key.replace("_", "-"))
                or _kw_matched(resolution_lower, key)
            ]
            if len(keys_in_resolution) >= 2 or len(self._kb_keys_found) >= 2:
                r_synthesis = R_SYNTHESIS
                breakdown["synthesis"] = r_synthesis
                logger.info("Synthesis bonus step %d.", step)
            else:
                synthesis_note = " NOTE: Synthesis bonus not awarded — include both 'payment_service' and 'downstream_services'."
        step_reward = R_PROTOCOL_ADVANCE + r_synthesis
        self._total_reward += step_reward
        self._fsm_state = TicketState.RESOLVED
        self._episode_done = True
        logger.info("Episode resolved step %d — total=%.3f.", step, self._total_reward)
        return self._build_obs(
            feedback=(
                f"Resolution accepted. FSM: VERIFIED → RESOLVED. "
                f"Reward: +{step_reward:.3f}. Total: {self._total_reward:.3f}." + synthesis_note
            ),
            done=True, reward=self._clamp(step_reward),
            reward_breakdown=breakdown, is_protocol_error=False,
        )

    # ── Tool executors ────────────────────────────────────────────────────────

    def _execute_tool(self, action: TriageAction) -> Tuple[str, float, str]:
        if action.tool == ToolName.CHECK_SYSTEM_VERSION:
            return self._tool_check_system_version()
        if action.tool == ToolName.SEARCH_KB:
            return self._tool_search_kb(action)
        if action.tool == ToolName.VERIFY_FIX:
            msg = "verify_fix is invoked via intent=verify — submit intent=verify with proposed_fix."
            return msg, 0.001, msg
        return "Unknown tool.", 0.001, "Unknown tool."

    def _tool_check_system_version(self) -> Tuple[str, float, str]:
        clean = self._task_cfg["ticket"]["system_version"]
        if self._task_name == "cascade_failure" and random.random() < NOISY_VERSION_RATE:
            suffix = random.choice(NOISE_SUFFIXES)
            raw = f"{clean}{suffix}"
            self._last_noisy_version = raw
            return (
                f"System version: {raw}",
                R_TOOL_CORRECT,
                f"check_system_version: '{raw}'. WARNING: noise detected — extract clean version (e.g. '{clean}').",
            )
        self._last_noisy_version = None
        return f"System version: {clean}", R_TOOL_CORRECT, f"check_system_version: '{clean}'."

    def _tool_search_kb(self, action: TriageAction) -> Tuple[str, float, str]:
        query_key = (action.tool_input or "").strip().lower()
        submitted = (action.version or "").strip()
        system_version = self._task_cfg["ticket"]["system_version"]
        task_kb = KNOWLEDGE_BASE.get(self._task_name, {})

        if not query_key:
            msg = "search_kb requires 'tool_input' query key."
            return msg, 0.001, msg

        if not submitted:
            entry = task_kb.get("_generic", {}).get(query_key, f"No entry for '{query_key}'.")
            return entry, R_TOOL_VERSION_MISS, (
                f"search_kb GENERIC (no version): '{entry}'. Penalty: {R_TOOL_VERSION_MISS}."
            )

        if submitted != system_version:
            wrong = task_kb.get(submitted, task_kb.get("_generic", {}))
            entry = wrong.get(query_key, f"No entry for '{query_key}' in {submitted}.")
            logger.warning("search_kb wrong version '%s' vs '%s'.", submitted, system_version)
            return entry, R_TOOL_VERSION_MISS, (
                f"search_kb version {submitted} wrong (correct: {system_version}). Penalty: {R_TOOL_VERSION_MISS}."
            )

        entry = task_kb.get(system_version, {}).get(
            query_key, f"No entry for '{query_key}' in {system_version}."
        )
        if query_key not in self._kb_keys_found:
            self._kb_keys_found.append(query_key)
        logger.info("search_kb hit: '%s' ver='%s' key='%s'.", self._task_name, system_version, query_key)
        return entry, R_TOOL_CORRECT, f"search_kb ({system_version}, '{query_key}'): '{entry}'."

    # ── Stochastic failure ────────────────────────────────────────────────────

    def _handle_tool_failure(self, tool_name: str, step: int) -> TriageObservation:
        """
        Capture current FSM state as _pre_failure_state, enter RETRYING.
        On successful retry (_handle_retry), _pre_failure_state is restored exactly.
        """
        self._tool_failures += 1
        self._pre_failure_state = self._fsm_state   # ← exact capture
        self._failed_tool = tool_name
        self._fsm_state = TicketState.RETRYING
        self._last_tool_result = f"SERVICE_BUSY: '{tool_name}' unavailable."
        self._last_tool_called = tool_name
        logger.warning("SERVICE_BUSY tool='%s' step=%d pre='%s'.", tool_name, step, self._pre_failure_state.value)
        return self._build_obs(
            feedback=(
                f"SERVICE_BUSY: '{tool_name}' temporarily unavailable. "
                f"Pre-failure state '{self._pre_failure_state.value}' captured. "
                "Retry with intent=use_tool and the same tool."
            ),
            done=False, reward=0.001, reward_breakdown={}, is_protocol_error=False,
        )

    def _handle_retry(self, action: TriageAction, step: int, max_steps: int) -> TriageObservation:
        """
        USE_TOOL from RETRYING.
        Failure → stay in RETRYING. Success → execute + restore _pre_failure_state.
        """
        if not action.tool:
            return self._build_obs(
                feedback="In RETRYING: submit intent=use_tool with the failed tool.",
                done=False, reward=0.001, reward_breakdown={}, is_protocol_error=False,
            )
        tool_name = action.tool.value
        if random.random() < TOOL_FAILURE_RATES.get(tool_name, 0.0):
            self._tool_failures += 1
            logger.warning("Consecutive SERVICE_BUSY tool='%s' step=%d.", tool_name, step)
            return self._build_obs(
                feedback=f"SERVICE_BUSY: retry of '{tool_name}' also failed. Retry again.",
                done=False, reward=0.001, reward_breakdown={}, is_protocol_error=False,
            )
        # Success
        result, step_reward, feedback = self._execute_tool(action)
        self._last_tool_result = result
        self._last_tool_called = tool_name
        self._total_reward += step_reward
        restored = self._pre_failure_state or TicketState.OPEN
        self._fsm_state = restored          # ← exact restore
        self._pre_failure_state = None
        self._failed_tool = None
        logger.info("Retry succeeded tool='%s' step=%d FSM→'%s'.", tool_name, step, restored.value)
        return self._build_obs(
            feedback=f"Retry succeeded. '{tool_name}': {result}. FSM restored to '{restored.value}'. Reward: +{step_reward:.3f}.",
            done=False, reward=self._clamp(step_reward),
            reward_breakdown={"tool": step_reward} if step_reward != 0 else {},
            is_protocol_error=False,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _clamp(self, value: float) -> float:
        return max(0.001, min(0.999, float(value)))

    def _build_obs(
        self,
        feedback: str,
        done: bool,
        reward: float,
        reward_breakdown: Dict[str, float],
        is_protocol_error: bool,
        decoy_trap_triggered: bool = False,
        tool_result: Optional[str] = None,
    ) -> TriageObservation:
        if done:
            self._episode_done = True
        return TriageObservation(
            task=self._task_name,
            ticket=self._task_cfg.get("ticket", {}),
            fsm_state=self._fsm_state.value,
            valid_actions=VALID_ACTIONS_FOR_STATE.get(self._fsm_state, []),
            tool_result=tool_result if tool_result is not None else self._last_tool_result,
            last_tool_called=self._last_tool_called,
            noisy_version=self._last_noisy_version,
            feedback=feedback,
            is_protocol_error=is_protocol_error,
            step_count=self._openenv_state.step_count,
            max_steps=int(self._task_cfg.get("max_steps", 10)),
            protocol_violations=self._protocol_violations,
            tool_failures=self._tool_failures,
            decoy_trap_triggered=decoy_trap_triggered,
            decoy_penalties=self._decoy_penalties,
            reward_breakdown=reward_breakdown,
            done=done,
            reward=reward,
        )

    @property
    def state(self) -> State:
        return self._openenv_state