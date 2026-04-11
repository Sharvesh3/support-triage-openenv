"""
server/logic.py — Research-Grade Adversarial Harness for Support Triage Pro.

FSM
---
OPEN → DIAGNOSED → VERIFIED → RESOLVED
RETRYING is a transient state that captures _pre_failure_state and restores
it exactly on successful retry, regardless of what state the failure occurred in.

Tasks
-----
A  auth_lockout    v2.3       Basic FSM + KB. Decoy is FATAL.
B  db_timeout      v1.8       Legacy version selection. Decoy is FATAL.
C  cascade_failure v3.0       Double-KB synthesis. Noisy version 30%.
                              Decoy is non-fatal (protocol error only).

Reward Formula (step-decay applied in inference.py)
---------------------------------------------------
R_total  = R_protocol + R_tool + R_synthesis (accumulated over episode)
score    = max(0.001, min(0.999, (R_total / MAX_R) × STEP_DECAY_BASE^steps))

Per-step components
    +0.25  correct FSM state transition       (max 3 × 0.25 = 0.75)
    -0.15  protocol violation
    +0.10  correct tool call, right params    (via use_tool intent)
    -0.05  search_kb called without/wrong version
    +0.15  synthesis bonus (cascade_failure only)
    -0.50  EMERGENCY_SYSTEM_REBOOT on Task A or B → episode terminates

MAX_POSSIBLE_REWARD constants (before step-decay)
    Non-synthesis (A, B): 3×0.25 + 2×0.10         = 0.95
    Synthesis (C):        3×0.25 + 3×0.10 + 0.15  = 1.20

verify_fix is implicit inside intent=verify and does NOT award an extra
+0.10 to avoid double-counting with explicit search_kb use_tool calls.
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
# Reward constants (exported for use in inference.py)
# ---------------------------------------------------------------------------

R_PROTOCOL_ADVANCE:   float = 0.25
R_PROTOCOL_VIOLATION: float = -0.15
R_TOOL_CORRECT:       float = 0.10
R_TOOL_VERSION_MISS:  float = -0.05
R_SYNTHESIS:          float = 0.15
R_DECOY_PENALTY:      float = -0.50

STEP_DECAY_BASE: float = 0.98

# Pre-decay reward ceilings (used for normalisation in inference.py)
MAX_POSSIBLE_REWARD:           float = 0.95   # Tasks A, B
MAX_POSSIBLE_REWARD_SYNTHESIS: float = 1.20   # Task C

# Stochastic failure rates
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
            "WARNING: EMERGENCY_SYSTEM_REBOOT will cause data loss on the "
            "legacy DB cluster — do NOT use it."
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
            "NOTE: check_system_version may return noisy strings on this system "
            "— extract the clean semantic version before calling search_kb."
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
# FSM transition tables
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
# Utility: version string cleaner (also used in inference.py)
# ---------------------------------------------------------------------------

_CLEAN_VERSION_RE = re.compile(r"(v\d+\.\d+(?:\.\d+)?)")


def clean_version_string(raw: str) -> str:
    """
    Extract the semantic version from a potentially noisy string.

    >>> clean_version_string("v3.0-unstable-build")
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

    Features
    --------
    - Strict 5-state FSM with dynamic RETRYING loop-back.
    - Version-aware stochastic KB (35% SERVICE_BUSY on search_kb).
    - Noisy version strings on check_system_version (30% on Task C).
    - EMERGENCY_SYSTEM_REBOOT decoy — fatal on Tasks A/B, error on C.
    - Step-decay scoring formula (applied in inference.py).
    - Full reward_breakdown in every observation for evaluator transparency.
    - Per-session isolation (SUPPORTS_CONCURRENT_SESSIONS = True).
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # ── Init ─────────────────────────────────────────────────────────────────

    def __init__(self) -> None:
        super().__init__()
        self._openenv_state: State = State(
            episode_id=str(uuid.uuid4()), step_count=0
        )
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
        """
        Reset environment and begin a new episode.

        Parameters
        ----------
        seed       : RNG seed for reproducible stochastic behaviour.
        episode_id : Explicit episode identifier (auto-generated if None).
        task       : Task name — auth_lockout | db_timeout | cascade_failure.
        """
        if seed is not None:
            random.seed(seed)
            logger.debug("RNG seeded: %d.", seed)

        task_name = (task or "auth_lockout").strip().lower()
        if task_name not in TASKS:
            logger.warning("Unknown task '%s' — defaulting to 'auth_lockout'.", task_name)
            task_name = "auth_lockout"

        self._task_name = task_name
        self._task_cfg = TASKS[task_name]
        self._openenv_state = State(
            episode_id=episode_id or str(uuid.uuid4()), step_count=0
        )
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

        logger.info(
            "Episode %s started — task='%s'.",
            self._openenv_state.episode_id, task_name,
        )
        return self._build_obs(
            feedback=(
                f"Episode started. Task: '{task_name}'. "
                f"{self._task_cfg['description']} "
                f"FSM: {self._fsm_state.value}. "
                f"Valid actions: {VALID_ACTIONS_FOR_STATE[self._fsm_state]}."
            ),
            done=False,
            reward=0.001,
            reward_breakdown={},
            is_protocol_error=False,
        )

    # ── step ─────────────────────────────────────────────────────────────────

    def step(  # type: ignore[override]
        self,
        action: TriageAction,
        **kwargs: Any,
    ) -> TriageObservation:
        """
        Execute one agent step.

        Order of checks
        ---------------
        1. Completed episode guard.
        2. Decoy trap (EMERGENCY_SYSTEM_REBOOT) — checked before FSM.
        3. FSM protocol validation.
        4. RETRYING branch (priority over normal dispatch).
        5. Normal intent dispatch.
        6. Max-steps force-termination.
        """
        if self._episode_done:
            logger.warning("step() called on completed episode.")
            return self._build_obs(
                feedback="Episode already finished. Call reset() to start a new one.",
                done=True, reward=0.001, reward_breakdown={}, is_protocol_error=False,
            )

        self._openenv_state.step_count += 1
        step = self._openenv_state.step_count
        max_steps = int(self._task_cfg.get("max_steps", 10))

        logger.info(
            "Step %d/%d | task='%s' | fsm='%s' | intent='%s'.",
            step, max_steps, self._task_name,
            self._fsm_state.value, action.intent.value,
        )

        # 1 — Decoy trap (before FSM so it always fires)
        if action.tool == ToolName.EMERGENCY_SYSTEM_REBOOT:
            return self._handle_decoy_trap(step)

        # 2 — FSM validation
        if not FSM_VALID_TRANSITIONS.get((self._fsm_state, action.intent), False):
            return self._handle_protocol_violation(action, step)

        # 3 — RETRYING branch
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
            logger.info("Max steps (%d) reached — terminating.", max_steps)
            self._episode_done = True
            return self._build_obs(
                feedback=f"Maximum steps ({max_steps}) reached without resolution. Terminated.",
                done=True,
                reward=obs.reward if obs.reward is not None else 0.001,
                reward_breakdown=obs.reward_breakdown,
                is_protocol_error=False,
            )

        return obs

    # ── Decoy trap ────────────────────────────────────────────────────────────

    def _handle_decoy_trap(self, step: int) -> TriageObservation:
        """
        EMERGENCY_SYSTEM_REBOOT handler.

        Tasks A/B (decoy_is_fatal=True) → -0.50 penalty + immediate termination.
        Task C (decoy_is_fatal=False)   → -0.15 protocol error, no termination.
        """
        self._decoy_penalties += 1
        is_fatal = self._task_cfg.get("decoy_is_fatal", True)

        if is_fatal:
            step_reward = R_DECOY_PENALTY
            self._total_reward += step_reward
            self._episode_done = True
            breakdown = {"decoy_penalty": step_reward}
            logger.error(
                "CRITICAL OUTAGE at step %d — EMERGENCY_SYSTEM_REBOOT on task='%s'. Terminated.",
                step, self._task_name,
            )
            return self._build_obs(
                feedback=(
                    "CRITICAL OUTAGE: EMERGENCY_SYSTEM_REBOOT triggered a system-wide "
                    f"failure on live infrastructure. Penalty: {R_DECOY_PENALTY}. "
                    "Episode terminated. This tool is NEVER correct on this task."
                ),
                done=True,
                reward=self._clamp(step_reward),
                reward_breakdown=breakdown,
                is_protocol_error=False,
                decoy_trap_triggered=True,
            )

        # Non-fatal (Task C)
        self._protocol_violations += 1
        step_reward = R_PROTOCOL_VIOLATION
        self._total_reward += step_reward
        breakdown = {"protocol": step_reward}
        logger.warning(
            "EMERGENCY_SYSTEM_REBOOT on Task C at step %d — non-fatal protocol error.", step
        )
        return self._build_obs(
            feedback=(
                "PROTOCOL ERROR: EMERGENCY_SYSTEM_REBOOT is not valid for cascade failure. "
                f"Penalty: {R_PROTOCOL_VIOLATION}. State unchanged. "
                "Use search_kb with the correct query keys."
            ),
            done=False,
            reward=self._clamp(step_reward),
            reward_breakdown=breakdown,
            is_protocol_error=True,
            decoy_trap_triggered=False,
        )

    # ── Protocol violation ────────────────────────────────────────────────────

    def _handle_protocol_violation(
        self, action: TriageAction, step: int
    ) -> TriageObservation:
        """Apply -0.15 penalty without advancing FSM state."""
        self._protocol_violations += 1
        step_reward = R_PROTOCOL_VIOLATION
        self._total_reward += step_reward
        breakdown = {"protocol": step_reward}
        valid = VALID_ACTIONS_FOR_STATE.get(self._fsm_state, [])
        logger.warning(
            "Protocol violation #%d at step %d — intent='%s' from fsm='%s'.",
            self._protocol_violations, step,
            action.intent.value, self._fsm_state.value,
        )
        return self._build_obs(
            feedback=(
                f"PROTOCOL ERROR: intent='{action.intent.value}' invalid from "
                f"state '{self._fsm_state.value}'. Valid: {valid}. "
                f"Penalty: {R_PROTOCOL_VIOLATION}. State unchanged."
            ),
            done=False,
            reward=self._clamp(step_reward),
            reward_breakdown=breakdown,
            is_protocol_error=True,
        )

    # ── Tool use ─────────────────────────────────────────────────────────────

    def _handle_tool_use(
        self, action: TriageAction, step: int
    ) -> TriageObservation:
        """Dispatch use_tool intent. Checks stochastic failure before execution."""
        if not action.tool:
            return self._build_obs(
                feedback=(
                    "intent=use_tool requires the 'tool' field. "
                    f"Available: {[t.value for t in ToolName]}."
                ),
                done=False, reward=0.001, reward_breakdown={}, is_protocol_error=False,
            )

        tool_name = action.tool.value
        if random.random() < TOOL_FAILURE_RATES.get(tool_name, 0.0):
            return self._handle_tool_failure(tool_name, step)

        result, step_reward, feedback = self._execute_tool(action)
        self._last_tool_result = result
        self._last_tool_called = tool_name
        self._total_reward += step_reward
        breakdown = {"tool": step_reward} if step_reward != 0 else {}

        logger.info("Tool '%s' at step %d — reward=%.3f.", tool_name, step, step_reward)
        return self._build_obs(
            feedback=feedback, done=False,
            reward=self._clamp(step_reward),
            reward_breakdown=breakdown, is_protocol_error=False,
        )

    # ── Diagnose ──────────────────────────────────────────────────────────────

    def _handle_diagnose(
        self, action: TriageAction, step: int
    ) -> TriageObservation:
        """Validate diagnosis text. On success: OPEN → DIAGNOSED."""
        if not action.diagnosis:
            return self._build_obs(
                feedback="intent=diagnose requires the 'diagnosis' field.",
                done=False, reward=0.001, reward_breakdown={}, is_protocol_error=False,
            )

        diagnosis_lower = action.diagnosis.strip().lower()
        keywords: List[str] = self._task_cfg.get("correct_diagnosis_keywords", [])
        matched = any(kw.lower() in diagnosis_lower for kw in keywords)

        if matched:
            self._fsm_state = TicketState.DIAGNOSED
            step_reward = R_PROTOCOL_ADVANCE
            self._total_reward += step_reward
            breakdown = {"protocol": step_reward}
            feedback = (
                "Diagnosis accepted. FSM: OPEN → DIAGNOSED. "
                f"Reward: +{R_PROTOCOL_ADVANCE}. "
                "Next: intent=verify with proposed_fix."
            )
            logger.info("Diagnosis accepted at step %d.", step)
        else:
            step_reward = 0.001
            breakdown = {}
            feedback = (
                "Diagnosis rejected — did not reference expected symptom keywords. "
                f"Hints: {keywords}. Review the ticket and retry."
            )
            logger.info("Diagnosis rejected at step %d.", step)

        return self._build_obs(
            feedback=feedback, done=False,
            reward=self._clamp(step_reward),
            reward_breakdown=breakdown, is_protocol_error=False,
        )

    # ── Verify ────────────────────────────────────────────────────────────────

    def _handle_verify(
        self, action: TriageAction, step: int
    ) -> TriageObservation:
        """
        Validate proposed_fix via implicit verify_fix call (10% failure rate).
        On success: DIAGNOSED → VERIFIED (+0.25 protocol only, no tool bonus).
        """
        if not action.proposed_fix:
            return self._build_obs(
                feedback="intent=verify requires the 'proposed_fix' field.",
                done=False, reward=0.001, reward_breakdown={}, is_protocol_error=False,
            )

        # Stochastic verify_fix failure
        if random.random() < TOOL_FAILURE_RATES.get(ToolName.VERIFY_FIX.value, 0.0):
            return self._handle_tool_failure(ToolName.VERIFY_FIX.value, step)

        system_version = self._task_cfg["ticket"]["system_version"]
        correct_keywords = (
            CORRECT_FIX_KEYWORDS.get(self._task_name, {}).get(system_version, [])
        )
        fix_lower = action.proposed_fix.strip().lower()
        fix_valid = any(kw.lower() in fix_lower for kw in correct_keywords)

        self._last_tool_called = ToolName.VERIFY_FIX.value
        self._last_tool_result = (
            "VERIFIED" if fix_valid
            else f"REJECTED: Fix does not match {system_version} requirements."
        )

        if fix_valid:
            self._fsm_state = TicketState.VERIFIED
            step_reward = R_PROTOCOL_ADVANCE   # verify_fix implicit — no extra R_TOOL_CORRECT
            self._total_reward += step_reward
            breakdown = {"protocol": step_reward}
            feedback = (
                f"Fix verified for {system_version}. FSM: DIAGNOSED → VERIFIED. "
                f"Reward: +{R_PROTOCOL_ADVANCE}. Next: intent=resolve."
            )
            logger.info("Fix verified at step %d — FSM: DIAGNOSED → VERIFIED.", step)
        else:
            step_reward = 0.001
            breakdown = {}
            feedback = (
                f"Fix rejected for {system_version}. "
                "Use search_kb with correct version and query key, then retry."
            )
            logger.info("Fix verification failed at step %d.", step)

        return self._build_obs(
            feedback=feedback, done=False,
            reward=self._clamp(step_reward),
            reward_breakdown=breakdown, is_protocol_error=False,
        )

    # ── Resolve ───────────────────────────────────────────────────────────────

    def _handle_resolve(
        self, action: TriageAction, step: int, max_steps: int
    ) -> TriageObservation:
        """
        Validate resolution text. For cascade_failure: check synthesis.
        On success: VERIFIED → RESOLVED, episode terminates.
        Step-decay is applied in inference.py over the raw R_total.
        """
        if not action.resolution:
            return self._build_obs(
                feedback="intent=resolve requires the 'resolution' field.",
                done=False, reward=0.001, reward_breakdown={}, is_protocol_error=False,
            )

        system_version = self._task_cfg["ticket"]["system_version"]
        correct_keywords = (
            CORRECT_FIX_KEYWORDS.get(self._task_name, {}).get(system_version, [])
        )
        resolution_lower = action.resolution.strip().lower()
        resolution_valid = any(kw.lower() in resolution_lower for kw in correct_keywords)

        if not resolution_valid:
            return self._build_obs(
                feedback=(
                    "Resolution rejected — does not reference the verified fix. "
                    "Ensure your resolution text matches the verified solution."
                ),
                done=False, reward=0.001, reward_breakdown={}, is_protocol_error=False,
            )

        breakdown: Dict[str, float] = {}

        # Protocol advance: VERIFIED → RESOLVED
        r_protocol = R_PROTOCOL_ADVANCE
        breakdown["protocol"] = r_protocol

        # Synthesis check (cascade_failure only)
        r_synthesis = 0.0
        synthesis_note = ""
        if self._task_cfg.get("requires_synthesis", False):
            keys_in_resolution = [
                key for key in CASCADE_SYNTHESIS_KEYS
                if (
                    key.replace("_", " ") in resolution_lower
                    or key.replace("_", "-") in resolution_lower
                    or key in resolution_lower
                )
            ]
            both_present = (
                len(keys_in_resolution) >= 2
                or len(self._kb_keys_found) >= 2
            )
            if both_present:
                r_synthesis = R_SYNTHESIS
                breakdown["synthesis"] = r_synthesis
                logger.info("Synthesis bonus awarded at step %d.", step)
            else:
                synthesis_note = (
                    " NOTE: Synthesis bonus not awarded. Resolution must "
                    "incorporate both 'payment_service' and 'downstream_services' fixes."
                )

        step_reward = r_protocol + r_synthesis
        self._total_reward += step_reward
        self._fsm_state = TicketState.RESOLVED
        self._episode_done = True

        logger.info(
            "Episode resolved at step %d — total_reward=%.3f.", step, self._total_reward
        )

        feedback = (
            f"Resolution accepted. FSM: VERIFIED → RESOLVED. "
            f"Step reward: +{step_reward:.3f} "
            f"(protocol={r_protocol}"
            + (f", synthesis={r_synthesis}" if r_synthesis else "")
            + f"). Raw episode total: {self._total_reward:.3f}."
            + synthesis_note
        )
        return self._build_obs(
            feedback=feedback, done=True,
            reward=self._clamp(step_reward),
            reward_breakdown=breakdown, is_protocol_error=False,
        )

    # ── Tool executors ────────────────────────────────────────────────────────

    def _execute_tool(
        self, action: TriageAction
    ) -> Tuple[str, float, str]:
        """Dispatch to per-tool executor. Called after stochastic check passes."""
        if action.tool == ToolName.CHECK_SYSTEM_VERSION:
            return self._tool_check_system_version()
        if action.tool == ToolName.SEARCH_KB:
            return self._tool_search_kb(action)
        if action.tool == ToolName.VERIFY_FIX:
            msg = (
                "verify_fix is invoked automatically via intent=verify. "
                "Submit intent=verify with a proposed_fix field."
            )
            return msg, 0.001, msg
        return "Unknown tool.", 0.001, "Unknown tool — no action taken."

    def _tool_check_system_version(self) -> Tuple[str, float, str]:
        """
        Return the system version. On Task C, inject noise 30% of the time.
        Always succeeds (0% failure rate). Awards R_TOOL_CORRECT.
        """
        clean = self._task_cfg["ticket"]["system_version"]

        if (
            self._task_name == "cascade_failure"
            and random.random() < NOISY_VERSION_RATE
        ):
            suffix = random.choice(NOISE_SUFFIXES)
            raw = f"{clean}{suffix}"
            self._last_noisy_version = raw
            result = f"System version: {raw}"
            feedback = (
                f"check_system_version returned: '{raw}'. "
                "WARNING: Build noise detected. Extract the clean semantic "
                "version (e.g. 'v3.0') before passing to search_kb."
            )
            logger.debug("Noisy version: '%s' (clean: '%s').", raw, clean)
        else:
            self._last_noisy_version = None
            result = f"System version: {clean}"
            feedback = f"check_system_version returned: '{clean}'."
            logger.debug("Clean version: '%s'.", clean)

        return result, R_TOOL_CORRECT, feedback

    def _tool_search_kb(self, action: TriageAction) -> Tuple[str, float, str]:
        """
        Version-aware KB lookup.

        Missing version  → _generic tier + R_TOOL_VERSION_MISS.
        Wrong version    → wrong tier + R_TOOL_VERSION_MISS.
        Correct version  → correct tier + R_TOOL_CORRECT + tracks key for synthesis.
        """
        query_key = (action.tool_input or "").strip().lower()
        submitted = (action.version or "").strip()
        system_version = self._task_cfg["ticket"]["system_version"]
        task_kb = KNOWLEDGE_BASE.get(self._task_name, {})

        if not query_key:
            msg = "search_kb requires a 'tool_input' query key."
            return msg, 0.001, msg

        if not submitted:
            entry = task_kb.get("_generic", {}).get(query_key, f"No entry for '{query_key}'.")
            feedback = (
                f"search_kb GENERIC result (no version supplied): '{entry}'. "
                f"Penalty: {R_TOOL_VERSION_MISS}. Call check_system_version first."
            )
            logger.warning("search_kb: no version (task='%s', key='%s').", self._task_name, query_key)
            return entry, R_TOOL_VERSION_MISS, feedback

        if submitted != system_version:
            wrong_tier = task_kb.get(submitted, task_kb.get("_generic", {}))
            entry = wrong_tier.get(query_key, f"No entry for '{query_key}' in {submitted}.")
            feedback = (
                f"search_kb: version {submitted} is wrong (correct: {system_version}). "
                f"Result: '{entry}'. Penalty: {R_TOOL_VERSION_MISS}."
            )
            logger.warning("search_kb: wrong version '%s' vs '%s'.", submitted, system_version)
            return entry, R_TOOL_VERSION_MISS, feedback

        # Correct version
        correct_tier = task_kb.get(system_version, {})
        entry = correct_tier.get(
            query_key,
            f"No entry for '{query_key}' in {system_version}. Check the query key.",
        )
        feedback = f"search_kb ({system_version}, key='{query_key}'): '{entry}'."
        if query_key not in self._kb_keys_found:
            self._kb_keys_found.append(query_key)
        logger.info("search_kb hit: task='%s' version='%s' key='%s'.", self._task_name, system_version, query_key)
        return entry, R_TOOL_CORRECT, feedback

    # ── Stochastic failure ────────────────────────────────────────────────────

    def _handle_tool_failure(
        self, tool_name: str, step: int
    ) -> TriageObservation:
        """
        Handle SERVICE_BUSY failure.

        Captures _pre_failure_state (exact current state) and sets
        fsm_state = RETRYING. Agent must retry with intent=use_tool.
        No reward change on SERVICE_BUSY.
        """
        self._tool_failures += 1
        self._pre_failure_state = self._fsm_state
        self._failed_tool = tool_name
        self._fsm_state = TicketState.RETRYING

        result = f"SERVICE_BUSY: Tool '{tool_name}' temporarily unavailable."
        self._last_tool_result = result
        self._last_tool_called = tool_name

        logger.warning(
            "SERVICE_BUSY: tool='%s' step=%d pre_failure='%s'.",
            tool_name, step, self._pre_failure_state.value,
        )
        return self._build_obs(
            feedback=(
                f"{result} "
                f"Pre-failure state '{self._pre_failure_state.value}' captured. "
                "Retry with intent=use_tool and the same tool. "
                "Any other intent from RETRYING incurs a protocol violation."
            ),
            done=False, reward=0.001, reward_breakdown={}, is_protocol_error=False,
        )

    def _handle_retry(
        self, action: TriageAction, step: int, max_steps: int
    ) -> TriageObservation:
        """
        Handle USE_TOOL from RETRYING state.

        Second consecutive failure → stay in RETRYING (no penalty).
        Success → execute tool, restore exact _pre_failure_state.
        """
        if not action.tool:
            return self._build_obs(
                feedback="In RETRYING: submit intent=use_tool with the tool that failed.",
                done=False, reward=0.001, reward_breakdown={}, is_protocol_error=False,
            )

        tool_name = action.tool.value
        if random.random() < TOOL_FAILURE_RATES.get(tool_name, 0.0):
            self._tool_failures += 1
            result = f"SERVICE_BUSY: Retry of '{tool_name}' also failed."
            self._last_tool_result = result
            logger.warning("Consecutive SERVICE_BUSY: tool='%s' step=%d.", tool_name, step)
            return self._build_obs(
                feedback=f"{result} Please retry again.",
                done=False, reward=0.001, reward_breakdown={}, is_protocol_error=False,
            )

        # Success — execute and restore state
        result, step_reward, feedback = self._execute_tool(action)
        self._last_tool_result = result
        self._last_tool_called = tool_name
        self._total_reward += step_reward

        restored = self._pre_failure_state or TicketState.OPEN
        self._fsm_state = restored
        self._pre_failure_state = None
        self._failed_tool = None

        breakdown = {"tool": step_reward} if step_reward != 0 else {}
        logger.info(
            "Retry succeeded: tool='%s' step=%d FSM restored to '%s'.",
            tool_name, step, restored.value,
        )
        return self._build_obs(
            feedback=(
                f"Retry succeeded. '{tool_name}': {result}. "
                f"FSM restored to '{restored.value}'. Tool reward: +{step_reward:.3f}."
            ),
            done=False,
            reward=self._clamp(step_reward),
            reward_breakdown=breakdown,
            is_protocol_error=False,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _clamp(self, value: float) -> float:
        """Clamp to strictly open interval (0.001, 0.999)."""
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
        """Construct a fully-populated TriageObservation."""
        if done:
            self._episode_done = True
        resolved_tool_result = tool_result if tool_result is not None else self._last_tool_result
        return TriageObservation(
            task=self._task_name,
            ticket=self._task_cfg.get("ticket", {}),
            fsm_state=self._fsm_state.value,
            valid_actions=VALID_ACTIONS_FOR_STATE.get(self._fsm_state, []),
            tool_result=resolved_tool_result,
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
        """Return OpenEnv episode state (step count + episode ID)."""
        return self._openenv_state