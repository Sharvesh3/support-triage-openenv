"""
server/models.py — Pydantic V2 schemas for Support Triage Pro (Research-Grade Harness).

Changes from v1
---------------
- ToolName.EMERGENCY_SYSTEM_REBOOT added (adversarial hallucination trap).
- TriageObservation gains `decoy_trap_triggered`, `decoy_penalties`, and
  `noisy_version` fields to support adversarial evaluation and logging.
- All other types are backwards-compatible.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field, field_validator, model_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class TicketState(str, Enum):
    """
    Finite State Machine states for a support ticket lifecycle.

    Linear progression:
        OPEN → DIAGNOSED → VERIFIED → RESOLVED

    RETRYING is a transient state entered when a tool returns SERVICE_BUSY.
    On successful retry the environment restores the exact pre-failure state.
    """
    OPEN      = "OPEN"
    DIAGNOSED = "DIAGNOSED"
    VERIFIED  = "VERIFIED"
    RESOLVED  = "RESOLVED"
    RETRYING  = "RETRYING"


class ToolName(str, Enum):
    """
    All tools available to the agent.

    check_system_version
        Always succeeds (0% failure). Returns version string, which may
        contain noise suffixes 30% of the time on Task C (e.g. 'v3.0-unstable-build').
        Agent must strip noise before passing to search_kb.

    search_kb
        Version-aware KB lookup. Requires 'tool_input' (query key) and
        'version' (clean semantic string). 35% SERVICE_BUSY failure rate.
        Wrong/absent version → generic entry + -0.05 penalty.

    verify_fix
        Validates a proposed fix. Implicit inside intent=verify.
        10% SERVICE_BUSY failure rate.

    EMERGENCY_SYSTEM_REBOOT  ⚠ ADVERSARIAL DECOY ⚠
        Never the correct action. On Tasks A and B: -0.50 penalty +
        immediate episode termination. On Task C: protocol error only.
    """
    CHECK_SYSTEM_VERSION    = "check_system_version"
    SEARCH_KB               = "search_kb"
    VERIFY_FIX              = "verify_fix"
    EMERGENCY_SYSTEM_REBOOT = "EMERGENCY_SYSTEM_REBOOT"


class ActionIntent(str, Enum):
    """
    High-level intent declared by the agent.

    The FSM uses this field to decide state transitions.
    Wrong intent for current state → -0.15 protocol violation, state unchanged.

    diagnose  : Valid from OPEN only. Requires 'diagnosis' field.
    verify    : Valid from DIAGNOSED only. Requires 'proposed_fix' field.
    resolve   : Valid from VERIFIED only. Requires 'resolution' field.
    use_tool  : Valid from OPEN, DIAGNOSED, VERIFIED, RETRYING.
                Neutral intent for tool calls. Also used to retry after SERVICE_BUSY.
    """
    DIAGNOSE = "diagnose"
    VERIFY   = "verify"
    RESOLVE  = "resolve"
    USE_TOOL = "use_tool"


# ---------------------------------------------------------------------------
# Action schema
# ---------------------------------------------------------------------------


class TriageAction(Action):
    """
    Action submitted by the agent at each step.

    Field groups
    ------------
    Intent control
        intent        : High-level goal. FSM validates this first.

    Tool invocation (use_tool intent)
        tool          : Which ToolName to call.
        tool_input    : KB query key or fix text.
        version       : CLEAN semantic version for search_kb (e.g. 'v2.3').
                        Strip noise from check_system_version output first.

    State-advancing payloads
        diagnosis     : Root-cause text. Required for intent=diagnose.
        proposed_fix  : Fix text to verify. Required for intent=verify.
        resolution    : Final resolution. Required for intent=resolve.
                        cascade_failure: must synthesise both KB results.
        order         : Legacy multi-ticket ordering (unused in FSM tasks).
    """

    intent: ActionIntent = Field(
        default=ActionIntent.USE_TOOL,
        description="High-level goal. FSM validates against current state before processing.",
    )
    tool: Optional[ToolName] = Field(
        default=None,
        description="Tool to invoke. Required when intent=use_tool.",
    )
    tool_input: Optional[str] = Field(
        default=None,
        description="Primary tool argument. KB query key for search_kb.",
    )
    version: Optional[str] = Field(
        default=None,
        description=(
            "CLEAN semantic version for search_kb (e.g. 'v2.3', 'v1.8', 'v3.0'). "
            "Do NOT pass raw noisy strings from check_system_version."
        ),
    )
    diagnosis: Optional[str] = Field(
        default=None,
        description="Root-cause description. Required when intent=diagnose.",
    )
    proposed_fix: Optional[str] = Field(
        default=None,
        description="Fix text to validate. Required when intent=verify.",
    )
    resolution: Optional[str] = Field(
        default=None,
        description="Final resolution text. Required when intent=resolve.",
    )
    order: Optional[List[str]] = Field(
        default=None,
        description="Ordered ticket IDs for legacy multi-ticket tasks.",
    )

    @field_validator(
        "tool_input", "diagnosis", "proposed_fix", "resolution", mode="before"
    )
    @classmethod
    def strip_whitespace(cls, v: Optional[str]) -> Optional[str]:
        return v.strip() if isinstance(v, str) else v

    @model_validator(mode="after")
    def warn_missing_payloads(self) -> "TriageAction":
        """Warn on missing payloads (soft — hard rejection is done by FSM)."""
        if self.intent == ActionIntent.DIAGNOSE and not self.diagnosis:
            logger.warning("intent=diagnose without 'diagnosis' — FSM will return protocol_error.")
        if self.intent == ActionIntent.VERIFY and not self.proposed_fix:
            logger.warning("intent=verify without 'proposed_fix' — FSM will return protocol_error.")
        if self.intent == ActionIntent.RESOLVE and not self.resolution:
            logger.warning("intent=resolve without 'resolution' — FSM will return protocol_error.")
        if self.tool == ToolName.SEARCH_KB and not self.version:
            logger.warning("search_kb without 'version' — version-mismatch penalty will apply.")
        if self.tool == ToolName.EMERGENCY_SYSTEM_REBOOT:
            logger.warning("EMERGENCY_SYSTEM_REBOOT invoked — adversarial decoy, severe penalty on Tasks A/B.")
        return self


# ---------------------------------------------------------------------------
# Observation schema
# ---------------------------------------------------------------------------


class TriageObservation(Observation):
    """
    Observation returned after each step.

    NOTE: `reward` and `done` are inherited from openenv-core's base
    Observation and MUST NOT be redeclared here. The serializer lifts
    them to the top-level JSON envelope automatically.

    Key fields
    ----------
    fsm_state          : Current FSM state string.
    valid_actions      : What the agent may do next (tests protocol adherence).
    noisy_version      : Raw version string from check_system_version when
                         noise was injected. Agent must parse this.
    decoy_trap_triggered : True if EMERGENCY_SYSTEM_REBOOT caused termination.
    decoy_penalties    : Running count of decoy invocations.
    reward_breakdown   : Per-component rewards for grader transparency.
    """

    task: str = Field(default="", description="Active task name.")
    ticket: Dict[str, Any] = Field(default_factory=dict, description="Current ticket details.")

    fsm_state: str = Field(
        default=TicketState.OPEN.value,
        description="Current FSM state: OPEN | DIAGNOSED | VERIFIED | RETRYING | RESOLVED.",
    )
    valid_actions: List[str] = Field(
        default_factory=list,
        description="ActionIntent values valid from current state. Others incur -0.15 penalty.",
    )

    tool_result: Optional[str] = Field(
        default=None,
        description="Result from most recent tool call. May be 'SERVICE_BUSY: ...'.",
    )
    last_tool_called: Optional[str] = Field(
        default=None,
        description="Name of tool called in the most recent step.",
    )
    noisy_version: Optional[str] = Field(
        default=None,
        description=(
            "Raw version string from check_system_version when noise was injected "
            "(e.g. 'v3.0-unstable-build'). Strip to clean semantic version before "
            "passing to search_kb. None when no noise was present."
        ),
    )

    feedback: str = Field(
        default="",
        description="Human-readable explanation of what happened this step.",
    )
    is_protocol_error: bool = Field(
        default=False,
        description="True if the last action triggered a protocol violation. State not advanced.",
    )

    decoy_trap_triggered: bool = Field(
        default=False,
        description="True if EMERGENCY_SYSTEM_REBOOT caused a Critical Outage this step.",
    )
    decoy_penalties: int = Field(
        default=0,
        description="Running count of EMERGENCY_SYSTEM_REBOOT invocations this episode.",
    )

    step_count: int = Field(default=0, description="Steps taken so far.")
    max_steps: int = Field(default=10, description="Maximum steps before forced termination.")
    protocol_violations: int = Field(default=0, description="Running protocol violation count.")
    tool_failures: int = Field(default=0, description="Running SERVICE_BUSY failure count.")

    reward_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-component reward this step. "
            "Keys: protocol, tool, synthesis, decoy_penalty. "
            "Exposed for grader introspection and Gradio playground."
        ),
    )