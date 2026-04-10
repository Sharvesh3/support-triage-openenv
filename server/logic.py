"""
SupportTriageEnvironment — core logic for all three tasks.

Tasks
-----
easy   : Classify a login-issue ticket  → Category=Auth, Priority=High
medium : Resolve Error 992 using search_kb  → resolution must include KB answer
hard   : Triage 3 tickets in correct urgency order (Outage first)

Reward shaping
--------------
+0.1  tool used (search_kb called)
+0.7  correct resolution / classification
-0.2  wrong order in Task 3 (Outage not first)
Episode terminates (done=True) when the correct answer is submitted.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from .models import TriageAction, TriageObservation
except ImportError:
    from models import TriageAction, TriageObservation

# ── Knowledge base ────────────────────────────────────────────────────────────

KNOWLEDGE_BASE: Dict[str, str] = {
    "992": "Restart background service",
    "500": "Check server logs for exceptions",
    "404": "Verify resource path and permissions",
}

# ── Task definitions ──────────────────────────────────────────────────────────

TASKS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "description": "Classify the following support ticket.",
        "ticket": {
            "id": "TKT-001",
            "title": "Login issue",
            "description": "User cannot log in — receives 'Invalid credentials' error.",
        },
        "target_category": "auth",       # stored lower-case for comparison
        "target_priority": "high",
        "max_steps": 4,
    },
    "medium": {
        "description": (
            "Investigate and resolve the reported error using available tools. "
            "Call search_kb with the error code, then submit a resolution."
        ),
        "ticket": {
            "id": "TKT-002",
            "title": "Error 992",
            "description": "Application crashes with error code 992 on startup.",
        },
        "error_code": "992",
        "required_resolution_keyword": "restart background service",  # lower-case
        "max_steps": 6,
    },
    "hard": {
        "description": (
            "Three tickets are open. Submit an 'order' list with their IDs "
            "sorted from most to least urgent."
        ),
        "tickets": [
            {
                "id": "TKT-010",
                "title": "Billing discrepancy",
                "description": "Customer was charged twice for the same invoice.",
            },
            {
                "id": "TKT-011",
                "title": "Production outage",
                "description": "All users are unable to access the platform.",
            },
            {
                "id": "TKT-012",
                "title": "Feature request",
                "description": "Add dark-mode support to the mobile app.",
            },
        ],
        "correct_order": ["TKT-011", "TKT-010", "TKT-012"],
        "max_steps": 5,
    },
}


# ── Environment ───────────────────────────────────────────────────────────────


class SupportTriageEnvironment(Environment):
    """Support-triage environment with three difficulty levels."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._task_name: str = "easy"
        self._task_cfg: Dict[str, Any] = TASKS["easy"]
        self._state: State = State(episode_id=str(uuid.uuid4()), step_count=0)
        self._episode_done: bool = False
        self._used_tool: bool = False
        self._kb_result: Optional[str] = None

    # ── reset ──────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task: Optional[str] = None,
        **kwargs: Any,
    ) -> TriageObservation:
        task_name = (task or "easy").strip().lower()
        if task_name not in TASKS:
            task_name = "easy"

        self._task_name = task_name
        self._task_cfg = TASKS[task_name]
        self._state = State(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
        )
        self._episode_done = False
        self._used_tool = False
        self._kb_result = None

        return self._build_obs(
            feedback=(
                f"Episode started. Task: '{task_name}'. "
                f"{self._task_cfg['description']}"
            ),
            tool_result=None,
            done=False,
            reward=0.0,
        )

    # ── step ───────────────────────────────────────────────────────────────────

    def step(  # type: ignore[override]
        self,
        action: TriageAction,
        **kwargs: Any,
    ) -> TriageObservation:

        if self._episode_done:
            return self._build_obs(
                feedback="Episode already finished. Call reset() to start a new one.",
                done=True,
                reward=0.0,
            )

        self._state.step_count += 1
        max_steps: int = int(self._task_cfg.get("max_steps", 6))

        # Dispatch ──────────────────────────────────────────────────────────
        if self._task_name == "easy":
            obs = self._step_easy(action)
        elif self._task_name == "medium":
            obs = self._step_medium(action)
        else:
            obs = self._step_hard(action)

        # Force-terminate at max_steps ──────────────────────────────────────
        if not obs.done and self._state.step_count >= max_steps:
            self._episode_done = True
            return TriageObservation(
                task=self._task_name,
                ticket=self._current_ticket(),
                tool_result=obs.tool_result,
                feedback=(
                    f"Maximum steps ({max_steps}) reached. Episode terminated."
                ),
                step_count=self._state.step_count,
                max_steps=max_steps,
                done=True,
                reward=float(obs.reward) if obs.reward is not None else 0.0,
            )

        return obs

    # ── task: easy ─────────────────────────────────────────────────────────────

    def _step_easy(self, action: TriageAction) -> TriageObservation:
        cfg = self._task_cfg
        reward = 0.0
        parts: list[str] = []

        # Optional tool bonus
        if action.tool and action.tool.strip().lower() == "search_kb" and action.tool_input:
            self._used_tool = True
            self._kb_result = KNOWLEDGE_BASE.get(
                action.tool_input.strip(), "No entry found."
            )
            parts.append(f"KB result: {self._kb_result}")
            reward += 0.1

        # Robust comparison: strip + lower on BOTH sides
        submitted_cat = (action.category or "").strip().lower()
        submitted_pri = (action.priority or "").strip().lower()
        target_cat    = cfg["target_category"].strip().lower()   # "auth"
        target_pri    = cfg["target_priority"].strip().lower()   # "high"

        correct_cat = submitted_cat == target_cat
        correct_pri = submitted_pri == target_pri

        if correct_cat and correct_pri:
            reward += 0.7
            parts.append(
                f"Correct! Category={action.category}, Priority={action.priority}."
            )
            self._episode_done = True
            return self._build_obs(
                feedback=" ".join(parts), done=True, reward=reward
            )

        # Partial feedback
        if submitted_cat and not correct_cat:
            parts.append(f"Category '{action.category}' is incorrect.")
        if submitted_pri and not correct_pri:
            parts.append(f"Priority '{action.priority}' is incorrect.")
        if not submitted_cat and not submitted_pri:
            parts.append(
                "Please provide both 'category' and 'priority' fields in your action."
            )
        parts.append("Try again.")

        return self._build_obs(
            feedback=" ".join(parts), done=False, reward=reward
        )

    # ── task: medium ───────────────────────────────────────────────────────────

    def _step_medium(self, action: TriageAction) -> TriageObservation:
        cfg = self._task_cfg
        reward = 0.0
        parts: list[str] = []
        tool_result: Optional[str] = None

        # Handle tool call
        if action.tool and action.tool.strip().lower() == "search_kb":
            if action.tool_input:
                self._used_tool = True
                result = KNOWLEDGE_BASE.get(
                    action.tool_input.strip(), "No entry found for that code."
                )
                self._kb_result = result
                tool_result = result
                reward += 0.1
                parts.append(
                    f"KB result for '{action.tool_input.strip()}': {result}"
                )
            else:
                parts.append("search_kb requires a 'tool_input' (error code).")

        # Evaluate resolution — robust: strip + lower on both sides
        if action.resolution:
            keyword = cfg["required_resolution_keyword"].strip().lower()
            submitted = action.resolution.strip().lower()
            if keyword in submitted:
                reward += 0.7
                parts.append("Resolution accepted — correct fix identified.")
                self._episode_done = True
                return self._build_obs(
                    feedback=" ".join(parts),
                    tool_result=tool_result,
                    done=True,
                    reward=reward,
                )
            else:
                parts.append(
                    "Resolution submitted but does not contain the expected fix. "
                    "Hint: use search_kb with the error code to find the correct answer."
                )

        if not action.tool and not action.resolution:
            parts.append(
                "No action taken. Use search_kb to look up the error code, "
                "then submit a resolution."
            )

        return self._build_obs(
            feedback=" ".join(parts),
            tool_result=tool_result,
            done=False,
            reward=reward,
        )

    # ── task: hard ─────────────────────────────────────────────────────────────

    def _step_hard(self, action: TriageAction) -> TriageObservation:
        cfg = self._task_cfg
        reward = 0.0
        parts: list[str] = []
        tool_result: Optional[str] = None

        # Optional tool bonus
        if action.tool and action.tool.strip().lower() == "search_kb" and action.tool_input:
            self._used_tool = True
            self._kb_result = KNOWLEDGE_BASE.get(
                action.tool_input.strip(), "No entry found."
            )
            tool_result = self._kb_result
            reward += 0.1
            parts.append(f"KB result: {self._kb_result}")

        if action.order is not None:
            submitted = [t.strip() for t in action.order]
            correct = cfg["correct_order"]

            if submitted == correct:
                reward += 0.7
                parts.append(
                    f"Correct triage order: {submitted}. Outage handled first."
                )
                self._episode_done = True
                return self._build_obs(
                    feedback=" ".join(parts),
                    tool_result=tool_result,
                    done=True,
                    reward=reward,
                )

            # Penalise if outage is not first
            if not submitted or submitted[0] != "TKT-011":
                reward -= 0.2
                parts.append(
                    f"-0.2 penalty: production outage (TKT-011) must be first. "
                    f"Your order: {submitted}."
                )
            else:
                parts.append(
                    f"Partial order {submitted} is incorrect. "
                    "Check the relative urgency of Billing vs Feature."
                )
        else:
            parts.append(
                "Submit an 'order' list with ticket IDs sorted by urgency. "
                "Tickets: TKT-010 (Billing), TKT-011 (Outage), TKT-012 (Feature)."
            )

        return self._build_obs(
            feedback=" ".join(parts),
            tool_result=tool_result,
            done=False,
            reward=reward,
        )

    # ── helpers ────────────────────────────────────────────────────────────────

    def _current_ticket(self) -> Dict[str, Any]:
        if self._task_name == "hard":
            return {"tickets": self._task_cfg.get("tickets", [])}
        return self._task_cfg.get("ticket", {})

    def _build_obs(
        self,
        feedback: str,
        done: bool,
        reward: float,
        tool_result: Optional[str] = None,
    ) -> TriageObservation:
        if done:
            self._episode_done = True
        
       
        safe_reward = max(0.01, min(0.99, float(reward)))

        return TriageObservation(
            task=self._task_name,
            ticket=self._current_ticket(),
            tool_result=tool_result if tool_result is not None else self._kb_result,
            feedback=feedback,
            step_count=self._state.step_count,
            max_steps=int(self._task_cfg.get("max_steps", 6)),
            done=done,
            reward=safe_reward,  # Use the safe_reward here
        )

    @property
    def state(self) -> State:
        return self._state