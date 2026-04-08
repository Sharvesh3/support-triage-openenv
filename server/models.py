"""
Pydantic Action and Observation models for the Support Triage environment.
"""

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class TriageAction(Action):
    """
    Action submitted by the agent each step.

    All fields are optional — submit whichever apply:
      category   : ticket category label  ("Auth", "Billing", "Outage", "Feature")
      priority   : ticket priority label  ("High", "Medium", "Low")
      tool       : tool to call           (only "search_kb" is supported)
      tool_input : argument for the tool  (e.g. error code "992")
      resolution : free-text fix description
      order      : ordered ticket-ID list for multi-ticket triage (Task 3)
    """

    category: Optional[str] = Field(default=None, description="Ticket category label")
    priority: Optional[str] = Field(default=None, description="Ticket priority label")
    tool: Optional[str] = Field(default=None, description="Tool to invoke (search_kb)")
    tool_input: Optional[str] = Field(default=None, description="Tool argument")
    resolution: Optional[str] = Field(default=None, description="Resolution text")
    order: Optional[List[str]] = Field(
        default=None, description="Ordered ticket IDs for triage (Task 3)"
    )


class TriageObservation(Observation):
    """
    Observation returned after each environment step.

    NOTE: `reward` (float) and `done` (bool) are inherited from the
    openenv-core base Observation class and MUST NOT be redeclared here.
    The serializer reads them directly from the base-class fields and
    places them at the top level of the JSON response envelope.
    Redeclaring them with a different type (e.g. bool instead of
    float | int | bool | None) causes Pydantic validation errors that
    silently zero-out the reward.
    """

    task: str = Field(default="", description="Active task name")
    ticket: Dict[str, Any] = Field(
        default_factory=dict, description="Current ticket details"
    )
    tool_result: Optional[str] = Field(
        default=None, description="Result from a tool call, if any"
    )
    feedback: str = Field(default="", description="Feedback on the last action")
    step_count: int = Field(default=0, description="Steps taken this episode")
    max_steps: int = Field(default=6, description="Max steps allowed per episode")