from __future__ import annotations

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# -----------------------------
# Possession LIST schemas
# -----------------------------

class PossessionSummary(BaseModel):
    possession_id: int

    # timing
    start_time_seconds: int
    end_time_seconds: int
    duration_seconds: Optional[int]

    # volume
    n_events: int

    # inferred dominant team
    team_id: Optional[int]

    # event ids for ordering/debug
    first_event_id: int
    last_event_id: int


class PossessionListFilters(BaseModel):
    only_possessions_with_shots: bool
    only_possessions_with_goals: bool


class PossessionListResponse(BaseModel):
    match_id: int
    team_id_filter: Optional[int]

    time_window: Dict[str, Optional[int]]  # {"from": int|None, "to": int|None}

    filters: PossessionListFilters

    n_possessions: int
    possessions: List[PossessionSummary]


# -----------------------------
# Possession CHAIN schemas
# -----------------------------

class PossessionEvent(BaseModel):
    # identity / ordering
    id: int
    t: Optional[int] = Field(description="Event time in comparable seconds")

    match_period: Optional[str]
    minute: Optional[int]
    second: Optional[int]

    # team / player
    team_id: Optional[int]
    player_id: Optional[int]
    player_name: Optional[str]

    # classification
    type_primary: Optional[str]
    type_secondary: Optional[List[Any]]
    label: str

    # spatial
    x: Optional[float]
    y: Optional[float]
    end_x: Optional[float]
    end_y: Optional[float]

    # pass extras
    recipient_id: Optional[int]
    recipient_name: Optional[str]

    # shot extras
    is_goal: Optional[bool]
    xg: Optional[float]
    body_part: Optional[str]
    # NOTE: goal_zone intentionally omitted here
    # add later if you expose it in JSON


class PossessionChainResponse(BaseModel):
    match_id: int
    possession_id: int

    team_id_filter: Optional[int]
    teams_in_events: List[int]

    n: int
    duration_seconds: Optional[int]

    events: List[PossessionEvent]
