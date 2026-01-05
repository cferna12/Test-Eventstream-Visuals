from __future__ import annotations

from typing import Optional, List, Literal
from pydantic import BaseModel


class Edge(BaseModel):
    passer_id: int
    receiver_id: int
    n: int
    n_accurate: int
    accuracy_pct: Optional[float]


class TeamEdgesResponse(BaseModel):
    match_id: int
    team_id: int
    edges: List[Edge]


class PassSegment(BaseModel):
    event_id: int
    minute: Optional[int] = None
    second: Optional[int] = None
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    accurate: Optional[bool] = None


class SegmentsResponse(BaseModel):
    passer_id: int
    receiver_id: int
    match_id: Optional[int] = None
    competition_id: Optional[int] = None
    season_id: Optional[int] = None
    direction: Literal["forward", "both"] = "forward"
    n: int
    segments: List[PassSegment]


class NodePos(BaseModel):
    player_id: int
    n: int
    avg_x: float
    avg_y: float


class NodePositionsResponse(BaseModel):
    match_id: Optional[int] = None
    competition_id: Optional[int] = None
    season_id: Optional[int] = None
    team_id: Optional[int] = None
    mode: Literal["pass_start", "reception_end"] = "pass_start"
    nodes: List[NodePos]
