from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel


class PointsResponse(BaseModel):
    player_id: int
    n: int
    points_xy: List[List[float]]


class PassPointsResponse(BaseModel):
    player_id: int
    n: int
    points_xy: List[List[float]]          # [[x,y], ...]
    end_points_xy: List[List[float]]      # [[end_x,end_y], ...]
    accurate: List[Optional[bool]]        # aligned by index


class CarryPointsResponse(BaseModel):
    player_id: int
    n: int
    points_xy: List[List[float]]
    end_points_xy: List[List[float]]
    progression: List[Optional[float]]    # aligned by index


class ShotPointsResponse(BaseModel):
    player_id: int
    n: int
    points_xy: List[List[float]]

    is_goal: List[Optional[bool]]
    on_target: List[Optional[bool]]
    goal_zone: List[Optional[str]]
    xg: List[Optional[float]]
    body_part: List[Optional[str]]


class GroundDuelPointsResponse(BaseModel):
    player_id: int
    n: int
    points_xy: List[List[float]]
    recovered_possession: List[Optional[bool]]
    kept_possession: List[Optional[bool]]
    type_secondary: List[Optional[list]]


class AerialDuelPointsResponse(BaseModel):
    player_id: int
    n: int
    points_xy: List[List[float]]
    first_touch: List[Optional[bool]]
