from __future__ import annotations

from pydantic import BaseModel
from typing import Optional, List


class HeatmapQuerySummary(BaseModel):
    player_id: Optional[int] = None
    competition_id: Optional[int] = None
    season_id: Optional[int] = None
    match_id: Optional[int] = None
    primary: Optional[str] = None
    primary_in: Optional[List[str]] = None
    secondary_any: Optional[List[str]] = None
