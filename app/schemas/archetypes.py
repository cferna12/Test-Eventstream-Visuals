from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class ArchetypeTop1Response(BaseModel):
    player_id: str
    season_name: str
    season_id: Optional[str] = None
    player_name: Optional[str] = None

    top1: str
    top1_p: Optional[float] = None

    updated_at: Optional[datetime] = None


class ArchetypeProbItem(BaseModel):
    label: str = Field(..., description="Archetype label")
    p: Optional[float] = Field(None, description="Probability (if available)")


class ArchetypeTop3Response(BaseModel):
    player_id: str
    season_name: str
    season_id: Optional[str] = None
    player_name: Optional[str] = None

    topk: List[ArchetypeProbItem]

    updated_at: Optional[datetime] = None
