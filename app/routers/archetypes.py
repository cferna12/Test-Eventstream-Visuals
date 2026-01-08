from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.db import POOL
from app.schemas.archetypes import ArchetypeTop1Response, ArchetypeTop3Response
from app.services.archetypes import get_top1_pred, get_top3_pred

router = APIRouter(prefix="/v1/archetypes", tags=["archetypes"])

DEFAULT_SEASON = "2025 Fall"


@router.get("/top1", response_model=ArchetypeTop1Response)
def archetype_top1(
    player_id: str = Query(..., description="Player ID (stored as TEXT in DB)"),
    season_name: str = Query(DEFAULT_SEASON, description="Season name, e.g. '2025 Fall'"),
):
    """
    Hard-label endpoint: returns Top1 archetype (and probability if stored).
    """
    try:
        data = get_top1_pred(POOL, player_id=player_id, season_name=season_name)
        return data
    except LookupError:
        raise HTTPException(status_code=404, detail="No archetype prediction found for this player/season.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Archetype top1 query failed: {e}")


@router.get("/top3", response_model=ArchetypeTop3Response)
def archetype_top3(
    player_id: str = Query(..., description="Player ID (stored as TEXT in DB)"),
    season_name: str = Query(DEFAULT_SEASON, description="Season name, e.g. '2025 Fall'"),
):
    """
    Top-3 endpoint: returns (Top1..Top3) labels + probabilities if stored.
    Note: this is NOT the full 18-way probability vector; it's only the stored top-3.
    """
    try:
        data = get_top3_pred(POOL, player_id=player_id, season_name=season_name)
        return data
    except LookupError:
        raise HTTPException(status_code=404, detail="No archetype prediction found for this player/season.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Archetype top3 query failed: {e}")
