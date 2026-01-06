from __future__ import annotations

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Query, Response, HTTPException

from app.db import POOL
from app.utils.possessions import (
    DEFAULT_PITCH_TYPE,
    DEFAULT_SHOT_TO_GOAL,
)
from app.services.possessions import (
    event_time_seconds,
    fetch_possession_list,
    fetch_possession_chain,
    render_possession_chain_image,
)

from app.schemas.possessions import (
    PossessionListResponse,
    PossessionChainResponse,
)


router = APIRouter(prefix="/v1/possessions", tags=["possessions"])


@router.get("/list", response_model=PossessionListResponse)
def possession_list(
    match_id: int,
    team_id: Optional[int] = None,

    from_minute: Optional[int] = Query(default=None, ge=0, le=200),
    from_second: int = Query(default=0, ge=0, le=59),
    from_period: str = Query(default="1H"),
    to_minute: Optional[int] = Query(default=None, ge=0, le=200),
    to_second: int = Query(default=0, ge=0, le=59),
    to_period: str = Query(default="1H"),

    only_possessions_with_shots: bool = Query(default=False),
    only_possessions_with_goals: bool = Query(default=False),

    limit: int = Query(default=500, ge=1, le=5000),
) -> Dict[str, Any]:
    try:
        return fetch_possession_list(
            pool=POOL,
            match_id=match_id,
            team_id=team_id,
            from_minute=from_minute,
            from_second=from_second,
            from_period=from_period,
            to_minute=to_minute,
            to_second=to_second,
            to_period=to_period,
            only_possessions_with_shots=only_possessions_with_shots,
            only_possessions_with_goals=only_possessions_with_goals,
            limit=limit,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"possession_list failed: {e}")


@router.get("/chain", response_model=PossessionChainResponse)
def possession_chain_json(
    match_id: int,
    possession_id: int,
    team_id: Optional[int] = None,
    include_other_team_events: bool = Query(default=False),
    limit: int = Query(default=600, ge=1, le=2000),
) -> Dict[str, Any]:
    try:
        return fetch_possession_chain(
            pool=POOL,
            match_id=match_id,
            possession_id=possession_id,
            team_id=team_id,
            include_other_team_events=include_other_team_events,
            limit=limit,
        )
    except LookupError:
        raise HTTPException(status_code=404, detail="No events found for that possession.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"possession_chain failed: {e}")


@router.get("/image_v2", response_class=Response)
def possession_chain_image_v2(
    match_id: int,
    possession_id: int,
    team_id: Optional[int] = None,
    include_other_team_events: bool = Query(default=False),

    # behavior params kept
    cutoff_minute: Optional[int] = Query(default=None, ge=0, le=200),
    cutoff_second: int = Query(default=0, ge=0, le=59),
    cutoff_period: str = Query(default="1H"),

    dpi: int = Query(default=180, ge=72, le=300),
    pitch_type: str = Query(default=DEFAULT_PITCH_TYPE),

    annotate_numbers: bool = Query(default=True),
    max_events: int = Query(default=120, ge=10, le=600),

    # behavior
    shot_to_goal: bool = Query(default=DEFAULT_SHOT_TO_GOAL),
) -> Response:
    cutoff_t = None
    if cutoff_minute is not None:
        cutoff_t = event_time_seconds(cutoff_period, cutoff_minute, cutoff_second)
        if cutoff_t is None:
            raise HTTPException(status_code=400, detail="Invalid cutoff time.")

    try:
        chain = fetch_possession_chain(
            pool=POOL,
            match_id=match_id,
            possession_id=possession_id,
            team_id=team_id,
            include_other_team_events=include_other_team_events,
            limit=2000,
        )
        png = render_possession_chain_image(
            chain=chain,
            cutoff_t=cutoff_t,
            dpi=dpi,
            annotate_numbers=annotate_numbers,
            max_events=max_events,
            pitch_type=pitch_type,
            shot_to_goal=shot_to_goal,
        )
        return Response(content=png, media_type="image/png")
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"possession_chain image failed: {e}")
