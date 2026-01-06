from __future__ import annotations

from typing import Optional, List
from fastapi import APIRouter, Query, Response, HTTPException

from app.db import POOL
from app.utils.pass_networks import DEFAULT_PITCH_TYPE
from app.services.pass_networks import render_pass_network_png

router = APIRouter(prefix="/v1/passnetwork", tags=["passnetwork"])


@router.get("/image_v2", response_class=Response)
def pass_network_image_v2(
    match_id: int,
    team_id: int,

    player_ids: Optional[List[int]] = Query(
        default=None,
        description="Preferred: list of player IDs to include (filters nodes + edges).",
    ),

    cutoff_minute: Optional[int] = Query(default=None, ge=0, le=200),
    cutoff_second: int = Query(0, ge=0, le=59),
    cutoff_period: str = Query("1H"),

    min_edge_count: int = Query(2, ge=1, le=50),
    only_accurate: bool = Query(False),

    dpi: int = Query(160, ge=72, le=300),
    show_labels: bool = Query(True),

    use_xgchain: bool = Query(True),
    node_color_mode: str = Query("made"),
    edge_color_mode: str = Query("mean"),

    show_colorbar: bool = Query(True),

    pitch_type: str = Query(DEFAULT_PITCH_TYPE),
):
    try:
        png = render_pass_network_png(
            pool=POOL,
            match_id=match_id,
            team_id=team_id,
            player_ids=player_ids,
            cutoff_period=cutoff_period,
            cutoff_minute=cutoff_minute,
            cutoff_second=cutoff_second,
            min_edge_count=min_edge_count,
            only_accurate=only_accurate,
            dpi=dpi,
            show_labels=show_labels,
            use_xgchain=use_xgchain,
            node_color_mode=node_color_mode,
            edge_color_mode=edge_color_mode,
            show_colorbar=show_colorbar,
            pitch_type=pitch_type,
        )
        return Response(content=png, media_type="image/png")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"passnetwork image failed: {e}")
