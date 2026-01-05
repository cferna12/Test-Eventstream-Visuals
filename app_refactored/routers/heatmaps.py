from __future__ import annotations

from typing import List, Optional, Literal
from fastapi import APIRouter, Query, Response, HTTPException

from app_refactored.db import POOL
from app_refactored.services.heatmaps import render_heatmap_image

router = APIRouter(prefix="/v1/heatmaps", tags=["heatmaps"])


@router.get("/heatmap/image_v2", response_class=Response)
def heatmap_image_binned_v2(
    # choose one (player OR team) — team not implemented here
    player_id: Optional[int] = None,

    competition_id: Optional[int] = None,
    season_id: Optional[int] = None,
    match_id: Optional[int] = None,

    primary: Optional[str] = Query(default=None),
    use_on_ball_default: bool = Query(default=True),
    secondary_any: Optional[List[str]] = Query(default=None),

    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,

    # kept as params (per your request)
    interpolation: Literal["bilinear", "nearest", "bicubic", "gaussian"] = Query(default="bilinear"),
    show_contours: bool = Query(default=False),
    contour_levels: int = Query(default=6, ge=3, le=15),
    show_colorbar: bool = Query(default=False),
    dpi: int = Query(default=150, ge=72, le=400),
):
    try:
        png_bytes = render_heatmap_image(
            pool=POOL,
            player_id=player_id,
            competition_id=competition_id,
            season_id=season_id,
            match_id=match_id,
            primary=primary,
            use_on_ball_default=use_on_ball_default,
            secondary_any=secondary_any,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            interpolation=interpolation,
            show_contours=show_contours,
            contour_levels=contour_levels,
            show_colorbar=show_colorbar,
            dpi=dpi,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except LookupError:
        raise HTTPException(status_code=404, detail="No events found for requested filters.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Heatmap render failed: {e}")

    return Response(content=png_bytes, media_type="image/png")
