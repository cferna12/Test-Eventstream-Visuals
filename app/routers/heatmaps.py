from __future__ import annotations

from typing import List, Optional
from fastapi import APIRouter, Query, Response, HTTPException
import os
import io

import numpy as np
import psycopg
from psycopg import sql

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mplsoccer import Pitch
from scipy.ndimage import gaussian_filter

from app.db import POOL

router = APIRouter(prefix="/v1/heatmaps", tags=["heatmaps"])


# MUST match what you used in sql/01_bins_and_indexes.sql
BINS_X = int(os.environ.get("HEATMAP_BINS_X", "60"))
BINS_Y = int(os.environ.get("HEATMAP_BINS_Y", "40"))

ON_BALL_PRIMARY_DEFAULT = [
    "touch", "pass", "received_pass", "duel", "interception", "shot", "carry",
    "clearance", "acceleration", "pressing_attempt", "infraction",
]

def build_where_and_params(
    *,
    player_id: int,
    competition_id: Optional[int],
    season_id: Optional[int],
    match_id: Optional[int],
    primary: Optional[str],
    primary_in: Optional[List[str]],
    secondary_any: Optional[List[str]],
    x_min: Optional[float],
    x_max: Optional[float],
    y_min: Optional[float],
    y_max: Optional[float],
):
    where = ["player_id = %s"]
    params = [player_id]

    if competition_id is not None:
        where.append("competition_id = %s")
        params.append(competition_id)

    if season_id is not None:
        where.append("season_id = %s")
        params.append(season_id)

    if match_id is not None:
        where.append("match_id = %s")
        params.append(match_id)

    if primary is not None:
        where.append("type_primary = %s")
        params.append(primary)
    elif primary_in:
        where.append("type_primary = ANY(%s)")
        params.append(primary_in)

    if secondary_any:
        where.append("type_secondary && %s")
        params.append(secondary_any)

    if x_min is not None:
        where.append("location_x >= %s")
        params.append(x_min)
    if x_max is not None:
        where.append("location_x <= %s")
        params.append(x_max)
    if y_min is not None:
        where.append("location_y >= %s")
        params.append(y_min)
    if y_max is not None:
        where.append("location_y <= %s")
        params.append(y_max)

    where.append("location_x IS NOT NULL")
    where.append("location_y IS NOT NULL")
    where.append("bin_x IS NOT NULL")
    where.append("bin_y IS NOT NULL")

    return " AND ".join(where), params


@router.get("/heatmap/image", response_class=Response)
def heatmap_image_binned(
    player_id: int,
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

    sigma: float = Query(default=1.6, ge=0.0, le=10.0),
    blur: bool = Query(default=True),

    dpi: int = Query(default=150, ge=72, le=300),
    pitch_type: str = "wyscout",
):
    primary_in = None
    if primary is None and use_on_ball_default:
        primary_in = ON_BALL_PRIMARY_DEFAULT

    where_sql, params = build_where_and_params(
        player_id=player_id,
        competition_id=competition_id,
        season_id=season_id,
        match_id=match_id,
        primary=primary,
        primary_in=primary_in,
        secondary_any=secondary_any,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )

    q = sql.SQL(f"""
        SELECT bin_x, bin_y, COUNT(*)::INT
        FROM eventstream_events
        WHERE {where_sql}
        GROUP BY bin_x, bin_y
    """)

    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(q, params)
            rows = cur.fetchall()

    grid = np.zeros((BINS_Y, BINS_X), dtype=np.float32)
    for bx, by, n in rows:
        if bx is None or by is None:
            continue
        bx_i = int(bx)
        by_i = int(by)
        if 0 <= bx_i < BINS_X and 0 <= by_i < BINS_Y:
            grid[by_i, bx_i] = n

    if blur and sigma > 0:
        grid = gaussian_filter(grid, sigma=sigma)

    if grid.max() > 0:
        grid /= grid.max()

    pitch = Pitch(pitch_type=pitch_type)
    fig, ax = pitch.draw(figsize=(10, 7))

    ax.imshow(
        grid,
        extent=(0, 100, 0, 100),
        origin="lower",
        aspect="auto",
        interpolation="bilinear",
        alpha=0.95,
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")
