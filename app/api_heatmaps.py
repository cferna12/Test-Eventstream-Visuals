# main.py
import os
from typing import List, Optional, Tuple, Literal

from fastapi import FastAPI, Query, Response, HTTPException
from pydantic import BaseModel
import psycopg
from psycopg import sql

import numpy as np

# Server-side rendering deps (install: matplotlib scipy)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from io import BytesIO


# -----------------------------
# Config
# -----------------------------
PG_DSN = os.environ.get("PG_DSN", "postgresql://postgres:postgres@localhost:5432/eventstream")

# Wyscout pitch coordinates are typically 0..100 for both axes
PITCH_X_MIN, PITCH_X_MAX = 0.0, 100.0
PITCH_Y_MIN, PITCH_Y_MAX = 0.0, 100.0

app = FastAPI(title="Eventstream Heatmaps (POC)", version="0.1.0")


# -----------------------------
# Models
# -----------------------------
class Point(BaseModel):
    x: float
    y: float


class HeatmapPointsResponse(BaseModel):
    player_id: int
    primary: str
    n_points: int
    points: List[Point]
    # optional: pass end points (if requested)
    end_points: Optional[List[Point]] = None


ON_BALL_PRIMARY_DEFAULT = [
    "touch",
    "pass",
    "received_pass",
    "duel",
    "interception",
    "shot",
    "carry",
    "clearance",
    "acceleration",
    "pressing_attempt",
    "infraction",
]
# -----------------------------
# DB helpers
# -----------------------------
def get_conn():
    return psycopg.connect(PG_DSN)




def _bbox_where(
    bbox: Optional[Tuple[float, float, float, float]],
    locx_col: str = "location_x",
    locy_col: str = "location_y",
):
    """
    bbox = (x_min, x_max, y_min, y_max)
    """
    if not bbox:
        return sql.SQL("TRUE"), []
    x_min, x_max, y_min, y_max = bbox
    clause = sql.SQL("{lx} BETWEEN %s AND %s AND {ly} BETWEEN %s AND %s").format(
        lx=sql.Identifier(locx_col),
        ly=sql.Identifier(locy_col),
    )
    return clause, [x_min, x_max, y_min, y_max]


def _secondary_any_where(secondary_any: Optional[List[str]]):
    """
    Match if any requested secondary appears in type_secondary array.
    Uses overlap operator: type_secondary && ARRAY[...]
    """
    if not secondary_any:
        return sql.SQL("TRUE"), []
    # psycopg will adapt Python list -> Postgres array
    return sql.SQL("type_secondary && %s"), [secondary_any]



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

    # “any of these secondary tags”
    if secondary_any:
        where.append("type_secondary && %s")
        params.append(secondary_any)

    # pitch windowing (defensive third etc.)
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

    # avoid null locations
    where.append("location_x IS NOT NULL")
    where.append("location_y IS NOT NULL")

    return " AND ".join(where), params

def _fetch_points(q: sql.SQL, params: list, limit: Optional[int] = None) -> np.ndarray:
    if limit is not None:
        q = q + sql.SQL(" LIMIT %s")
        params = params + [limit]

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q, params)
            rows = cur.fetchall()

    if not rows:
        return np.zeros((0, 2), dtype=float)

    arr = np.array(rows, dtype=float)
    # arr[:,0]=x, arr[:,1]=y
    return arr


def _fetch_pass_end_points(
    *,
    player_id: int,
    match_id: Optional[int],
    competition_id: Optional[int],
    season_id: Optional[int],
    secondary_any: Optional[List[str]],
    bbox: Optional[Tuple[float, float, float, float]],
    limit: Optional[int] = None,
) -> np.ndarray:
    """
    Fetch end points for passes using eventstream_passes join.
    bbox applies to *start* location in events (common heatmap interpretation).
    If you want bbox on end location instead, change bbox columns to p.end_x/p.end_y.
    """
    where_parts = [
        sql.SQL("e.player_id = %s"),
        sql.SQL("e.type_primary = 'pass'"),
    ]
    params = [player_id]

    if match_id is not None:
        where_parts.append(sql.SQL("e.match_id = %s"))
        params.append(match_id)

    if competition_id is not None:
        where_parts.append(sql.SQL("e.competition_id = %s"))
        params.append(competition_id)

    if season_id is not None:
        where_parts.append(sql.SQL("e.season_id = %s"))
        params.append(season_id)

    sec_clause, sec_params = _secondary_any_where(secondary_any)
    # sec_clause references type_secondary column directly; qualify it:
    sec_clause = sql.SQL(str(sec_clause)).replace("type_secondary", "e.type_secondary")
    where_parts.append(sec_clause)
    params.extend(sec_params)

    bbox_clause, bbox_params = _bbox_where(bbox, "location_x", "location_y")
    bbox_clause = sql.SQL(str(bbox_clause)).replace("location_x", "e.location_x").replace("location_y", "e.location_y")
    where_parts.append(bbox_clause)
    params.extend(bbox_params)

    q = sql.SQL("""
        SELECT p.end_x, p.end_y
        FROM eventstream_events e
        JOIN eventstream_passes p ON p.event_id = e.id
        WHERE {where}
          AND p.end_x IS NOT NULL
          AND p.end_y IS NOT NULL
    """).format(where=sql.SQL(" AND ").join(where_parts))

    if limit is not None:
        q = q + sql.SQL(" LIMIT %s")
        params = params + [limit]

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q, params)
            rows = cur.fetchall()

    if not rows:
        return np.zeros((0, 2), dtype=float)

    return np.array(rows, dtype=float)



@app.get("/v1/heatmaps/points", response_model=HeatmapPointsResponse)
def get_heatmap_points(
    player_id: int = Query(..., description="player_id"),

    # CHANGED: primary is optional now
    primary: Optional[str] = Query(default=None, description="Primary action type (type_primary)"),

    # NEW: default behavior matches /heatmaps/image
    use_on_ball_default: bool = Query(default=True),

    match_id: Optional[int] = Query(None),
    competition_id: Optional[int] = Query(None),
    season_id: Optional[int] = Query(None),

    secondary_any: Optional[List[str]] = Query(None, description="Match if any secondary types appear"),
    bbox: Optional[str] = Query(None, description="bbox as 'x_min,x_max,y_min,y_max' (e.g. '0,33.3,0,100')"),
    limit: Optional[int] = Query(None, description="Optional cap for debugging"),

    include_pass_ends: bool = Query(False, description="If primary=pass, also return pass end points"),
):
    bbox_t: Optional[Tuple[float, float, float, float]] = None
    if bbox:
        try:
            parts = [float(x.strip()) for x in bbox.split(",")]
            if len(parts) != 4:
                raise ValueError()
            bbox_t = (parts[0], parts[1], parts[2], parts[3])
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid bbox. Use 'x_min,x_max,y_min,y_max'")

    # NEW: interpret default “touches” as on-ball primaries
    primary_in = None
    if primary is None and use_on_ball_default:
        primary_in = ON_BALL_PRIMARY_DEFAULT

    # If they ask for pass end points, we must be in "pass only" mode.
    # Otherwise you'd be returning end points for passes while points includes many other actions.
    if include_pass_ends and primary != "pass":
        raise HTTPException(status_code=400, detail="include_pass_ends=true requires primary=pass")

    # Reuse your existing helper that already supports primary_in
    where_sql, params = build_where_and_params(
        player_id=player_id,
        competition_id=competition_id,
        season_id=season_id,
        match_id=match_id,
        primary=primary,
        primary_in=primary_in,
        secondary_any=secondary_any,
        x_min=bbox_t[0] if bbox_t else None,
        x_max=bbox_t[1] if bbox_t else None,
        y_min=bbox_t[2] if bbox_t else None,
        y_max=bbox_t[3] if bbox_t else None,
    )

    q = sql.SQL(f"""
        SELECT location_x, location_y
        FROM eventstream_events
        WHERE {where_sql}
          AND location_x IS NOT NULL
          AND location_y IS NOT NULL
    """)

    pts = _fetch_points(q, params, limit=limit)

    end_pts = None
    if include_pass_ends and primary == "pass":
        end = _fetch_pass_end_points(
            player_id=player_id,
            match_id=match_id,
            competition_id=competition_id,
            season_id=season_id,
            secondary_any=secondary_any,
            bbox=bbox_t,
            limit=limit,
        )
        end_pts = [Point(x=float(x), y=float(y)) for x, y in end]

    # Response: primary is now either the explicit primary or the “group default”
    primary_label = primary if primary is not None else "on_ball_default"

    return HeatmapPointsResponse(
        player_id=player_id,
        primary=primary_label,
        n_points=int(pts.shape[0]),
        points=[Point(x=float(x), y=float(y)) for x, y in pts],
        end_points=end_pts,
    )


import io
from mplsoccer import Pitch

@app.get("/heatmaps/image")
def heatmap_image(
    player_id: int,
    competition_id: Optional[int] = None,
    season_id: Optional[int] = None,
    match_id: Optional[int] = None,

    # If you pass `primary`, it overrides the group default.
    primary: Optional[str] = Query(default=None),

    # Default behavior: “on-ball actions”
    use_on_ball_default: bool = Query(default=True),

    # Secondary filters
    secondary_any: Optional[List[str]] = Query(default=None),

    # Region filters (example: defensive third = x_max=33.3 if using 0-100 scale)
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,

    # KDE controls
    levels: int = 10,
    alpha: float = 0.8,

    # Pitch settings: Wyscout is typically 0..100.
    pitch_type: str = "wyscout",  # good default for 0-100 coords
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

    sql = f"""
        SELECT location_x, location_y
        FROM eventstream_events
        WHERE {where_sql}
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    # If no points, return a blank pitch (still valid PNG)
    x = [r[0] for r in rows] if rows else []
    y = [r[1] for r in rows] if rows else []

    pitch = Pitch(pitch_type=pitch_type)
    fig, ax = pitch.draw(figsize=(10, 7))

    if rows:
        pitch.kdeplot(
            x,
            y,
            fill=True,
            levels=levels,
            alpha=alpha,
            ax=ax,
        )

    # Render to PNG bytes
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)

    # Important: close figure to avoid memory leak under load
    import matplotlib.pyplot as plt
    plt.close(fig)

    return Response(content=buf.getvalue(), media_type="image/png")