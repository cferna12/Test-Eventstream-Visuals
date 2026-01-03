# Binning and speed improvements as well as division of primary types


# '''
# Binning and speed improvements (theoretically)
# '''

# import os
# import io
# import time
# import hashlib
# from typing import List, Optional, Tuple, Literal, Dict, Any

# from fastapi import FastAPI, Query, Response, HTTPException
# from pydantic import BaseModel
# import psycopg
# from psycopg import sql

# import numpy as np

# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from scipy.ndimage import gaussian_filter  # <-- use this instead of gaussian_kde
# from mplsoccer import Pitch


# # -----------------------------
# # Config
# # -----------------------------
# PG_DSN = os.environ.get("PG_DSN", "postgresql://postgres:postgres@localhost:5432/eventstream")

# PITCH_X_MIN, PITCH_X_MAX = 0.0, 100.0
# PITCH_Y_MIN, PITCH_Y_MAX = 0.0, 100.0

# # Must match your GENERATED columns in SQL
# BINS_X = int(os.environ.get("HEATMAP_BINS_X", "60"))
# BINS_Y = int(os.environ.get("HEATMAP_BINS_Y", "40"))

# # Simple in-process cache for PNG bytes (works great for dev / single worker).
# # For multi-worker or distributed, swap this for Redis (fastapi-cache2).
# IMAGE_CACHE_MAX = int(os.environ.get("HEATMAP_IMAGE_CACHE_MAX", "2000"))
# _IMAGE_CACHE: Dict[str, Tuple[float, bytes]] = {}  # key -> (ts, png_bytes)


# app = FastAPI(title="Eventstream Heatmaps (fast bins)", version="0.2.0")


# # -----------------------------
# # Models
# # -----------------------------
# class Point(BaseModel):
#     x: float
#     y: float


# class HeatmapPointsResponse(BaseModel):
#     player_id: int
#     primary: str
#     n_points: int
#     points: List[Point]
#     end_points: Optional[List[Point]] = None


# ON_BALL_PRIMARY_DEFAULT = [
#     "touch",
#     "pass",
#     "received_pass",
#     "duel",
#     "interception",
#     "shot",
#     "carry",
#     "clearance",
#     "acceleration",
#     "pressing_attempt",
#     "infraction",
# ]


# # -----------------------------
# # DB helpers
# # -----------------------------
# def get_conn():
#     # IMPORTANT: for speed, keep this simple; production should use a pool
#     return psycopg.connect(PG_DSN)


# def build_where_and_params(
#     *,
#     player_id: int,
#     competition_id: Optional[int],
#     season_id: Optional[int],
#     match_id: Optional[int],
#     primary: Optional[str],
#     primary_in: Optional[List[str]],
#     secondary_any: Optional[List[str]],
#     x_min: Optional[float],
#     x_max: Optional[float],
#     y_min: Optional[float],
#     y_max: Optional[float],
# ):
#     where = ["player_id = %s"]
#     params = [player_id]

#     if competition_id is not None:
#         where.append("competition_id = %s")
#         params.append(competition_id)

#     if season_id is not None:
#         where.append("season_id = %s")
#         params.append(season_id)

#     if match_id is not None:
#         where.append("match_id = %s")
#         params.append(match_id)

#     if primary is not None:
#         where.append("type_primary = %s")
#         params.append(primary)
#     elif primary_in:
#         where.append("type_primary = ANY(%s)")
#         params.append(primary_in)

#     if secondary_any:
#         where.append("type_secondary && %s")
#         params.append(secondary_any)

#     if x_min is not None:
#         where.append("location_x >= %s")
#         params.append(x_min)
#     if x_max is not None:
#         where.append("location_x <= %s")
#         params.append(x_max)
#     if y_min is not None:
#         where.append("location_y >= %s")
#         params.append(y_min)
#     if y_max is not None:
#         where.append("location_y <= %s")
#         params.append(y_max)

#     where.append("location_x IS NOT NULL")
#     where.append("location_y IS NOT NULL")

#     return " AND ".join(where), params


# def _fetch_points(q: sql.SQL, params: list, limit: Optional[int] = None) -> np.ndarray:
#     if limit is not None:
#         q = q + sql.SQL(" LIMIT %s")
#         params = params + [limit]

#     with get_conn() as conn:
#         with conn.cursor() as cur:
#             cur.execute(q, params)
#             rows = cur.fetchall()

#     if not rows:
#         return np.zeros((0, 2), dtype=float)

#     return np.array(rows, dtype=float)


# def _fetch_bin_counts(where_sql: str, params: list) -> np.ndarray:
#     """
#     Returns rows of (bin_x, bin_y, n).
#     This is the fast path for season-level (and most) heatmaps.
#     """
#     q = sql.SQL(f"""
#         SELECT bin_x, bin_y, COUNT(*)::INT AS n
#         FROM eventstream_events
#         WHERE {where_sql}
#         GROUP BY bin_x, bin_y
#     """)

#     with get_conn() as conn:
#         with conn.cursor() as cur:
#             cur.execute(q, params)
#             rows = cur.fetchall()

#     if not rows:
#         return np.zeros((0, 3), dtype=int)

#     return np.array(rows, dtype=int)


# def _rows_to_grid(rows_bxy_n: np.ndarray, bins_x: int, bins_y: int) -> np.ndarray:
#     """
#     Build a (bins_y, bins_x) grid where grid[y, x] = count.
#     """
#     grid = np.zeros((bins_y, bins_x), dtype=np.float32)
#     if rows_bxy_n.size == 0:
#         return grid

#     bx = rows_bxy_n[:, 0]
#     by = rows_bxy_n[:, 1]
#     n = rows_bxy_n[:, 2].astype(np.float32)

#     # Guard against any out-of-range values (shouldn't happen with GENERATED cols)
#     m = (bx >= 0) & (bx < bins_x) & (by >= 0) & (by < bins_y)
#     bx, by, n = bx[m], by[m], n[m]

#     grid[by, bx] = n
#     return grid


# def _cache_get(key: str) -> Optional[bytes]:
#     v = _IMAGE_CACHE.get(key)
#     if not v:
#         return None
#     return v[1]


# def _cache_put(key: str, png_bytes: bytes) -> None:
#     _IMAGE_CACHE[key] = (time.time(), png_bytes)
#     # crude eviction
#     if len(_IMAGE_CACHE) > IMAGE_CACHE_MAX:
#         # drop ~10% oldest
#         items = sorted(_IMAGE_CACHE.items(), key=lambda kv: kv[1][0])
#         drop = max(1, len(items) // 10)
#         for k, _ in items[:drop]:
#             _IMAGE_CACHE.pop(k, None)


# def _make_cache_key(payload: Dict[str, Any]) -> str:
#     s = repr(sorted(payload.items())).encode("utf-8")
#     return hashlib.sha1(s).hexdigest()


# # -----------------------------
# # Endpoints
# # -----------------------------
# @app.get("/v1/heatmaps/points", response_model=HeatmapPointsResponse)
# def get_heatmap_points(
#     player_id: int = Query(...),
#     primary: Optional[str] = Query(default=None),
#     use_on_ball_default: bool = Query(default=True),
#     match_id: Optional[int] = Query(None),
#     competition_id: Optional[int] = Query(None),
#     season_id: Optional[int] = Query(None),
#     secondary_any: Optional[List[str]] = Query(None),
#     bbox: Optional[str] = Query(None, description="bbox 'x_min,x_max,y_min,y_max'"),
#     limit: Optional[int] = Query(None),
# ):
#     bbox_t: Optional[Tuple[float, float, float, float]] = None
#     if bbox:
#         try:
#             parts = [float(x.strip()) for x in bbox.split(",")]
#             if len(parts) != 4:
#                 raise ValueError()
#             bbox_t = (parts[0], parts[1], parts[2], parts[3])
#         except Exception:
#             raise HTTPException(status_code=400, detail="Invalid bbox. Use 'x_min,x_max,y_min,y_max'")

#     primary_in = None
#     if primary is None and use_on_ball_default:
#         primary_in = ON_BALL_PRIMARY_DEFAULT

#     where_sql, params = build_where_and_params(
#         player_id=player_id,
#         competition_id=competition_id,
#         season_id=season_id,
#         match_id=match_id,
#         primary=primary,
#         primary_in=primary_in,
#         secondary_any=secondary_any,
#         x_min=bbox_t[0] if bbox_t else None,
#         x_max=bbox_t[1] if bbox_t else None,
#         y_min=bbox_t[2] if bbox_t else None,
#         y_max=bbox_t[3] if bbox_t else None,
#     )

#     q = sql.SQL(f"""
#         SELECT location_x, location_y
#         FROM eventstream_events
#         WHERE {where_sql}
#     """)

#     pts = _fetch_points(q, params, limit=limit)
#     primary_label = primary if primary is not None else "on_ball_default"

#     return HeatmapPointsResponse(
#         player_id=player_id,
#         primary=primary_label,
#         n_points=int(pts.shape[0]),
#         points=[Point(x=float(x), y=float(y)) for x, y in pts],
#         end_points=None,
#     )


# @app.get("/heatmaps/image")
# def heatmap_image(
#     player_id: int,
#     competition_id: Optional[int] = None,
#     season_id: Optional[int] = None,
#     match_id: Optional[int] = None,

#     primary: Optional[str] = Query(default=None),
#     use_on_ball_default: bool = Query(default=True),
#     secondary_any: Optional[List[str]] = Query(default=None),

#     x_min: Optional[float] = None,
#     x_max: Optional[float] = None,
#     y_min: Optional[float] = None,
#     y_max: Optional[float] = None,

#     # New: blur controls for gaussian-filtered histogram
#     sigma: float = Query(default=1.6, ge=0.0, le=10.0),
#     # Optional: for very sparse filters you can skip blur
#     blur: bool = Query(default=True),

#     # Render quality
#     dpi: int = Query(default=150, ge=72, le=300),
#     pitch_type: str = "wyscout",

#     # Cache control
#     cache: bool = Query(default=True),
# ):
#     primary_in = None
#     if primary is None and use_on_ball_default:
#         primary_in = ON_BALL_PRIMARY_DEFAULT

#     where_sql, params = build_where_and_params(
#         player_id=player_id,
#         competition_id=competition_id,
#         season_id=season_id,
#         match_id=match_id,
#         primary=primary,
#         primary_in=primary_in,
#         secondary_any=secondary_any,
#         x_min=x_min,
#         x_max=x_max,
#         y_min=y_min,
#         y_max=y_max,
#     )

#     cache_key = _make_cache_key(
#         dict(
#             player_id=player_id,
#             competition_id=competition_id,
#             season_id=season_id,
#             match_id=match_id,
#             primary=primary,
#             use_on_ball_default=use_on_ball_default,
#             secondary_any=tuple(secondary_any) if secondary_any else None,
#             x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
#             sigma=sigma, blur=blur, dpi=dpi, pitch_type=pitch_type,
#             bins_x=BINS_X, bins_y=BINS_Y,
#         )
#     )

#     if cache:
#         hit = _cache_get(cache_key)
#         if hit is not None:
#             return Response(content=hit, media_type="image/png")

#     # 1) DB: fetch aggregated bin counts (FAST)
#     rows_bxy_n = _fetch_bin_counts(where_sql, params)

#     # 2) Build grid and blur
#     grid = _rows_to_grid(rows_bxy_n, BINS_X, BINS_Y)

#     if blur and sigma > 0:
#         grid = gaussian_filter(grid, sigma=sigma)

#     # 3) Render
#     pitch = Pitch(pitch_type=pitch_type)
#     fig, ax = pitch.draw(figsize=(10, 7))

#     # Map bins to pitch extent.
#     # We use imshow over the pitch area; origin='lower' to match coordinate convention.
#     ax.imshow(
#         grid,
#         extent=(PITCH_X_MIN, PITCH_X_MAX, PITCH_Y_MIN, PITCH_Y_MAX),
#         origin="lower",
#         aspect="auto",
#         interpolation="bilinear",
#         alpha=0.95,
#     )

#     buf = io.BytesIO()
#     fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
#     plt.close(fig)
#     buf.seek(0)
#     png_bytes = buf.getvalue()

#     if cache:
#         _cache_put(cache_key, png_bytes)

#     return Response(content=png_bytes, media_type="image/png")

# main.py
import os
import time
import io
from typing import List, Optional, Tuple

from fastapi import FastAPI, Query, Response, HTTPException
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mplsoccer import Pitch

import psycopg
from psycopg import sql
from psycopg.rows import tuple_row
from psycopg_pool import ConnectionPool

# -----------------------------
# Config
# -----------------------------
PG_DSN = os.environ.get("PG_DSN", "postgresql://postgres:postgres@localhost:5432/eventstream")

app = FastAPI(title="Eventstream Heatmaps (POC)", version="0.2.0", default_response_class=ORJSONResponse)

# IMPORTANT: pool (keeps connections warm)
POOL: ConnectionPool = ConnectionPool(
    conninfo=PG_DSN,
    min_size=1,
    max_size=10,
    timeout=10,
    kwargs={
        # tuple rows are faster than dict rows
        "row_factory": tuple_row,
        # keep it simple
        "autocommit": True,
    },
)

@app.on_event("shutdown")
def _shutdown_pool():
    POOL.close()


# -----------------------------
# Models (only keep if you really want typed docs)
# -----------------------------
# class Point(BaseModel):
#     x: float
#     y: float


from pydantic import BaseModel
from typing import List, Optional

class ShotMeta(BaseModel):
    is_goal: Optional[bool] = None
    on_target: Optional[bool] = None
    goal_zone: Optional[str] = None
    xg: Optional[float] = None
    body_part: Optional[str] = None

class HeatmapPointsCompactResponse(BaseModel):
    player_id: int
    primary: str
    n_points: int
    points_xy: List[List[float]]              # [[x,y], ...]
    end_points_xy: Optional[List[List[float]]] = None  # pass/carry ends
    shot_meta: Optional[List[ShotMeta]] = None         # aligned with points_xy


from psycopg import sql
from typing import Optional, Tuple, List

def _secondary_any_where(secondary_any: Optional[List[str]], alias: str = "e"):
    """
    Match if any requested secondary appears in type_secondary array.
    Uses overlap operator: <alias>.type_secondary && %s
    """
    if not secondary_any:
        return sql.SQL("TRUE"), []
    clause = sql.SQL("{a}.type_secondary && %s").format(a=sql.Identifier(alias))
    return clause, [secondary_any]


def _bbox_where(
    bbox: Optional[Tuple[float, float, float, float]],
    alias: str = "e",
    locx_col: str = "location_x",
    locy_col: str = "location_y",
):
    """
    bbox = (x_min, x_max, y_min, y_max)
    Applies to <alias>.<locx_col>, <alias>.<locy_col>
    """
    if not bbox:
        return sql.SQL("TRUE"), []
    x_min, x_max, y_min, y_max = bbox
    clause = sql.SQL(
        "{a}.{lx} BETWEEN %s AND %s AND {a}.{ly} BETWEEN %s AND %s"
    ).format(
        a=sql.Identifier(alias),
        lx=sql.Identifier(locx_col),
        ly=sql.Identifier(locy_col),
    )
    return clause, [x_min, x_max, y_min, y_max]




BINS_X = int(os.environ.get("HEATMAP_BINS_X", "60"))
BINS_Y = int(os.environ.get("HEATMAP_BINS_Y", "40"))

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
# Helpers
# -----------------------------
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

    return " AND ".join(where), params


def _fetch_points_fast(query_text: str, params: list, limit: Optional[int] = None) -> list[tuple[float, float]]:
    """
    Returns list of (x,y) tuples. Avoids numpy + pydantic overhead.
    """
    if limit is not None:
        query_text += " LIMIT %s"
        params = params + [limit]

    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query_text, params)
            return cur.fetchall()  # list[tuple]


def _fetch_pass_end_points_fast(
    *,
    player_id: int,
    match_id: Optional[int],
    competition_id: Optional[int],
    season_id: Optional[int],
    secondary_any: Optional[List[str]],
    x_min: Optional[float],
    x_max: Optional[float],
    y_min: Optional[float],
    y_max: Optional[float],
    limit: Optional[int] = None,
) -> list[tuple[float, float]]:
    where = ["e.player_id = %s", "e.type_primary = 'pass'"]
    params = [player_id]

    if match_id is not None:
        where.append("e.match_id = %s")
        params.append(match_id)
    if competition_id is not None:
        where.append("e.competition_id = %s")
        params.append(competition_id)
    if season_id is not None:
        where.append("e.season_id = %s")
        params.append(season_id)
    if secondary_any:
        where.append("e.type_secondary && %s")
        params.append(secondary_any)

    if x_min is not None:
        where.append("e.location_x >= %s")
        params.append(x_min)
    if x_max is not None:
        where.append("e.location_x <= %s")
        params.append(x_max)
    if y_min is not None:
        where.append("e.location_y >= %s")
        params.append(y_min)
    if y_max is not None:
        where.append("e.location_y <= %s")
        params.append(y_max)

    where.append("p.end_x IS NOT NULL")
    where.append("p.end_y IS NOT NULL")

    q = f"""
        SELECT p.end_x, p.end_y
        FROM eventstream_events e
        JOIN eventstream_passes p ON p.event_id = e.id
        WHERE {" AND ".join(where)}
    """
    if limit is not None:
        q += " LIMIT %s"
        params = params + [limit]

    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(q, params)
            return cur.fetchall()


def _fetch_carry_end_points(
    *,
    player_id: int,
    match_id: Optional[int],
    competition_id: Optional[int],
    season_id: Optional[int],
    secondary_any: Optional[List[str]],
    bbox: Optional[Tuple[float, float, float, float]],
    limit: Optional[int] = None,
):
    where_parts = [
        sql.SQL("e.player_id = %s"),
        sql.SQL("e.type_primary = 'carry'"),
    ]
    params = [player_id]

    if match_id is not None:
        where_parts.append(sql.SQL("e.match_id = %s")); params.append(match_id)
    if competition_id is not None:
        where_parts.append(sql.SQL("e.competition_id = %s")); params.append(competition_id)
    if season_id is not None:
        where_parts.append(sql.SQL("e.season_id = %s")); params.append(season_id)

    sec_clause, sec_params = _secondary_any_where(secondary_any, alias="e")
    where_parts.append(sec_clause); params.extend(sec_params)

    bbox_clause, bbox_params = _bbox_where(bbox, alias="e", locx_col="location_x", locy_col="location_y")
    where_parts.append(bbox_clause); params.extend(bbox_params)

    q = sql.SQL("""
        SELECT c.end_x, c.end_y
        FROM eventstream_events e
        JOIN eventstream_carries c ON c.event_id = e.id
        WHERE {where}
          AND c.end_x IS NOT NULL
          AND c.end_y IS NOT NULL
    """).format(where=sql.SQL(" AND ").join(where_parts))

    if limit is not None:
        q = q + sql.SQL(" LIMIT %s")
        params.append(limit)

    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(q, params)
            return cur.fetchall()

import numpy as np
from typing import Optional, List, Tuple
from psycopg import sql

def _fetch_shot_points_with_meta(
    *,
    player_id: int,
    match_id: Optional[int],
    competition_id: Optional[int],
    season_id: Optional[int],
    secondary_any: Optional[List[str]],
    bbox: Optional[Tuple[float, float, float, float]],
    limit: Optional[int] = None,
):
    where_parts = [
        sql.SQL("e.player_id = %s"),
        sql.SQL("e.type_primary = 'shot'"),
        sql.SQL("e.location_x IS NOT NULL"),
        sql.SQL("e.location_y IS NOT NULL"),
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

    sec_clause, sec_params = _secondary_any_where(secondary_any, alias="e")
    where_parts.append(sec_clause)
    params.extend(sec_params)

    bbox_clause, bbox_params = _bbox_where(bbox, alias="e", locx_col="location_x", locy_col="location_y")
    where_parts.append(bbox_clause)
    params.extend(bbox_params)

    q = sql.SQL("""
        SELECT
            e.location_x, e.location_y,
            s.is_goal, s.on_target, s.goal_zone, s.xg, s.body_part
        FROM eventstream_events e
        LEFT JOIN eventstream_shots s ON s.event_id = e.id
        WHERE {where}
    """).format(where=sql.SQL(" AND ").join(where_parts))

    if limit is not None:
        q = q + sql.SQL(" LIMIT %s")
        params.append(limit)

    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(q, params)
            return cur.fetchall()

def _row_xy(row) -> List[float]:
    return [float(row[0]), float(row[1])]

def _row_shot_meta(row) -> dict:
    # row = (x,y,is_goal,on_target,goal_zone,xg,body_part)
    return {
        "is_goal": row[2],
        "on_target": row[3],
        "goal_zone": row[4],
        "xg": float(row[5]) if row[5] is not None else None,
        "body_part": row[6],
    }

@app.get("/v1/heatmaps/points_compact", response_model=HeatmapPointsCompactResponse)
def get_heatmap_points_compact(
    player_id: int = Query(...),
    primary: Optional[str] = Query(default=None),
    use_on_ball_default: bool = Query(default=True),

    match_id: Optional[int] = Query(None),
    competition_id: Optional[int] = Query(None),
    season_id: Optional[int] = Query(None),

    secondary_any: Optional[List[str]] = Query(None),
    bbox: Optional[str] = Query(None, description="x_min,x_max,y_min,y_max"),
    limit: Optional[int] = Query(None),

    include_pass_ends: bool = Query(False),
    include_carry_ends: bool = Query(False),
    include_shot_meta: bool = Query(False),
):
    bbox_t = None
    if bbox:
        parts = [float(x.strip()) for x in bbox.split(",")]
        if len(parts) != 4:
            raise HTTPException(status_code=400, detail="Invalid bbox. Use 'x_min,x_max,y_min,y_max'")
        bbox_t = (parts[0], parts[1], parts[2], parts[3])

    primary_in = None
    if primary is None and use_on_ball_default:
        primary_in = ON_BALL_PRIMARY_DEFAULT

    # validate combos
    if include_pass_ends and primary != "pass":
        raise HTTPException(status_code=400, detail="include_pass_ends=true requires primary=pass")
    if include_carry_ends and primary != "carry":
        raise HTTPException(status_code=400, detail="include_carry_ends=true requires primary=carry")
    if include_shot_meta and primary != "shot":
        raise HTTPException(status_code=400, detail="include_shot_meta=true requires primary=shot")

    # Special handling for shots so points/meta ordering is guaranteed
    if primary == "shot":
        rows = _fetch_shot_points_with_meta(
            player_id=player_id,
            match_id=match_id,
            competition_id=competition_id,
            season_id=season_id,
            secondary_any=secondary_any,
            bbox=bbox_t,
            limit=limit,
        )

        points_xy = [_row_xy(r) for r in rows]
        shot_meta = [_row_shot_meta(r) for r in rows]

        return HeatmapPointsCompactResponse(
            player_id=player_id,
            primary="shot",
            n_points=len(points_xy),
            points_xy=points_xy,
            shot_meta=shot_meta,
        )


    # Normal path: just fetch start points
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
    """)

    pts = _fetch_points_fast(q, params, limit=limit)
    points_xy = pts
    # pts.astype(float).tolist()  # [[x,y], ...]

    end_points_xy = None
    if include_pass_ends and primary == "pass":
        end = _fetch_pass_end_points_fast(
            player_id=player_id,
            match_id=match_id,
            competition_id=competition_id,
            season_id=season_id,
            secondary_any=secondary_any,
            # bbox=bbox_t,
            limit=limit,
        )
        end_points_xy = end.astype(float).tolist()

    if include_carry_ends and primary == "carry":
        end = _fetch_carry_end_points(
            player_id=player_id,
            match_id=match_id,
            competition_id=competition_id,
            season_id=season_id,
            secondary_any=secondary_any,
            bbox=bbox_t,
            limit=limit,
        )
        end_points_xy = end.astype(float).tolist()

    primary_label = primary if primary is not None else "on_ball_default"
    return HeatmapPointsCompactResponse(
        player_id=player_id,
        primary=primary_label,
        n_points=len(points_xy),
        points_xy=points_xy,
        end_points_xy=end_points_xy,
    )


# -----------------------------
# Endpoints
# -----------------------------
@app.get("/v1/heatmaps/points")
def get_heatmap_points(
    player_id: int = Query(...),

    primary: Optional[str] = Query(default=None),
    use_on_ball_default: bool = Query(default=True),

    match_id: Optional[int] = Query(None),
    competition_id: Optional[int] = Query(None),
    season_id: Optional[int] = Query(None),

    secondary_any: Optional[List[str]] = Query(None),
    bbox: Optional[str] = Query(None, description="x_min,x_max,y_min,y_max"),
    limit: Optional[int] = Query(None),

    include_pass_ends: bool = Query(False),
):
    t0 = time.perf_counter()

    x_min = x_max = y_min = y_max = None
    if bbox:
        try:
            parts = [float(x.strip()) for x in bbox.split(",")]
            if len(parts) != 4:
                raise ValueError()
            x_min, x_max, y_min, y_max = parts
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid bbox. Use 'x_min,x_max,y_min,y_max'")

    primary_in = None
    if primary is None and use_on_ball_default:
        primary_in = ON_BALL_PRIMARY_DEFAULT

    if include_pass_ends and primary != "pass":
        raise HTTPException(status_code=400, detail="include_pass_ends=true requires primary=pass")

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

    q = f"""
        SELECT location_x, location_y
        FROM eventstream_events
        WHERE {where_sql}
    """

    t1 = time.perf_counter()
    rows = _fetch_points_fast(q, params, limit=limit)
    t2 = time.perf_counter()

    end_rows = None
    if include_pass_ends and primary == "pass":
        end_rows = _fetch_pass_end_points_fast(
            player_id=player_id,
            match_id=match_id,
            competition_id=competition_id,
            season_id=season_id,
            secondary_any=secondary_any,
            x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
            limit=limit,
        )

    # Build response as plain dicts (MUCH faster than Point() * N)
    # points = [{"x": float(x), "y": float(y)} for (x, y) in rows]
    points = rows
    payload = {
        "player_id": player_id,
        "primary": primary if primary is not None else "on_ball_default",
        "n_points": len(points),
        "points": points,
    }
    if end_rows is not None:
        # payload["end_points"] = [{"x": float(x), "y": float(y)} for (x, y) in end_rows]
        payload["end_points"] = end_rows

    t3 = time.perf_counter()
    payload["_timing_ms"] = {
        "build_query": round((t1 - t0) * 1000, 2),
        "db_fetch": round((t2 - t1) * 1000, 2),
        "serialize": round((t3 - t2) * 1000, 2),
        "total": round((t3 - t0) * 1000, 2),
    }
    return payload


from scipy.ndimage import gaussian_filter

@app.get("/heatmaps/image")
def heatmap_image(
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

    # blur controls
    sigma: float = Query(default=1.6, ge=0.0, le=10.0),
    blur: bool = Query(default=True),

    # render
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

    # 1) Fetch binned counts (FAST)
    q = f"""
        SELECT bin_x, bin_y, COUNT(*)::INT
        FROM eventstream_events
        WHERE {where_sql}
        GROUP BY bin_x, bin_y
    """

    with POOL.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(q, params)
            rows = cur.fetchall()

    # 2) Build grid
    grid = np.zeros((BINS_Y, BINS_X), dtype=np.float32)
    for bx, by, n in rows:
        if 0 <= bx < BINS_X and 0 <= by < BINS_Y:
            grid[by, bx] = n

    # 3) Gaussian blur
    if blur and sigma > 0:
        grid = gaussian_filter(grid, sigma=sigma)

    # Optional normalization (comment out if you prefer raw intensity)
    if grid.max() > 0:
        grid /= grid.max()

    # 4) Render
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
