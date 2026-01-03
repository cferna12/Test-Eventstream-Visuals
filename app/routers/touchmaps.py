# # app/routers/heatmaps.py
# from __future__ import annotations

# from typing import List, Optional, Tuple, Dict, Any
# from fastapi import APIRouter, Query, HTTPException
# from pydantic import BaseModel
# import os
# import time

# import psycopg
# from psycopg import sql

# router = APIRouter(prefix="/v1/touchmaps", tags=["touchmaps"])

# PG_DSN = os.environ.get("PG_DSN", "postgresql://postgres:postgres@localhost:5432/eventstream")


# # -----------------------------
# # DB
# # -----------------------------
# def get_conn():
#     return psycopg.connect(PG_DSN)


# def _secondary_any_where(secondary_any: Optional[List[str]]):
#     """
#     Returns (clause_sql, params)
#     Matches if any requested secondary appears in type_secondary array:
#       type_secondary && ARRAY[...]
#     """
#     if not secondary_any:
#         return sql.SQL("TRUE"), []
#     return sql.SQL("type_secondary && %s"), [secondary_any]


# def _bbox_where(
#     bbox: Optional[Tuple[float, float, float, float]],
#     locx_col: str = "location_x",
#     locy_col: str = "location_y",
# ):
#     """
#     bbox = (x_min, x_max, y_min, y_max)
#     Returns (clause_sql, params)
#     """
#     if not bbox:
#         return sql.SQL("TRUE"), []
#     x_min, x_max, y_min, y_max = bbox
#     clause = sql.SQL("{lx} BETWEEN %s AND %s AND {ly} BETWEEN %s AND %s").format(
#         lx=sql.Identifier(locx_col),
#         ly=sql.Identifier(locy_col),
#     )
#     return clause, [x_min, x_max, y_min, y_max]


# # -----------------------------
# # Response models (compact)
# # -----------------------------
# class PointsXYResponse(BaseModel):
#     player_id: int
#     n_points: int
#     points_xy: List[List[float]]
#     _timing_ms: Optional[Dict[str, float]] = None


# class PassesResponse(BaseModel):
#     player_id: int
#     n_passes: int
#     start_xy: List[List[float]]      # [[x,y], ...]
#     end_xy: List[List[float]]        # [[x,y], ...]
#     _timing_ms: Optional[Dict[str, float]] = None


# class CarriesResponse(BaseModel):
#     player_id: int
#     n_carries: int
#     start_xy: List[List[float]]
#     end_xy: List[List[float]]
#     _timing_ms: Optional[Dict[str, float]] = None


# class ShotsResponse(BaseModel):
#     player_id: int
#     n_shots: int
#     points_xy: List[List[float]]     # [[x,y], ...]
#     shot_meta: List[Dict[str, Any]]  # aligned with points_xy
#     _timing_ms: Optional[Dict[str, float]] = None


# # -----------------------------
# # Parsing helpers
# # -----------------------------
# def _parse_bbox(bbox: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
#     if not bbox:
#         return None
#     try:
#         parts = [float(x.strip()) for x in bbox.split(",")]
#         if len(parts) != 4:
#             raise ValueError()
#         return (parts[0], parts[1], parts[2], parts[3])
#     except Exception:
#         raise HTTPException(status_code=400, detail="Invalid bbox. Use 'x_min,x_max,y_min,y_max'")


# # -----------------------------
# # Core WHERE builder (generic)
# # -----------------------------
# def build_where_and_params(
#     *,
#     player_id: int,
#     competition_id: Optional[int],
#     season_id: Optional[int],
#     match_id: Optional[int],
#     type_primary: Optional[str],
#     secondary_any: Optional[List[str]],
#     bbox_t: Optional[Tuple[float, float, float, float]],
# ):
#     where_parts = [sql.SQL("e.player_id = %s")]
#     params: List[Any] = [player_id]

#     if competition_id is not None:
#         where_parts.append(sql.SQL("e.competition_id = %s"))
#         params.append(competition_id)

#     if season_id is not None:
#         where_parts.append(sql.SQL("e.season_id = %s"))
#         params.append(season_id)

#     if match_id is not None:
#         where_parts.append(sql.SQL("e.match_id = %s"))
#         params.append(match_id)

#     if type_primary is not None:
#         where_parts.append(sql.SQL("e.type_primary = %s"))
#         params.append(type_primary)

#     sec_clause, sec_params = _secondary_any_where(secondary_any)
#     # NOTE: sec_clause references "type_secondary" unqualified; build it qualified here
#     # safest: just write it qualified directly
#     if sec_params:
#         where_parts.append(sql.SQL("e.type_secondary && %s"))
#         params.extend(sec_params)

#     bbox_clause, bbox_params = _bbox_where(bbox_t, "location_x", "location_y")
#     # qualify columns for bbox clause:
#     if bbox_params:
#         where_parts.append(sql.SQL("e.location_x BETWEEN %s AND %s AND e.location_y BETWEEN %s AND %s"))
#         params.extend(bbox_params)

#     # avoid null locations
#     where_parts.append(sql.SQL("e.location_x IS NOT NULL"))
#     where_parts.append(sql.SQL("e.location_y IS NOT NULL"))

#     return where_parts, params


# # -----------------------------
# # 1) Generic points (primary or secondary filters)
# # -----------------------------
# @router.get("/points", response_model=PointsXYResponse)
# def get_points(
#     player_id: int = Query(...),
#     competition_id: Optional[int] = None,
#     season_id: Optional[int] = None,
#     match_id: Optional[int] = None,

#     # use for “normal” queries
#     type_primary: Optional[str] = Query(default=None),

#     # use for secondary-based queries (e.g., ground_duel, aerial_duel, carry if you want)
#     secondary_any: Optional[List[str]] = Query(default=None),

#     bbox: Optional[str] = Query(default=None),
#     limit: Optional[int] = Query(default=None),
# ):
#     t0 = time.perf_counter()
#     bbox_t = _parse_bbox(bbox)

#     where_parts, params = build_where_and_params(
#         player_id=player_id,
#         competition_id=competition_id,
#         season_id=season_id,
#         match_id=match_id,
#         type_primary=type_primary,
#         secondary_any=secondary_any,
#         bbox_t=bbox_t,
#     )

#     q = sql.SQL("""
#         SELECT e.location_x, e.location_y
#         FROM eventstream_events e
#         WHERE {where}
#     """).format(where=sql.SQL(" AND ").join(where_parts))

#     if limit is not None:
#         q = q + sql.SQL(" LIMIT %s")
#         params = params + [limit]

#     t1 = time.perf_counter()
#     with get_conn() as conn:
#         with conn.cursor() as cur:
#             cur.execute(q, params)
#             rows = cur.fetchall()
#     t2 = time.perf_counter()

#     pts = [[float(x), float(y)] for (x, y) in rows]
#     t3 = time.perf_counter()

#     return PointsXYResponse(
#         player_id=player_id,
#         n_points=len(pts),
#         points_xy=pts,
#         _timing_ms={
#             "build_query": round((t1 - t0) * 1000, 2),
#             "db_fetch": round((t2 - t1) * 1000, 2),
#             "serialize": round((t3 - t2) * 1000, 2),
#             "total": round((t3 - t0) * 1000, 2),
#         },
#     )


# # -----------------------------
# # 2) Passes (start + end)
# # -----------------------------
# @router.get("/passes", response_model=PassesResponse)
# def get_passes(
#     player_id: int = Query(...),
#     competition_id: Optional[int] = None,
#     season_id: Optional[int] = None,
#     match_id: Optional[int] = None,

#     secondary_any: Optional[List[str]] = Query(default=None),
#     bbox: Optional[str] = Query(default=None),
#     limit: Optional[int] = Query(default=None),
# ):
#     t0 = time.perf_counter()
#     bbox_t = _parse_bbox(bbox)

#     where_parts, params = build_where_and_params(
#         player_id=player_id,
#         competition_id=competition_id,
#         season_id=season_id,
#         match_id=match_id,
#         type_primary="pass",
#         secondary_any=secondary_any,
#         bbox_t=bbox_t,
#     )

#     q = sql.SQL("""
#         SELECT
#           e.location_x, e.location_y,
#           p.end_x, p.end_y
#         FROM eventstream_events e
#         JOIN eventstream_passes p ON p.event_id = e.id
#         WHERE {where}
#           AND p.end_x IS NOT NULL
#           AND p.end_y IS NOT NULL
#     """).format(where=sql.SQL(" AND ").join(where_parts))

#     if limit is not None:
#         q = q + sql.SQL(" LIMIT %s")
#         params = params + [limit]

#     t1 = time.perf_counter()
#     with get_conn() as conn:
#         with conn.cursor() as cur:
#             cur.execute(q, params)
#             rows = cur.fetchall()
#     t2 = time.perf_counter()

#     start_xy = [[float(sx), float(sy)] for (sx, sy, ex, ey) in rows]
#     end_xy = [[float(ex), float(ey)] for (sx, sy, ex, ey) in rows]
#     t3 = time.perf_counter()

#     return PassesResponse(
#         player_id=player_id,
#         n_passes=len(rows),
#         start_xy=start_xy,
#         end_xy=end_xy,
#         _timing_ms={
#             "build_query": round((t1 - t0) * 1000, 2),
#             "db_fetch": round((t2 - t1) * 1000, 2),
#             "serialize": round((t3 - t2) * 1000, 2),
#             "total": round((t3 - t0) * 1000, 2),
#         },
#     )


# # -----------------------------
# # 3) Carries (start + end)
# #    For now, carries are identified by type_secondary containing "carry"
# # -----------------------------
# @router.get("/carries", response_model=CarriesResponse)
# def get_carries(
#     player_id: int = Query(...),
#     competition_id: Optional[int] = None,
#     season_id: Optional[int] = None,
#     match_id: Optional[int] = None,

#     # your carry filter lives in secondary tags for now:
#     secondary_any: Optional[List[str]] = Query(default=["carry"]),

#     bbox: Optional[str] = Query(default=None),
#     limit: Optional[int] = Query(default=None),
# ):
#     t0 = time.perf_counter()
#     bbox_t = _parse_bbox(bbox)

#     # NOTE: do NOT constrain type_primary here; it may not be "carry"
#     where_parts, params = build_where_and_params(
#         player_id=player_id,
#         competition_id=competition_id,
#         season_id=season_id,
#         match_id=match_id,
#         type_primary=None,
#         secondary_any=secondary_any,
#         bbox_t=bbox_t,
#     )

#     q = sql.SQL("""
#         SELECT
#           e.location_x, e.location_y,
#           c.end_x, c.end_y
#         FROM eventstream_events e
#         JOIN eventstream_carries c ON c.event_id = e.id
#         WHERE {where}
#           AND c.end_x IS NOT NULL
#           AND c.end_y IS NOT NULL
#     """).format(where=sql.SQL(" AND ").join(where_parts))

#     if limit is not None:
#         q = q + sql.SQL(" LIMIT %s")
#         params = params + [limit]

#     t1 = time.perf_counter()
#     with get_conn() as conn:
#         with conn.cursor() as cur:
#             cur.execute(q, params)
#             rows = cur.fetchall()
#     t2 = time.perf_counter()

#     start_xy = [[float(sx), float(sy)] for (sx, sy, ex, ey) in rows]
#     end_xy = [[float(ex), float(ey)] for (sx, sy, ex, ey) in rows]
#     t3 = time.perf_counter()

#     return CarriesResponse(
#         player_id=player_id,
#         n_carries=len(rows),
#         start_xy=start_xy,
#         end_xy=end_xy,
#         _timing_ms={
#             "build_query": round((t1 - t0) * 1000, 2),
#             "db_fetch": round((t2 - t1) * 1000, 2),
#             "serialize": round((t3 - t2) * 1000, 2),
#             "total": round((t3 - t0) * 1000, 2),
#         },
#     )


# # -----------------------------
# # 4) Shots (points + metadata)
# # -----------------------------
# @router.get("/shots", response_model=ShotsResponse)
# def get_shots(
#     player_id: int = Query(...),
#     competition_id: Optional[int] = None,
#     season_id: Optional[int] = None,
#     match_id: Optional[int] = None,

#     secondary_any: Optional[List[str]] = Query(default=None),
#     bbox: Optional[str] = Query(default=None),
#     limit: Optional[int] = Query(default=None),
# ):
#     t0 = time.perf_counter()
#     bbox_t = _parse_bbox(bbox)

#     where_parts, params = build_where_and_params(
#         player_id=player_id,
#         competition_id=competition_id,
#         season_id=season_id,
#         match_id=match_id,
#         type_primary="shot",
#         secondary_any=secondary_any,
#         bbox_t=bbox_t,
#     )

#     q = sql.SQL("""
#         SELECT
#           e.location_x, e.location_y,
#           s.is_goal, s.on_target, s.goal_zone, s.xg, s.body_part
#         FROM eventstream_events e
#         JOIN eventstream_shots s ON s.event_id = e.id
#         WHERE {where}
#     """).format(where=sql.SQL(" AND ").join(where_parts))

#     if limit is not None:
#         q = q + sql.SQL(" LIMIT %s")
#         params = params + [limit]

#     t1 = time.perf_counter()
#     with get_conn() as conn:
#         with conn.cursor() as cur:
#             cur.execute(q, params)
#             rows = cur.fetchall()
#     t2 = time.perf_counter()

#     points_xy: List[List[float]] = []
#     shot_meta: List[Dict[str, Any]] = []

#     for (x, y, is_goal, on_target, goal_zone, xg, body_part) in rows:
#         points_xy.append([float(x), float(y)])
#         shot_meta.append(
#             {
#                 "is_goal": bool(is_goal) if is_goal is not None else None,
#                 "on_target": bool(on_target) if on_target is not None else None,
#                 "goal_zone": goal_zone,
#                 "xg": float(xg) if xg is not None else None,
#                 "body_part": body_part,
#             }
#         )

#     t3 = time.perf_counter()
#     return ShotsResponse(
#         player_id=player_id,
#         n_shots=len(points_xy),
#         points_xy=points_xy,
#         shot_meta=shot_meta,
#         _timing_ms={
#             "build_query": round((t1 - t0) * 1000, 2),
#             "db_fetch": round((t2 - t1) * 1000, 2),
#             "serialize": round((t3 - t2) * 1000, 2),
#             "total": round((t3 - t0) * 1000, 2),
#         },
#     )

# app/routers/heatmaps.py
from __future__ import annotations

from typing import List, Optional, Tuple, Dict, Any
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
import os
import time

import psycopg
from psycopg import sql

router = APIRouter(prefix="/v1/touchmaps", tags=["touchmaps"])

PG_DSN = os.environ.get("PG_DSN", "postgresql://postgres:postgres@localhost:5432/eventstream")

class PointsWithEndsAndMetaResponse(BaseModel):
    player_id: int
    n: int
    points_xy: List[List[float]]
    end_points_xy: Optional[List[List[float]]] = None
    meta: Optional[List[dict]] = None



# -----------------------------
# DB
# -----------------------------
def get_conn():
    return psycopg.connect(PG_DSN)


def _secondary_any_where(secondary_any: Optional[List[str]]):
    """
    Returns (clause_sql, params)
    Matches if any requested secondary appears in type_secondary array:
      type_secondary && ARRAY[...]
    """
    if not secondary_any:
        return sql.SQL("TRUE"), []
    return sql.SQL("type_secondary && %s"), [secondary_any]


def _bbox_where(
    bbox: Optional[Tuple[float, float, float, float]],
    locx_col: str = "location_x",
    locy_col: str = "location_y",
):
    """
    bbox = (x_min, x_max, y_min, y_max)
    Returns (clause_sql, params)
    """
    if not bbox:
        return sql.SQL("TRUE"), []
    x_min, x_max, y_min, y_max = bbox
    clause = sql.SQL("{lx} BETWEEN %s AND %s AND {ly} BETWEEN %s AND %s").format(
        lx=sql.Identifier(locx_col),
        ly=sql.Identifier(locy_col),
    )
    return clause, [x_min, x_max, y_min, y_max]

def _fetch_pass_points_with_meta(
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
        sql.SQL("e.type_primary = 'pass'"),
        sql.SQL("e.location_x IS NOT NULL"),
        sql.SQL("e.location_y IS NOT NULL"),
        sql.SQL("p.end_x IS NOT NULL"),
        sql.SQL("p.end_y IS NOT NULL"),
    ]
    params = [player_id]

    if match_id is not None:
        where_parts.append(sql.SQL("e.match_id = %s")); params.append(match_id)
    if competition_id is not None:
        where_parts.append(sql.SQL("e.competition_id = %s")); params.append(competition_id)
    if season_id is not None:
        where_parts.append(sql.SQL("e.season_id = %s")); params.append(season_id)

    # secondary_any: uses overlap operator against e.type_secondary
    if secondary_any:
        where_parts.append(sql.SQL("e.type_secondary && %s"))
        params.append(secondary_any)

    if bbox:
        x_min, x_max, y_min, y_max = bbox
        where_parts.append(sql.SQL("e.location_x BETWEEN %s AND %s")); params += [x_min, x_max]
        where_parts.append(sql.SQL("e.location_y BETWEEN %s AND %s")); params += [y_min, y_max]

    q = sql.SQL("""
        SELECT
          e.location_x, e.location_y,
          p.end_x, p.end_y,
          p.accurate
        FROM eventstream_events e
        JOIN eventstream_passes p ON p.event_id = e.id
        WHERE {where}
        ORDER BY e.id
    """).format(where=sql.SQL(" AND ").join(where_parts))

    if limit is not None:
        q = q + sql.SQL(" LIMIT %s")
        params = params + [limit]

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q, params)
            return cur.fetchall()


def _fetch_carry_points_with_meta(
    *,
    player_id: int,
    match_id: Optional[int],
    competition_id: Optional[int],
    season_id: Optional[int],
    bbox: Optional[Tuple[float, float, float, float]],
    limit: Optional[int] = None,
):
    where_parts = [
        sql.SQL("e.player_id = %s"),
        # carry can be primary or secondary; keep both
        sql.SQL("(e.type_primary = 'carry' OR e.type_secondary && %s)"),
        sql.SQL("e.location_x IS NOT NULL"),
        sql.SQL("e.location_y IS NOT NULL"),
        sql.SQL("c.end_x IS NOT NULL"),
        sql.SQL("c.end_y IS NOT NULL"),
    ]
    params = [player_id, ["carry"]]

    if match_id is not None:
        where_parts.append(sql.SQL("e.match_id = %s")); params.append(match_id)
    if competition_id is not None:
        where_parts.append(sql.SQL("e.competition_id = %s")); params.append(competition_id)
    if season_id is not None:
        where_parts.append(sql.SQL("e.season_id = %s")); params.append(season_id)

    if bbox:
        x_min, x_max, y_min, y_max = bbox
        where_parts.append(sql.SQL("e.location_x BETWEEN %s AND %s")); params += [x_min, x_max]
        where_parts.append(sql.SQL("e.location_y BETWEEN %s AND %s")); params += [y_min, y_max]

    q = sql.SQL("""
        SELECT
          e.location_x, e.location_y,
          c.end_x, c.end_y,
          c.progression
        FROM eventstream_events e
        JOIN eventstream_carries c ON c.event_id = e.id
        WHERE {where}
        ORDER BY e.id
    """).format(where=sql.SQL(" AND ").join(where_parts))

    if limit is not None:
        q = q + sql.SQL(" LIMIT %s")
        params = params + [limit]

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q, params)
            return cur.fetchall()


def _fetch_aerial_duel_points_with_meta(
    *,
    player_id: int,
    match_id: Optional[int],
    competition_id: Optional[int],
    season_id: Optional[int],
    bbox: Optional[Tuple[float, float, float, float]],
    limit: Optional[int] = None,
):
    where_parts = [
        sql.SQL("e.player_id = %s"),
        # aerial duel is usually secondary; allow primary too
        sql.SQL("(e.type_primary = 'aerial_duel' OR e.type_secondary && %s)"),
        sql.SQL("e.location_x IS NOT NULL"),
        sql.SQL("e.location_y IS NOT NULL"),
    ]
    params = [player_id, ["aerial_duel"]]

    if match_id is not None:
        where_parts.append(sql.SQL("e.match_id = %s")); params.append(match_id)
    if competition_id is not None:
        where_parts.append(sql.SQL("e.competition_id = %s")); params.append(competition_id)
    if season_id is not None:
        where_parts.append(sql.SQL("e.season_id = %s")); params.append(season_id)

    if bbox:
        x_min, x_max, y_min, y_max = bbox
        where_parts.append(sql.SQL("e.location_x BETWEEN %s AND %s")); params += [x_min, x_max]
        where_parts.append(sql.SQL("e.location_y BETWEEN %s AND %s")); params += [y_min, y_max]

    q = sql.SQL("""
        SELECT
          e.location_x, e.location_y,
          a.first_touch
        FROM eventstream_events e
        JOIN eventstream_aerial_duels a ON a.event_id = e.id
        WHERE {where}
        ORDER BY e.id
    """).format(where=sql.SQL(" AND ").join(where_parts))

    if limit is not None:
        q = q + sql.SQL(" LIMIT %s")
        params = params + [limit]

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q, params)
            return cur.fetchall()


def _fetch_ground_duel_points_with_meta(
    *,
    player_id: int,
    match_id: Optional[int],
    competition_id: Optional[int],
    season_id: Optional[int],
    bbox: Optional[Tuple[float, float, float, float]],
    limit: Optional[int] = None,
):
    where_parts = [
        sql.SQL("e.player_id = %s"),
        sql.SQL("(e.type_primary = 'ground_duel' OR e.type_secondary && %s)"),
        sql.SQL("e.location_x IS NOT NULL"),
        sql.SQL("e.location_y IS NOT NULL"),
    ]
    params = [player_id, ["ground_duel"]]

    if match_id is not None:
        where_parts.append(sql.SQL("e.match_id = %s")); params.append(match_id)
    if competition_id is not None:
        where_parts.append(sql.SQL("e.competition_id = %s")); params.append(competition_id)
    if season_id is not None:
        where_parts.append(sql.SQL("e.season_id = %s")); params.append(season_id)

    if bbox:
        x_min, x_max, y_min, y_max = bbox
        where_parts.append(sql.SQL("e.location_x BETWEEN %s AND %s")); params += [x_min, x_max]
        where_parts.append(sql.SQL("e.location_y BETWEEN %s AND %s")); params += [y_min, y_max]

    q = sql.SQL("""
        SELECT
          e.location_x, e.location_y,
          g.recovered_possession,
          g.kept_possession,
          e.type_secondary
        FROM eventstream_events e
        JOIN eventstream_ground_duels g ON g.event_id = e.id
        WHERE {where}
        ORDER BY e.id
    """).format(where=sql.SQL(" AND ").join(where_parts))

    if limit is not None:
        q = q + sql.SQL(" LIMIT %s")
        params = params + [limit]

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q, params)
            return cur.fetchall()


@router.get("/v1/points/passes", response_model=PointsWithEndsAndMetaResponse)
def get_pass_points(
    player_id: int,
    match_id: Optional[int] = None,
    competition_id: Optional[int] = None,
    season_id: Optional[int] = None,
    secondary_any: Optional[List[str]] = None,
    bbox: Optional[str] = None,
    limit: Optional[int] = None,
):
    bbox_t = None
    if bbox:
        x_min, x_max, y_min, y_max = [float(x.strip()) for x in bbox.split(",")]
        bbox_t = (x_min, x_max, y_min, y_max)

    rows = _fetch_pass_points_with_meta(
        player_id=player_id,
        match_id=match_id,
        competition_id=competition_id,
        season_id=season_id,
        secondary_any=secondary_any,
        bbox=bbox_t,
        limit=limit,
    )

    points_xy = [[float(r[0]), float(r[1])] for r in rows]
    end_points_xy = [[float(r[2]), float(r[3])] for r in rows]
    meta = [{"accurate": r[4]} for r in rows]

    return {
        "player_id": player_id,
        "n": len(rows),
        "points_xy": points_xy,
        "end_points_xy": end_points_xy,
        "meta": meta,
    }


@router.get("/v1/points/carries", response_model=PointsWithEndsAndMetaResponse)
def get_carry_points(
    player_id: int,
    match_id: Optional[int] = None,
    competition_id: Optional[int] = None,
    season_id: Optional[int] = None,
    bbox: Optional[str] = None,
    limit: Optional[int] = None,
):
    bbox_t = None
    if bbox:
        x_min, x_max, y_min, y_max = [float(x.strip()) for x in bbox.split(",")]
        bbox_t = (x_min, x_max, y_min, y_max)

    rows = _fetch_carry_points_with_meta(
        player_id=player_id,
        match_id=match_id,
        competition_id=competition_id,
        season_id=season_id,
        bbox=bbox_t,
        limit=limit,
    )

    points_xy = [[float(r[0]), float(r[1])] for r in rows]
    end_points_xy = [[float(r[2]), float(r[3])] for r in rows]
    meta = [{"progression": r[4]} for r in rows]

    return {
        "player_id": player_id,
        "n": len(rows),
        "points_xy": points_xy,
        "end_points_xy": end_points_xy,
        "meta": meta,
    }
