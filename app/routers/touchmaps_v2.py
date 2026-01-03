from __future__ import annotations

from typing import List, Optional, Tuple, Dict, Any, Sequence
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
import os
import time

import psycopg
from psycopg import sql


# -----------------------------
# Router
# -----------------------------
router = APIRouter(prefix="/v1/touchmaps", tags=["touchmaps"])

PG_DSN = os.environ.get("PG_DSN", "postgresql://postgres:postgres@localhost:5432/eventstream")


# -----------------------------
# Models
# -----------------------------
class PointsResponse(BaseModel):
    player_id: int
    n: int
    points_xy: List[List[float]]


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


# -----------------------------
# Shared helpers (YOUR existing ones)
# -----------------------------
def _parse_bbox(bbox: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
    if not bbox:
        return None
    try:
        parts = [float(x.strip()) for x in bbox.split(",")]
        if len(parts) != 4:
            raise ValueError
        return (parts[0], parts[1], parts[2], parts[3])
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid bbox. Use 'x_min,x_max,y_min,y_max'")


def _add_common_filters(
    *,
    where_parts: List[sql.SQL],
    params: List[Any],
    player_id: int,
    match_id: Optional[int],
    competition_id: Optional[int],
    season_id: Optional[int],
    bbox: Optional[Tuple[float, float, float, float]],
):
    where_parts.append(sql.SQL("e.player_id = %s"))
    params.append(player_id)

    if match_id is not None:
        where_parts.append(sql.SQL("e.match_id = %s"))
        params.append(match_id)
    if competition_id is not None:
        where_parts.append(sql.SQL("e.competition_id = %s"))
        params.append(competition_id)
    if season_id is not None:
        where_parts.append(sql.SQL("e.season_id = %s"))
        params.append(season_id)

    where_parts.append(sql.SQL("e.location_x IS NOT NULL"))
    where_parts.append(sql.SQL("e.location_y IS NOT NULL"))

    if bbox:
        x_min, x_max, y_min, y_max = bbox
        where_parts.append(sql.SQL("e.location_x BETWEEN %s AND %s"))
        params += [x_min, x_max]
        where_parts.append(sql.SQL("e.location_y BETWEEN %s AND %s"))
        params += [y_min, y_max]


def _add_secondary_any_filter(
    *,
    where_parts: List[sql.SQL],
    params: List[Any],
    secondary_any: Optional[List[str]],
    table_alias: str = "e",
):
    if secondary_any:
        where_parts.append(sql.SQL(f"{table_alias}.type_secondary && %s"))
        params.append(secondary_any)


def _add_primary_or_secondary_tag(
    *,
    where_parts: List[sql.SQL],
    params: List[Any],
    primary_value: str,
    secondary_tag: str,
):
    where_parts.append(sql.SQL("(e.type_primary = %s OR e.type_secondary && %s)"))
    params.append(primary_value)
    params.append([secondary_tag])


def _fetch_action_rows(
    *,
    player_id: int,
    match_id: Optional[int],
    competition_id: Optional[int],
    season_id: Optional[int],
    bbox: Optional[Tuple[float, float, float, float]],
    limit: Optional[int],
    action_table: str,
    action_alias: str,
    action_join_on: str,
    select_cols: Sequence[str],
    extra_where: Optional[Sequence[str]] = None,
    event_primary: Optional[str] = None,
    event_primary_or_secondary: Optional[Tuple[str, str]] = None,
    secondary_any: Optional[List[str]] = None,
    order_by: str = "e.id",
) -> list[tuple]:
    where_parts: List[sql.SQL] = []
    params: List[Any] = []

    _add_common_filters(
        where_parts=where_parts,
        params=params,
        player_id=player_id,
        match_id=match_id,
        competition_id=competition_id,
        season_id=season_id,
        bbox=bbox,
    )

    if event_primary is not None:
        where_parts.append(sql.SQL("e.type_primary = %s"))
        params.append(event_primary)

    if event_primary_or_secondary is not None:
        primary_val, secondary_tag = event_primary_or_secondary
        _add_primary_or_secondary_tag(
            where_parts=where_parts,
            params=params,
            primary_value=primary_val,
            secondary_tag=secondary_tag,
        )

    _add_secondary_any_filter(where_parts=where_parts, params=params, secondary_any=secondary_any, table_alias="e")

    if extra_where:
        for w in extra_where:
            where_parts.append(sql.SQL(w))

    q = sql.SQL("""
        SELECT {select_list}
        FROM eventstream_events e
        JOIN {action_table} {a} ON {join_on}
        WHERE {where}
        ORDER BY {order_by}
    """).format(
        select_list=sql.SQL(", ").join(sql.SQL(c) for c in select_cols),
        action_table=sql.Identifier(action_table),
        a=sql.Identifier(action_alias),
        join_on=sql.SQL(action_join_on),
        where=sql.SQL(" AND ").join(where_parts),
        order_by=sql.SQL(order_by),
    )

    if limit is not None:
        q = q + sql.SQL(" LIMIT %s")
        params.append(limit)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q, params)
            return cur.fetchall()


# -----------------------------
# DEFAULT points endpoint (re-included)
# -----------------------------
@router.get("/points", response_model=PointsResponse)
def get_points(
    player_id: int,
    match_id: Optional[int] = None,
    competition_id: Optional[int] = None,
    season_id: Optional[int] = None,
    primary: Optional[str] = Query(default=None, description="e.type_primary"),
    secondary_any: Optional[List[str]] = Query(default=None),
    bbox: Optional[str] = None,
    limit: Optional[int] = None,
):
    bbox_t = _parse_bbox(bbox)

    where_parts: List[sql.SQL] = []
    params: List[Any] = []

    _add_common_filters(
        where_parts=where_parts,
        params=params,
        player_id=player_id,
        match_id=match_id,
        competition_id=competition_id,
        season_id=season_id,
        bbox=bbox_t,
    )

    if primary is not None:
        where_parts.append(sql.SQL("e.type_primary = %s"))
        params.append(primary)

    _add_secondary_any_filter(where_parts=where_parts, params=params, secondary_any=secondary_any, table_alias="e")

    q = sql.SQL("""
        SELECT e.location_x, e.location_y
        FROM eventstream_events e
        WHERE {where}
        ORDER BY e.id
    """).format(where=sql.SQL(" AND ").join(where_parts))

    if limit is not None:
        q = q + sql.SQL(" LIMIT %s")
        params.append(limit)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q, params)
            rows = cur.fetchall()

    points_xy = [[float(x), float(y)] for (x, y) in rows]
    return {"player_id": player_id, "n": len(rows), "points_xy": points_xy}


# -----------------------------
# PASS points + end points + meta (accurate)
# -----------------------------
@router.get("/points/passes", response_model=PointsWithEndsAndMetaResponse)
def get_pass_points(
    player_id: int,
    match_id: Optional[int] = None,
    competition_id: Optional[int] = None,
    season_id: Optional[int] = None,
    secondary_any: Optional[List[str]] = Query(None),
    bbox: Optional[str] = None,
    limit: Optional[int] = None,
):
    bbox_t = _parse_bbox(bbox)

    rows = _fetch_action_rows(
        player_id=player_id,
        match_id=match_id,
        competition_id=competition_id,
        season_id=season_id,
        bbox=bbox_t,
        limit=limit,
        action_table="eventstream_passes",
        action_alias="p",
        action_join_on="p.event_id = e.id",
        select_cols=[
            "e.location_x", "e.location_y",
            "p.end_x", "p.end_y",
            "p.accurate",
        ],
        event_primary="pass",
        secondary_any=secondary_any,
        extra_where=[
            "p.end_x IS NOT NULL",
            "p.end_y IS NOT NULL",
        ],
    )

    points_xy = [[float(r[0]), float(r[1])] for r in rows]
    end_points_xy = [[float(r[2]), float(r[3])] for r in rows]
    meta = [{"accurate": r[4]} for r in rows]

    return {"player_id": player_id, "n": len(rows), "points_xy": points_xy, "end_points_xy": end_points_xy, "meta": meta}


# -----------------------------
# CARRY points + end points + meta (progression)
# carries are type_secondary for now (but supports primary too)
# -----------------------------
@router.get("/points/carries", response_model=PointsWithEndsAndMetaResponse)
def get_carry_points(
    player_id: int,
    match_id: Optional[int] = None,
    competition_id: Optional[int] = None,
    season_id: Optional[int] = None,
    bbox: Optional[str] = None,
    limit: Optional[int] = None,
):
    bbox_t = _parse_bbox(bbox)

    rows = _fetch_action_rows(
        player_id=player_id,
        match_id=match_id,
        competition_id=competition_id,
        season_id=season_id,
        bbox=bbox_t,
        limit=limit,
        action_table="eventstream_carries",
        action_alias="c",
        action_join_on="c.event_id = e.id",
        select_cols=[
            "e.location_x", "e.location_y",
            "c.end_x", "c.end_y",
            "c.progression",
        ],
        event_primary_or_secondary=("carry", "carry"),
        extra_where=[
            "c.end_x IS NOT NULL",
            "c.end_y IS NOT NULL",
        ],
    )

    points_xy = [[float(r[0]), float(r[1])] for r in rows]
    end_points_xy = [[float(r[2]), float(r[3])] for r in rows]
    meta = [{"progression": (None if r[4] is None else float(r[4]))} for r in rows]

    return {"player_id": player_id, "n": len(rows), "points_xy": points_xy, "end_points_xy": end_points_xy, "meta": meta}


# -----------------------------
# SHOT points + meta (xg/body_part/etc.)
# -----------------------------
@router.get("/points/shots", response_model=PointsWithEndsAndMetaResponse)
def get_shot_points(
    player_id: int,
    match_id: Optional[int] = None,
    competition_id: Optional[int] = None,
    season_id: Optional[int] = None,
    secondary_any: Optional[List[str]] = Query(None),
    bbox: Optional[str] = None,
    limit: Optional[int] = None,
):
    bbox_t = _parse_bbox(bbox)

    rows = _fetch_action_rows(
        player_id=player_id,
        match_id=match_id,
        competition_id=competition_id,
        season_id=season_id,
        bbox=bbox_t,
        limit=limit,
        action_table="eventstream_shots",
        action_alias="s",
        action_join_on="s.event_id = e.id",
        select_cols=[
            "e.location_x", "e.location_y",
            "s.is_goal", "s.on_target", "s.goal_zone", "s.xg", "s.body_part",
        ],
        event_primary="shot",
        secondary_any=secondary_any,
    )

    points_xy = [[float(r[0]), float(r[1])] for r in rows]
    meta = [
        {
            "is_goal": r[2],
            "on_target": r[3],
            "goal_zone": r[4],
            "xg": (None if r[5] is None else float(r[5])),
            "body_part": r[6],
        }
        for r in rows
    ]

    return {"player_id": player_id, "n": len(rows), "points_xy": points_xy, "meta": meta}


# -----------------------------
# AERIAL DUEL points + meta (first_touch)
# -----------------------------
@router.get("/points/aerial-duels", response_model=PointsWithEndsAndMetaResponse)
def get_aerial_duel_points(
    player_id: int,
    match_id: Optional[int] = None,
    competition_id: Optional[int] = None,
    season_id: Optional[int] = None,
    bbox: Optional[str] = None,
    limit: Optional[int] = None,
):
    bbox_t = _parse_bbox(bbox)

    rows = _fetch_action_rows(
        player_id=player_id,
        match_id=match_id,
        competition_id=competition_id,
        season_id=season_id,
        bbox=bbox_t,
        limit=limit,
        action_table="eventstream_aerial_duels",
        action_alias="a",
        action_join_on="a.event_id = e.id",
        select_cols=[
            "e.location_x", "e.location_y",
            "a.first_touch",
        ],
        event_primary_or_secondary=("aerial_duel", "aerial_duel"),
    )

    points_xy = [[float(r[0]), float(r[1])] for r in rows]
    meta = [{"first_touch": r[2]} for r in rows]

    return {"player_id": player_id, "n": len(rows), "points_xy": points_xy, "meta": meta}


# -----------------------------
# GROUND DUEL points + meta (recovered/kept + raw type_secondary)
# -----------------------------
# @router.get("/points/ground-duels", response_model=PointsWithEndsAndMetaResponse)
# def get_ground_duel_points(
#     player_id: int,
#     match_id: Optional[int] = None,
#     competition_id: Optional[int] = None,
#     season_id: Optional[int] = None,
#     bbox: Optional[str] = None,
#     limit: Optional[int] = None,
# ):
#     bbox_t = _parse_bbox(bbox)

#     rows = _fetch_action_rows(
#         player_id=player_id,
#         match_id=match_id,
#         competition_id=competition_id,
#         season_id=season_id,
#         bbox=bbox_t,
#         limit=limit,
#         action_table="eventstream_ground_duels",
#         action_alias="g",
#         action_join_on="g.event_id = e.id",
#         select_cols=[
#             "e.location_x", "e.location_y",
#             "g.recovered_possession", "g.kept_possession",
#             "e.type_secondary",
#         ],
#         event_primary_or_secondary=("ground_duel", "ground_duel"),
#     )

#     points_xy = [[float(r[0]), float(r[1])] for r in rows]
#     meta = [
#         {
#             "recovered_possession": r[2],
#             "kept_possession": r[3],
#             "type_secondary": r[4],
#         }
#         for r in rows
#     ]

#     return {"player_id": player_id, "n": len(rows), "points_xy": points_xy, "meta": meta}


'''Add secondary filtering'''
@router.get("/points/ground_duels", response_model=PointsWithEndsAndMetaResponse)
def get_ground_duel_points(
    player_id: int,
    match_id: Optional[int] = None,
    competition_id: Optional[int] = None,
    season_id: Optional[int] = None,
    secondary_any: Optional[List[str]] = None,  # ✅ allow frontend to filter (tackle/dribble/etc)
    bbox: Optional[str] = None,
    limit: Optional[int] = None,
):
    bbox_t = _parse_bbox(bbox)

    rows = _fetch_action_rows(
        player_id=player_id,
        match_id=match_id,
        competition_id=competition_id,
        season_id=season_id,
        bbox=bbox_t,
        limit=limit,

        action_table="eventstream_ground_duels",
        action_alias="g",
        action_join_on="g.event_id = e.id",
        select_cols=[
            "e.location_x", "e.location_y",
            "g.recovered_possession",
            "g.kept_possession",
            "e.type_secondary",
        ],

        # ✅ identify “ground duel events”
        # if your data stores these as secondary tags:
        event_primary_or_secondary=("ground_duel", "ground_duel"),

        # ✅ additionally filter within them
        secondary_any=secondary_any,
    )

    points_xy = [[float(r[0]), float(r[1])] for r in rows]
    meta = [
        {
            "recovered_possession": r[2],
            "kept_possession": r[3],
            "type_secondary": r[4],  # psycopg returns list for TEXT[]
        }
        for r in rows
    ]

    return {
        "player_id": player_id,
        "n": len(rows),
        "points_xy": points_xy,
        "end_points_xy": None,
        "meta": meta,
    }

