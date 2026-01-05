from __future__ import annotations

from typing import Optional, List, Any, Literal
from fastapi import APIRouter, Query, HTTPException
from psycopg import sql

from app_refactored.db import POOL

from app_refactored.schemas.passlinks import (
    Edge,
    TeamEdgesResponse,
    PassSegment,
    SegmentsResponse,
    NodePos,
    NodePositionsResponse,
)

from app_refactored.utils.passlinks import (
    build_team_edges_query,
    build_pass_segments_query,
    build_node_positions_query,
)

router = APIRouter(prefix="/v1/passlinks", tags=["passlinks"])


def _get_pool():
    return POOL


# -----------------------------
# A) EDGES for pass network
# -----------------------------
@router.get("/edges/team", response_model=TeamEdgesResponse)
def edges_for_team_in_match(
    match_id: int,
    team_id: int,
    only_accurate: bool = Query(False),
    min_count: int = Query(1, ge=1),
):
    where = [
        sql.SQL("e.match_id = %s"),
        sql.SQL("e.team_id = %s"),
    ]
    params: List[Any] = [match_id, team_id]

    q_base, where_base = build_team_edges_query(only_accurate=only_accurate)
    where_all = where_base + where

    q = q_base.format(where=sql.SQL(" AND ").join(where_all))
    params.append(min_count)

    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(q, params)
            rows = cur.fetchall()

    edges: List[Edge] = []
    for passer_id, receiver_id, n, n_acc in rows:
        n = int(n)
        n_acc = int(n_acc or 0)
        acc_pct = (n_acc / n * 100.0) if n else None
        edges.append(
            Edge(
                passer_id=int(passer_id),
                receiver_id=int(receiver_id),
                n=n,
                n_accurate=n_acc,
                accuracy_pct=acc_pct,
            )
        )

    return {"match_id": match_id, "team_id": team_id, "edges": edges}


# -----------------------------
# B) GEOMETRY: SEGMENTS between two players
# -----------------------------
@router.get("/segments", response_model=SegmentsResponse)
def pass_segments_between_players(
    passer_id: int,
    receiver_id: int,
    match_id: Optional[int] = None,
    competition_id: Optional[int] = None,
    season_id: Optional[int] = None,
    direction: Literal["forward", "both"] = "forward",
    only_accurate: bool = Query(False),
    limit: Optional[int] = Query(None, ge=1, le=5000),
):
    # Build forward query
    try:
        q1, params1 = build_pass_segments_query(
            passer_id=passer_id,
            receiver_id=receiver_id,
            match_id=match_id,
            competition_id=competition_id,
            season_id=season_id,
            only_accurate=only_accurate,
            limit=limit,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    segments: List[PassSegment] = []

    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(q1, params1)
            rows = cur.fetchall()
            for eid, minute, second, sx, sy, ex, ey, acc in rows:
                segments.append(
                    PassSegment(
                        event_id=int(eid),
                        minute=minute,
                        second=second,
                        start_x=float(sx),
                        start_y=float(sy),
                        end_x=float(ex),
                        end_y=float(ey),
                        accurate=acc,
                    )
                )

            # Reverse direction
            if direction == "both":
                q2, params2 = build_pass_segments_query(
                    passer_id=receiver_id,
                    receiver_id=passer_id,
                    match_id=match_id,
                    competition_id=competition_id,
                    season_id=season_id,
                    only_accurate=only_accurate,
                    limit=limit,
                )
                cur.execute(q2, params2)
                rows2 = cur.fetchall()
                for eid, minute, second, sx, sy, ex, ey, acc in rows2:
                    segments.append(
                        PassSegment(
                            event_id=int(eid),
                            minute=minute,
                            second=second,
                            start_x=float(sx),
                            start_y=float(sy),
                            end_x=float(ex),
                            end_y=float(ey),
                            accurate=acc,
                        )
                    )

    return {
        "passer_id": passer_id,
        "receiver_id": receiver_id,
        "match_id": match_id,
        "competition_id": competition_id,
        "season_id": season_id,
        "direction": direction,
        "n": len(segments),
        "segments": segments,
    }


# -----------------------------
# C) NODE POSITIONS for network layout
# -----------------------------
@router.get("/nodes/positions", response_model=NodePositionsResponse)
def node_positions(
    match_id: Optional[int] = None,
    competition_id: Optional[int] = None,
    season_id: Optional[int] = None,
    team_id: Optional[int] = None,
    mode: Literal["pass_start", "reception_end"] = "pass_start",
    min_n: int = Query(5, ge=1),
):
    try:
        q, params = build_node_positions_query(
            match_id=match_id,
            competition_id=competition_id,
            season_id=season_id,
            team_id=team_id,
            mode=mode,
            min_n=min_n,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(q, params)
            rows = cur.fetchall()

    nodes = [
        NodePos(player_id=int(pid), n=int(n), avg_x=float(ax), avg_y=float(ay))
        for pid, n, ax, ay in rows
    ]

    return {
        "match_id": match_id,
        "competition_id": competition_id,
        "season_id": season_id,
        "team_id": team_id,
        "mode": mode,
        "nodes": nodes,
    }
