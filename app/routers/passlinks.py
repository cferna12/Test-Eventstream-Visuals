from __future__ import annotations

from typing import Optional, List, Any, Literal, Dict
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from psycopg import sql

from app.db import POOL

router = APIRouter(prefix="/v1/passlinks", tags=["passlinks"])


def _get_pool():
    return POOL


# -----------------------------
# Models
# -----------------------------
class Edge(BaseModel):
    passer_id: int
    receiver_id: int
    n: int
    n_accurate: int
    accuracy_pct: Optional[float]


class TeamEdgesResponse(BaseModel):
    match_id: int
    team_id: int
    edges: List[Edge]


class PassSegment(BaseModel):
    event_id: int
    minute: Optional[int] = None
    second: Optional[int] = None
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    accurate: Optional[bool] = None


class SegmentsResponse(BaseModel):
    passer_id: int
    receiver_id: int
    match_id: Optional[int] = None
    competition_id: Optional[int] = None
    season_id: Optional[int] = None
    direction: Literal["forward", "both"] = "forward"
    n: int
    segments: List[PassSegment]


class NodePos(BaseModel):
    player_id: int
    n: int
    avg_x: float
    avg_y: float


class NodePositionsResponse(BaseModel):
    match_id: Optional[int] = None
    competition_id: Optional[int] = None
    season_id: Optional[int] = None
    team_id: Optional[int] = None
    mode: Literal["pass_start", "reception_end"] = "pass_start"
    nodes: List[NodePos]


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
    """
    Network edges for a team in a specific match.
    passer = e.player_id
    receiver = p.recipient_id
    """
    where = [
        sql.SQL("e.type_primary = 'pass'"),
        sql.SQL("e.match_id = %s"),
        sql.SQL("e.team_id = %s"),
        sql.SQL("p.recipient_id IS NOT NULL"),
    ]
    params: List[Any] = [match_id, team_id]

    if only_accurate:
        where.append(sql.SQL("p.accurate IS TRUE"))

    q = sql.SQL("""
        SELECT
          e.player_id AS passer_id,
          p.recipient_id AS receiver_id,
          COUNT(*)::INT AS n,
          SUM(CASE WHEN p.accurate IS TRUE THEN 1 ELSE 0 END)::INT AS n_accurate
        FROM eventstream_passes p
        JOIN eventstream_events e ON e.id = p.event_id
        WHERE {where}
        GROUP BY e.player_id, p.recipient_id
        HAVING COUNT(*) >= %s
        ORDER BY n DESC
    """).format(where=sql.SQL(" AND ").join(where))

    params.append(min_count)

    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(q, params)
            rows = cur.fetchall()

    edges: List[Edge] = []
    for passer_id, receiver_id, n, n_acc in rows:
        acc_pct = (n_acc / n * 100.0) if n else None
        edges.append(Edge(
            passer_id=int(passer_id),
            receiver_id=int(receiver_id),
            n=int(n),
            n_accurate=int(n_acc or 0),
            accuracy_pct=acc_pct,
        ))

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
    """
    Returns per-pass arrows (start->end).
    Start coords: eventstream_events.location_x/y
    End coords:   eventstream_passes.end_x/y
    """
    if match_id is None and (competition_id is None or season_id is None):
        raise HTTPException(
            status_code=400,
            detail="Provide either match_id OR (competition_id AND season_id)."
        )

    def build_where(passer: int, receiver: int):
        where = [
            sql.SQL("e.type_primary = 'pass'"),
            sql.SQL("e.player_id = %s"),
            sql.SQL("p.recipient_id = %s"),
            sql.SQL("e.location_x IS NOT NULL"),
            sql.SQL("e.location_y IS NOT NULL"),
            sql.SQL("p.end_x IS NOT NULL"),
            sql.SQL("p.end_y IS NOT NULL"),
        ]
        params: List[Any] = [passer, receiver]

        if match_id is not None:
            where.append(sql.SQL("e.match_id = %s"))
            params.append(match_id)
        else:
            where.append(sql.SQL("e.competition_id = %s"))
            params.append(competition_id)
            where.append(sql.SQL("e.season_id = %s"))
            params.append(season_id)

        if only_accurate:
            where.append(sql.SQL("p.accurate IS TRUE"))

        return where, params

    base_select = sql.SQL("""
        SELECT
          e.id,
          e.minute,
          e.second,
          e.location_x, e.location_y,
          p.end_x, p.end_y,
          p.accurate
        FROM eventstream_passes p
        JOIN eventstream_events e ON e.id = p.event_id
        WHERE {where}
        ORDER BY e.id
    """)

    segments: List[PassSegment] = []

    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            # forward passer->receiver
            where1, params1 = build_where(passer_id, receiver_id)
            q1 = base_select.format(where=sql.SQL(" AND ").join(where1))
            if limit is not None:
                q1 = q1 + sql.SQL(" LIMIT %s")
                params1 = params1 + [limit]

            cur.execute(q1, params1)
            rows = cur.fetchall()
            for eid, minute, second, sx, sy, ex, ey, acc in rows:
                segments.append(PassSegment(
                    event_id=int(eid),
                    minute=minute,
                    second=second,
                    start_x=float(sx),
                    start_y=float(sy),
                    end_x=float(ex),
                    end_y=float(ey),
                    accurate=acc,
                ))

            # reverse receiver->passer if direction=both
            if direction == "both":
                where2, params2 = build_where(receiver_id, passer_id)
                q2 = base_select.format(where=sql.SQL(" AND ").join(where2))
                if limit is not None:
                    q2 = q2 + sql.SQL(" LIMIT %s")
                    params2 = params2 + [limit]

                cur.execute(q2, params2)
                rows2 = cur.fetchall()
                for eid, minute, second, sx, sy, ex, ey, acc in rows2:
                    segments.append(PassSegment(
                        event_id=int(eid),
                        minute=minute,
                        second=second,
                        start_x=float(sx),
                        start_y=float(sy),
                        end_x=float(ex),
                        end_y=float(ey),
                        accurate=acc,
                    ))

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
    """
    Returns average positions for players so you can place nodes on the pitch.

    mode:
      - pass_start: use pass start coords (events.location_x/y) for each passer
      - reception_end: use pass end coords (passes.end_x/y) for each recipient
    """
    if match_id is None and (competition_id is None or season_id is None):
        raise HTTPException(
            status_code=400,
            detail="Provide either match_id OR (competition_id AND season_id)."
        )

    where = [sql.SQL("e.type_primary = 'pass'")]
    params: List[Any] = []

    if match_id is not None:
        where.append(sql.SQL("e.match_id = %s"))
        params.append(match_id)
    else:
        where.append(sql.SQL("e.competition_id = %s"))
        params.append(competition_id)
        where.append(sql.SQL("e.season_id = %s"))
        params.append(season_id)

    if team_id is not None:
        where.append(sql.SQL("e.team_id = %s"))
        params.append(team_id)

    if mode == "pass_start":
        # average start locations by passer
        where += [sql.SQL("e.location_x IS NOT NULL"), sql.SQL("e.location_y IS NOT NULL")]
        q = sql.SQL("""
            SELECT
              e.player_id,
              COUNT(*)::INT AS n,
              AVG(e.location_x)::FLOAT AS avg_x,
              AVG(e.location_y)::FLOAT AS avg_y
            FROM eventstream_events e
            WHERE {where}
            GROUP BY e.player_id
            HAVING COUNT(*) >= %s
            ORDER BY n DESC
        """).format(where=sql.SQL(" AND ").join(where))
        params2 = params + [min_n]

    else:
        # average end locations by recipient
        where += [
            sql.SQL("p.recipient_id IS NOT NULL"),
            sql.SQL("p.end_x IS NOT NULL"),
            sql.SQL("p.end_y IS NOT NULL"),
        ]
        q = sql.SQL("""
            SELECT
              p.recipient_id AS player_id,
              COUNT(*)::INT AS n,
              AVG(p.end_x)::FLOAT AS avg_x,
              AVG(p.end_y)::FLOAT AS avg_y
            FROM eventstream_passes p
            JOIN eventstream_events e ON e.id = p.event_id
            WHERE {where}
            GROUP BY p.recipient_id
            HAVING COUNT(*) >= %s
            ORDER BY n DESC
        """).format(where=sql.SQL(" AND ").join(where))
        params2 = params + [min_n]

    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(q, params2)
            rows = cur.fetchall()

    nodes = [NodePos(player_id=int(pid), n=int(n), avg_x=float(ax), avg_y=float(ay)) for pid, n, ax, ay in rows]
    return {
        "match_id": match_id,
        "competition_id": competition_id,
        "season_id": season_id,
        "team_id": team_id,
        "mode": mode,
        "nodes": nodes,
    }
