from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from psycopg import sql

from app_refactored.utils.sql_filters import (
    BBox,
    add_common_event_filters,
    add_secondary_any_filter,
    add_primary_or_secondary_tag,
)


# -----------------------------
# Low-level fetchers
# -----------------------------
def fetch_action_rows(
    *,
    pool,
    player_id: int,
    match_id: Optional[int],
    competition_id: Optional[int],
    season_id: Optional[int],
    bbox: Optional[BBox],
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

    add_common_event_filters(
        where_parts=where_parts,
        params=params,
        player_id=player_id,
        match_id=match_id,
        competition_id=competition_id,
        season_id=season_id,
        bbox=bbox,
        event_alias="e",
        require_xy=True,
    )

    if event_primary is not None:
        where_parts.append(sql.SQL("e.type_primary = %s"))
        params.append(event_primary)

    if event_primary_or_secondary is not None:
        primary_val, secondary_tag = event_primary_or_secondary
        add_primary_or_secondary_tag(
            where_parts=where_parts,
            params=params,
            primary_value=primary_val,
            secondary_tag=secondary_tag,
            event_alias="e",
        )

    add_secondary_any_filter(where_parts=where_parts, params=params, secondary_any=secondary_any, table_alias="e")

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

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(q, params)
            return cur.fetchall()


def fetch_reception_rows(
    *,
    pool,
    recipient_id: int,
    match_id: Optional[int],
    competition_id: Optional[int],
    season_id: Optional[int],
    bbox: Optional[BBox],
    limit: Optional[int],
    secondary_any: Optional[List[str]] = None,
    order_by: str = "e.id",
) -> list[tuple]:
    """
    Receptions are encoded on pass events:
      receiver = eventstream_passes.recipient_id
      reception location = eventstream_passes.end_x/end_y
    """
    where_parts: List[sql.SQL] = []
    params: List[Any] = []

    where_parts.append(sql.SQL("p.recipient_id = %s"))
    params.append(recipient_id)

    if match_id is not None:
        where_parts.append(sql.SQL("e.match_id = %s"))
        params.append(match_id)
    if competition_id is not None:
        where_parts.append(sql.SQL("e.competition_id = %s"))
        params.append(competition_id)
    if season_id is not None:
        where_parts.append(sql.SQL("e.season_id = %s"))
        params.append(season_id)

    where_parts.append(sql.SQL("p.end_x IS NOT NULL"))
    where_parts.append(sql.SQL("p.end_y IS NOT NULL"))

    if bbox:
        x_min, x_max, y_min, y_max = bbox
        where_parts.append(sql.SQL("p.end_x BETWEEN %s AND %s"))
        params += [x_min, x_max]
        where_parts.append(sql.SQL("p.end_y BETWEEN %s AND %s"))
        params += [y_min, y_max]

    if secondary_any:
        where_parts.append(sql.SQL("e.type_secondary && %s"))
        params.append(secondary_any)

    where_parts.append(sql.SQL("e.type_primary = %s"))
    params.append("pass")

    q = sql.SQL("""
        SELECT p.end_x, p.end_y
        FROM eventstream_passes p
        JOIN eventstream_events e ON e.id = p.event_id
        WHERE {where}
        ORDER BY {order_by}
    """).format(
        where=sql.SQL(" AND ").join(where_parts),
        order_by=sql.SQL(order_by),
    )

    if limit is not None:
        q = q + sql.SQL(" LIMIT %s")
        params.append(limit)

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(q, params)
            return cur.fetchall()


# -----------------------------
# High-level services (return dicts)
# -----------------------------
def get_points_service(
    *,
    pool,
    player_id: int,
    match_id: Optional[int],
    competition_id: Optional[int],
    season_id: Optional[int],
    primary: Optional[str],
    secondary_any: Optional[List[str]],
    bbox: Optional[BBox],
    limit: Optional[int],
) -> Dict[str, Any]:
    if primary == "pass_reception":
        rows = fetch_reception_rows(
            pool=pool,
            recipient_id=player_id,
            match_id=match_id,
            competition_id=competition_id,
            season_id=season_id,
            bbox=bbox,
            limit=limit,
            secondary_any=secondary_any,
        )
        points_xy = [[float(x), float(y)] for (x, y) in rows]
        return {"player_id": player_id, "n": len(rows), "points_xy": points_xy}

    where_parts: List[sql.SQL] = []
    params: List[Any] = []

    add_common_event_filters(
        where_parts=where_parts,
        params=params,
        player_id=player_id,
        match_id=match_id,
        competition_id=competition_id,
        season_id=season_id,
        bbox=bbox,
        event_alias="e",
        require_xy=True,
    )

    if primary is not None:
        where_parts.append(sql.SQL("e.type_primary = %s"))
        params.append(primary)

    add_secondary_any_filter(where_parts=where_parts, params=params, secondary_any=secondary_any, table_alias="e")

    q = sql.SQL("""
        SELECT e.location_x, e.location_y
        FROM eventstream_events e
        WHERE {where}
        ORDER BY e.id
    """).format(where=sql.SQL(" AND ").join(where_parts))

    if limit is not None:
        q = q + sql.SQL(" LIMIT %s")
        params.append(limit)

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(q, params)
            rows = cur.fetchall()

    points_xy = [[float(x), float(y)] for (x, y) in rows]
    return {"player_id": player_id, "n": len(rows), "points_xy": points_xy}


def get_pass_points_service(
    *,
    pool,
    player_id: int,
    match_id: Optional[int],
    competition_id: Optional[int],
    season_id: Optional[int],
    secondary_any: Optional[List[str]],
    bbox: Optional[BBox],
    limit: Optional[int],
) -> Dict[str, Any]:
    rows = fetch_action_rows(
        pool=pool,
        player_id=player_id,
        match_id=match_id,
        competition_id=competition_id,
        season_id=season_id,
        bbox=bbox,
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
    accurate = [r[4] for r in rows]
    return {
        "player_id": player_id,
        "n": len(rows),
        "points_xy": points_xy,
        "end_points_xy": end_points_xy,
        "accurate": accurate,
    }


def get_carry_points_service(
    *,
    pool,
    player_id: int,
    match_id: Optional[int],
    competition_id: Optional[int],
    season_id: Optional[int],
    bbox: Optional[BBox],
    limit: Optional[int],
) -> Dict[str, Any]:
    rows = fetch_action_rows(
        pool=pool,
        player_id=player_id,
        match_id=match_id,
        competition_id=competition_id,
        season_id=season_id,
        bbox=bbox,
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
    progression = [r[4] for r in rows]
    return {
        "player_id": player_id,
        "n": len(rows),
        "points_xy": points_xy,
        "end_points_xy": end_points_xy,
        "progression": progression,
    }


def get_shot_points_service(
    *,
    pool,
    player_id: int,
    match_id: Optional[int],
    competition_id: Optional[int],
    season_id: Optional[int],
    secondary_any: Optional[List[str]],
    bbox: Optional[BBox],
    limit: Optional[int],
) -> Dict[str, Any]:
    rows = fetch_action_rows(
        pool=pool,
        player_id=player_id,
        match_id=match_id,
        competition_id=competition_id,
        season_id=season_id,
        bbox=bbox,
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
    is_goal = [r[2] for r in rows]
    on_target = [r[3] for r in rows]
    goal_zone = [r[4] for r in rows]
    xg = [r[5] for r in rows]
    body_part = [r[6] for r in rows]

    return {
        "player_id": player_id,
        "n": len(rows),
        "points_xy": points_xy,
        "is_goal": is_goal,
        "on_target": on_target,
        "goal_zone": goal_zone,
        "xg": xg,
        "body_part": body_part,
    }


def get_aerial_duel_points_service(
    *,
    pool,
    player_id: int,
    match_id: Optional[int],
    competition_id: Optional[int],
    season_id: Optional[int],
    bbox: Optional[BBox],
    limit: Optional[int],
) -> Dict[str, Any]:
    rows = fetch_action_rows(
        pool=pool,
        player_id=player_id,
        match_id=match_id,
        competition_id=competition_id,
        season_id=season_id,
        bbox=bbox,
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
    first_touch = [r[2] for r in rows]

    return {"player_id": player_id, "n": len(rows), "points_xy": points_xy, "first_touch": first_touch}


def get_ground_duel_points_service(
    *,
    pool,
    player_id: int,
    match_id: Optional[int],
    competition_id: Optional[int],
    season_id: Optional[int],
    secondary_any: Optional[List[str]],
    bbox: Optional[BBox],
    limit: Optional[int],
) -> Dict[str, Any]:
    rows = fetch_action_rows(
        pool=pool,
        player_id=player_id,
        match_id=match_id,
        competition_id=competition_id,
        season_id=season_id,
        bbox=bbox,
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
        event_primary_or_secondary=("ground_duel", "ground_duel"),
        secondary_any=secondary_any,
    )

    points_xy = [[float(r[0]), float(r[1])] for r in rows]
    recovered_possession = [r[2] for r in rows]
    kept_possession = [r[3] for r in rows]
    type_secondary = [r[4] for r in rows]

    return {
        "player_id": player_id,
        "n": len(rows),
        "points_xy": points_xy,
        "recovered_possession": recovered_possession,
        "kept_possession": kept_possession,
        "type_secondary": type_secondary,
    }
