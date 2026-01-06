from __future__ import annotations

from typing import Optional, List, Any, Literal, Tuple
from psycopg import sql


def _require_match_or_comp_season(
    *,
    match_id: Optional[int],
    competition_id: Optional[int],
    season_id: Optional[int],
) -> None:
    if match_id is None and (competition_id is None or season_id is None):
        raise ValueError("Provide either match_id OR (competition_id AND season_id).")


def _scope_where_and_params(
    *,
    match_id: Optional[int],
    competition_id: Optional[int],
    season_id: Optional[int],
    table_alias: str = "e",
) -> Tuple[List[sql.SQL], List[Any]]:
    """
    Returns WHERE parts + params for either:
    - match scope
    - (competition_id, season_id) scope

    table_alias: typically "e" (events), but keep flexible.
    """
    where: List[sql.SQL] = []
    params: List[Any] = []

    if match_id is not None:
        where.append(sql.SQL(f"{table_alias}.match_id = %s"))
        params.append(match_id)
    else:
        where.append(sql.SQL(f"{table_alias}.competition_id = %s"))
        params.append(competition_id)
        where.append(sql.SQL(f"{table_alias}.season_id = %s"))
        params.append(season_id)

    return where, params


def build_team_edges_query(
    *,
    only_accurate: bool,
) -> Tuple[sql.SQL, List[sql.SQL]]:
    """
    Query text is fixed; caller supplies where list + params.
    Returns:
      - base query template with {where}
      - base where clauses that always apply
    """
    where_base = [
        sql.SQL("e.type_primary = 'pass'"),
        sql.SQL("p.recipient_id IS NOT NULL"),
    ]
    if only_accurate:
        where_base.append(sql.SQL("p.accurate IS TRUE"))

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
    """)

    return q, where_base


def build_pass_segments_query(
    *,
    passer_id: int,
    receiver_id: int,
    match_id: Optional[int],
    competition_id: Optional[int],
    season_id: Optional[int],
    only_accurate: bool,
    limit: Optional[int],
) -> Tuple[sql.SQL, List[Any]]:
    """
    Returns a fully-formed query + params for (passer -> receiver).
    """
    _require_match_or_comp_season(match_id=match_id, competition_id=competition_id, season_id=season_id)

    where: List[sql.SQL] = [
        sql.SQL("e.type_primary = 'pass'"),
        sql.SQL("e.player_id = %s"),
        sql.SQL("p.recipient_id = %s"),
        sql.SQL("e.location_x IS NOT NULL"),
        sql.SQL("e.location_y IS NOT NULL"),
        sql.SQL("p.end_x IS NOT NULL"),
        sql.SQL("p.end_y IS NOT NULL"),
    ]
    params: List[Any] = [passer_id, receiver_id]

    scope_where, scope_params = _scope_where_and_params(
        match_id=match_id, competition_id=competition_id, season_id=season_id, table_alias="e"
    )
    where.extend(scope_where)
    params.extend(scope_params)

    if only_accurate:
        where.append(sql.SQL("p.accurate IS TRUE"))

    q = sql.SQL("""
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
    """).format(where=sql.SQL(" AND ").join(where))

    if limit is not None:
        q = q + sql.SQL(" LIMIT %s")
        params.append(limit)

    return q, params


def build_node_positions_query(
    *,
    match_id: Optional[int],
    competition_id: Optional[int],
    season_id: Optional[int],
    team_id: Optional[int],
    mode: Literal["pass_start", "reception_end"],
    min_n: int,
) -> Tuple[sql.SQL, List[Any]]:
    """
    Returns query + params for:
      - pass_start: avg events.location (group by passer)
      - reception_end: avg passes.end (group by recipient)
    """
    _require_match_or_comp_season(match_id=match_id, competition_id=competition_id, season_id=season_id)

    where: List[sql.SQL] = [sql.SQL("e.type_primary = 'pass'")]
    params: List[Any] = []

    scope_where, scope_params = _scope_where_and_params(
        match_id=match_id, competition_id=competition_id, season_id=season_id, table_alias="e"
    )
    where.extend(scope_where)
    params.extend(scope_params)

    if team_id is not None:
        where.append(sql.SQL("e.team_id = %s"))
        params.append(team_id)

    if mode == "pass_start":
        where.extend([sql.SQL("e.location_x IS NOT NULL"), sql.SQL("e.location_y IS NOT NULL")])
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
        params.append(min_n)
        return q, params

    # reception_end
    where.extend([
        sql.SQL("p.recipient_id IS NOT NULL"),
        sql.SQL("p.end_x IS NOT NULL"),
        sql.SQL("p.end_y IS NOT NULL"),
    ])
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
    params.append(min_n)
    return q, params
