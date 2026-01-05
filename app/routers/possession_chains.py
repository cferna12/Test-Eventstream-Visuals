# app/routers/possession_chain.py
from __future__ import annotations

from typing import Optional, List, Dict, Any, Tuple
from fastapi import APIRouter, Query, Response, HTTPException
from psycopg import sql
import io

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mplsoccer import Pitch

from app.db import POOL

router = APIRouter(prefix="/v1/possessions", tags=["possessions"])


def _get_pool():
    return POOL


def _event_time_seconds(match_period: Optional[str], minute: Optional[int], second: Optional[int]) -> Optional[int]:
    """
    Convert (period, minute, second) into comparable seconds.
    Defensive because feeds vary (minute can be half-relative or cumulative).
    """
    if minute is None:
        return None
    s = int(second or 0)
    m = int(minute)
    p = (match_period or "").lower()

    # If period suggests 2nd half, add 45m offset *if minutes look half-relative*
    if ("2" in p) or ("second" in p) or ("2h" in p):
        if m <= 45:
            return (45 * 60) + (m * 60) + s
        return (m * 60) + s

    # First half or unknown: treat minute as comparable
    return (m * 60) + s


def _classify_event(type_primary: Optional[str], type_secondary: Optional[list]) -> str:
    """
    Lightweight label for visualization / filtering.
    Extend as you like.
    """
    tp = (type_primary or "").lower()
    sec = [str(x).lower() for x in (type_secondary or [])]

    if tp == "pass":
        return "pass"
    if tp == "carry" or ("carry" in sec):
        return "carry"
    if tp == "shot":
        return "shot"
    if tp in {"duel", "ground_duel", "aerial_duel"} or ("duel" in sec):
        return "duel"
    if tp in {"interception", "clearance"} or ("interception" in sec):
        return "defensive"
    if tp in {"touch", "received_pass"}:
        return "touch"
    return tp or "event"


# @router.get("/list")
# def possession_list(
#     match_id: int,
#     team_id: Optional[int] = None,
#     # optional time window (lets you do "first 30 minutes", etc.)
#     from_minute: Optional[int] = Query(default=None, ge=0, le=200),
#     from_second: int = Query(default=0, ge=0, le=59),
#     from_period: str = Query(default="1H"),
#     to_minute: Optional[int] = Query(default=None, ge=0, le=200),
#     to_second: int = Query(default=0, ge=0, le=59),
#     to_period: str = Query(default="1H"),
#     limit: int = Query(default=500, ge=1, le=5000),
# ) -> Dict[str, Any]:
#     """
#     Returns possessions in chronological order for a match.
#     We do NOT assume possession_id is ordered. We order by earliest event time.

#     If team_id is provided, we restrict to possessions that include that team,
#     and we also return the "dominant" team_id for the possession (most events).
#     """
#     # Convert optional time bounds into seconds
#     t_from = _event_time_seconds(from_period, from_minute, from_second) if from_minute is not None else None
#     t_to = _event_time_seconds(to_period, to_minute, to_second) if to_minute is not None else None

#     # Basic filters
#     where = [
#         sql.SQL("match_id = %s"),
#         sql.SQL("possession_id IS NOT NULL"),
#         sql.SQL("minute IS NOT NULL"),
#     ]
#     params: List[Any] = [match_id]

#     if team_id is not None:
#         where.append(sql.SQL("team_id = %s"))
#         params.append(team_id)

#     # Pull minimal fields; we’ll compute time in Python (robust to period/minute format)
#     q = sql.SQL("""
#         SELECT
#           possession_id,
#           team_id,
#           match_period,
#           minute,
#           second,
#           id
#         FROM eventstream_events
#         WHERE {where}
#         ORDER BY match_period, minute, second, id
#     """).format(where=sql.SQL(" AND ").join(where))

#     with _get_pool().connection() as conn:
#         with conn.cursor() as cur:
#             cur.execute(q, params)
#             rows = cur.fetchall()

#     if not rows:
#         return {"match_id": match_id, "team_id_filter": team_id, "n_possessions": 0, "possessions": []}

#     # Aggregate
#     # possession_id -> stats
#     agg: Dict[int, Dict[str, Any]] = {}

#     for poss_id, ev_team_id, mp, m, s, ev_id in rows:
#         if poss_id is None:
#             continue
#         poss_id = int(poss_id)
#         ev_team_id = int(ev_team_id) if ev_team_id is not None else None

#         t = _event_time_seconds(mp, m, s)
#         if t is None:
#             continue

#         # time-window filtering
#         if t_from is not None and t < t_from:
#             continue
#         if t_to is not None and t > t_to:
#             continue

#         if poss_id not in agg:
#             agg[poss_id] = {
#                 "possession_id": poss_id,
#                 "start_time_seconds": t,
#                 "end_time_seconds": t,
#                 "n_events": 0,
#                 "team_event_counts": {},  # team_id -> count
#                 "first_event_id": int(ev_id),
#                 "last_event_id": int(ev_id),
#             }

#         d = agg[poss_id]
#         d["n_events"] += 1
#         if t < d["start_time_seconds"]:
#             d["start_time_seconds"] = t
#             d["first_event_id"] = int(ev_id)
#         if t > d["end_time_seconds"]:
#             d["end_time_seconds"] = t
#             d["last_event_id"] = int(ev_id)

#         if ev_team_id is not None:
#             tec = d["team_event_counts"]
#             tec[ev_team_id] = tec.get(ev_team_id, 0) + 1

#     possessions = list(agg.values())

#     # Determine "dominant" team per possession (team with most events)
#     for d in possessions:
#         tec = d.pop("team_event_counts", {})
#         if tec:
#             dom_team = max(tec.items(), key=lambda kv: kv[1])[0]
#         else:
#             dom_team = None
#         d["team_id"] = dom_team
#         d["duration_seconds"] = d["end_time_seconds"] - d["start_time_seconds"]

#     # Sort chronologically by start time, with tie-breaker on first_event_id
#     possessions.sort(key=lambda d: (d["start_time_seconds"], d["first_event_id"]))

#     # limit
#     if len(possessions) > limit:
#         possessions = possessions[:limit]

#     return {
#         "match_id": match_id,
#         "team_id_filter": team_id,
#         "time_window": {
#             "from": t_from,
#             "to": t_to,
#         },
#         "n_possessions": len(possessions),
#         "possessions": possessions,
#     }


'''TODO filter for only shots and goals'''
@router.get("/list")
def possession_list(
    match_id: int,
    team_id: Optional[int] = None,

    # optional time window
    from_minute: Optional[int] = Query(default=None, ge=0, le=200),
    from_second: int = Query(default=0, ge=0, le=59),
    from_period: str = Query(default="1H"),
    to_minute: Optional[int] = Query(default=None, ge=0, le=200),
    to_second: int = Query(default=0, ge=0, le=59),
    to_period: str = Query(default="1H"),

    # NEW filters
    only_possessions_with_shots: bool = Query(
        default=False,
        description="If true, only returns possessions that contain >=1 shot event (for the filtered team if team_id is provided).",
    ),
    only_possessions_with_goals: bool = Query(
        default=False,
        description="If true, only returns possessions that contain >=1 goal (shot with is_goal=true) (for the filtered team if team_id is provided).",
    ),

    limit: int = Query(default=500, ge=1, le=5000),
) -> Dict[str, Any]:
    """
    Returns possessions in chronological order for a match.

    IMPORTANT SEMANTICS:
    - If team_id is provided, we DO NOT filter events to team_id.
      Instead, we filter *possessions* to those that include that team (via EXISTS),
      while still retrieving all events in that possession to compute dominant team, duration, etc.

    Shot/goal filters:
    - If only_possessions_with_shots/goals is true, we restrict to possessions that contain
      at least one shot/goal for the team_id (if provided) otherwise for either team.
    """
    # Convert optional time bounds into seconds
    t_from = _event_time_seconds(from_period, from_minute, from_second) if from_minute is not None else None
    t_to = _event_time_seconds(to_period, to_minute, to_second) if to_minute is not None else None

    if only_possessions_with_goals:
        # goals implies shots; keep behavior intuitive
        only_possessions_with_shots = True

    # Base WHERE
    where = [
        sql.SQL("e.match_id = %s"),
        sql.SQL("e.possession_id IS NOT NULL"),
        sql.SQL("e.minute IS NOT NULL"),
    ]
    params: List[Any] = [match_id]

    # Possession membership filter by team (but DON'T filter events)
    team_exists_sql = sql.SQL("")
    if team_id is not None:
        team_exists_sql = sql.SQL("""
          AND EXISTS (
            SELECT 1
            FROM eventstream_events e2
            WHERE e2.match_id = e.match_id
              AND e2.possession_id = e.possession_id
              AND e2.team_id = %s
          )
        """)
        params.append(team_id)

    # Shot / goal possession filters
    shot_goal_exists_sql = sql.SQL("")
    if only_possessions_with_shots or only_possessions_with_goals:
        # If team_id is provided, require the shot/goal to be taken by that team.
        # If team_id is not provided, allow either team.
        team_clause = sql.SQL("")
        if team_id is not None:
            team_clause = sql.SQL("AND es.team_id = %s")

        # Shot filter (type_primary = 'shot')
        if only_possessions_with_shots and not only_possessions_with_goals:
            shot_goal_exists_sql = sql.SQL("""
              AND EXISTS (
                SELECT 1
                FROM eventstream_events es
                WHERE es.match_id = e.match_id
                  AND es.possession_id = e.possession_id
                  AND es.type_primary = 'shot'
                  {team_clause}
              )
            """).format(team_clause=team_clause)

        # Goal filter (join shots table is safest)
        if only_possessions_with_goals:
            shot_goal_exists_sql = sql.SQL("""
              AND EXISTS (
                SELECT 1
                FROM eventstream_events es
                JOIN eventstream_shots sh ON sh.event_id = es.id
                WHERE es.match_id = e.match_id
                  AND es.possession_id = e.possession_id
                  AND es.type_primary = 'shot'
                  AND sh.is_goal IS TRUE
                  {team_clause}
              )
            """).format(team_clause=team_clause)

        # If team_id is used in the EXISTS clause, add it to params again
        # (because it appears in a separate EXISTS subquery)
        if team_id is not None:
            params.append(team_id)

    # Pull minimal fields; compute time in Python
    q = sql.SQL("""
        SELECT
          e.possession_id,
          e.team_id,
          e.match_period,
          e.minute,
          e.second,
          e.id
        FROM eventstream_events e
        WHERE {where}
        {team_exists}
        {shot_goal_exists}
        ORDER BY e.match_period, e.minute, e.second, e.id
    """).format(
        where=sql.SQL(" AND ").join(where),
        team_exists=team_exists_sql,
        shot_goal_exists=shot_goal_exists_sql,
    )

    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(q, params)
            rows = cur.fetchall()

    if not rows:
        return {
            "match_id": match_id,
            "team_id_filter": team_id,
            "filters": {
                "only_possessions_with_shots": only_possessions_with_shots,
                "only_possessions_with_goals": only_possessions_with_goals,
            },
            "n_possessions": 0,
            "possessions": [],
        }

    # Aggregate possession stats
    agg: Dict[int, Dict[str, Any]] = {}

    for poss_id, ev_team_id, mp, m, s, ev_id in rows:
        if poss_id is None:
            continue
        poss_id = int(poss_id)
        ev_team_id = int(ev_team_id) if ev_team_id is not None else None

        t = _event_time_seconds(mp, m, s)
        if t is None:
            continue

        # time-window filtering
        if t_from is not None and t < t_from:
            continue
        if t_to is not None and t > t_to:
            continue

        if poss_id not in agg:
            agg[poss_id] = {
                "possession_id": poss_id,
                "start_time_seconds": t,
                "end_time_seconds": t,
                "n_events": 0,
                "team_event_counts": {},  # team_id -> count
                "first_event_id": int(ev_id),
                "last_event_id": int(ev_id),
            }

        d = agg[poss_id]
        d["n_events"] += 1
        if t < d["start_time_seconds"]:
            d["start_time_seconds"] = t
            d["first_event_id"] = int(ev_id)
        if t > d["end_time_seconds"]:
            d["end_time_seconds"] = t
            d["last_event_id"] = int(ev_id)

        if ev_team_id is not None:
            tec = d["team_event_counts"]
            tec[ev_team_id] = tec.get(ev_team_id, 0) + 1

    possessions = list(agg.values())

    # Determine dominant team per possession
    for d in possessions:
        tec = d.pop("team_event_counts", {})
        dom_team = max(tec.items(), key=lambda kv: kv[1])[0] if tec else None
        d["team_id"] = dom_team
        d["duration_seconds"] = d["end_time_seconds"] - d["start_time_seconds"]

    possessions.sort(key=lambda d: (d["start_time_seconds"], d["first_event_id"]))

    if len(possessions) > limit:
        possessions = possessions[:limit]

    return {
        "match_id": match_id,
        "team_id_filter": team_id,
        "time_window": {"from": t_from, "to": t_to},
        "filters": {
            "only_possessions_with_shots": only_possessions_with_shots,
            "only_possessions_with_goals": only_possessions_with_goals,
        },
        "n_possessions": len(possessions),
        "possessions": possessions,
    }

@router.get("/chain")
def possession_chain_json(
    match_id: int,
    possession_id: int,
    team_id: Optional[int] = None,
    include_other_team_events: bool = Query(
        default=False,
        description="If false (default), filters to team_id when provided. If true, includes all teams for the possession_id.",
    ),
    limit: int = Query(default=600, ge=1, le=2000),
) -> Dict[str, Any]:
    """
    Return ordered possession events as JSON.
    Includes:
      - event start x/y
      - pass/carry end x/y (if present)
      - recipient_id (passes)
      - shot meta (if present)
    """
    where_parts = [sql.SQL("e.match_id = %s"), sql.SQL("e.possession_id = %s")]
    params: List[Any] = [match_id, possession_id]

    if team_id is not None and not include_other_team_events:
        where_parts.append(sql.SQL("e.team_id = %s"))
        params.append(team_id)

    # Keep this filter so we don't flood with non-spatial rows, but you can remove if you want “full logs”
    where_parts.append(sql.SQL("e.location_x IS NOT NULL"))
    where_parts.append(sql.SQL("e.location_y IS NOT NULL"))

    q = sql.SQL("""
        SELECT
          e.id,
          e.team_id,
          e.player_id,
          e.player_name,
          e.type_primary,
          e.type_secondary,
          e.match_period,
          e.minute,
          e.second,
          e.location_x, e.location_y,

          p.end_x, p.end_y,
          p.recipient_id,
          p.recipient_name,

          c.end_x, c.end_y,

          s.is_goal, s.xg, s.body_part

        FROM eventstream_events e
        LEFT JOIN eventstream_passes  p ON p.event_id = e.id
        LEFT JOIN eventstream_carries c ON c.event_id = e.id
        LEFT JOIN eventstream_shots   s ON s.event_id = e.id
        WHERE {where}
        ORDER BY e.match_period, e.minute, e.second, e.id
        LIMIT %s
    """).format(where=sql.SQL(" AND ").join(where_parts))

    params.append(limit)

    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(q, params)
            rows = cur.fetchall()

    if not rows:
        raise HTTPException(status_code=404, detail="No events found for that possession (with current filters).")

    events: List[Dict[str, Any]] = []
    for r in rows:
        (
            ev_id, ev_team_id, player_id, player_name, type_primary, type_secondary,
            mp, minute, second, x, y,
            pass_ex, pass_ey, recipient_id, recipient_name,
            carry_ex, carry_ey,
            is_goal, xg, body_part
        ) = r

        t = _event_time_seconds(mp, minute, second)

        # prefer pass end coords if present, else carry end coords if present
        end_x = pass_ex if pass_ex is not None else carry_ex
        end_y = pass_ey if pass_ey is not None else carry_ey

        label = _classify_event(type_primary, type_secondary)

        events.append({
            "id": int(ev_id),
            "t": t,
            "match_period": mp,
            "minute": minute,
            "second": second,
            "team_id": int(ev_team_id) if ev_team_id is not None else None,
            "player_id": int(player_id) if player_id is not None else None,
            "player_name": player_name,
            "type_primary": type_primary,
            "type_secondary": type_secondary,
            "label": label,
            "x": float(x) if x is not None else None,
            "y": float(y) if y is not None else None,
            "end_x": float(end_x) if end_x is not None else None,
            "end_y": float(end_y) if end_y is not None else None,

            # pass extras
            "recipient_id": int(recipient_id) if recipient_id not in (None, 0) else None,
            "recipient_name": recipient_name if recipient_id not in (None, 0) else None,

            # shot extras
            "is_goal": bool(is_goal) if is_goal is not None else None,
            "xg": float(xg) if xg is not None else None,
            "body_part": body_part,
        })

    # enrich top-level summary
    inferred_team_ids = sorted({e["team_id"] for e in events if e["team_id"] is not None})
    t0 = next((e["t"] for e in events if e["t"] is not None), None)
    t1 = next((e["t"] for e in reversed(events) if e["t"] is not None), None)
    duration_s = (t1 - t0) if (t0 is not None and t1 is not None) else None

    return {
        "match_id": match_id,
        "possession_id": possession_id,
        "team_id_filter": team_id,
        "teams_in_events": inferred_team_ids,
        "n": len(events),
        "duration_seconds": duration_s,
        "events": events,
    }


@router.get("/image", response_class=Response)
def possession_chain_image(
    match_id: int,
    possession_id: int,
    team_id: Optional[int] = None,
    include_other_team_events: bool = Query(default=False),
    # time window (optional)
    cutoff_minute: Optional[int] = Query(default=None, ge=0, le=200),
    cutoff_second: int = Query(default=0, ge=0, le=59),
    cutoff_period: str = Query(default="1H"),

    # viz controls
    dpi: int = Query(default=170, ge=72, le=300),
    pitch_type: str = Query(default="wyscout"),
    annotate_numbers: bool = Query(default=True),
    show_player_names: bool = Query(default=False),
    max_events: int = Query(default=120, ge=10, le=600),
) -> Response:
    """
    Render a polished possession chain image overlayed on the pitch.

    - Passes: solid arrows
    - Carries: dashed arrows
    - Shots: star marker
    - Everything else: point
    - Optional numbering and player labels
    """
    chain = possession_chain_json(
        match_id=match_id,
        possession_id=possession_id,
        team_id=team_id,
        include_other_team_events=include_other_team_events,
        limit=2000,
    )
    events = chain["events"]

    # optional cutoff
    if cutoff_minute is not None:
        cutoff_t = _event_time_seconds(cutoff_period, cutoff_minute, cutoff_second)
        if cutoff_t is None:
            raise HTTPException(status_code=400, detail="Invalid cutoff time.")
        events = [e for e in events if e["t"] is not None and e["t"] <= cutoff_t]

    if not events:
        raise HTTPException(status_code=404, detail="No events after applying cutoff/filtering.")

    # clip to max_events (keep the most recent segment if it’s huge)
    if len(events) > max_events:
        events = events[-max_events:]

    # build node positions for (optional) labels: avg of start locations per player in this chain
    pos_sum: Dict[int, Tuple[float, float, int]] = {}
    name_map: Dict[int, str] = {}
    for e in events:
        pid = e.get("player_id")
        x = e.get("x")
        y = e.get("y")
        if pid is None or x is None or y is None:
            continue
        if e.get("player_name"):
            name_map[int(pid)] = e["player_name"]
        if pid not in pos_sum:
            pos_sum[pid] = (x, y, 1)
        else:
            sx, sy, n = pos_sum[pid]
            pos_sum[pid] = (sx + x, sy + y, n + 1)

    avg_pos: Dict[int, Tuple[float, float]] = {}
    for pid, (sx, sy, n) in pos_sum.items():
        avg_pos[pid] = (sx / n, sy / n)

    # --- draw
    # pitch = Pitch(pitch_type=pitch_type)
    # fig, ax = pitch.draw(figsize=(10, 7))

    # # alpha gradient: early faint -> late strong
    # n = len(events)
    # alphas = np.linspace(0.25, 0.95, n)

    # # draw arrows + points
    # for i, e in enumerate(events, start=1):
    #     x, y = e.get("x"), e.get("y")
    #     if x is None or y is None:
    #         continue

    #     label = e.get("label", "event")
    #     ex, ey = e.get("end_x"), e.get("end_y")
    #     a = float(alphas[i - 1])

    #     # choose marker style
    #     is_shot = (label == "shot")
    #     is_pass = (label == "pass") and (ex is not None and ey is not None)
    #     is_carry = (label == "carry") and (ex is not None and ey is not None)

    #     if is_pass:
    #         # solid arrow
    #         pitch.arrows(x, y, ex, ey, ax=ax, alpha=a, width=2, headwidth=4, headlength=4)
    #     elif is_carry:
    #         # dashed arrow: mplsoccer arrows doesn’t support dashed nicely; draw as a line + tiny arrow
    #         pitch.lines(x, y, ex, ey, ax=ax, alpha=a, lw=2, linestyle="--")
    #         pitch.arrows(
    #             (x + ex) / 2, (y + ey) / 2, ex, ey,
    #             ax=ax, alpha=a, width=1.6, headwidth=3.5, headlength=3.5
    #         )

    #     # point marker
    #     if is_shot:
    #         ax.scatter([x], [y], marker="*", s=180, alpha=min(1.0, a + 0.1))
    #     else:
    #         ax.scatter([x], [y], s=40, alpha=a)

    #     # numbering
    #     if annotate_numbers:
    #         ax.text(
    #             x, y,
    #             str(i),
    #             ha="center", va="center",
    #             fontsize=8,
    #             bbox=dict(boxstyle="circle,pad=0.15", alpha=0.6),
    #         )

    '''TODO improve visual'''
    pitch = Pitch(pitch_type=pitch_type)
    fig, ax = pitch.draw(figsize=(10, 7))

    STYLE = {
        "pass":      {"color": "#1f77b4", "marker": "o", "size": 42,  "lw": 2.2, "ls": "-"},
        "carry":     {"color": "#ff7f0e", "marker": "o", "size": 38,  "lw": 2.0, "ls": "--"},
        "shot":      {"color": "#d62728", "marker": "*", "size": 220, "lw": 0.0, "ls": "-"},
        "duel":      {"color": "#7f7f7f", "marker": "x", "size": 70,  "lw": 2.0, "ls": "-"},
        "defensive": {"color": "#7f7f7f", "marker": "x", "size": 70,  "lw": 2.0, "ls": "-"},
        "touch":     {"color": "#bbbbbb", "marker": ".", "size": 40,  "lw": 0.0, "ls": "-"},
        "event":     {"color": "#bbbbbb", "marker": ".", "size": 40,  "lw": 0.0, "ls": "-"},
    }

    n = len(events)
    tseq = np.linspace(0, 1, n)
    alphas = 0.10 + 0.90 * (tseq ** 1.8)  # early faint, late strong

    # subtle backbone path
    path = [(e["x"], e["y"]) for e in events if e.get("x") is not None and e.get("y") is not None]
    if len(path) >= 2:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        pitch.lines(path_x[:-1], path_y[:-1], path_x[1:], path_y[1:], ax=ax, lw=1.0, alpha=0.22, color="#222222", zorder=1)

    for i, e in enumerate(events, start=1):
        x, y = e.get("x"), e.get("y")
        if x is None or y is None:
            continue

        label = e.get("label", "event")
        sty = STYLE.get(label, STYLE["event"])
        ex, ey = e.get("end_x"), e.get("end_y")
        a = float(alphas[i - 1])
        z = 5 + i  # newer on top

        is_shot = (label == "shot")
        is_pass = (label == "pass") and (ex is not None and ey is not None)
        is_carry = (label == "carry") and (ex is not None and ey is not None)

        # time-weighted thickness
        ti = float(tseq[i - 1])
        lw = sty["lw"] * (0.8 + 0.6 * (ti ** 1.2))

        if is_pass:
            pitch.arrows(
                x, y, ex, ey, ax=ax,
                alpha=a, color=sty["color"],
                width=1.8 + 1.4 * (ti ** 1.4),
                headwidth=4.2, headlength=4.2,
                zorder=z
            )
            ax.scatter([ex], [ey], s=18, alpha=a, color=sty["color"], edgecolors="none", zorder=z)

        elif is_carry:
            pitch.lines(x, y, ex, ey, ax=ax, alpha=a, lw=lw, linestyle="--", color=sty["color"], zorder=z)
            pitch.arrows(
                (x + ex) / 2, (y + ey) / 2, ex, ey,
                ax=ax, alpha=a, color=sty["color"],
                width=1.4 + 1.2 * (ti ** 1.4),
                headwidth=3.8, headlength=3.8,
                zorder=z
            )
            ax.scatter([ex], [ey], s=16, alpha=a, color=sty["color"], edgecolors="none", zorder=z)

        # node marker
        ax.scatter(
            [x], [y],
            s=sty["size"],
            marker=sty["marker"],
            alpha=a,
            color=sty["color"],
            edgecolors="black" if sty["marker"] in {"o", "*"} else "none",
            linewidths=0.6 if sty["marker"] in {"o", "*"} else 0.0,
            zorder=z
        )

        # xG label for shots
        if is_shot and e.get("xg") is not None:
            ax.text(
                x, y - 2.0,
                f'xG {e["xg"]:.2f}',
                ha="center", va="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", alpha=0.55),
                zorder=z + 1
            )

        # numbering (less clutter)
        if annotate_numbers and (label in {"pass", "carry", "shot"} or i in {1, n}):
            ax.text(
                x, y,
                str(i),
                ha="center", va="center",
                fontsize=8,
                bbox=dict(boxstyle="circle,pad=0.12", alpha=0.35 + 0.35 * a),
                zorder=z + 2
            )

    # highlight final event
    last = next((e for e in reversed(events) if e.get("x") is not None and e.get("y") is not None), None)
    if last is not None:
        ax.scatter([last["x"]], [last["y"]], s=260, facecolors="none", edgecolors="black", linewidths=2.0, zorder=999)


    # optional player labels (at avg position)
    if show_player_names:
        for pid, (x, y) in avg_pos.items():
            nm = name_map.get(pid, str(pid))
            ax.text(
                x, y,
                nm,
                ha="center", va="center",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25", alpha=0.55),
            )

    title_bits = [f"Possession Chain | match={match_id} poss={possession_id}"]
    if team_id is not None and not include_other_team_events:
        title_bits.append(f"team={team_id}")
    if cutoff_minute is not None:
        title_bits.append(f"<= {cutoff_period} {cutoff_minute}:{cutoff_second:02d}")
    ax.set_title(" | ".join(title_bits))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")

@router.get("/image_v2", response_class=Response)
def possession_chain_image_v2(
    match_id: int,
    possession_id: int,
    team_id: Optional[int] = None,
    include_other_team_events: bool = Query(default=False),

    # optional cutoff (keep)
    cutoff_minute: Optional[int] = Query(default=None, ge=0, le=200),
    cutoff_second: int = Query(default=0, ge=0, le=59),
    cutoff_period: str = Query(default="1H"),

    # viz controls
    dpi: int = Query(default=180, ge=72, le=300),
    pitch_type: str = Query(default="wyscout"),

    annotate_numbers: bool = Query(default=True),
    max_events: int = Query(default=120, ge=10, le=600),

    # --- Soccermatics-ish style controls (optional params)
    pitch_bg_color: str = Query(default="#16313a", description="Pitch background color (dark works best)."),
    pitch_line_color: str = Query(default="#7f8c8d", description="Pitch line color."),
    base_event_color: str = Query(default="#cfd8dc", description="Default node/line color."),
    start_color: str = Query(default="#4da3ff", description="Start event marker color."),
    end_color: str = Query(default="#ff4d4d", description="End event marker color."),
    shot_color: str = Query(default="#ff4d4d", description="Shot highlight color."),

    pass_lw: float = Query(default=2.4, ge=0.5, le=6.0),
    carry_lw: float = Query(default=2.0, ge=0.5, le=6.0),

    node_size: float = Query(default=175, ge=50, le=600),
    node_edge_lw: float = Query(default=2.0, ge=0.5, le=6.0),

    # opacity / sequencing
    alpha_min: float = Query(default=0.30, ge=0.0, le=1.0),
    alpha_max: float = Query(default=0.95, ge=0.0, le=1.0),

    # label styling
    number_fontsize: int = Query(default=9, ge=6, le=14),

    # NEW: shot arrow controls
    shot_to_goal: bool = Query(default=True, description="If true, draws shot arrow to goal mouth using goal_zone."),
    goal_x: float = Query(default=100.0, ge=0.0, le=100.0, description="Attacking goal x-position (0..100)."),
    goal_y_left: float = Query(default=62.0, ge=0.0, le=100.0),
    goal_y_center: float = Query(default=50.0, ge=0.0, le=100.0),
    goal_y_right: float = Query(default=38.0, ge=0.0, le=100.0),
) -> Response:
    """
    Soccermatics-style possession chain (+ shot arrow to goal).

    - Dark pitch background + muted lines
    - Passes: solid arrows
    - Carries: dotted lines + small arrow head at end
    - Sequence nodes: outlined circles with numbers
    - Start event: blue dot
    - End event: red dot
    - Shot: red star + arrow to goal mouth derived from goal_zone (L/C/R)
    - Legend at bottom
    """

    def _goal_zone_lr(goal_zone: Optional[str]) -> Optional[str]:
        """
        Wyscout goal zones like: gtl, gt, gtr, gl, gc, gr, glb, gb, gbr
        We bucket by Left/Center/Right using the first letter after 'g'.
        - g[l]... => left
        - g[c]... => center
        - g[r]... => right
        - g[t]/g[b] alone => treat as center (no L/R info)
        """
        if not goal_zone:
            return None
        z = str(goal_zone).strip().lower()
        if not z.startswith("g") or len(z) < 2:
            return None
        c = z[1]  # first letter after 'g'
        if c == "l":
            return "l"
        if c == "r":
            return "r"
        if c == "c":
            return "c"
        # gt / gb / g? => center fallback
        if c in {"t", "b"}:
            return "c"
        return None

    def _goal_mouth_target(lr: Optional[str]) -> Tuple[float, float]:
        if lr == "l":
            return (goal_x, goal_y_left)
        if lr == "r":
            return (goal_x, goal_y_right)
        return (goal_x, goal_y_center)

    chain = possession_chain_json(
        match_id=match_id,
        possession_id=possession_id,
        team_id=team_id,
        include_other_team_events=include_other_team_events,
        limit=2000,
    )
    events: List[Dict[str, Any]] = chain["events"]

    # optional cutoff
    if cutoff_minute is not None:
        cutoff_t = _event_time_seconds(cutoff_period, cutoff_minute, cutoff_second)
        if cutoff_t is None:
            raise HTTPException(status_code=400, detail="Invalid cutoff time.")
        events = [e for e in events if e.get("t") is not None and e["t"] <= cutoff_t]

    if not events:
        raise HTTPException(status_code=404, detail="No events after applying cutoff/filtering.")

    # clip to max_events (keep the most recent segment if huge)
    if len(events) > max_events:
        events = events[-max_events:]

    # filter to spatial events only (defensive)
    events = [e for e in events if e.get("x") is not None and e.get("y") is not None]
    if len(events) < 2:
        raise HTTPException(status_code=404, detail="Not enough spatial events to render a chain.")

    # Identify start/end indices after filtering
    start_idx = 0
    end_idx = len(events) - 1

    # alpha gradient: early faint -> late strong
    n = len(events)
    alphas = np.linspace(alpha_min, alpha_max, n)

    # --- draw pitch (dark theme)
    pitch = Pitch(
        pitch_type=pitch_type,
        pitch_color=pitch_bg_color,
        line_color=pitch_line_color,
    )
    fig, ax = pitch.draw(figsize=(10.8, 6.6))

    # --- draw movement (passes/carries/shots)
    for i, e in enumerate(events):
        x, y = float(e["x"]), float(e["y"])
        ex, ey = e.get("end_x"), e.get("end_y")
        a = float(alphas[i])

        label = (e.get("label") or "event").lower()
        is_pass = (label == "pass") and (ex is not None and ey is not None)
        is_carry = (label == "carry") and (ex is not None and ey is not None)
        is_shot = (label == "shot")

        action_color = base_event_color

        if is_pass:
            pitch.arrows(
                x, y, float(ex), float(ey),
                ax=ax,
                color=action_color,
                alpha=a,
                lw=pass_lw,
                headwidth=5.0,
                headlength=5.0,
                zorder=2,
            )

        elif is_carry:
            pitch.lines(
                x, y, float(ex), float(ey),
                ax=ax,
                color=action_color,
                alpha=a,
                lw=carry_lw,
                linestyle=":",
                zorder=2,
            )
            dx = float(ex) - x
            dy = float(ey) - y
            seg = 0.22
            hx = float(ex) - seg * dx
            hy = float(ey) - seg * dy
            pitch.arrows(
                hx, hy, float(ex), float(ey),
                ax=ax,
                color=action_color,
                alpha=a,
                lw=carry_lw,
                headwidth=4.5,
                headlength=4.5,
                zorder=3,
            )

        elif is_shot:
            # preferred: arrow to goal mouth using goal_zone
            if shot_to_goal:
                lr = _goal_zone_lr(e.get("goal_zone"))
                tx, ty = _goal_mouth_target(lr)

                pitch.arrows(
                    x, y, float(tx), float(ty),
                    ax=ax,
                    color=shot_color,
                    alpha=min(1.0, a + 0.12),
                    lw=max(2.8, pass_lw + 0.4),
                    headwidth=6.2,
                    headlength=6.2,
                    zorder=4,
                )
            # fallback: arrow to end coords if present
            elif ex is not None and ey is not None:
                pitch.arrows(
                    x, y, float(ex), float(ey),
                    ax=ax,
                    color=shot_color,
                    alpha=min(1.0, a + 0.10),
                    lw=max(2.6, pass_lw),
                    headwidth=6.0,
                    headlength=6.0,
                    zorder=4,
                )

            ax.scatter(
                [x], [y],
                s=150,
                marker="*",
                c=shot_color,
                edgecolors="none",
                alpha=min(1.0, a + 0.15),
                zorder=6
            )

        else:
            pass

    # --- draw nodes (outlined circles with numbers)
    xs = [float(e["x"]) for e in events]
    ys = [float(e["y"]) for e in events]

    ax.scatter(
        xs, ys,
        s=node_size,
        facecolors=pitch_bg_color,
        edgecolors=base_event_color,
        linewidths=node_edge_lw,
        alpha=0.95,
        zorder=5
    )

    # Start marker overlay (blue filled)
    sx, sy = xs[start_idx], ys[start_idx]
    ax.scatter(
        [sx], [sy],
        s=node_size * 0.70,
        c=start_color,
        edgecolors="none",
        alpha=0.95,
        zorder=7
    )

    # End marker overlay (red filled)
    ex_, ey_ = xs[end_idx], ys[end_idx]
    ax.scatter(
        [ex_], [ey_],
        s=node_size * 0.70,
        c=end_color,
        edgecolors="none",
        alpha=0.95,
        zorder=7
    )

    # Numbers (sequence)
    if annotate_numbers:
        for i, (x, y) in enumerate(zip(xs, ys), start=1):
            ax.text(
                x, y, str(i),
                ha="center", va="center",
                fontsize=number_fontsize,
                color=base_event_color,
                zorder=8
            )

    # --- title
    title_bits = [f"Possession Chain | match={match_id} poss={possession_id}"]
    if team_id is not None and not include_other_team_events:
        title_bits.append(f"team={team_id}")
    if cutoff_minute is not None:
        title_bits.append(f"<= {cutoff_period} {cutoff_minute}:{cutoff_second:02d}")
    ax.set_title(" | ".join(title_bits), color=base_event_color, fontsize=14, pad=14)

    # --- legend (bottom)
    y0 = -0.08
    ax.scatter([0.12], [y0], transform=ax.transAxes, s=110, c=start_color, clip_on=False)
    ax.text(0.16, y0, "Start Event", transform=ax.transAxes, color=base_event_color,
            va="center", fontsize=10, clip_on=False)

    ax.scatter([0.36], [y0], transform=ax.transAxes, s=110, c=end_color, clip_on=False)
    ax.text(0.40, y0, "End Event", transform=ax.transAxes, color=base_event_color,
            va="center", fontsize=10, clip_on=False)

    ax.annotate(
        "",
        xy=(0.63, y0), xytext=(0.54, y0),
        xycoords=ax.transAxes,
        arrowprops=dict(arrowstyle="->", lw=2.2, color=base_event_color),
        annotation_clip=False,
    )
    ax.text(0.64, y0, "Pass", transform=ax.transAxes, color=base_event_color,
            va="center", fontsize=10, clip_on=False)

    ax.plot([0.78, 0.90], [y0, y0], transform=ax.transAxes,
            linestyle=":", lw=2.0, color=base_event_color, clip_on=False)
    ax.text(0.91, y0, "Carry", transform=ax.transAxes, color=base_event_color,
            va="center", fontsize=10, clip_on=False)

    # NEW: shot legend
    ax.scatter([0.02], [y0], transform=ax.transAxes, s=140, marker="*", c=shot_color, clip_on=False)
    ax.text(0.05, y0, "Shot", transform=ax.transAxes, color=base_event_color,
            va="center", fontsize=10, clip_on=False)

    # export
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")
