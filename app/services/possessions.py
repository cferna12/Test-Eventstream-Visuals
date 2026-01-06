from __future__ import annotations

from typing import Optional, List, Dict, Any, Tuple, Literal

import io
import numpy as np
from psycopg import sql

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mplsoccer import Pitch

from app.utils.possessions import (
    DEFAULT_PITCH_TYPE,
    DEFAULT_PITCH_BG_COLOR,
    DEFAULT_PITCH_LINE_COLOR,
    DEFAULT_BASE_EVENT_COLOR,
    DEFAULT_START_COLOR,
    DEFAULT_END_COLOR,
    DEFAULT_SHOT_COLOR,
    DEFAULT_PASS_LW,
    DEFAULT_CARRY_LW,
    DEFAULT_NODE_SIZE,
    DEFAULT_NODE_EDGE_LW,
    DEFAULT_ALPHA_MIN,
    DEFAULT_ALPHA_MAX,
    DEFAULT_NUMBER_FONTSIZE,
    DEFAULT_SHOT_TO_GOAL,
    DEFAULT_GOAL_X,
    DEFAULT_GOAL_Y_LEFT,
    DEFAULT_GOAL_Y_CENTER,
    DEFAULT_GOAL_Y_RIGHT,
)


# -----------------------------
# Shared helpers (logic-level)
# -----------------------------
def event_time_seconds(match_period: Optional[str], minute: Optional[int], second: Optional[int]) -> Optional[int]:
    if minute is None:
        return None
    s = int(second or 0)
    m = int(minute)
    p = (match_period or "").lower()

    if ("2" in p) or ("second" in p) or ("2h" in p):
        if m <= 45:
            return (45 * 60) + (m * 60) + s
        return (m * 60) + s

    return (m * 60) + s


def classify_event(type_primary: Optional[str], type_secondary: Optional[list]) -> str:
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


# -----------------------------
# DB: possession list
# -----------------------------
def fetch_possession_list(
    *,
    pool,
    match_id: int,
    team_id: Optional[int],
    from_minute: Optional[int],
    from_second: int,
    from_period: str,
    to_minute: Optional[int],
    to_second: int,
    to_period: str,
    only_possessions_with_shots: bool,
    only_possessions_with_goals: bool,
    limit: int,
) -> Dict[str, Any]:
    t_from = event_time_seconds(from_period, from_minute, from_second) if from_minute is not None else None
    t_to = event_time_seconds(to_period, to_minute, to_second) if to_minute is not None else None

    if only_possessions_with_goals:
        only_possessions_with_shots = True

    where = [
        sql.SQL("e.match_id = %s"),
        sql.SQL("e.possession_id IS NOT NULL"),
        sql.SQL("e.minute IS NOT NULL"),
    ]
    params: List[Any] = [match_id]

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

    shot_goal_exists_sql = sql.SQL("")
    if only_possessions_with_shots or only_possessions_with_goals:
        team_clause = sql.SQL("")
        if team_id is not None:
            team_clause = sql.SQL("AND es.team_id = %s")

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

        if team_id is not None:
            params.append(team_id)

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

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(q, params)
            rows = cur.fetchall()

    if not rows:
        return {
            "match_id": match_id,
            "team_id_filter": team_id,
            "time_window": {"from": t_from, "to": t_to},
            "filters": {
                "only_possessions_with_shots": only_possessions_with_shots,
                "only_possessions_with_goals": only_possessions_with_goals,
            },
            "n_possessions": 0,
            "possessions": [],
        }

    agg: Dict[int, Dict[str, Any]] = {}

    for poss_id, ev_team_id, mp, m, s, ev_id in rows:
        if poss_id is None:
            continue
        poss_id = int(poss_id)
        ev_team_id = int(ev_team_id) if ev_team_id is not None else None

        t = event_time_seconds(mp, m, s)
        if t is None:
            continue

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
                "team_event_counts": {},
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


# -----------------------------
# DB: possession chain json
# -----------------------------
def fetch_possession_chain(
    *,
    pool,
    match_id: int,
    possession_id: int,
    team_id: Optional[int],
    include_other_team_events: bool,
    limit: int,
) -> Dict[str, Any]:
    where_parts = [sql.SQL("e.match_id = %s"), sql.SQL("e.possession_id = %s")]
    params: List[Any] = [match_id, possession_id]

    if team_id is not None and not include_other_team_events:
        where_parts.append(sql.SQL("e.team_id = %s"))
        params.append(team_id)

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

          s.is_goal, s.xg, s.body_part, s.goal_zone

        FROM eventstream_events e
        LEFT JOIN eventstream_passes  p ON p.event_id = e.id
        LEFT JOIN eventstream_carries c ON c.event_id = e.id
        LEFT JOIN eventstream_shots   s ON s.event_id = e.id
        WHERE {where}
        ORDER BY e.match_period, e.minute, e.second, e.id
        LIMIT %s
    """).format(where=sql.SQL(" AND ").join(where_parts))

    params.append(limit)

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(q, params)
            rows = cur.fetchall()

    if not rows:
        raise LookupError("No events found for that possession (with current filters).")

    events: List[Dict[str, Any]] = []
    for r in rows:
        (
            ev_id, ev_team_id, player_id, player_name, type_primary, type_secondary,
            mp, minute, second, x, y,
            pass_ex, pass_ey, recipient_id, recipient_name,
            carry_ex, carry_ey,
            is_goal, xg, body_part, goal_zone
        ) = r

        t = event_time_seconds(mp, minute, second)

        end_x = pass_ex if pass_ex is not None else carry_ex
        end_y = pass_ey if pass_ey is not None else carry_ey

        label = classify_event(type_primary, type_secondary)

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
            "recipient_id": int(recipient_id) if recipient_id not in (None, 0) else None,
            "recipient_name": recipient_name if recipient_id not in (None, 0) else None,
            "is_goal": bool(is_goal) if is_goal is not None else None,
            "xg": float(xg) if xg is not None else None,
            "body_part": body_part,
            "goal_zone": goal_zone,
        })

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


# -----------------------------
# Render: possession chain image
# -----------------------------
def _goal_zone_lr(goal_zone: Optional[str]) -> Optional[str]:
    if not goal_zone:
        return None
    z = str(goal_zone).strip().lower()
    if not z.startswith("g") or len(z) < 2:
        return None
    c = z[1]
    if c == "l":
        return "l"
    if c == "r":
        return "r"
    if c == "c":
        return "c"
    if c in {"t", "b"}:
        return "c"
    return None


def _goal_mouth_target(lr: Optional[str]) -> Tuple[float, float]:
    if lr == "l":
        return (DEFAULT_GOAL_X, DEFAULT_GOAL_Y_LEFT)
    if lr == "r":
        return (DEFAULT_GOAL_X, DEFAULT_GOAL_Y_RIGHT)
    return (DEFAULT_GOAL_X, DEFAULT_GOAL_Y_CENTER)


def render_possession_chain_image(
    *,
    chain: Dict[str, Any],
    # behavior params kept on endpoint
    cutoff_t: Optional[int],
    dpi: int,
    annotate_numbers: bool,
    max_events: int,
    # allow overriding pitch_type as an endpoint param if you want; otherwise keep constant
    pitch_type: str,
    # allow overriding shot_to_goal as a behavior param; default constant
    shot_to_goal: bool,
) -> bytes:
    events: List[Dict[str, Any]] = chain["events"]

    if cutoff_t is not None:
        events = [e for e in events if e.get("t") is not None and e["t"] <= cutoff_t]

    if not events:
        raise LookupError("No events after applying cutoff/filtering.")

    if len(events) > max_events:
        events = events[-max_events:]

    events = [e for e in events if e.get("x") is not None and e.get("y") is not None]
    if len(events) < 2:
        raise LookupError("Not enough spatial events to render a chain.")

    start_idx = 0
    end_idx = len(events) - 1

    n = len(events)
    alphas = np.linspace(DEFAULT_ALPHA_MIN, DEFAULT_ALPHA_MAX, n)

    pitch = Pitch(
        pitch_type=pitch_type,
        pitch_color=DEFAULT_PITCH_BG_COLOR,
        line_color=DEFAULT_PITCH_LINE_COLOR,
    )
    fig, ax = pitch.draw(figsize=(10.8, 6.6))

    # movement
    for i, e in enumerate(events):
        x, y = float(e["x"]), float(e["y"])
        ex, ey = e.get("end_x"), e.get("end_y")
        a = float(alphas[i])

        label = (e.get("label") or "event").lower()
        is_pass = (label == "pass") and (ex is not None and ey is not None)
        is_carry = (label == "carry") and (ex is not None and ey is not None)
        is_shot = (label == "shot")

        if is_pass:
            pitch.arrows(
                x, y, float(ex), float(ey),
                ax=ax,
                color=DEFAULT_BASE_EVENT_COLOR,
                alpha=a,
                lw=DEFAULT_PASS_LW,
                headwidth=5.0,
                headlength=5.0,
                zorder=2,
            )

        elif is_carry:
            pitch.lines(
                x, y, float(ex), float(ey),
                ax=ax,
                color=DEFAULT_BASE_EVENT_COLOR,
                alpha=a,
                lw=DEFAULT_CARRY_LW,
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
                color=DEFAULT_BASE_EVENT_COLOR,
                alpha=a,
                lw=DEFAULT_CARRY_LW,
                headwidth=4.5,
                headlength=4.5,
                zorder=3,
            )

        elif is_shot:
            if shot_to_goal:
                lr = _goal_zone_lr(e.get("goal_zone"))
                tx, ty = _goal_mouth_target(lr)
                pitch.arrows(
                    x, y, float(tx), float(ty),
                    ax=ax,
                    color=DEFAULT_SHOT_COLOR,
                    alpha=min(1.0, a + 0.12),
                    lw=max(2.8, DEFAULT_PASS_LW + 0.4),
                    headwidth=6.2,
                    headlength=6.2,
                    zorder=4,
                )
            elif ex is not None and ey is not None:
                pitch.arrows(
                    x, y, float(ex), float(ey),
                    ax=ax,
                    color=DEFAULT_SHOT_COLOR,
                    alpha=min(1.0, a + 0.10),
                    lw=max(2.6, DEFAULT_PASS_LW),
                    headwidth=6.0,
                    headlength=6.0,
                    zorder=4,
                )

            ax.scatter(
                [x], [y],
                s=150,
                marker="*",
                c=DEFAULT_SHOT_COLOR,
                edgecolors="none",
                alpha=min(1.0, a + 0.15),
                zorder=6
            )

    # nodes
    xs = [float(e["x"]) for e in events]
    ys = [float(e["y"]) for e in events]

    ax.scatter(
        xs, ys,
        s=DEFAULT_NODE_SIZE,
        facecolors=DEFAULT_PITCH_BG_COLOR,
        edgecolors=DEFAULT_BASE_EVENT_COLOR,
        linewidths=DEFAULT_NODE_EDGE_LW,
        alpha=0.95,
        zorder=5
    )

    # start overlay
    sx, sy = xs[start_idx], ys[start_idx]
    ax.scatter(
        [sx], [sy],
        s=DEFAULT_NODE_SIZE * 0.70,
        c=DEFAULT_START_COLOR,
        edgecolors="none",
        alpha=0.95,
        zorder=7
    )

    # end overlay
    ex_, ey_ = xs[end_idx], ys[end_idx]
    ax.scatter(
        [ex_], [ey_],
        s=DEFAULT_NODE_SIZE * 0.70,
        c=DEFAULT_END_COLOR,
        edgecolors="none",
        alpha=0.95,
        zorder=7
    )

    if annotate_numbers:
        for i, (x, y) in enumerate(zip(xs, ys), start=1):
            ax.text(
                x, y, str(i),
                ha="center", va="center",
                fontsize=DEFAULT_NUMBER_FONTSIZE,
                color=DEFAULT_BASE_EVENT_COLOR,
                zorder=8
            )

    # title (use existing chain info if you want)
    match_id = chain.get("match_id")
    poss_id = chain.get("possession_id")
    ax.set_title(f"Possession Chain | match={match_id} poss={poss_id}",
                 color=DEFAULT_BASE_EVENT_COLOR, fontsize=14, pad=14)

    # legend
    y0 = -0.08
    ax.scatter([0.12], [y0], transform=ax.transAxes, s=110, c=DEFAULT_START_COLOR, clip_on=False)
    ax.text(0.16, y0, "Start", transform=ax.transAxes, color=DEFAULT_BASE_EVENT_COLOR,
            va="center", fontsize=10, clip_on=False)

    ax.scatter([0.28], [y0], transform=ax.transAxes, s=110, c=DEFAULT_END_COLOR, clip_on=False)
    ax.text(0.32, y0, "End", transform=ax.transAxes, color=DEFAULT_BASE_EVENT_COLOR,
            va="center", fontsize=10, clip_on=False)

    ax.scatter([0.42], [y0], transform=ax.transAxes, s=140, marker="*", c=DEFAULT_SHOT_COLOR, clip_on=False)
    ax.text(0.46, y0, "Shot", transform=ax.transAxes, color=DEFAULT_BASE_EVENT_COLOR,
            va="center", fontsize=10, clip_on=False)

    ax.annotate(
        "",
        xy=(0.66, y0), xytext=(0.58, y0),
        xycoords=ax.transAxes,
        arrowprops=dict(arrowstyle="->", lw=2.2, color=DEFAULT_BASE_EVENT_COLOR),
        annotation_clip=False,
    )
    ax.text(0.67, y0, "Pass", transform=ax.transAxes, color=DEFAULT_BASE_EVENT_COLOR,
            va="center", fontsize=10, clip_on=False)

    ax.plot([0.82, 0.92], [y0, y0], transform=ax.transAxes,
            linestyle=":", lw=2.0, color=DEFAULT_BASE_EVENT_COLOR, clip_on=False)
    ax.text(0.93, y0, "Carry", transform=ax.transAxes, color=DEFAULT_BASE_EVENT_COLOR,
            va="center", fontsize=10, clip_on=False)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
