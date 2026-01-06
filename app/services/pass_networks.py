from __future__ import annotations

from typing import Optional, List, Dict, Tuple, Any, Set

import io
import math
import numpy as np
from psycopg import sql

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from mplsoccer import Pitch
import matplotlib.patheffects as pe

from app.utils.pass_networks import (
    DEFAULT_PITCH_TYPE,
    DEFAULT_PITCH_BG,
    DEFAULT_PITCH_LINE_COLOR,
    DEFAULT_CMAP_NAME,
    DEFAULT_COLOR_SOFTEN,
    DEFAULT_COLOR_ALPHA,
    DEFAULT_EDGE_ALPHA_MIN,
    DEFAULT_EDGE_ALPHA_MAX,
    DEFAULT_EDGE_LW_MIN,
    DEFAULT_EDGE_LW_MAX,
    DEFAULT_DONUT_RING_LW,
    DEFAULT_DONUT_HOLE_RATIO,
    DEFAULT_LABEL_FONTSIZE,
    DEFAULT_LABEL_OPACITY,
    DEFAULT_LABEL_BBOX,
    DEFAULT_LABEL_BBOX_ALPHA,
    DEFAULT_LABEL_BBOX_PAD,
    DEFAULT_LABEL_Y_OFFSET,
    DEFAULT_NODE_SIZE_MIN,
    DEFAULT_NODE_SIZE_MAX,
    DEFAULT_NODE_SIZE_GAMMA,
)

# -----------------------------
# Shared helpers (pure-ish)
# -----------------------------
def event_time_seconds(match_period: Optional[str], minute: Optional[int], second: Optional[int]) -> Optional[int]:
    if minute is None:
        return None
    s = int(second or 0)
    m = int(minute)
    p = (match_period or "").lower()

    if "2" in p or "second" in p:
        if m <= 45:
            return (45 * 60) + (m * 60) + s
        return (m * 60) + s

    return (m * 60) + s


def normalize_xi(player_ids: Optional[List[int]], allowed_players: Set[int]) -> Optional[Set[int]]:
    if not player_ids:
        return None
    cleaned: List[int] = []
    for x in player_ids:
        try:
            xi = int(x)
        except Exception:
            continue
        if xi > 0:
            cleaned.append(xi)

    xi_set = set(cleaned)
    if len(xi_set) < 2:
        raise ValueError("player_ids must contain at least 2 valid IDs.")

    xi_set = {p for p in xi_set if p in allowed_players}
    if len(xi_set) < 2:
        raise ValueError("After intersecting with team players, player_ids has < 2 valid IDs for this match/team.")

    return xi_set


def safe_norm(vals: np.ndarray) -> mpl.colors.Normalize:
    if vals.size == 0:
        return mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6
    return mpl.colors.Normalize(vmin=vmin, vmax=vmax)


def mix_with_bg(rgba, bg_rgba, mix: float) -> Tuple[float, float, float, float]:
    r, g, b, a = rgba
    br, bg, bb, ba = bg_rgba
    r2 = (1 - mix) * r + mix * br
    g2 = (1 - mix) * g + mix * bg
    b2 = (1 - mix) * b + mix * bb
    return (float(r2), float(g2), float(b2), float(a))


def power_scale(v: float, v_max: float, gamma: float) -> float:
    if v_max <= 0:
        return 0.0
    x = max(0.0, float(v) / float(v_max))
    return x ** float(gamma)


def aggregate_passes(
    rows: List[Tuple],
    allowed_players: Set[int],
    xi_set: Optional[Set[int]],
    cutoff_t: Optional[int],
) -> Tuple[
    List[Tuple[int, int, float, float, float, float, float]],
    Dict[int, str],
]:
    filt: List[Tuple[int, int, float, float, float, float, float]] = []
    name_map: Dict[int, str] = {}

    def _is_valid_pid(x) -> bool:
        if x is None:
            return False
        try:
            xi = int(x)
        except Exception:
            return False
        if xi == 0:
            return False
        if isinstance(x, float) and math.isnan(x):
            return False
        return True

    for (
        passer_id, receiver_id, receiver_name, passer_name, mp, m, s,
        sx, sy, ex, ey, acc, poss_id, xg_chain
    ) in rows:
        if not _is_valid_pid(passer_id) or not _is_valid_pid(receiver_id):
            continue

        passer_id = int(passer_id)
        receiver_id = int(receiver_id)

        if passer_id not in allowed_players or receiver_id not in allowed_players:
            continue

        if xi_set is not None:
            if passer_id not in xi_set or receiver_id not in xi_set:
                continue
        else:
            t = event_time_seconds(mp, m, s)
            if t is None:
                continue
            if cutoff_t is None or t > cutoff_t:
                continue

        if passer_name:
            name_map[passer_id] = passer_name
        if receiver_name:
            name_map[receiver_id] = receiver_name

        filt.append(
            (passer_id, receiver_id, float(sx), float(sy), float(ex), float(ey), float(xg_chain or 0.0))
        )

    return filt, name_map


def compute_network(filt):
    edge_counts: Dict[Tuple[int, int], int] = {}
    out_counts: Dict[int, int] = {}
    in_counts: Dict[int, int] = {}

    start_sum: Dict[int, Tuple[float, float, int]] = {}
    end_sum: Dict[int, Tuple[float, float, int]] = {}

    edge_xg_sum: Dict[Tuple[int, int], float] = {}
    edge_xg_n: Dict[Tuple[int, int], int] = {}
    node_xg_made: Dict[int, float] = {}
    node_xg_recv: Dict[int, float] = {}

    for passer, receiver, sx, sy, ex, ey, xg_chain in filt:
        k = (passer, receiver)
        edge_counts[k] = edge_counts.get(k, 0) + 1
        out_counts[passer] = out_counts.get(passer, 0) + 1
        in_counts[receiver] = in_counts.get(receiver, 0) + 1

        xgv = float(xg_chain or 0.0)
        edge_xg_sum[k] = edge_xg_sum.get(k, 0.0) + xgv
        edge_xg_n[k] = edge_xg_n.get(k, 0) + 1
        node_xg_made[passer] = node_xg_made.get(passer, 0.0) + xgv
        node_xg_recv[receiver] = node_xg_recv.get(receiver, 0.0) + xgv

        if passer not in start_sum:
            start_sum[passer] = (sx, sy, 1)
        else:
            a, b, n = start_sum[passer]
            start_sum[passer] = (a + sx, b + sy, n + 1)

        if receiver not in end_sum:
            end_sum[receiver] = (ex, ey, 1)
        else:
            a, b, n = end_sum[receiver]
            end_sum[receiver] = (a + ex, b + ey, n + 1)

    players = sorted(set(out_counts.keys()) | set(in_counts.keys()))

    node_pos: Dict[int, Tuple[float, float]] = {}
    for pid in players:
        parts: List[Tuple[float, float]] = []
        if pid in start_sum:
            sx, sy, n = start_sum[pid]
            parts.append((sx / n, sy / n))
        if pid in end_sum:
            ex, ey, n = end_sum[pid]
            parts.append((ex / n, ey / n))
        if not parts:
            continue
        node_pos[pid] = (
            float(np.mean([p[0] for p in parts])),
            float(np.mean([p[1] for p in parts])),
        )

    involvement = {pid: out_counts.get(pid, 0) + in_counts.get(pid, 0) for pid in players}

    return (
        players,
        node_pos,
        involvement,
        edge_counts,
        edge_xg_sum,
        edge_xg_n,
        node_xg_made,
        node_xg_recv,
    )


def node_edge_xg_values(
    players,
    edge_items,
    node_xg_made,
    node_xg_recv,
    edge_xg_sum,
    edge_xg_n,
    node_color_mode: str,
    edge_color_mode: str,
):
    node_color_mode = (node_color_mode or "made").lower()
    edge_color_mode = (edge_color_mode or "mean").lower()

    node_color_val: Dict[int, float] = {}
    for pid in players:
        v = float(node_xg_made.get(pid, 0.0))
        if node_color_mode in {"made_plus_received", "made+received", "both"}:
            v += float(node_xg_recv.get(pid, 0.0))
        node_color_val[pid] = v

    edge_color_val: Dict[Tuple[int, int], float] = {}
    for a, b, _ in edge_items:
        k = (a, b)
        s = float(edge_xg_sum.get(k, 0.0))
        if edge_color_mode == "sum":
            edge_color_val[k] = s
        else:
            n = int(edge_xg_n.get(k, 1)) or 1
            edge_color_val[k] = s / n

    return node_color_val, edge_color_val


def draw_donut_nodes(ax, xs, ys, sizes, ring_colors, pitch_face_rgba):
    ax.scatter(xs, ys, s=sizes, facecolors="none", edgecolors=ring_colors,
               linewidths=DEFAULT_DONUT_RING_LW, alpha=0.98, zorder=5)
    inner_sizes = [max(8.0, DEFAULT_DONUT_HOLE_RATIO * s) for s in sizes]
    ax.scatter(xs, ys, s=inner_sizes, c=[pitch_face_rgba] * len(xs),
               edgecolors="none", alpha=1.0, zorder=6)


def draw_labels(ax, pids_plot, xs, ys, ring_colors, name_map):
    for pid, x, y, col in zip(pids_plot, xs, ys, ring_colors):
        r, g, b, a = col
        txt_col = (r, g, b, float(DEFAULT_LABEL_OPACITY))

        bbox = None
        if DEFAULT_LABEL_BBOX:
            bbox = dict(
                boxstyle=f"round,pad={DEFAULT_LABEL_BBOX_PAD}",
                facecolor=(0, 0, 0, DEFAULT_LABEL_BBOX_ALPHA),
                edgecolor="none",
            )

        ax.text(
            x, y + DEFAULT_LABEL_Y_OFFSET,
            name_map.get(pid, str(pid)),
            ha="center", va="center",
            fontsize=DEFAULT_LABEL_FONTSIZE,
            color=txt_col,
            bbox=bbox,
            zorder=9,
            path_effects=[
            pe.Stroke(linewidth=2.6, foreground=(0,0,0,0.55)),
            pe.Normal()
        ],
        )


# -----------------------------
# DB helpers
# -----------------------------
def fetch_allowed_team_players(*, pool, match_id: int, team_id: int) -> Set[int]:
    q = """
    SELECT DISTINCT player_id
    FROM eventstream_events
    WHERE match_id = %s AND team_id = %s AND player_id IS NOT NULL
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(q, [match_id, team_id])
            return {int(r[0]) for r in cur.fetchall()}


def fetch_pass_rows(*, pool, match_id: int, team_id: int, only_accurate: bool, use_xgchain: bool) -> Tuple[List[Tuple], List[Any]]:
    where = [
        sql.SQL("e.match_id = %s"),
        sql.SQL("e.team_id = %s"),
        sql.SQL("e.type_primary = 'pass'"),
        sql.SQL("p.recipient_id IS NOT NULL"),
        sql.SQL("e.location_x IS NOT NULL"),
        sql.SQL("e.location_y IS NOT NULL"),
        sql.SQL("p.end_x IS NOT NULL"),
        sql.SQL("p.end_y IS NOT NULL"),
        sql.SQL("e.minute IS NOT NULL"),
    ]
    if only_accurate:
        where.append(sql.SQL("p.accurate IS TRUE"))

    if use_xgchain:
        q = sql.SQL("""
        WITH poss_xg AS (
          SELECT
            e.match_id,
            e.team_id,
            e.possession_id,
            COALESCE(SUM(s.xg), 0.0) AS possession_xg
          FROM eventstream_events e
          JOIN eventstream_shots s ON s.event_id = e.id
          WHERE e.match_id = %s
            AND e.team_id = %s
            AND e.possession_id IS NOT NULL
          GROUP BY e.match_id, e.team_id, e.possession_id
        )
        SELECT
          e.player_id AS passer_id,
          p.recipient_id AS receiver_id,
          p.recipient_name,
          e.player_name,
          e.match_period,
          e.minute,
          e.second,
          e.location_x, e.location_y,
          p.end_x, p.end_y,
          p.accurate,
          e.possession_id,
          COALESCE(px.possession_xg, 0.0) AS xg_chain
        FROM eventstream_passes p
        JOIN eventstream_events e ON e.id = p.event_id
        LEFT JOIN poss_xg px
          ON px.match_id = e.match_id
         AND px.team_id = e.team_id
         AND px.possession_id = e.possession_id
        WHERE {where}
        ORDER BY e.id
        """).format(where=sql.SQL(" AND ").join(where))
        params: List[Any] = [match_id, team_id, match_id, team_id]
    else:
        q = sql.SQL("""
        SELECT
          e.player_id AS passer_id,
          p.recipient_id AS receiver_id,
          p.recipient_name,
          e.player_name,
          e.match_period,
          e.minute,
          e.second,
          e.location_x, e.location_y,
          p.end_x, p.end_y,
          p.accurate,
          e.possession_id,
          0.0::double precision AS xg_chain
        FROM eventstream_passes p
        JOIN eventstream_events e ON e.id = p.event_id
        WHERE {where}
        ORDER BY e.id
        """).format(where=sql.SQL(" AND ").join(where))
        params = [match_id, team_id]

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(q, params)
            rows = cur.fetchall()

    return rows, params


# -----------------------------
# Render main
# -----------------------------
def render_pass_network_png(
    *,
    pool,
    match_id: int,
    team_id: int,
    player_ids: Optional[List[int]],
    cutoff_period: str,
    cutoff_minute: Optional[int],
    cutoff_second: int,
    min_edge_count: int,
    only_accurate: bool,
    dpi: int,
    show_labels: bool,
    use_xgchain: bool,
    node_color_mode: str,
    edge_color_mode: str,
    show_colorbar: bool,
    pitch_type: str = DEFAULT_PITCH_TYPE,
) -> bytes:
    xi_required = player_ids is not None and len(player_ids) > 0
    cutoff_t: Optional[int] = None
    if not xi_required:
        if cutoff_minute is None:
            raise ValueError("Provide player_ids OR cutoff_minute/cutoff_period.")
        cutoff_t = event_time_seconds(cutoff_period, cutoff_minute, cutoff_second)
        if cutoff_t is None:
            raise ValueError("Invalid cutoff time.")

    allowed_players = fetch_allowed_team_players(pool=pool, match_id=match_id, team_id=team_id)
    xi_set = normalize_xi(player_ids, allowed_players)

    rows, _params = fetch_pass_rows(
        pool=pool,
        match_id=match_id,
        team_id=team_id,
        only_accurate=only_accurate,
        use_xgchain=use_xgchain,
    )

    filt, name_map = aggregate_passes(rows, allowed_players, xi_set, cutoff_t)
    if not filt:
        raise LookupError("No passes found under the current filters.")

    (
        players,
        node_pos,
        involvement,
        edge_counts,
        edge_xg_sum,
        edge_xg_n,
        node_xg_made,
        node_xg_recv,
    ) = compute_network(filt)

    if xi_set is not None:
        players = [p for p in sorted(xi_set) if p in node_pos]

    edge_items = [(a, b, c) for (a, b), c in edge_counts.items()
                  if c >= min_edge_count and a in node_pos and b in node_pos]
    if not edge_items:
        raise LookupError(f"No edges with count >= {min_edge_count} under current filters.")
    edge_items.sort(key=lambda t: t[2])

    node_color_val, edge_color_val = node_edge_xg_values(
        players, edge_items,
        node_xg_made, node_xg_recv,
        edge_xg_sum, edge_xg_n,
        node_color_mode, edge_color_mode,
    )

    node_vals = np.array([node_color_val.get(pid, 0.0) for pid in players if pid in node_pos], dtype=float)
    edge_vals = np.array([edge_color_val.get((a, b), 0.0) for a, b, _ in edge_items], dtype=float)
    node_norm = safe_norm(node_vals) if use_xgchain else mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    edge_norm = safe_norm(edge_vals) if use_xgchain else mpl.colors.Normalize(vmin=0.0, vmax=1.0)

    cmap = mpl.cm.get_cmap(DEFAULT_CMAP_NAME)

    pitch = Pitch(pitch_type=pitch_type, pitch_color=DEFAULT_PITCH_BG, line_color=DEFAULT_PITCH_LINE_COLOR)
    fig, ax = pitch.draw(figsize=(10, 7))
    pitch_face = ax.get_facecolor()

    inv_vals = np.array([involvement.get(pid, 0) for pid in players if pid in node_pos], dtype=float)
    inv_max = float(inv_vals.max()) if inv_vals.size else 1.0
    edge_max = max(c for _, _, c in edge_items) if edge_items else 1

    def node_size(v: float) -> float:
        x = power_scale(v, inv_max, DEFAULT_NODE_SIZE_GAMMA)
        return float(DEFAULT_NODE_SIZE_MIN + (DEFAULT_NODE_SIZE_MAX - DEFAULT_NODE_SIZE_MIN) * x)

    def edge_width(c: int) -> float:
        if edge_max <= 0:
            return float(DEFAULT_EDGE_LW_MAX)
        x = float(c) / float(edge_max)
        return float(DEFAULT_EDGE_LW_MIN + (DEFAULT_EDGE_LW_MAX - DEFAULT_EDGE_LW_MIN) * x)

    def edge_alpha(c: int) -> float:
        if edge_max <= 0:
            return float(DEFAULT_EDGE_ALPHA_MAX)
        x = float(c) / float(edge_max)
        return float(np.clip(DEFAULT_EDGE_ALPHA_MIN + (DEFAULT_EDGE_ALPHA_MAX - DEFAULT_EDGE_ALPHA_MIN) * x, 0.0, 1.0))

    def map_color(v: float, norm: mpl.colors.Normalize):
        rgba = cmap(norm(v))
        rgba = mix_with_bg(rgba, pitch_face, DEFAULT_COLOR_SOFTEN)
        r, g, b, a = rgba
        return (float(r), float(g), float(b), float(DEFAULT_COLOR_ALPHA))

    # edges
    for a, b, c in edge_items:
        ax1, ay1 = node_pos[a]
        bx1, by1 = node_pos[b]
        lw = edge_width(c)
        al = edge_alpha(c)
        v = edge_color_val.get((a, b), 0.0)
        col = map_color(v, edge_norm) if use_xgchain else (0.85, 0.85, 0.85, 0.6)

        pitch.lines(ax1, ay1, bx1, by1, lw=lw, ax=ax, alpha=al, color=col, zorder=2)

    # nodes
    xs, ys, sizes, ring_colors, pids_plot = [], [], [], [], []
    for pid in players:
        if pid not in node_pos:
            continue
        x, y = node_pos[pid]
        xs.append(x)
        ys.append(y)
        sizes.append(node_size(involvement.get(pid, 0)))
        pids_plot.append(pid)
        v = node_color_val.get(pid, 0.0)
        ring_colors.append(map_color(v, node_norm) if use_xgchain else (0.9, 0.9, 0.9, 0.8))

    draw_donut_nodes(ax=ax, xs=xs, ys=ys, sizes=sizes, ring_colors=ring_colors, pitch_face_rgba=pitch_face)

    if show_labels:
        draw_labels(ax=ax, pids_plot=pids_plot, xs=xs, ys=ys, ring_colors=ring_colors, name_map=name_map)

    title_bits = [f"Pass Network | match={match_id} team={team_id}"]
    if xi_set is not None:
        title_bits.append(f"XI n={len(xi_set)}")
    else:
        title_bits.append(f"<= {cutoff_period} {cutoff_minute}:{cutoff_second:02d}")
    if use_xgchain:
        title_bits.append(f"xGChain node={node_color_mode} edge={edge_color_mode}")
    ax.set_title(" | ".join(title_bits))

    if show_colorbar and use_xgchain:
        sm = mpl.cm.ScalarMappable(norm=node_norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("xGChain (possession xG proxy)", rotation=90)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
