# app/routers/passnetwork_image.py

from __future__ import annotations

from typing import Optional, List, Dict, Tuple, Any, Set
from fastapi import APIRouter, Query, Response, HTTPException
from psycopg import sql
import io
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from mplsoccer import Pitch

from app.db import POOL

router = APIRouter(prefix="/v1/passnetwork", tags=["passnetwork"])


def _get_pool():
    return POOL


def _event_time_seconds(match_period: Optional[str], minute: Optional[int], second: Optional[int]) -> Optional[int]:
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


# -----------------------------
# Helpers (refactor)
# -----------------------------


cmap_name: str ="RdYlGn" #, description="Use RdYlGn for red-yellow-green (softened by bg-mix)."),
color_soften: float = 0 #Query(0.28, ge=0.0, le=0.8, description="Mix colormap colors toward pitch background."),
color_alpha: float =  1 #Query(0.78, ge=0.1, le=1.0, description="Overall alpha applied to colormap colors."),

# edges
edge_alpha_min: float = 0.45 # Query(0.45, ge=0.0, le=1.0),
edge_alpha_max: float = 0.7 #Query(0.92, ge=0.0, le=1.0),
edge_lw_min: float = 0.3 #Query(0.30, ge=0.1, le=4.0),
edge_lw_max: float = 4 #Query(2.60, ge=0.2, le=8.0),

# donut node styling
donut_ring_lw: float = 8#Query(2.6, ge=0.5, le=8.0),
donut_hole_ratio: float = 0.9 #Query(0.55, ge=0.1, le=0.9),

# labels
label_fontsize: int = 9 #Query(9, ge=6, le=14),
label_opacity: float = 1#Query(0.92, ge=0.1, le=1.0),
label_bbox: bool =  True # Query(True, description="If true, adds subtle backing so text never disappears."),

# node size scaling (power scale emphasis)
node_size_min: float = 420 #Query(420.0, ge=50, le=3000),
node_size_max: float = 1850 #Query(1850.0, ge=100, le=6000),
node_size_gamma: float = 2.4#Query(0.70, ge=0.20, le=2.50, description="Power scale (<1 emphasizes big nodes)."),

pitch_type: str = "wyscout" #Query("wyscout"),

def _normalize_xi(player_ids: Optional[List[int]], allowed_players: Set[int]) -> Optional[Set[int]]:
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
        raise HTTPException(status_code=400, detail="player_ids must contain at least 2 valid IDs.")
    xi_set = {p for p in xi_set if p in allowed_players}
    if len(xi_set) < 2:
        raise HTTPException(
            status_code=400,
            detail="After intersecting with team players, player_ids has < 2 valid IDs for this match/team.",
        )
    return xi_set


def _safe_norm(vals: np.ndarray) -> mpl.colors.Normalize:
    if vals.size == 0:
        return mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6
    return mpl.colors.Normalize(vmin=vmin, vmax=vmax)


def _mix_with_bg(rgba, bg_rgba, mix: float) -> Tuple[float, float, float, float]:
    """
    mix=0 -> original color, mix=1 -> background
    """
    r, g, b, a = rgba
    br, bg, bb, ba = bg_rgba
    r2 = (1 - mix) * r + mix * br
    g2 = (1 - mix) * g + mix * bg
    b2 = (1 - mix) * b + mix * bb
    return (float(r2), float(g2), float(b2), float(a))


def _power_scale(v: float, v_max: float, gamma: float) -> float:
    if v_max <= 0:
        return 0.0
    x = max(0.0, float(v) / float(v_max))
    return x ** float(gamma)


def _aggregate_passes(
    rows: List[Tuple],
    allowed_players: Set[int],
    xi_set: Optional[Set[int]],
    cutoff_t: Optional[int],
) -> Tuple[
    List[Tuple[int, int, float, float, float, float, float]],
    Dict[int, str],
]:
    """
    Return filt rows: (passer, receiver, sx, sy, ex, ey, xg_chain)
    """
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

        # team guard
        if passer_id not in allowed_players or receiver_id not in allowed_players:
            continue

        # XI filter (primary)
        if xi_set is not None:
            if passer_id not in xi_set or receiver_id not in xi_set:
                continue
        else:
            # cutoff fallback
            t = _event_time_seconds(mp, m, s)
            if t is None:
                continue
            if cutoff_t is None or t > cutoff_t:
                continue

        # names
        if passer_name:
            name_map[passer_id] = passer_name
        if receiver_name:
            name_map[receiver_id] = receiver_name

        filt.append(
            (passer_id, receiver_id, float(sx), float(sy), float(ex), float(ey), float(xg_chain or 0.0))
        )

    return filt, name_map


def _compute_network(filt):
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

    # blended node positions (avg of start + end means)
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
        out_counts,
        in_counts,
        edge_xg_sum,
        edge_xg_n,
        node_xg_made,
        node_xg_recv,
    )


def _node_edge_xg_values(
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


def _draw_donut_nodes(
    ax,
    xs,
    ys,
    sizes,
    ring_colors,
    pitch_face_rgba,
    donut_ring_lw: float,
    donut_hole_ratio: float,
    node_alpha: float,
):
    # Outer ring
    ax.scatter(
        xs, ys,
        s=sizes,
        facecolors="none",
        edgecolors=ring_colors,
        linewidths=donut_ring_lw,
        alpha=node_alpha,
        zorder=5
    )
    # Inner hole
    inner_sizes = [max(8.0, donut_hole_ratio * s) for s in sizes]
    ax.scatter(
        xs, ys,
        s=inner_sizes,
        c=[pitch_face_rgba] * len(xs),
        edgecolors="none",
        alpha=1.0,
        zorder=6
    )


def _draw_labels(
    ax,
    pids_plot,
    xs,
    ys,
    ring_colors,
    name_map,
    fontsize: int,
    opacity: float,
    with_bbox: bool = True,
    bbox_alpha: float = 0.35,
    bbox_pad: float = 0.18,
    y_offset: float = 0.0,
):
    """
    label color = slight offset of node ring color (and/or just less opaque).
    We also add a subtle bbox so text never disappears on donut.
    """
    for pid, x, y, col in zip(pids_plot, xs, ys, ring_colors):
        # Slightly dim text color (keep hue)
        r, g, b, a = col
        txt_col = (r, g, b, float(opacity))

        bbox = None
        if with_bbox:
            bbox = dict(
                boxstyle=f"round,pad={bbox_pad}",
                facecolor=(0, 0, 0, bbox_alpha),
                edgecolor="none",
            )

        ax.text(
            x, y + y_offset,
            name_map.get(pid, str(pid)),
            ha="center", va="center",
            fontsize=fontsize,
            color=txt_col,
            bbox=bbox,
            zorder=9,
        )


# -----------------------------
# Endpoint
# -----------------------------

@router.get("/image_v2", response_class=Response)
def pass_network_image_v2(
    match_id: int,
    team_id: int,

    player_ids: Optional[List[int]] = Query(
        default=None,
        description="Preferred: list of 11 player IDs to include in the network (filters nodes + edges).",
    ),

    cutoff_minute: Optional[int] = Query(default=None, ge=0, le=200),
    cutoff_second: int = Query(0, ge=0, le=59),
    cutoff_period: str = Query("1H"),

    min_edge_count: int = Query(2, ge=1, le=50),
    only_accurate: bool = Query(False),

    dpi: int = Query(160, ge=72, le=300),
    show_labels: bool = Query(True),

    use_xgchain: bool = Query(True),
    node_color_mode: str = Query("made"), # optional: made_plus_receieved
    edge_color_mode: str = Query("mean"), # optional: sum

    # --- Polished color ramp defaults (soft red->yellow->green)
    # cmap_name: str = Query("RdYlGn", description="Use RdYlGn for red-yellow-green (softened by bg-mix)."),
    # color_soften: float = Query(0.28, ge=0.0, le=0.8, description="Mix colormap colors toward pitch background."),
    # color_alpha: float = Query(0.78, ge=0.1, le=1.0, description="Overall alpha applied to colormap colors."),
    show_colorbar: bool = Query(True),

    # edges
    # edge_alpha_min: float = Query(0.45, ge=0.0, le=1.0),
    # edge_alpha_max: float = Query(0.92, ge=0.0, le=1.0),
    # edge_lw_min: float = Query(0.30, ge=0.1, le=4.0),
    # edge_lw_max: float = Query(2.60, ge=0.2, le=8.0),

    # # donut node styling
    # donut_ring_lw: float = Query(2.6, ge=0.5, le=8.0),
    # donut_hole_ratio: float = Query(0.55, ge=0.1, le=0.9),

    # # labels
    # label_fontsize: int = Query(9, ge=6, le=14),
    # label_opacity: float = Query(0.92, ge=0.1, le=1.0),
    # label_bbox: bool = Query(True, description="If true, adds subtle backing so text never disappears."),

    # # node size scaling (power scale emphasis)
    # node_size_min: float = Query(420.0, ge=50, le=3000),
    # node_size_max: float = Query(1850.0, ge=100, le=6000),
    # node_size_gamma: float = Query(0.70, ge=0.20, le=2.50, description="Power scale (<1 emphasizes big nodes)."),
):
    """
    Polished pass network (StatsBomb-ish donut nodes, soft R->Y->G xGChain ramp, readable labels).

    Primary filter: player_ids (XI).
    Fallback: cutoff time.
    """

    # --- cutoff fallback
    xi_required = player_ids is not None and len(player_ids) > 0
    cutoff_t: Optional[int] = None
    if not xi_required:
        if cutoff_minute is None:
            raise HTTPException(status_code=400, detail="Provide player_ids OR cutoff_minute/cutoff_period.")
        cutoff_t = _event_time_seconds(cutoff_period, cutoff_minute, cutoff_second)
        if cutoff_t is None:
            raise HTTPException(status_code=400, detail="Invalid cutoff time.")

    # --- allowed team players
    q_team_players = """
    SELECT DISTINCT player_id
    FROM eventstream_events
    WHERE match_id = %s AND team_id = %s AND player_id IS NOT NULL
    """
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(q_team_players, [match_id, team_id])
            allowed_players = {int(r[0]) for r in cur.fetchall()}

    # XI normalize
    xi_set = _normalize_xi(player_ids, allowed_players)

    # --- query
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

    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(q, params)
            rows = cur.fetchall()

    # --- filter + aggregate
    filt, name_map = _aggregate_passes(rows, allowed_players, xi_set, cutoff_t)
    if not filt:
        raise HTTPException(status_code=404, detail="No passes found under the current filters.")

    (
        players,
        node_pos,
        involvement,
        edge_counts,
        out_counts,
        in_counts,
        edge_xg_sum,
        edge_xg_n,
        node_xg_made,
        node_xg_recv,
    ) = _compute_network(filt)

    # If XI mode, lock nodes to XI set order (keeps network stable visually)
    if xi_set is not None:
        players = [p for p in sorted(xi_set) if p in node_pos]

    # edges filtered by min count + existing nodes
    edge_items = [(a, b, c) for (a, b), c in edge_counts.items() if c >= min_edge_count and a in node_pos and b in node_pos]
    if not edge_items:
        raise HTTPException(status_code=404, detail=f"No edges with count >= {min_edge_count} under current filters.")
    edge_items.sort(key=lambda t: t[2])  # thin -> thick

    # xg values for coloring
    node_color_val, edge_color_val = _node_edge_xg_values(
        players, edge_items,
        node_xg_made, node_xg_recv,
        edge_xg_sum, edge_xg_n,
        node_color_mode, edge_color_mode,
    )

    # norms
    node_vals = np.array([node_color_val.get(pid, 0.0) for pid in players if pid in node_pos], dtype=float)
    edge_vals = np.array([edge_color_val.get((a, b), 0.0) for a, b, _ in edge_items], dtype=float)
    node_norm = _safe_norm(node_vals) if use_xgchain else mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    edge_norm = _safe_norm(edge_vals) if use_xgchain else mpl.colors.Normalize(vmin=0.0, vmax=1.0)

    cmap = mpl.cm.get_cmap(cmap_name) if cmap_name else mpl.cm.get_cmap("RdYlGn")

    # --- pitch styling (dark recommended for RYG)
    pitch_bg = "#1f2326"
    line_col = "#4a4f52"
    pitch = Pitch(pitch_type=pitch_type, pitch_color=pitch_bg, line_color=line_col)
    fig, ax = pitch.draw(figsize=(10, 7))
    pitch_face = ax.get_facecolor()  # RGBA

    # scaling helpers
    inv_vals = np.array([involvement.get(pid, 0) for pid in players if pid in node_pos], dtype=float)
    inv_max = float(inv_vals.max()) if inv_vals.size else 1.0
    edge_max = max(c for _, _, c in edge_items) if edge_items else 1

    def node_size(v: float) -> float:
        # power scale: emphasize differences
        x = _power_scale(v, inv_max, node_size_gamma)
        return float(node_size_min + (node_size_max - node_size_min) * x)

    def edge_width(c: int) -> float:
        if edge_max <= 0:
            return float(edge_lw_max)
        x = float(c) / float(edge_max)
        return float(edge_lw_min + (edge_lw_max - edge_lw_min) * x)

    def edge_alpha(c: int) -> float:
        if edge_max <= 0:
            return float(edge_alpha_max)
        x = float(c) / float(edge_max)
        return float(np.clip(edge_alpha_min + (edge_alpha_max - edge_alpha_min) * x, 0.0, 1.0))

    def map_color(v: float, norm: mpl.colors.Normalize):
        rgba = cmap(norm(v))
        # soften by mixing toward pitch bg + apply alpha
        rgba = _mix_with_bg(rgba, pitch_face, color_soften)
        r, g, b, a = rgba
        return (float(r), float(g), float(b), float(color_alpha))

    # --- draw edges
    for a, b, c in edge_items:
        ax1, ay1 = node_pos[a]
        bx1, by1 = node_pos[b]
        lw = edge_width(c)
        al = edge_alpha(c)
        v = edge_color_val.get((a, b), 0.0)
        col = map_color(v, edge_norm) if use_xgchain else (0.85, 0.85, 0.85, 0.6)

        pitch.lines(
            ax1, ay1, bx1, by1,
            lw=lw,
            ax=ax,
            alpha=al,       # alpha scaled by count
            color=col,
            zorder=2
        )

    # --- draw donut nodes
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

    _draw_donut_nodes(
        ax=ax,
        xs=xs,
        ys=ys,
        sizes=sizes,
        ring_colors=ring_colors,
        pitch_face_rgba=pitch_face,
        donut_ring_lw=donut_ring_lw,
        donut_hole_ratio=donut_hole_ratio,
        node_alpha=0.98
    )

    # --- labels
    if show_labels:
        _draw_labels(
            ax=ax,
            pids_plot=pids_plot,
            xs=xs,
            ys=ys,
            ring_colors=ring_colors,
            name_map=name_map,
            fontsize=label_fontsize,
            opacity=label_opacity,
            with_bbox=bool(label_bbox),
            bbox_alpha=0.35,
            bbox_pad=0.16,
            y_offset=0.0,
        )

    # title
    title_bits = [f"Pass Network | match={match_id} team={team_id}"]
    if xi_set is not None:
        title_bits.append(f"XI n={len(xi_set)}")
    else:
        title_bits.append(f"<= {cutoff_period} {cutoff_minute}:{cutoff_second:02d}")
    if use_xgchain:
        title_bits.append(f"xGChain node={node_color_mode} edge={edge_color_mode}")
    ax.set_title(" | ".join(title_bits))

    # optional colorbar
    if show_colorbar and use_xgchain:
        sm = mpl.cm.ScalarMappable(norm=node_norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("xGChain (possession xG proxy)", rotation=90)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")
