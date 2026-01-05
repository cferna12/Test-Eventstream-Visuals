# app/routers/passnetwork_image.py
from __future__ import annotations

from typing import Optional, List, Dict, Tuple, Any
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
    """Convert (period, minute, second) into a comparable second counter.

    This is intentionally defensive because feeds vary:
    - Some give minute=0..45 each half
    - Some give minute=0..90 cumulative
    """
    if minute is None:
        return None
    s = int(second or 0)
    m = int(minute)

    p = (match_period or "").lower()

    # If period suggests 2nd half, add 45 minutes *if minutes look half-relative*.
    if "2" in p or "second" in p:
        # If minute looks like 0..45 (half-relative), offset by 45
        if m <= 45:
            return (45 * 60) + (m * 60) + s
        # else it's probably already cumulative
        return (m * 60) + s

    # First half or unknown: assume minute is already comparable
    return (m * 60) + s


@router.get("/image", response_class=Response)
def pass_network_image(
    match_id: int,
    team_id: int,

    # cutoff time: “pre-sub” window end
    cutoff_minute: int = Query(..., ge=0, le=200),
    cutoff_second: int = Query(0, ge=0, le=59),
    cutoff_period: str = Query("1H", description="Typically '1H' or '2H'"),

    # viz controls
    min_edge_count: int = Query(2, ge=1, le=50),
    only_accurate: bool = Query(False),
    dpi: int = Query(150, ge=72, le=300),
    pitch_type: str = Query("wyscout"),
    show_labels: bool = Query(False),
):
    """
    Returns a standard pass network image for (match_id, team_id) within a time window.
    Node positions = average of (pass start positions + reception end positions).
    Node size      = proportional to passes made + receptions.
    Edge width     = proportional to number of passes from A->B.
    """
    cutoff_t = _event_time_seconds(cutoff_period, cutoff_minute, cutoff_second)
    if cutoff_t is None:
        raise HTTPException(status_code=400, detail="Invalid cutoff time.")

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
    params: List[Any] = [match_id, team_id]

    if only_accurate:
        where.append(sql.SQL("p.accurate IS TRUE"))

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
          p.accurate
        FROM eventstream_passes p
        JOIN eventstream_events e ON e.id = p.event_id
        WHERE {where}
        ORDER BY e.id
    """).format(where=sql.SQL(" AND ").join(where))

    q_team_players = """
    SELECT DISTINCT player_id
    FROM eventstream_events
    WHERE match_id = %s AND team_id = %s AND player_id IS NOT NULL
    """
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(q_team_players, [match_id, team_id])
            allowed_players = {int(r[0]) for r in cur.fetchall()}

    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(q, params)
            rows = cur.fetchall()

    # --- filter by time window in Python (simple + robust for now)
    filt: List[Tuple] = []
    name_map: dict[int, str] = {}

    def _is_valid_pid(x) -> bool:
        if x is None:
            return False
        # psycopg sometimes returns Decimal/str; normalize
        try:
            xi = int(x)
        except Exception:
            return False
        # guard against pandas/numpy nan if it ever appears
        if isinstance(x, float) and math.isnan(x):
            return False
        
        if x == 0:
            return False
        return True

    for passer_id, receiver_id, receiver_name, passer_name, mp, m, s, sx, sy, ex, ey, acc in rows:
        if not _is_valid_pid(passer_id) or not _is_valid_pid(receiver_id):
            continue

        passer_id = int(passer_id)
        receiver_id = int(receiver_id)
        
        if passer_id is not None and passer_name:
            name_map[passer_id] = passer_name

        # Receiver
        if receiver_id is not None and receiver_name:
            name_map[receiver_id] = receiver_name
        t = _event_time_seconds(mp, m, s)
        if t is None:
            continue
        if t <= cutoff_t:
            if passer_id in allowed_players and receiver_id in allowed_players:
                filt.append((int(passer_id), int(receiver_id), float(sx), float(sy), float(ex), float(ey)))

    if not filt:
        raise HTTPException(status_code=404, detail="No passes found in the requested time window.")

    # --- aggregate edges + node involvement + node positions
    edge_counts: Dict[Tuple[int, int], int] = {}
    out_counts: Dict[int, int] = {}
    in_counts: Dict[int, int] = {}

    # accumulate positions
    start_sum: Dict[int, Tuple[float, float, int]] = {}   # passer: sumx,sumy,n
    end_sum: Dict[int, Tuple[float, float, int]] = {}     # receiver: sumx,sumy,n

    
    for passer, receiver, sx, sy, ex, ey in filt:
        edge_counts[(passer, receiver)] = edge_counts.get((passer, receiver), 0) + 1
        out_counts[passer] = out_counts.get(passer, 0) + 1
        in_counts[receiver] = in_counts.get(receiver, 0) + 1

        # passer avg start
        if passer not in start_sum:
            start_sum[passer] = (sx, sy, 1)
        else:
            a, b, n = start_sum[passer]
            start_sum[passer] = (a + sx, b + sy, n + 1)

        # receiver avg end (reception location)
        if receiver not in end_sum:
            end_sum[receiver] = (ex, ey, 1)
        else:
            a, b, n = end_sum[receiver]
            end_sum[receiver] = (a + ex, b + ey, n + 1)

    players = sorted(set(list(out_counts.keys()) + list(in_counts.keys())))

    # blended node positions: average of start and end means (if both exist)
    node_pos: Dict[int, Tuple[float, float]] = {}
    for pid in players:
        parts = []
        if pid in start_sum:
            sx, sy, n = start_sum[pid]
            parts.append((sx / n, sy / n))
        if pid in end_sum:
            ex, ey, n = end_sum[pid]
            parts.append((ex / n, ey / n))

        if not parts:
            continue
        x = float(np.mean([p[0] for p in parts]))
        y = float(np.mean([p[1] for p in parts]))
        node_pos[pid] = (x, y)

    # --- build viz scaling
    # node size based on involvement (passes made + receptions)
    involvement = {pid: out_counts.get(pid, 0) + in_counts.get(pid, 0) for pid in players}
    inv_vals = np.array([involvement.get(pid, 0) for pid in players], dtype=float)
    inv_max = float(inv_vals.max()) if len(inv_vals) else 1.0

    # edge widths based on counts
    edge_items = [(a, b, c) for (a, b), c in edge_counts.items() if c >= min_edge_count]
    if not edge_items:
        raise HTTPException(
            status_code=404,
            detail=f"No edges with count >= {min_edge_count} in the requested time window."
        )
    edge_max = max(c for _, _, c in edge_items)

    def scale_node_size(v: float) -> float:
        # area in points^2 for matplotlib scatter
        # gentle scaling: 200..1400
        if inv_max <= 0:
            return 200.0
        return 200.0 + (1200.0 * (v / inv_max))

    def scale_edge_width(c: int) -> float:
        # line width: 0.5..6
        if edge_max <= 0:
            return 1.0
        return 0.5 + (5.5 * (c / edge_max))

    # --- draw
    pitch = Pitch(pitch_type=pitch_type)
    fig, ax = pitch.draw(figsize=(10, 7))

    # edges first (so nodes are on top)
    for a, b, c in edge_items:
        if a not in node_pos or b not in node_pos:
            continue
        ax1, ay1 = node_pos[a]
        bx1, by1 = node_pos[b]
        lw = scale_edge_width(c)
        # Use pitch.lines for correct pitch coordinate mapping
        pitch.lines(ax1, ay1, bx1, by1, lw=lw, ax=ax, alpha=0.8)

    # nodes
    xs = []
    ys = []
    ss = []
    labels = []
    for pid in players:
        if pid not in node_pos:
            continue
        x, y = node_pos[pid]
        xs.append(x)
        ys.append(y)
        ss.append(scale_node_size(involvement.get(pid, 0)))
        labels.append(str(pid))

    ax.scatter(xs, ys, s=ss, alpha=0.9)

    if show_labels:
        for pid in players:
            if pid not in node_pos:
                continue
            x, y = node_pos[pid]
            ax.text(
                x, y,
                name_map.get(pid, str(pid)),
                ha="center", va="center",
                fontsize=8, 
                bbox=dict(boxstyle="round,pad=0.2", alpha=0.6)
            )

    # title (kept simple)
    ax.set_title(f"Pass Network | match={match_id} team={team_id} | <= {cutoff_period} {cutoff_minute}:{cutoff_second:02d}")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")


'''TODO improve visualization'''
@router.get("/image_v2", response_class=Response)
def pass_network_image_v2(
    match_id: int,
    team_id: int,

    # NEW PRIMARY FILTER: explicit XI (11 player_ids)
    # FastAPI accepts: ?player_ids=1&player_ids=2&... or comma-separated depending on client.
    player_ids: Optional[List[int]] = Query(
        default=None,
        description="Preferred: list of 11 player IDs to include in the network (filters nodes + edges).",
    ),

    # optional fallback cutoff window (kept for backward compatibility)
    cutoff_minute: Optional[int] = Query(default=None, ge=0, le=200),
    cutoff_second: int = Query(0, ge=0, le=59),
    cutoff_period: str = Query("1H", description="Typically '1H' or '2H'"),

    # viz controls
    min_edge_count: int = Query(2, ge=1, le=50),
    only_accurate: bool = Query(False),

    dpi: int = Query(160, ge=72, le=300),
    pitch_type: str = Query("wyscout"),
    show_labels: bool = Query(True, description="Overlay player names on donut nodes."),

    # xgChain encodings
    use_xgchain: bool = Query(True, description="If true, colors nodes/edges by possession xG (xGChain proxy)."),
    node_color_mode: str = Query("made", description="xgChain credit for nodes: 'made' or 'made_plus_received'"),
    edge_color_mode: str = Query("mean", description="xgChain on edges: 'mean' or 'sum'"),
    cmap_name: str = Query("viridis", description="Matplotlib colormap for xgChain encoding."),
    show_colorbar: bool = Query(False, description="If true, adds a colorbar legend for xgChain."),

    # styling tweaks requested
    edge_alpha_min: float = Query(0.35, ge=0.0, le=1.0),
    edge_alpha_max: float = Query(0.90, ge=0.0, le=1.0),

    # donut node styling (StatsBomb-ish)
    donut_ring_lw: float = Query(2.6, ge=0.5, le=8.0),
    donut_hole_ratio: float = Query(0.55, ge=0.1, le=0.9),  # inner hole size relative to outer
    label_fontsize: int = Query(9, ge=6, le=14),
):
    """
    Pass network image for (match_id, team_id).

    PRIMARY MODE (recommended):
      - Provide player_ids (ideally 11 IDs).
      - Network is filtered to passes where BOTH passer and receiver are in player_ids.

    FALLBACK MODE:
      - If player_ids is not provided, you can use a cutoff time window (<= cutoff_period/minute/second).

    Node positions:
      average of (pass start positions + reception end positions) per player.

    Node size:
      proportional to passes made + receptions.

    Edge width:
      proportional to number of passes from A->B (dialed back a bit vs previous).

    Enhancements:
      - Edge alpha scaled by edge count (increased overall opacity)
      - Edge/node coloring by xGChain proxy (possession xG)
      - Donut nodes with overlaid player names (StatsBomb-like cleanliness)
    """

    # normalize / validate XI filter
    xi_set: Optional[set[int]] = None
    if player_ids:
        # de-dupe + remove zeros/Nones defensively
        cleaned = []
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
        # we won't hard fail if not exactly 11, but this hints at intended usage
        # (keeps it flexible for red cards / late subs / partial networks)
        # if len(xi_set) != 11: ... (optional strictness)

    # cutoff fallback (only required if no player list)
    cutoff_t: Optional[int] = None
    if xi_set is None:
        if cutoff_minute is None:
            raise HTTPException(
                status_code=400,
                detail="Provide player_ids (preferred) OR provide cutoff_minute/cutoff_period to use time-window fallback.",
            )
        cutoff_t = _event_time_seconds(cutoff_period, cutoff_minute, cutoff_second)
        if cutoff_t is None:
            raise HTTPException(status_code=400, detail="Invalid cutoff time.")

    # Basic filters for query
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

    # Allowed players for this team (also helps avoid weird nulls)
    q_team_players = """
    SELECT DISTINCT player_id
    FROM eventstream_events
    WHERE match_id = %s AND team_id = %s AND player_id IS NOT NULL
    """
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(q_team_players, [match_id, team_id])
            allowed_players = {int(r[0]) for r in cur.fetchall()}

    # If XI provided, intersect with team players (prevents cross-team IDs)
    if xi_set is not None:
        xi_set = set(p for p in xi_set if p in allowed_players)
        if len(xi_set) < 2:
            raise HTTPException(
                status_code=400,
                detail="After intersecting with team players, player_ids has < 2 valid IDs for this match/team.",
            )

    # Main pass query, optionally with xgChain proxy
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

        # params: first pair for poss_xg CTE, second pair for WHERE filters
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

    # --- filter rows in Python (time + XI)
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

    for passer_id, receiver_id, receiver_name, passer_name, mp, m, s, sx, sy, ex, ey, acc, poss_id, xg_chain in rows:
        if not _is_valid_pid(passer_id) or not _is_valid_pid(receiver_id):
            continue

        passer_id = int(passer_id)
        receiver_id = int(receiver_id)

        # team guard
        if passer_id not in allowed_players or receiver_id not in allowed_players:
            continue

        # name map
        if passer_name:
            name_map[passer_id] = passer_name
        if receiver_name:
            name_map[receiver_id] = receiver_name

        # PRIMARY: XI filter
        if xi_set is not None:
            if passer_id not in xi_set or receiver_id not in xi_set:
                continue
            # no time cutoff required here; you can add one later if you want
            filt.append((passer_id, receiver_id, float(sx), float(sy), float(ex), float(ey), float(xg_chain or 0.0)))
            continue

        # FALLBACK: cutoff filter
        t = _event_time_seconds(mp, m, s)
        if t is None:
            continue
        if cutoff_t is not None and t <= cutoff_t:
            filt.append((passer_id, receiver_id, float(sx), float(sy), float(ex), float(ey), float(xg_chain or 0.0)))

    if not filt:
        raise HTTPException(status_code=404, detail="No passes found under the current filters.")

    # --- aggregate edges + node involvement + node positions
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

        # passer avg start
        if passer not in start_sum:
            start_sum[passer] = (sx, sy, 1)
        else:
            a, b, n = start_sum[passer]
            start_sum[passer] = (a + sx, b + sy, n + 1)

        # receiver avg end
        if receiver not in end_sum:
            end_sum[receiver] = (ex, ey, 1)
        else:
            a, b, n = end_sum[receiver]
            end_sum[receiver] = (a + ex, b + ey, n + 1)

    # In XI mode, keep players in XI order as much as possible (stable visual set)
    if xi_set is not None:
        players = sorted(xi_set)
    else:
        players = sorted(set(out_counts.keys()) | set(in_counts.keys()))

    # blended node positions: average of start and end means (if both exist)
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
        node_pos[pid] = (float(np.mean([p[0] for p in parts])), float(np.mean([p[1] for p in parts])))

    # --- scaling
    involvement = {pid: out_counts.get(pid, 0) + in_counts.get(pid, 0) for pid in players}
    inv_vals = np.array([involvement.get(pid, 0) for pid in players if pid in node_pos], dtype=float)
    inv_max = float(inv_vals.max()) if inv_vals.size else 1.0

    # filter edges by min count (and keep only edges whose nodes exist)
    edge_items = [(a, b, c) for (a, b), c in edge_counts.items() if c >= min_edge_count and a in node_pos and b in node_pos]
    if not edge_items:
        raise HTTPException(status_code=404, detail=f"No edges with count >= {min_edge_count} under current filters.")

    # draw thin->thick so thick edges sit on top, but we're dialing thickness back
    edge_items.sort(key=lambda t: t[2])
    edge_max = max(c for _, _, c in edge_items)

    def scale_node_size(v: float) -> float:
        # keep nodes a bit cleaner for donut style
        # area in points^2
        if inv_max <= 0:
            return 480.0
        return 480.0 + (1150.0 * (v / inv_max))  # ~480..1630

    def scale_edge_width(c: int) -> float:
        # dialed back vs previous (roughly 0.35..3.2)
        if edge_max <= 0:
            return 1.2
        return 0.35 + (2.85 * (c / edge_max))

    def scale_edge_alpha(c: int) -> float:
        if edge_max <= 0:
            return float(np.clip(edge_alpha_max, 0.0, 1.0))
        a = edge_alpha_min + (edge_alpha_max - edge_alpha_min) * (c / edge_max)
        return float(np.clip(a, 0.0, 1.0))

    # --- xgChain values for coloring
    if use_xgchain:
        mode = (node_color_mode or "made").lower()
        node_color_val: Dict[int, float] = {}
        for pid in players:
            v = float(node_xg_made.get(pid, 0.0))
            if mode in {"made_plus_received", "made+received", "both"}:
                v += float(node_xg_recv.get(pid, 0.0))
            node_color_val[pid] = v

        edge_color_val: Dict[Tuple[int, int], float] = {}
        emode = (edge_color_mode or "mean").lower()
        for a, b, _ in edge_items:
            k = (a, b)
            s = float(edge_xg_sum.get(k, 0.0))
            if emode == "sum":
                edge_color_val[k] = s
            else:
                n = int(edge_xg_n.get(k, 1)) or 1
                edge_color_val[k] = s / n
    else:
        node_color_val = {pid: 0.0 for pid in players}
        edge_color_val = {(a, b): 0.0 for a, b, _ in edge_items}

    cmap = mpl.cm.get_cmap(cmap_name) if cmap_name else mpl.cm.viridis

    def _safe_norm(vals: np.ndarray) -> mpl.colors.Normalize:
        if vals.size == 0:
            return mpl.colors.Normalize(vmin=0.0, vmax=1.0)
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-6
        return mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    node_vals = np.array([node_color_val.get(pid, 0.0) for pid in players if pid in node_pos], dtype=float)
    edge_vals = np.array([edge_color_val.get((a, b), 0.0) for a, b, _ in edge_items], dtype=float)
    node_norm = _safe_norm(node_vals) if use_xgchain else mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    edge_norm = _safe_norm(edge_vals) if use_xgchain else mpl.colors.Normalize(vmin=0.0, vmax=1.0)

    # --- draw
    pitch_bg = "#1f2326"   # charcoal green-gray
    line_color = "#4a4f52"
    pitch = Pitch(pitch_type=pitch_type, pitch_color=pitch_bg, line_color=line_color)
    fig, ax = pitch.draw(figsize=(10, 7))

    # edges first
    for a, b, c in edge_items:
        ax1, ay1 = node_pos[a]
        bx1, by1 = node_pos[b]
        lw = scale_edge_width(c)
        alpha = scale_edge_alpha(c)

        v = edge_color_val.get((a, b), 0.0)
        color = cmap(edge_norm(v)) if use_xgchain else (0.12, 0.12, 0.12, 1.0)

        pitch.lines(ax1, ay1, bx1, by1, lw=lw, ax=ax, alpha=alpha, color=color, zorder=2)

    # donut nodes: outer ring + inner "hole" + text overlay
    # We don't rely on pitch background color (varies), but a transparent fill still reads as a donut ring.
    # The "hole" is a second scatter matching pitch facecolor to create a clean donut.
    pitch_face = ax.get_facecolor()  # RGBA
    xs, ys, sizes, ring_colors, pids_plot = [], [], [], [], []
    for pid in players:
        if pid not in node_pos:
            continue
        x, y = node_pos[pid]
        xs.append(x)
        ys.append(y)
        s = scale_node_size(involvement.get(pid, 0))
        sizes.append(s)
        pids_plot.append(pid)
        v = node_color_val.get(pid, 0.0)
        ring_colors.append(cmap(node_norm(v)) if use_xgchain else (0.2, 0.2, 0.2, 1.0))

    # Outer ring
    ax.scatter(
        xs, ys,
        s=sizes,
        facecolors="none",
        edgecolors=ring_colors,
        linewidths=donut_ring_lw,
        alpha=0.98,
        zorder=5
    )

    # Inner hole (smaller filled circle matching pitch background)
    # This is what creates the StatsBomb-like donut.
    inner_sizes = [max(10.0, donut_hole_ratio * s) for s in sizes]
    ax.scatter(
        xs, ys,
        s=inner_sizes,
        c=[pitch_face] * len(xs),
        edgecolors="none",
        alpha=1.0,
        zorder=6
    )

    # labels on top (player names by default, fallback to pid)
    if show_labels:
        for pid, x, y, col in zip(pids_plot, xs, ys, ring_colors):
            ax.text(
                x, y,
                name_map.get(pid, str(pid)),
                ha="center", va="center",
                fontsize=label_fontsize,
                color=col,  # matches ring color (SB-like)
                zorder=7
            )

    # title
    title_bits = [f"Pass Network | match={match_id} team={team_id}"]
    if xi_set is not None:
        title_bits.append(f"XI-filter n={len(xi_set)}")
    else:
        title_bits.append(f"<= {cutoff_period} {cutoff_minute}:{cutoff_second:02d}")
    if use_xgchain:
        title_bits.append("xGChain colors")
    ax.set_title(" | ".join(title_bits))

    # optional colorbar
    if show_colorbar and use_xgchain:
        sm = mpl.cm.ScalarMappable(norm=node_norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("xGChain (possession xG)", rotation=90)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")