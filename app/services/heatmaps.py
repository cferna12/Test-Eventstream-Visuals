from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Literal

import io
import numpy as np
from psycopg import sql

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from mplsoccer import Pitch
from scipy.ndimage import gaussian_filter

from app.utils.heatmaps import (
    BINS_X, BINS_Y,
    ON_BALL_PRIMARY_DEFAULT,
    DEFAULT_PITCH_TYPE,
    DEFAULT_BLUR, DEFAULT_SIGMA, DEFAULT_TRUNCATE, DEFAULT_BLUR_MODE,
    DEFAULT_CLIP_HI, DEFAULT_SCALE, DEFAULT_GAMMA,
    DEFAULT_CMAP, DEFAULT_ALPHA,
)


def build_where_and_params(
    *,
    player_id: Optional[int],
    competition_id: Optional[int],
    season_id: Optional[int],
    match_id: Optional[int],
    primary: Optional[str],
    primary_in: Optional[List[str]],
    secondary_any: Optional[List[str]],
    x_min: Optional[float],
    x_max: Optional[float],
    y_min: Optional[float],
    y_max: Optional[float],
) -> Tuple[str, List[Any]]:
    if player_id is None:
        # You said you only support player right now (team commented out).
        # We keep this strict to avoid accidental full-table scans.
        raise ValueError("player_id is required (team_id mode not implemented).")

    where = ["player_id = %s"]
    params: List[Any] = [player_id]

    if competition_id is not None:
        where.append("competition_id = %s")
        params.append(competition_id)

    if season_id is not None:
        where.append("season_id = %s")
        params.append(season_id)

    if match_id is not None:
        where.append("match_id = %s")
        params.append(match_id)

    if primary is not None:
        where.append("type_primary = %s")
        params.append(primary)
    elif primary_in:
        where.append("type_primary = ANY(%s)")
        params.append(primary_in)

    if secondary_any:
        where.append("type_secondary && %s")
        params.append(secondary_any)

    if x_min is not None:
        where.append("location_x >= %s")
        params.append(x_min)
    if x_max is not None:
        where.append("location_x <= %s")
        params.append(x_max)
    if y_min is not None:
        where.append("location_y >= %s")
        params.append(y_min)
    if y_max is not None:
        where.append("location_y <= %s")
        params.append(y_max)

    where.append("location_x IS NOT NULL")
    where.append("location_y IS NOT NULL")
    where.append("bin_x IS NOT NULL")
    where.append("bin_y IS NOT NULL")

    return " AND ".join(where), params


def _rows_to_grid(rows: list[tuple]) -> np.ndarray:
    grid = np.zeros((BINS_Y, BINS_X), dtype=np.float32)
    for bx, by, n in rows:
        if bx is None or by is None:
            continue
        bx_i = int(bx)
        by_i = int(by)
        if 0 <= bx_i < BINS_X and 0 <= by_i < BINS_Y:
            grid[by_i, bx_i] = float(n)
    return grid


def _apply_smoothing_and_scaling(
    grid: np.ndarray,
    *,
    blur: bool,
    sigma: float,
    truncate: float,
    mode: Literal["nearest", "constant", "reflect", "mirror", "wrap"],
    clip_hi: float,
    scale: Literal["linear", "log", "power"],
    gamma: float,
) -> Tuple[np.ndarray, Optional[colors.Normalize]]:
    if blur and sigma > 0:
        grid = gaussian_filter(grid, sigma=sigma, mode=mode, truncate=truncate)

    # percentile clip for contrast
    if clip_hi < 100.0:
        hi = float(np.percentile(grid, clip_hi))
        if hi > 0:
            grid = np.clip(grid, 0.0, hi)

    norm = None
    if scale == "log":
        grid = np.log1p(grid)
    elif scale == "power":
        vmax = float(np.percentile(grid, clip_hi)) if clip_hi < 100.0 else float(grid.max())
        vmax = vmax if vmax > 0 else float(grid.max())
        norm = colors.PowerNorm(gamma=gamma, vmin=0.0, vmax=vmax if vmax > 0 else 1.0)
    else:
        # linear: keep grid as-is
        pass

    return grid, norm


def render_heatmap_image(
    *,
    pool,
    player_id: Optional[int],
    competition_id: Optional[int],
    season_id: Optional[int],
    match_id: Optional[int],
    primary: Optional[str],
    use_on_ball_default: bool,
    secondary_any: Optional[List[str]],
    x_min: Optional[float],
    x_max: Optional[float],
    y_min: Optional[float],
    y_max: Optional[float],
    # kept as params
    interpolation: Literal["bilinear", "nearest", "bicubic", "gaussian"],
    show_contours: bool,
    contour_levels: int,
    show_colorbar: bool,
    dpi: int,
) -> bytes:
    # default primary list
    primary_in: Optional[List[str]] = None
    if primary is None and use_on_ball_default:
        primary_in = ON_BALL_PRIMARY_DEFAULT

    where_sql, params = build_where_and_params(
        player_id=player_id,
        competition_id=competition_id,
        season_id=season_id,
        match_id=match_id,
        primary=primary,
        primary_in=primary_in,
        secondary_any=secondary_any,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )

    q = sql.SQL(f"""
        SELECT bin_x, bin_y, COUNT(*)::INT
        FROM eventstream_events
        WHERE {where_sql}
        GROUP BY bin_x, bin_y
    """)

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(q, params)
            rows = cur.fetchall()

    grid = _rows_to_grid(rows)

    if grid.max() <= 0:
        # caller will translate to HTTP 404
        raise LookupError("No events found for requested filters.")

    # Apply defaults from settings (not API params)
    grid, norm = _apply_smoothing_and_scaling(
        grid,
        blur=DEFAULT_BLUR,
        sigma=DEFAULT_SIGMA,
        truncate=DEFAULT_TRUNCATE,
        mode=DEFAULT_BLUR_MODE,
        clip_hi=DEFAULT_CLIP_HI,
        scale=DEFAULT_SCALE,
        gamma=DEFAULT_GAMMA,
    )

    pitch = Pitch(pitch_type=DEFAULT_PITCH_TYPE)
    fig, ax = pitch.draw(figsize=(10, 7))

    cmap_obj = plt.get_cmap(DEFAULT_CMAP).copy()

    im = ax.imshow(
        grid,
        extent=(0, 100, 0, 100),
        origin="lower",
        aspect="auto",
        interpolation=interpolation,
        alpha=DEFAULT_ALPHA,
        cmap=cmap_obj,
        norm=norm,
    )

    if show_contours:
        ax.contour(
            grid,
            levels=contour_levels,
            extent=(0, 100, 0, 100),
            origin="lower",
            alpha=0.35,
            linewidths=1.0,
        )

    if show_colorbar:
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    title_bits = []
    if player_id is not None:
        title_bits.append(f"player={player_id}")
    if match_id is not None:
        title_bits.append(f"match={match_id}")
    if season_id is not None:
        title_bits.append(f"season={season_id}")
    if competition_id is not None:
        title_bits.append(f"comp={competition_id}")
    if primary is not None:
        title_bits.append(f"primary={primary}")
    ax.set_title("Heatmap | " + " ".join(title_bits))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
