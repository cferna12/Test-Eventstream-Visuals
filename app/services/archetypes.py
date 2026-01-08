from __future__ import annotations

from typing import Any, Dict, Optional

from psycopg.rows import dict_row


TABLE = "player_archetype_preds"


def _fetch_one_pred_row(pool, player_id: str, season_name: str) -> Dict[str, Any]:
    """
    Fetch a single prediction row for (player_id, season_name).
    Raises LookupError if not found.
    """
    sql = f"""
    SELECT
        player_id, season_name, season_id, player_name,
        top1, top1_p, top2, top2_p, top3, top3_p,
        updated_at
    FROM {TABLE}
    WHERE player_id = %s AND season_name = %s
    LIMIT 1;
    """

    with pool.connection() as conn:
        conn.row_factory = dict_row
        with conn.cursor() as cur:
            cur.execute(sql, (player_id, season_name))
            row = cur.fetchone()

    if not row:
        raise LookupError(f"No archetype preds found for player_id={player_id}, season_name={season_name}")

    return row


def get_top1_pred(pool, player_id: str, season_name: str) -> Dict[str, Any]:
    row = _fetch_one_pred_row(pool, player_id, season_name)
    return {
        "player_id": row["player_id"],
        "season_name": row["season_name"],
        "season_id": row.get("season_id"),
        "player_name": row.get("player_name"),
        "top1": row.get("top1"),
        "top1_p": row.get("top1_p"),
        "updated_at": row.get("updated_at"),
    }


def get_top3_pred(pool, player_id: str, season_name: str) -> Dict[str, Any]:
    row = _fetch_one_pred_row(pool, player_id, season_name)

    topk = []
    if row.get("top1"):
        topk.append({"label": row["top1"], "p": row.get("top1_p")})
    if row.get("top2"):
        topk.append({"label": row["top2"], "p": row.get("top2_p")})
    if row.get("top3"):
        topk.append({"label": row["top3"], "p": row.get("top3_p")})

    return {
        "player_id": row["player_id"],
        "season_name": row["season_name"],
        "season_id": row.get("season_id"),
        "player_name": row.get("player_name"),
        "topk": topk,
        "updated_at": row.get("updated_at"),
    }
