from __future__ import annotations

from typing import Any, List, Optional, Tuple
from fastapi import HTTPException
from psycopg import sql

BBox = Tuple[float, float, float, float]


def parse_bbox(bbox: Optional[str]) -> Optional[BBox]:
    if not bbox:
        return None
    try:
        parts = [float(x.strip()) for x in bbox.split(",")]
        if len(parts) != 4:
            raise ValueError
        return (parts[0], parts[1], parts[2], parts[3])
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid bbox. Use 'x_min,x_max,y_min,y_max'")


def add_common_event_filters(
    *,
    where_parts: List[sql.SQL],
    params: List[Any],
    player_id: int,
    match_id: Optional[int],
    competition_id: Optional[int],
    season_id: Optional[int],
    bbox: Optional[BBox],
    event_alias: str = "e",
    require_xy: bool = True,
) -> None:
    where_parts.append(sql.SQL(f"{event_alias}.player_id = %s"))
    params.append(player_id)

    if match_id is not None:
        where_parts.append(sql.SQL(f"{event_alias}.match_id = %s"))
        params.append(match_id)
    if competition_id is not None:
        where_parts.append(sql.SQL(f"{event_alias}.competition_id = %s"))
        params.append(competition_id)
    if season_id is not None:
        where_parts.append(sql.SQL(f"{event_alias}.season_id = %s"))
        params.append(season_id)

    if require_xy:
        where_parts.append(sql.SQL(f"{event_alias}.location_x IS NOT NULL"))
        where_parts.append(sql.SQL(f"{event_alias}.location_y IS NOT NULL"))

        if bbox:
            x_min, x_max, y_min, y_max = bbox
            where_parts.append(sql.SQL(f"{event_alias}.location_x BETWEEN %s AND %s"))
            params += [x_min, x_max]
            where_parts.append(sql.SQL(f"{event_alias}.location_y BETWEEN %s AND %s"))
            params += [y_min, y_max]


def add_secondary_any_filter(
    *,
    where_parts: List[sql.SQL],
    params: List[Any],
    secondary_any: Optional[List[str]],
    table_alias: str = "e",
) -> None:
    if secondary_any:
        where_parts.append(sql.SQL(f"{table_alias}.type_secondary && %s"))
        params.append(secondary_any)


def add_primary_or_secondary_tag(
    *,
    where_parts: List[sql.SQL],
    params: List[Any],
    primary_value: str,
    secondary_tag: str,
    event_alias: str = "e",
) -> None:
    where_parts.append(sql.SQL(f"({event_alias}.type_primary = %s OR {event_alias}.type_secondary && %s)"))
    params.append(primary_value)
    params.append([secondary_tag])
