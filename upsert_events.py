# # upsert_events.py (FIXED quoting + JSONB)
import ast
import os
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd
import psycopg
from psycopg.types.json import Jsonb

def parse_list_str(v: Any) -> Optional[list]:
    if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
        return None
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        s = v.strip()
        if s == "" or s.lower() in {"nan", "none"}:
            return None
        try:
            out = ast.literal_eval(s)
            if isinstance(out, list):
                return out
        except Exception:
            pass
    return None

def parse_dict_xy(v: Any) -> Tuple[Optional[float], Optional[float]]:
    if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
        return (None, None)
    if isinstance(v, dict):
        return (float(v.get("x")) if v.get("x") is not None else None,
                float(v.get("y")) if v.get("y") is not None else None)
    if isinstance(v, str):
        s = v.strip()
        if s == "" or s.lower() in {"nan", "none"}:
            return (None, None)
        try:
            d = ast.literal_eval(s)
            if isinstance(d, dict):
                return (float(d.get("x")) if d.get("x") is not None else None,
                        float(d.get("y")) if d.get("y") is not None else None)
        except Exception:
            return (None, None)
    return (None, None)

def to_bool(v: Any) -> Optional[bool]:
    if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "t", "1"}:
            return True
        if s in {"false", "f", "0"}:
            return False
    return None

def to_int(v: Any) -> Optional[int]:
    if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
        return None
    try:
        return int(v)
    except Exception:
        return None

def to_float(v: Any) -> Optional[float]:
    if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
        return None
    try:
        return float(v)
    except Exception:
        return None

INSERT_COLS = [
    "id",
    "matchId",
    "competition_id",
    "season_id",
    "matchPeriod",
    "minute",
    "second",
    "matchTimestamp",
    "videoTimestamp",
    "relatedEventId",
    "type_primary",
    "type_secondary",
    "location_x",
    "location_y",
    "team.id",
    "team.name",
    "player.id",
    "player.name",
    "player.position",
    "possession.id",
    "pass_accurate",
    "pass_end_x",
    "pass_end_y",
    "carry_end_x",
    "carry_end_y",
    "carry_progression",
    "shot_isGoal",
    "shot_onTarget",
    "shot_goalZone",
    "shot_xg",
    "shot_bodyPart",
    "aerialDuel_firstTouch",
    "groundDuel_recoveredPossession",
    "groundDuel_keptPossession",
    "raw",
]

def q(col: str) -> str:
    # Always quote. This matches your schema which used quoted mixed-case and dotted names.
    return f'"{col}"'

UPSERT_SQL = f"""
INSERT INTO eventstream_events ({", ".join(q(c) for c in INSERT_COLS)})
VALUES ({", ".join(["%s"] * len(INSERT_COLS))})
ON CONFLICT (id) DO UPDATE SET
{", ".join(f"{q(c)} = EXCLUDED.{q(c)}" for c in INSERT_COLS if c != "id")}
;
"""

def row_to_record(r: pd.Series) -> Tuple[Any, ...]:
    sec = parse_list_str(r.get("type.secondary"))
    pass_ex, pass_ey = parse_dict_xy(r.get("pass.endLocation"))
    carry_ex, carry_ey = parse_dict_xy(r.get("carry.endLocation"))

    raw = {}
    for k, v in r.to_dict().items():
        if isinstance(v, float) and pd.isna(v):
            raw[k] = None
        else:
            raw[k] = v

    rec: Dict[str, Any] = {
        "id": to_int(r.get("id")),
        "matchId": to_int(r.get("matchId")),
        "competition_id": to_int(r.get("competition_id")),
        "season_id": to_int(r.get("season_id")),
        "matchPeriod": r.get("matchPeriod"),
        "minute": to_int(r.get("minute")),
        "second": to_int(r.get("second")),
        "matchTimestamp": r.get("matchTimestamp"),
        "videoTimestamp": to_float(r.get("videoTimestamp")),
        "relatedEventId": to_int(r.get("relatedEventId")),

        "type_primary": r.get("type.primary"),
        "type_secondary": sec,

        "location_x": to_float(r.get("location.x")),
        "location_y": to_float(r.get("location.y")),

        "team.id": to_int(r.get("team.id")),
        "team.name": r.get("team.name"),
        "player.id": to_int(r.get("player.id")),
        "player.name": r.get("player.name"),
        "player.position": r.get("player.position"),
        "possession.id": to_int(r.get("possession.id")),

        "pass_accurate": to_bool(r.get("pass.accurate")),
        "pass_end_x": pass_ex,
        "pass_end_y": pass_ey,

        "carry_end_x": carry_ex,
        "carry_end_y": carry_ey,
        "carry_progression": to_float(r.get("carry.progression")),

        "shot_isGoal": to_bool(r.get("shot.isGoal")),
        "shot_onTarget": to_bool(r.get("shot.onTarget")),
        "shot_goalZone": r.get("shot.goalZone"),
        "shot_xg": to_float(r.get("shot.xg")),
        "shot_bodyPart": r.get("shot.bodyPart"),

        "aerialDuel_firstTouch": to_bool(r.get("aerialDuel.firstTouch")),
        "groundDuel_recoveredPossession": to_bool(r.get("groundDuel.recoveredPossession")),
        "groundDuel_keptPossession": to_bool(r.get("groundDuel.keptPossession")),

        "raw": Jsonb(raw),
    }

    return tuple(rec[c] for c in INSERT_COLS)

def upsert_csv(csv_path: str, dsn: str, chunksize: int = 20000) -> None:
    df = pd.read_csv(csv_path)

    missing = [c for c in ["id", "matchId", "type.primary", "location.x", "location.y"] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    total = len(df)
    print(f"Loaded {total:,} rows from {csv_path}")

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            for start in range(0, total, chunksize):
                chunk = df.iloc[start:start + chunksize]
                records = [row_to_record(r) for _, r in chunk.iterrows()]
                cur.executemany(UPSERT_SQL, records)
                conn.commit()
                print(f"Upserted rows {start:,}..{min(start+chunksize, total):,}")

    print("Done.")

if __name__ == "__main__":
    dsn = os.environ.get("PG_DSN", "")
    if not dsn:
        raise RuntimeError("Set PG_DSN env var, e.g. postgresql://postgres:postgres@localhost:5432/eventstream")

    csv_path = os.environ.get("EVENTS_CSV", "data/two_games.csv")
    upsert_csv(csv_path, dsn)
