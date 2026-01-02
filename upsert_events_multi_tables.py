# upsert_events.py (MULTI-TABLE)
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
            return out if isinstance(out, list) else None
        except Exception:
            return None
    return None

def parse_dict(v: Any) -> Optional[dict]:
    if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
        return None
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        s = v.strip()
        if s == "" or s.lower() in {"nan", "none"}:
            return None
        try:
            out = ast.literal_eval(s)
            return out if isinstance(out, dict) else None
        except Exception:
            return None
    return None

def parse_dict_xy(v: Any) -> Tuple[Optional[float], Optional[float]]:
    d = parse_dict(v)
    if not d:
        return (None, None)
    x = d.get("x")
    y = d.get("y")
    return (float(x) if x is not None else None, float(y) if y is not None else None)

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

def q(col: str) -> str:
    return f'"{col}"'

# ----------------------------
# Base events table upsert
# ----------------------------
EVENT_COLS = [
    "id",
    "matchId",
    "competition_id",
    "season_id",
    "matchPeriod",
    "minute",
    "second",
    "matchTimestamp",
    # "videoTimestamp",
    # "relatedEventId",
    "type_primary",
    "type_secondary",
    "location_x",
    "location_y",
    "team.id",
    "player.id",
    "possession.id",
    # "team.name",
    # "player.name",
    "player.position",
    "raw",
]

EVENT_UPSERT = f"""
INSERT INTO eventstream_events ({", ".join(q(c) for c in EVENT_COLS)})
VALUES ({", ".join(["%s"] * len(EVENT_COLS))})
ON CONFLICT (id) DO UPDATE SET
{", ".join(f"{q(c)} = EXCLUDED.{q(c)}" for c in EVENT_COLS if c != "id")}
;
"""

# ----------------------------
# Child table upserts
# ----------------------------
PASS_COLS = ["id", "pass_accurate", "pass_angle", "pass_height", "pass_length", "pass_end_x", "pass_end_y"]#, "pass_recipient"]
PASS_UPSERT = f"""
INSERT INTO eventstream_passes ({", ".join(q(c) for c in PASS_COLS)})
VALUES ({", ".join(["%s"] * len(PASS_COLS))})
ON CONFLICT (id) DO UPDATE SET
{", ".join(f"{q(c)} = EXCLUDED.{q(c)}" for c in PASS_COLS if c != "id")}
;
"""

CARRY_COLS = ["id", "carry_progression", "carry_end_x", "carry_end_y"]
CARRY_UPSERT = f"""
INSERT INTO eventstream_carries ({", ".join(q(c) for c in CARRY_COLS)})
VALUES ({", ".join(["%s"] * len(CARRY_COLS))})
ON CONFLICT (id) DO UPDATE SET
{", ".join(f"{q(c)} = EXCLUDED.{q(c)}" for c in CARRY_COLS if c != "id")}
;
"""

SHOT_COLS = ["id", "shot_isGoal", "shot_onTarget", "shot_goalZone", "shot_xg", "shot_postShotXg", "shot_bodyPart"]
SHOT_UPSERT = f"""
INSERT INTO eventstream_shots ({", ".join(q(c) for c in SHOT_COLS)})
VALUES ({", ".join(["%s"] * len(SHOT_COLS))})
ON CONFLICT (id) DO UPDATE SET
{", ".join(f"{q(c)} = EXCLUDED.{q(c)}" for c in SHOT_COLS if c != "id")}
;
"""

AERIAL_COLS = ["id", "aerialDuel_firstTouch", "aerialDuel_height"] #, "aerialDuel_opponent"]
AERIAL_UPSERT = f"""
INSERT INTO eventstream_aerial_duels ({", ".join(q(c) for c in AERIAL_COLS)})
VALUES ({", ".join(["%s"] * len(AERIAL_COLS))})
ON CONFLICT (id) DO UPDATE SET
{", ".join(f"{q(c)} = EXCLUDED.{q(c)}" for c in AERIAL_COLS if c != "id")}
;
"""

GROUND_COLS = [
    "id",
    "groundDuel_recoveredPossession",
    "groundDuel_keptPossession",
    "groundDuel_progressedWithBall",
    "groundDuel_stoppedProgress",
    "groundDuel_takeOn",
    "groundDuel_duelType",
    "groundDuel_side",
    # "groundDuel_opponent",
]
GROUND_UPSERT = f"""
INSERT INTO eventstream_ground_duels ({", ".join(q(c) for c in GROUND_COLS)})
VALUES ({", ".join(["%s"] * len(GROUND_COLS))})
ON CONFLICT (id) DO UPDATE SET
{", ".join(f"{q(c)} = EXCLUDED.{q(c)}" for c in GROUND_COLS if c != "id")}
;
"""

def build_event_row(r: pd.Series) -> Tuple[Any, ...]:
    sec = parse_list_str(r.get("type.secondary"))
    raw = {}
    for k, v in r.to_dict().items():
        raw[k] = None if (isinstance(v, float) and pd.isna(v)) else v

    rec = {
        "id": to_int(r.get("id")),
        "matchId": to_int(r.get("matchId")),
        "competition_id": to_int(r.get("competition_id")),
        "season_id": to_int(r.get("season_id")),
        "matchPeriod": r.get("matchPeriod"),
        "minute": to_int(r.get("minute")),
        "second": to_int(r.get("second")),
        "matchTimestamp": r.get("matchTimestamp"),
        # "videoTimestamp": to_float(r.get("videoTimestamp")),
        # "relatedEventId": to_int(r.get("relatedEventId")),
        "type_primary": r.get("type.primary"),
        "type_secondary": sec,
        "location_x": to_float(r.get("location.x")),
        "location_y": to_float(r.get("location.y")),
        "team.id": to_int(r.get("team.id")),
        "player.id": to_int(r.get("player.id")),
        "possession.id": to_int(r.get("possession.id")),
        # "team.name": r.get("team.name"),
        # "player.name": r.get("player.name"),
        "player.position": r.get("player.position"),
        "raw": Jsonb(raw),
    }
    return tuple(rec[c] for c in EVENT_COLS)

def build_pass_row(r: pd.Series) -> Optional[Tuple[Any, ...]]:
    pid = to_int(r.get("id"))
    if pid is None:
        return None

    end_x, end_y = parse_dict_xy(r.get("pass.endLocation"))
    recipient = parse_dict(r.get("pass.recipient"))

    # Only write row if we actually have any pass-like fields
    has_any = any(
        v is not None
        for v in [
            r.get("pass.accurate"),
            r.get("pass.angle"),
            r.get("pass.height"),
            r.get("pass.length"),
            r.get("pass.endLocation"),
            r.get("pass.recipient"),
        ]
    )
    if not has_any:
        return None

    rec = {
        "id": pid,
        "pass_accurate": to_bool(r.get("pass.accurate")),
        "pass_angle": to_float(r.get("pass.angle")),
        "pass_height": r.get("pass.height") if pd.notna(r.get("pass.height")) else None,
        "pass_length": to_float(r.get("pass.length")),
        "pass_end_x": end_x,
        "pass_end_y": end_y,
        # "pass_recipient": Jsonb(recipient) if recipient is not None else None,
    }
    return tuple(rec[c] for c in PASS_COLS)

def build_carry_row(r: pd.Series) -> Optional[Tuple[Any, ...]]:
    pid = to_int(r.get("id"))
    if pid is None:
        return None
    end_x, end_y = parse_dict_xy(r.get("carry.endLocation"))
    prog = to_float(r.get("carry.progression"))

    has_any = any(v is not None for v in [r.get("carry.endLocation"), r.get("carry.progression")])
    if not has_any:
        return None

    rec = {"id": pid, "carry_progression": prog, "carry_end_x": end_x, "carry_end_y": end_y}
    return tuple(rec[c] for c in CARRY_COLS)

def build_shot_row(r: pd.Series) -> Optional[Tuple[Any, ...]]:
    pid = to_int(r.get("id"))
    if pid is None:
        return None

    has_any = any(
        v is not None
        for v in [
            r.get("shot.isGoal"),
            r.get("shot.onTarget"),
            r.get("shot.goalZone"),
            r.get("shot.xg"),
            r.get("shot.postShotXg"),
            r.get("shot.bodyPart"),
        ]
    )
    if not has_any:
        return None

    rec = {
        "id": pid,
        "shot_isGoal": to_bool(r.get("shot.isGoal")),
        "shot_onTarget": to_bool(r.get("shot.onTarget")),
        "shot_goalZone": r.get("shot.goalZone") if pd.notna(r.get("shot.goalZone")) else None,
        "shot_xg": to_float(r.get("shot.xg")),
        "shot_postShotXg": to_float(r.get("shot.postShotXg")),
        "shot_bodyPart": r.get("shot.bodyPart") if pd.notna(r.get("shot.bodyPart")) else None,
    }
    return tuple(rec[c] for c in SHOT_COLS)

def build_aerial_row(r: pd.Series) -> Optional[Tuple[Any, ...]]:
    pid = to_int(r.get("id"))
    if pid is None:
        return None

    opp = parse_dict(r.get("aerialDuel.opponent"))
    has_any = any(v is not None for v in [r.get("aerialDuel.firstTouch"), r.get("aerialDuel.height"), r.get("aerialDuel.opponent")])
    if not has_any:
        return None

    rec = {
        "id": pid,
        "aerialDuel_firstTouch": to_bool(r.get("aerialDuel.firstTouch")),
        "aerialDuel_height": to_float(r.get("aerialDuel.height")),
        "aerialDuel_opponent": Jsonb(opp) if opp is not None else None,
    }
    return tuple(rec[c] for c in AERIAL_COLS)

def build_ground_row(r: pd.Series) -> Optional[Tuple[Any, ...]]:
    pid = to_int(r.get("id"))
    if pid is None:
        return None

    opp = parse_dict(r.get("groundDuel.opponent"))
    has_any = any(
        v is not None
        for v in [
            r.get("groundDuel.recoveredPossession"),
            r.get("groundDuel.keptPossession"),
            r.get("groundDuel.progressedWithBall"),
            r.get("groundDuel.stoppedProgress"),
            r.get("groundDuel.takeOn"),
            r.get("groundDuel.duelType"),
            r.get("groundDuel.side"),
            r.get("groundDuel.opponent"),
        ]
    )
    if not has_any:
        return None

    rec = {
        "id": pid,
        "groundDuel_recoveredPossession": to_bool(r.get("groundDuel.recoveredPossession")),
        "groundDuel_keptPossession": to_bool(r.get("groundDuel.keptPossession")),
        "groundDuel_progressedWithBall": to_bool(r.get("groundDuel.progressedWithBall")),
        "groundDuel_stoppedProgress": to_bool(r.get("groundDuel.stoppedProgress")),
        "groundDuel_takeOn": to_bool(r.get("groundDuel.takeOn")),
        "groundDuel_duelType": r.get("groundDuel.duelType") if pd.notna(r.get("groundDuel.duelType")) else None,
        "groundDuel_side": r.get("groundDuel.side") if pd.notna(r.get("groundDuel.side")) else None,
        # "groundDuel_opponent": Jsonb(opp) if opp is not None else None,
    }
    return tuple(rec[c] for c in GROUND_COLS)

def upsert_csv(csv_path: str, dsn: str, chunksize: int = 20000) -> None:
    df = pd.read_csv(csv_path)

    required = ["id", "matchId", "type.primary", "location.x", "location.y"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    total = len(df)
    print(f"Loaded {total:,} rows from {csv_path}")

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            for start in range(0, total, chunksize):
                chunk = df.iloc[start:start + chunksize]

                # 1) Base events first
                event_rows = [build_event_row(r) for _, r in chunk.iterrows()]
                cur.executemany(EVENT_UPSERT, event_rows)

                # 2) Child tables
                pass_rows = []
                carry_rows = []
                shot_rows = []
                aerial_rows = []
                ground_rows = []

                for _, r in chunk.iterrows():
                    pr = build_pass_row(r)
                    if pr is not None: pass_rows.append(pr)

                    cr = build_carry_row(r)
                    if cr is not None: carry_rows.append(cr)

                    sr = build_shot_row(r)
                    if sr is not None: shot_rows.append(sr)

                    ar = build_aerial_row(r)
                    if ar is not None: aerial_rows.append(ar)

                    gr = build_ground_row(r)
                    if gr is not None: ground_rows.append(gr)

                if pass_rows:
                    cur.executemany(PASS_UPSERT, pass_rows)
                if carry_rows:
                    cur.executemany(CARRY_UPSERT, carry_rows)
                if shot_rows:
                    cur.executemany(SHOT_UPSERT, shot_rows)
                if aerial_rows:
                    cur.executemany(AERIAL_UPSERT, aerial_rows)
                if ground_rows:
                    cur.executemany(GROUND_UPSERT, ground_rows)

                conn.commit()
                print(f"Upserted rows {start:,}..{min(start+chunksize, total):,}")

    print("Done.")

if __name__ == "__main__":
    dsn = os.environ.get("PG_DSN", "")
    if not dsn:
        raise RuntimeError("Set PG_DSN env var, e.g. postgresql://postgres:postgres@localhost:5432/eventstream")

    csv_path = os.environ.get("EVENTS_CSV", "data/two_games.csv")
    upsert_csv(csv_path, dsn)
