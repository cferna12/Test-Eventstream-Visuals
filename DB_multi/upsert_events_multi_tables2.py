# upsert_events.py
import ast
import json
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
            return None
    return None


def parse_dict_xy(v: Any) -> Tuple[Optional[float], Optional[float]]:
    if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
        return (None, None)
    if isinstance(v, dict):
        return (
            float(v.get("x")) if v.get("x") is not None else None,
            float(v.get("y")) if v.get("y") is not None else None,
        )
    if isinstance(v, str):
        s = v.strip()
        if s == "" or s.lower() in {"nan", "none"}:
            return (None, None)
        try:
            d = ast.literal_eval(s)
            if isinstance(d, dict):
                return (
                    float(d.get("x")) if d.get("x") is not None else None,
                    float(d.get("y")) if d.get("y") is not None else None,
                )
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


# may edit this insertion to cut down on total columns
# what are %s?
# what is excluded?
UPSERT_EVENTS_SQL = """
INSERT INTO eventstream_events (
  id,
  match_id, competition_id, season_id,
  match_period, minute, second, match_timestamp, video_timestamp,
  related_event_id,
  type_primary, type_secondary,
  location_x, location_y,
  team_id, player_id, possession_id,
  player_position,
  raw
) 
VALUES (
  %s,
  %s, %s, %s,
  %s, %s, %s, %s, %s,
  %s,
  %s, %s,
  %s, %s,
  %s, %s, %s,
  %s, %s, %s,
  %s
)
ON CONFLICT (id) DO UPDATE SET
  match_id = EXCLUDED.match_id,
  competition_id = EXCLUDED.competition_id,
  season_id = EXCLUDED.season_id,
  match_period = EXCLUDED.match_period,
  minute = EXCLUDED.minute,
  second = EXCLUDED.second,
  match_timestamp = EXCLUDED.match_timestamp,
  video_timestamp = EXCLUDED.video_timestamp,
  related_event_id = EXCLUDED.related_event_id,
  type_primary = EXCLUDED.type_primary,
  type_secondary = EXCLUDED.type_secondary,
  location_x = EXCLUDED.location_x,
  location_y = EXCLUDED.location_y,
  team_id = EXCLUDED.team_id,
  player_id = EXCLUDED.player_id,
  possession_id = EXCLUDED.possession_id,
  team_name = EXCLUDED.team_name,
  player_name = EXCLUDED.player_name,
  player_position = EXCLUDED.player_position,
  raw = EXCLUDED.raw
;
"""

UPSERT_PASS_SQL = """
INSERT INTO eventstream_passes (event_id, accurate, end_x, end_y)
VALUES (%s, %s, %s, %s)
ON CONFLICT (event_id) DO UPDATE SET
  accurate = EXCLUDED.accurate,
  end_x = EXCLUDED.end_x,
  end_y = EXCLUDED.end_y
;
"""

UPSERT_CARRY_SQL = """
INSERT INTO eventstream_carries (event_id, progression, end_x, end_y)
VALUES (%s, %s, %s, %s)
ON CONFLICT (event_id) DO UPDATE SET
  progression = EXCLUDED.progression,
  end_x = EXCLUDED.end_x,
  end_y = EXCLUDED.end_y
;
"""

UPSERT_SHOT_SQL = """
INSERT INTO eventstream_shots (event_id, is_goal, on_target, goal_zone, xg, body_part)
VALUES (%s, %s, %s, %s, %s, %s)
ON CONFLICT (event_id) DO UPDATE SET
  is_goal = EXCLUDED.is_goal,
  on_target = EXCLUDED.on_target,
  goal_zone = EXCLUDED.goal_zone,
  xg = EXCLUDED.xg,
  body_part = EXCLUDED.body_part
;
"""

UPSERT_AERIAL_DUEL_SQL = """
INSERT INTO eventstream_aerial_duels (event_id, first_touch)
VALUES (%s, %s)
ON CONFLICT (event_id) DO UPDATE SET
  first_touch = EXCLUDED.first_touch
;
"""

UPSERT_GROUND_DUEL_SQL = """
INSERT INTO eventstream_ground_duels (event_id, recovered_possession, kept_possession)
VALUES (%s, %s, %s)
ON CONFLICT (event_id) DO UPDATE SET
  recovered_possession = EXCLUDED.recovered_possession,
  kept_possession = EXCLUDED.kept_possession
;
"""


def row_to_event_record(r: pd.Series):
    # secondary list
    sec = parse_list_str(r.get("type.secondary"))

    # raw JSONB: keep original row
    raw = {k: (None if (isinstance(v, float) and pd.isna(v)) else v) for k, v in r.to_dict().items()}

    return (
        to_int(r.get("id")),
        to_int(r.get("matchId")),
        to_int(r.get("competition_id")),
        to_int(r.get("season_id")),
        r.get("matchPeriod"),
        to_int(r.get("minute")),
        to_int(r.get("second")),
        r.get("matchTimestamp"),
        to_float(r.get("videoTimestamp")),
        to_int(r.get("relatedEventId")),
        r.get("type.primary"),
        sec,
        to_float(r.get("location.x")),
        to_float(r.get("location.y")),
        to_int(r.get("team.id")),
        to_int(r.get("player.id")),
        to_int(r.get("possession.id")),
        r.get("team.name"),
        r.get("player.name"),
        r.get("player.position"),
        Jsonb(raw),
    )


def row_to_pass_record(r: pd.Series):
    ex, ey = parse_dict_xy(r.get("pass.endLocation"))
    return (
        to_int(r.get("id")),
        to_bool(r.get("pass.accurate")),
        ex,
        ey,
    )


def row_to_carry_record(r: pd.Series):
    ex, ey = parse_dict_xy(r.get("carry.endLocation"))
    return (
        to_int(r.get("id")),
        to_float(r.get("carry.progression")),
        ex,
        ey,
    )


def row_to_shot_record(r: pd.Series):
    return (
        to_int(r.get("id")),
        to_bool(r.get("shot.isGoal")),
        to_bool(r.get("shot.onTarget")),
        r.get("shot.goalZone"),
        to_float(r.get("shot.xg")),
        r.get("shot.bodyPart"),
    )


def row_to_aerial_duel_record(r: pd.Series):
    return (
        to_int(r.get("id")),
        to_bool(r.get("aerialDuel.firstTouch")),
    )


def row_to_ground_duel_record(r: pd.Series):
    return (
        to_int(r.get("id")),
        to_bool(r.get("groundDuel.recoveredPossession")),
        to_bool(r.get("groundDuel.keptPossession")),
    )


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

                # 1) Base events
                ev_records = [row_to_event_record(r) for _, r in chunk.iterrows()]
                cur.executemany(UPSERT_EVENTS_SQL, ev_records)

                # 2) Action tables
                # Passes
                pass_mask = chunk["type.primary"] == "pass"
                if pass_mask.any():
                    pass_records = [row_to_pass_record(r) for _, r in chunk.loc[pass_mask].iterrows()]
                    cur.executemany(UPSERT_PASS_SQL, pass_records)

                # Carries
                # carry_mask = chunk["type.primary"] == "carry"
                # if carry_mask.any():
                #     carry_records = [row_to_carry_record(r) for _, r in chunk.loc[carry_mask].iterrows()]
                #     cur.executemany(UPSERT_CARRY_SQL, carry_records)

                if "type.secondary" in chunk.columns:
                    carry_mask = chunk["type.secondary"].astype(str).str.contains("carry", na=False)
                carry_mask = carry_mask | (chunk["type.primary"] == "carry")


                # Shots
                shot_mask = chunk["type.primary"] == "shot"
                if shot_mask.any():
                    shot_records = [row_to_shot_record(r) for _, r in chunk.loc[shot_mask].iterrows()]
                    cur.executemany(UPSERT_SHOT_SQL, shot_records)

                # Aerial duels (some exports use primary=duel with secondary aerial_duel;
                # your list includes "duel" primary and "aerial_duel" secondary.)
                # We'll capture both cases:
                sec_series = chunk.get("type.secondary")
                aerial_mask = False
                if "type.secondary" in chunk.columns:
                    aerial_mask = chunk["type.secondary"].astype(str).str.contains("aerial_duel", na=False)
                aerial_mask = aerial_mask | (chunk["type.primary"] == "aerial_duel")

                if hasattr(aerial_mask, "any") and aerial_mask.any():
                    aerial_records = [row_to_aerial_duel_record(r) for _, r in chunk.loc[aerial_mask].iterrows()]
                    cur.executemany(UPSERT_AERIAL_DUEL_SQL, aerial_records)

                ground_mask = False
                if "type.secondary" in chunk.columns:
                    ground_mask = chunk["type.secondary"].astype(str).str.contains("ground_duel", na=False)
                ground_mask = ground_mask | (chunk["type.primary"] == "ground_duel")

                if hasattr(ground_mask, "any") and ground_mask.any():
                    ground_records = [row_to_ground_duel_record(r) for _, r in chunk.loc[ground_mask].iterrows()]
                    cur.executemany(UPSERT_GROUND_DUEL_SQL, ground_records)

                conn.commit()
                print(f"Upserted rows {start:,}..{min(start + chunksize, total):,}")

    print("Done.")


if __name__ == "__main__":
    dsn = os.environ.get("PG_DSN", "")
    if not dsn:
        raise RuntimeError("Set PG_DSN env var, e.g. postgresql://postgres:postgres@localhost:5432/eventstream")

    csv_path = os.environ.get("EVENTS_CSV", r"C:\Users\cruzf\Okapi\OKAPI_october\Wyscout_v3\notebooks\matches\wyscout_dumps\USA\v3\comp_869\events_season_190578.csv")
    upsert_csv(csv_path, dsn)
