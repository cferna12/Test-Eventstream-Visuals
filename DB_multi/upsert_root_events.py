#!/usr/bin/env python3
"""
upsert_root_events.py

Upserts Wyscout eventstream CSVs from a root directory (e.g. data/)
into Postgres, including pass recipient information.

Expected layout example:
  data/
    comp_869/
      events_season_190578.csv
      events_season_190579.csv
    comp_123/
      events_season_555.csv

Usage:
  set PG_DSN=postgresql://postgres:postgres@localhost:5432/eventstream
  python upsert_root_events.py --root data

Optional:
  python upsert_root_events.py --root data --pattern "events_season_*.csv" --chunksize 10000 --max-files 3
"""

import ast
import os
import time
import argparse
from pathlib import Path
from typing import Any, Optional, Tuple, List, Dict

import pandas as pd
import psycopg
from psycopg.types.json import Jsonb


# -----------------------------
# Parsing helpers
# -----------------------------
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


def parse_dict_id_name_pos(v: Any) -> tuple[Optional[int], Optional[str], Optional[str]]:
    """
    Parses {'id': 21020, 'name': 'G. Brugman', 'position': 'LCMF3'}
    from dict or stringified dict.
    """
    if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
        return (None, None, None)

    d = None
    if isinstance(v, dict):
        d = v
    elif isinstance(v, str):
        s = v.strip()
        if s == "" or s.lower() in {"nan", "none"}:
            return (None, None, None)
        try:
            out = ast.literal_eval(s)
            if isinstance(out, dict):
                d = out
        except Exception:
            return (None, None, None)

    if not isinstance(d, dict):
        return (None, None, None)

    rid = d.get("id")
    rname = d.get("name")
    rpos = d.get("position")

    return (
        int(rid) if rid is not None else None,
        str(rname) if rname is not None else None,
        str(rpos) if rpos is not None else None,
    )


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


# -----------------------------
# SQL
# -----------------------------
UPSERT_EVENTS_SQL = """
INSERT INTO eventstream_events (
  id,
  match_id, competition_id, season_id,
  match_period, minute, second, match_timestamp, video_timestamp,
  related_event_id,
  type_primary, type_secondary,
  location_x, location_y,
  team_id, player_id, possession_id,
  player_name, team_name, player_position,
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

# Updated: includes pass recipient fields (ensure your DB has these columns)
UPSERT_PASS_SQL = """
INSERT INTO eventstream_passes (
  event_id, accurate, end_x, end_y,
  recipient_id, recipient_name, recipient_position
)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (event_id) DO UPDATE SET
  accurate = EXCLUDED.accurate,
  end_x = EXCLUDED.end_x,
  end_y = EXCLUDED.end_y,
  recipient_id = EXCLUDED.recipient_id,
  recipient_name = EXCLUDED.recipient_name,
  recipient_position = EXCLUDED.recipient_position
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


# -----------------------------
# Row -> record (dict-based, faster than iterrows Series)
# -----------------------------
def row_to_event_record(d: Dict[str, Any]):
    sec = parse_list_str(d.get("type.secondary"))

    # Keep original row for debugging; convert NaN to None for JSONB
    raw = {k: (None if (isinstance(v, float) and pd.isna(v)) else v) for k, v in d.items()}

    return (
        to_int(d.get("id")),
        to_int(d.get("matchId")),
        to_int(d.get("competition_id")),
        to_int(d.get("season_id")),
        d.get("matchPeriod"),
        to_int(d.get("minute")),
        to_int(d.get("second")),
        d.get("matchTimestamp"),
        to_float(d.get("videoTimestamp")),
        to_int(d.get("relatedEventId")),
        d.get("type.primary"),
        sec,
        to_float(d.get("location.x")),
        to_float(d.get("location.y")),
        to_int(d.get("team.id")),
        to_int(d.get("player.id")),
        to_int(d.get("possession.id")),
        # IMPORTANT: correct order = player_name, team_name, player_position
        d.get("player.name"),
        d.get("team.name"),
        d.get("player.position"),
        Jsonb(raw),
    )


def row_to_pass_record(d: Dict[str, Any]):
    ex, ey = parse_dict_xy(d.get("pass.endLocation"))
    rid, rname, rpos = parse_dict_id_name_pos(d.get("pass.recipient"))
    return (
        to_int(d.get("id")),
        to_bool(d.get("pass.accurate")),
        ex,
        ey,
        rid,
        rname,
        rpos,
    )


def row_to_carry_record(d: Dict[str, Any]):
    ex, ey = parse_dict_xy(d.get("carry.endLocation"))
    return (
        to_int(d.get("id")),
        to_float(d.get("carry.progression")),
        ex,
        ey,
    )


def row_to_shot_record(d: Dict[str, Any]):
    return (
        to_int(d.get("id")),
        to_bool(d.get("shot.isGoal")),
        to_bool(d.get("shot.onTarget")),
        d.get("shot.goalZone"),
        to_float(d.get("shot.xg")),
        d.get("shot.bodyPart"),
    )


def row_to_aerial_duel_record(d: Dict[str, Any]):
    return (
        to_int(d.get("id")),
        to_bool(d.get("aerialDuel.firstTouch")),
    )


def row_to_ground_duel_record(d: Dict[str, Any]):
    return (
        to_int(d.get("id")),
        to_bool(d.get("groundDuel.recoveredPossession")),
        to_bool(d.get("groundDuel.keptPossession")),
    )


# -----------------------------
# File discovery
# -----------------------------
def find_event_csvs(root: Path, pattern: str) -> List[Path]:
    # Recursively find matching files
    files = sorted(root.rglob(pattern))
    return [p for p in files if p.is_file()]


# -----------------------------
# Upsert runner
# -----------------------------
def upsert_one_csv(
    *,
    csv_path: Path,
    cur: psycopg.Cursor,
    chunksize: int,
    verbose: bool,
) -> int:
    """
    Returns number of rows processed from this CSV (sum across chunks).
    """
    # stream chunks
    total_rows = 0
    t0 = time.time()

    # dtype=None keeps pandas inference; low_memory=False avoids mixed-type warnings
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
        # quick schema guard
        required = ["id", "matchId", "type.primary"]
        missing = [c for c in required if c not in chunk.columns]
        if missing:
            raise ValueError(f"{csv_path} missing required columns: {missing}")

        # Build dict records once (faster than iterrows)
        records: List[Dict[str, Any]] = chunk.to_dict(orient="records")
        n = len(records)
        total_rows += n

        # 1) events
        ev_records = [row_to_event_record(d) for d in records]
        cur.executemany(UPSERT_EVENTS_SQL, ev_records)

        # 2) passes
        # avoid pandas masks; cheap filter in Python on dicts
        pass_records = [row_to_pass_record(d) for d in records if d.get("type.primary") == "pass"]
        if pass_records:
            cur.executemany(UPSERT_PASS_SQL, pass_records)

        # 3) shots
        shot_records = [row_to_shot_record(d) for d in records if d.get("type.primary") == "shot"]
        if shot_records:
            cur.executemany(UPSERT_SHOT_SQL, shot_records)

        # 4) aerial/ground/carry via secondary tags or primary
        # Note: type.secondary is stringified list in many exports, so do str-contains fast path.
        for d in records:
            pass

        aerial_records = []
        ground_records = []
        carry_records = []

        for d in records:
            primary = d.get("type.primary")
            sec = d.get("type.secondary")
            sec_s = str(sec) if sec is not None else ""

            if primary == "aerial_duel" or ("aerial_duel" in sec_s):
                aerial_records.append(row_to_aerial_duel_record(d))
            if primary == "ground_duel" or ("ground_duel" in sec_s):
                ground_records.append(row_to_ground_duel_record(d))
            if primary == "carry" or ("carry" in sec_s):
                carry_records.append(row_to_carry_record(d))

        if aerial_records:
            cur.executemany(UPSERT_AERIAL_DUEL_SQL, aerial_records)
        if ground_records:
            cur.executemany(UPSERT_GROUND_DUEL_SQL, ground_records)
        if carry_records:
            cur.executemany(UPSERT_CARRY_SQL, carry_records)

        if verbose:
            dt = time.time() - t0
            rps = total_rows / dt if dt > 0 else 0.0
            print(f"  ...{csv_path.name}: {total_rows:,} rows so far ({rps:,.0f} rows/s)")

    dt = time.time() - t0
    rps = total_rows / dt if dt > 0 else 0.0
    print(f"[FILE OK] {csv_path} | rows={total_rows:,} | {dt:.1f}s | {rps:,.0f} rows/s")
    return total_rows


def upsert_root(
    *,
    root: Path,
    dsn: str,
    pattern: str,
    chunksize: int,
    max_files: Optional[int],
    commit_every_file: bool,
    verbose: bool,
) -> None:
    files = find_event_csvs(root, pattern)
    if not files:
        raise RuntimeError(f"No files found under {root} matching pattern: {pattern}")

    if max_files is not None:
        files = files[:max_files]

    print(f"Found {len(files)} file(s) under {root} matching {pattern}")

    grand_rows = 0
    t0 = time.time()

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            for i, csv_path in enumerate(files, start=1):
                print(f"\n[{i}/{len(files)}] Upserting {csv_path} ...")
                n_rows = upsert_one_csv(
                    csv_path=csv_path,
                    cur=cur,
                    chunksize=chunksize,
                    verbose=verbose,
                )
                grand_rows += n_rows

                if commit_every_file:
                    conn.commit()
                    print(f"[COMMIT] after file {i}/{len(files)}")

            # final commit
            conn.commit()

    dt = time.time() - t0
    rps = grand_rows / dt if dt > 0 else 0.0
    print(f"\n[ALL DONE] files={len(files)} total_rows={grand_rows:,} total_time={dt:.1f}s ({rps:,.0f} rows/s)")


# -----------------------------
# CLI
# -----------------------------
def main():
    p = argparse.ArgumentParser(description="Upsert all Wyscout event CSVs under a root directory into Postgres.")
    p.add_argument("--root", default="data", help="Root directory to scan (default: data)")
    p.add_argument("--pattern", default="*_events_season_*.csv", help="Glob pattern to match (default: events_season_*.csv)")
    p.add_argument("--chunksize", type=int, default=20000, help="CSV chunksize (default: 20000)")
    p.add_argument("--max-files", type=int, default=None, help="Limit number of files (for testing)")
    p.add_argument("--commit-every-file", action="store_true", help="Commit after each file (safer, slightly slower)")
    p.add_argument("--verbose", action="store_true", help="Print progress within files")

    args = p.parse_args()

    dsn = os.environ.get("PG_DSN", "")
    if not dsn:
        raise RuntimeError("Set PG_DSN env var, e.g. postgresql://postgres:postgres@localhost:5432/eventstream")

    root = Path(args.root).resolve()
    if not root.exists():
        raise RuntimeError(f"Root directory does not exist: {root}")

    upsert_root(
        root=root,
        dsn=dsn,
        pattern=args.pattern,
        chunksize=args.chunksize,
        max_files=args.max_files,
        commit_every_file=args.commit_every_file,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
