import os
import pandas as pd
import psycopg

CSV_PATH = r"C:\Users\cruzf\Okapi\OKAPI_october\Heatmaps\data\archetypes\NCAA_2025_archetypes_0_min_cutoff.csv"
EXPECTED_SEASON = "2025 Fall"

DSN = os.environ.get(
    "PG_DSN",
    "postgresql://postgres:postgres@localhost:5432/postgres"
)

TABLE = "player_archetype_preds"

def main():
    df = pd.read_csv(CSV_PATH)

    # ---- column checks ----
    required_cols = [
        "playerId", "playerName",
        "Top1", "Top1_p", "Top2", "Top2_p", "Top3", "Top3_p",
        "season.name"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"CSV missing required columns: {missing}")

    # ---- filter by season.name ----
    valid = df["season.name"].astype(int) == EXPECTED_SEASON
    invalid_df = df.loc[~valid].copy()
    df = df.loc[valid].copy()

    print(f"[INFO] rows with season.name == '{EXPECTED_SEASON}': {len(df)}")
    print(f"[WARN] rows skipped (wrong season): {len(invalid_df)}")

    if len(invalid_df) > 0:
        invalid_path = os.path.splitext(CSV_PATH)[0] + "__SKIPPED_wrong_season.csv"
        invalid_df.to_csv(invalid_path, index=False, encoding="utf-8-sig")
        print(f"[WARN] wrote skipped rows to: {invalid_path}")

    if df.empty:
        raise RuntimeError("No rows left after season filter — nothing to upsert.")

    # ---- normalize columns ----
    slim = pd.DataFrame({
        "player_id": df["playerId"].astype(str),
        "season_name": EXPECTED_SEASON,
        "season_id": df.get("seasonId"),
        "player_name": df["playerName"],
        "top1": df["Top1"],
        "top1_p": df["Top1_p"],
        "top2": df["Top2"],
        "top2_p": df["Top2_p"],
        "top3": df["Top3"],
        "top3_p": df["Top3_p"],
    })

    # ---- enforce 1 row per player per season ----
    dup = slim.duplicated(subset=["player_id", "season_name"], keep=False)
    if dup.any():
        dup_path = os.path.splitext(CSV_PATH)[0] + "__DUP_player_season.csv"
        slim.loc[dup].to_csv(dup_path, index=False, encoding="utf-8-sig")
        raise ValueError(
            "Duplicate (player_id, season_name) rows detected.\n"
            f"Wrote duplicates to: {dup_path}"
        )

    # ---- upsert ----
    sql = f"""
    INSERT INTO {TABLE} (
        player_id, season_name, season_id, player_name,
        top1, top1_p, top2, top2_p, top3, top3_p
    )
    VALUES (
        %(player_id)s, %(season_name)s, %(season_id)s, %(player_name)s,
        %(top1)s, %(top1_p)s, %(top2)s, %(top2_p)s, %(top3)s, %(top3_p)s
    )
    ON CONFLICT (player_id, season_name)
    DO UPDATE SET
        season_id = EXCLUDED.season_id,
        player_name = EXCLUDED.player_name,
        top1 = EXCLUDED.top1,
        top1_p = EXCLUDED.top1_p,
        top2 = EXCLUDED.top2,
        top2_p = EXCLUDED.top2_p,
        top3 = EXCLUDED.top3,
        top3_p = EXCLUDED.top3_p,
        updated_at = NOW();
    """

    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, slim.to_dict(orient="records"))
        conn.commit()

    print(f"✅ Upserted {len(slim)} rows into {TABLE} for season '{EXPECTED_SEASON}'")

if __name__ == "__main__":
    main()
