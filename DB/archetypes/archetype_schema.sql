-- Archetype probability outputs (slim)
CREATE TABLE IF NOT EXISTS player_archetype_preds (
    player_id   TEXT NOT NULL,
    season_name TEXT NOT NULL,

    -- keep season_id for future joins/queries
    season_id   TEXT,

    player_name TEXT,

    top1   TEXT,
    top1_p DOUBLE PRECISION,
    top2   TEXT,
    top2_p DOUBLE PRECISION,
    top3   TEXT,
    top3_p DOUBLE PRECISION,

    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (player_id, season_name)
);

-- Common queries now
CREATE INDEX IF NOT EXISTS idx_player_archetype_preds_seasonname
ON player_archetype_preds (season_name);

-- Future: seasonId support (filter/join)
CREATE INDEX IF NOT EXISTS idx_player_archetype_preds_player_seasonid
ON player_archetype_preds (player_id, season_id);
