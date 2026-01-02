-- -- schema.sql
-- CREATE TABLE IF NOT EXISTS eventstream_events (
--   -- Canonical Wyscout: id
--   id BIGINT PRIMARY KEY,

--   -- Canonical identifiers
--   "matchId" BIGINT NOT NULL,
--   competition_id BIGINT,
--   season_id BIGINT,

--   "matchPeriod" TEXT,
--   minute INT,
--   second INT,
--   "matchTimestamp" TEXT,
--   "videoTimestamp" DOUBLE PRECISION,

--   "relatedEventId" BIGINT,

--   -- Canonical: type.primary, type.secondary (list)
--   type_primary TEXT NOT NULL,
--   type_secondary TEXT[],

--   -- Canonical: location.x, location.y
--   location_x REAL,
--   location_y REAL,

--   -- Canonical IDs
--   "team.id" BIGINT,
--   "player.id" BIGINT,
--   "possession.id" BIGINT,

--   -- Optional “nice to have” for debugging/labels (not required for viz)
--   "team.name" TEXT,
--   "player.name" TEXT,
--   "player.position" TEXT,

--   -- Action-specific typed fields (thin but fast)
--   pass_accurate BOOLEAN,
--   pass_end_x REAL,
--   pass_end_y REAL,

--   carry_end_x REAL,
--   carry_end_y REAL,
--   carry_progression REAL,

--   shot_isGoal BOOLEAN,
--   shot_onTarget BOOLEAN,
--   shot_goalZone TEXT,
--   shot_xg REAL,
--   shot_bodyPart TEXT,

--   aerialDuel_firstTouch BOOLEAN,
--   groundDuel_recoveredPossession BOOLEAN,
--   groundDuel_keptPossession BOOLEAN,

--   -- Optional “escape hatch” for any extra fields you later want
--   raw JSONB
-- );

-- -- Common query patterns for heatmaps
-- CREATE INDEX IF NOT EXISTS idx_events_match_primary
--   ON eventstream_events ("matchId", type_primary);

-- CREATE INDEX IF NOT EXISTS idx_events_player_primary
--   ON eventstream_events ("player.id", type_primary);

-- CREATE INDEX IF NOT EXISTS idx_events_player_comp_season_primary
--   ON eventstream_events ("player.id", competition_id, season_id, type_primary);

-- -- Location filtering is very common
-- CREATE INDEX IF NOT EXISTS idx_events_location
--   ON eventstream_events (location_x, location_y);

-- -- Secondary type membership (GIN for array)
-- CREATE INDEX IF NOT EXISTS idx_events_secondary_gin
--   ON eventstream_events USING GIN (type_secondary);

-- -- If you ever query raw JSON a lot, you can add:
-- -- CREATE INDEX IF NOT EXISTS idx_events_raw_gin ON eventstream_events USING GIN (raw);


DROP TABLE IF EXISTS eventstream_events;

CREATE TABLE eventstream_events (
  id BIGINT PRIMARY KEY,

  -- Canonical identifiers (quoted because Wyscout style / mixed-case)
  "matchId" BIGINT NOT NULL,
  competition_id BIGINT,
  season_id BIGINT,

  "matchPeriod" TEXT,
  minute INT,
  second INT,
  "matchTimestamp" TEXT,
  "videoTimestamp" DOUBLE PRECISION,

  "relatedEventId" BIGINT,

  -- Canonical: type.primary, type.secondary (list)
  type_primary TEXT NOT NULL,
  type_secondary TEXT[],

  -- Canonical: location.x, location.y
  location_x REAL,
  location_y REAL,

  -- Canonical IDs (quoted because they contain dots)
  "team.id" BIGINT,
  "player.id" BIGINT,
  "possession.id" BIGINT,

  -- Optional labels
  "team.name" TEXT,
  "player.name" TEXT,
  "player.position" TEXT,

  -- Action-specific (MAKE THESE ALL LOWERCASE SNAKE_CASE)
  pass_accurate BOOLEAN,
  pass_end_x REAL,
  pass_end_y REAL,

  carry_end_x REAL,
  carry_end_y REAL,
  carry_progression REAL,

  shot_is_goal BOOLEAN,
  shot_on_target BOOLEAN,
  shot_goal_zone TEXT,
  shot_xg REAL,
  shot_body_part TEXT,

  aerial_duel_first_touch BOOLEAN,
  ground_duel_recovered_possession BOOLEAN,
  ground_duel_kept_possession BOOLEAN,

  raw JSONB
);

CREATE INDEX idx_events_match_primary
  ON eventstream_events ("matchId", type_primary);

CREATE INDEX idx_events_player_primary
  ON eventstream_events ("player.id", type_primary);

CREATE INDEX idx_events_player_comp_season_primary
  ON eventstream_events ("player.id", competition_id, season_id, type_primary);

CREATE INDEX idx_events_location
  ON eventstream_events (location_x, location_y);

CREATE INDEX idx_events_secondary_gin
  ON eventstream_events USING GIN (type_secondary);
