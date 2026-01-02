-- DB/schema.sql

CREATE TABLE IF NOT EXISTS eventstream_events (
  id BIGINT PRIMARY KEY,

  "matchId" BIGINT NOT NULL,
  competition_id BIGINT,
  season_id BIGINT,

  "matchPeriod" TEXT,
  minute INT,
  second INT,
  "matchTimestamp" TEXT,
--   "videoTimestamp" DOUBLE PRECISION,
--   "relatedEventId" BIGINT,

  type_primary TEXT NOT NULL,
  type_secondary TEXT[],

  location_x REAL,
  location_y REAL,

  "team.id" BIGINT,
  "player.id" BIGINT,
  "possession.id" BIGINT,

--   "team.name" TEXT,
--   "player.name" TEXT,
  "player.position" TEXT,

  raw JSONB
);

-- PASS
CREATE TABLE IF NOT EXISTS eventstream_passes (
  id BIGINT PRIMARY KEY REFERENCES eventstream_events(id) ON DELETE CASCADE,
  pass_accurate BOOLEAN,
  pass_angle REAL,
  pass_height TEXT,
  pass_length REAL,
  pass_end_x REAL,
  pass_end_y REAL
--   pass_recipient JSONB
);

-- CARRY
CREATE TABLE IF NOT EXISTS eventstream_carries (
  id BIGINT PRIMARY KEY REFERENCES eventstream_events(id) ON DELETE CASCADE,
  carry_progression REAL,
  carry_end_x REAL,
  carry_end_y REAL
);

-- SHOT
CREATE TABLE IF NOT EXISTS eventstream_shots (
  id BIGINT PRIMARY KEY REFERENCES eventstream_events(id) ON DELETE CASCADE,
  "shot_isGoal" BOOLEAN,
  "shot_onTarget" BOOLEAN,
  "shot_goalZone" TEXT,
  shot_xg REAL,
  "shot_postShotXg" REAL,
  "shot_bodyPart" TEXT
);

-- AERIAL DUEL
CREATE TABLE IF NOT EXISTS eventstream_aerial_duels (
  id BIGINT PRIMARY KEY REFERENCES eventstream_events(id) ON DELETE CASCADE,
  "aerialDuel_firstTouch" BOOLEAN,
  "aerialDuel_height" REAL
--   "aerialDuel_opponent" JSONB
);

-- GROUND DUEL
CREATE TABLE IF NOT EXISTS eventstream_ground_duels (
  id BIGINT PRIMARY KEY REFERENCES eventstream_events(id) ON DELETE CASCADE,
  "groundDuel_recoveredPossession" BOOLEAN,
  "groundDuel_keptPossession" BOOLEAN,
  "groundDuel_progressedWithBall" BOOLEAN,
  "groundDuel_stoppedProgress" BOOLEAN,
  "groundDuel_takeOn" BOOLEAN,
  "groundDuel_duelType" TEXT,
  "groundDuel_side" TEXT
--   groundDuel_opponent JSONB
);

-- Indexes for heatmap queries
CREATE INDEX IF NOT EXISTS idx_events_match_primary
  ON eventstream_events ("matchId", type_primary);

CREATE INDEX IF NOT EXISTS idx_events_player_primary
  ON eventstream_events ("player.id", type_primary);

CREATE INDEX IF NOT EXISTS idx_events_player_comp_season_primary
  ON eventstream_events ("player.id", competition_id, season_id, type_primary);

CREATE INDEX IF NOT EXISTS idx_events_location
  ON eventstream_events (location_x, location_y);

CREATE INDEX IF NOT EXISTS idx_events_secondary_gin
  ON eventstream_events USING GIN (type_secondary);
