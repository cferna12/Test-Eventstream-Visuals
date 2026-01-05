-- schema.sql (recommended: all-lower, snake_case)
BEGIN;

-- Base events table: thin common fields for heatmaps + joins
CREATE TABLE IF NOT EXISTS eventstream_events (
  id                BIGINT PRIMARY KEY, 

  match_id          BIGINT NOT NULL,
  competition_id    BIGINT,
  season_id         BIGINT,

  match_period      TEXT, 
  minute            INT,
  second            INT,
  match_timestamp   TEXT,
  video_timestamp   DOUBLE PRECISION, --  dont need

  related_event_id  BIGINT, -- dont need likely

  type_primary      TEXT NOT NULL,
  type_secondary    TEXT[],

  location_x        REAL,
  location_y        REAL,

  team_id           BIGINT,
  player_id         BIGINT,
  possession_id     BIGINT,

  -- optional labels (handy for debugging; not required for viz)
  team_name         TEXT, -- dont need
  player_name       TEXT, -- dont need
  player_position   TEXT,

  raw               JSONB --need? 
);

-- Action tables: 1:1 with event id (only rows where that action applies)
CREATE TABLE IF NOT EXISTS eventstream_passes (
  event_id        BIGINT PRIMARY KEY REFERENCES eventstream_events(id) ON DELETE CASCADE,
  accurate        BOOLEAN,
  end_x           REAL,
  end_y           REAL,
  -- maybe add length/angle
  -- add recipient
  recipient_id          BIGINT,
  recipient_name        TEXT,
  recipient_position    TEXT
);

CREATE TABLE IF NOT EXISTS eventstream_carries (
  event_id        BIGINT PRIMARY KEY REFERENCES eventstream_events(id) ON DELETE CASCADE,
  progression     REAL,
  end_x           REAL,
  end_y           REAL
);

CREATE TABLE IF NOT EXISTS eventstream_shots (
  event_id        BIGINT PRIMARY KEY REFERENCES eventstream_events(id) ON DELETE CASCADE,
  is_goal         BOOLEAN,
  on_target       BOOLEAN,
  goal_zone       TEXT,
  xg              REAL,
  body_part       TEXT
  --maybe post thot xg as well
);

CREATE TABLE IF NOT EXISTS eventstream_aerial_duels (
  event_id        BIGINT PRIMARY KEY REFERENCES eventstream_events(id) ON DELETE CASCADE,
  first_touch     BOOLEAN
);

CREATE TABLE IF NOT EXISTS eventstream_ground_duels (
  event_id               BIGINT PRIMARY KEY REFERENCES eventstream_events(id) ON DELETE CASCADE,
  recovered_possession   BOOLEAN,
  kept_possession        BOOLEAN
);

-- Indexes for your common heatmap queries

--- shohuldnt need this one
--- not sure which ones I actually need t
CREATE INDEX IF NOT EXISTS idx_ev_match_primary
  ON eventstream_events (match_id, type_primary);

CREATE INDEX IF NOT EXISTS idx_ev_player_primary
  ON eventstream_events (player_id, type_primary);


CREATE INDEX IF NOT EXISTS idx_ev_player_comp_season_primary
  ON eventstream_events (player_id, competition_id, season_id, type_primary);

CREATE INDEX IF NOT EXISTS idx_ev_location
  ON eventstream_events (location_x, location_y);

CREATE INDEX IF NOT EXISTS idx_ev_secondary_gin
  ON eventstream_events USING GIN (type_secondary);

  -- recipient-based queries (receptions, pass networks)
CREATE INDEX IF NOT EXISTS idx_pass_recipient_id
  ON eventstream_passes (recipient_id);

-- pass edges: (passer -> recipient) often filters by passer in events, recipient in passes
CREATE INDEX IF NOT EXISTS idx_pass_recipient_event
  ON eventstream_passes (recipient_id, event_id);

CREATE INDEX IF NOT EXISTS idx_ev_match_player_primary
  ON eventstream_events (match_id, player_id, type_primary);

COMMIT;
