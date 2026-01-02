BEGIN;

-- 1) Add STORED generated bin columns (fast to group by)
--    Assumes location_x/y are 0..100
--    60x40 grid
ALTER TABLE eventstream_events
  ADD COLUMN IF NOT EXISTS bin_x SMALLINT
  GENERATED ALWAYS AS (
    LEAST(
      59,
      GREATEST(0, FLOOR((location_x / 100.0) * 60))
    )::SMALLINT
  ) STORED;

ALTER TABLE eventstream_events
  ADD COLUMN IF NOT EXISTS bin_y SMALLINT
  GENERATED ALWAYS AS (
    LEAST(
      39,
      GREATEST(0, FLOOR((location_y / 100.0) * 40))
    )::SMALLINT
  ) STORED;

-- 2) Replace idx_ev_location with something actually used for your predicates:
--    Most of your queries filter by (player/season/comp/match/primary) and then group by bins.
--    Partial index avoids NULL locations.
CREATE INDEX IF NOT EXISTS idx_ev_player_comp_season_primary_bins
  ON eventstream_events (player_id, competition_id, season_id, type_primary, bin_x, bin_y)
  WHERE location_x IS NOT NULL AND location_y IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_ev_player_match_primary_bins
  ON eventstream_events (player_id, match_id, type_primary, bin_x, bin_y)
  WHERE location_x IS NOT NULL AND location_y IS NOT NULL;

-- If you often query without primary (your default “on ball” list), this helps too:
CREATE INDEX IF NOT EXISTS idx_ev_player_comp_season_bins
  ON eventstream_events (player_id, competition_id, season_id, bin_x, bin_y)
  WHERE location_x IS NOT NULL AND location_y IS NOT NULL;

-- Your existing GIN is good for && overlap:
-- CREATE INDEX idx_ev_secondary_gin ON eventstream_events USING GIN (type_secondary);

-- Optional: reduce table bloat from jsonb-heavy rows in hot path (planner can use index-only scans more often)
-- (You already SELECT only location fields, so this is mostly about index coverage.)
-- No change required here.

COMMIT;

-- After large loads, do this once:
ANALYZE eventstream_events;
