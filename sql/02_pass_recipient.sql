-- migrations/02_pass_recipient.sql
-- not used since already handled in upsertion script
BEGIN;

ALTER TABLE eventstream_passes
  ADD COLUMN IF NOT EXISTS recipient_id BIGINT,
  ADD COLUMN IF NOT EXISTS recipient_name TEXT,
  ADD COLUMN IF NOT EXISTS recipient_position TEXT;

-- recipient-based queries (receptions, pass networks)
CREATE INDEX IF NOT EXISTS idx_pass_recipient_id
  ON eventstream_passes (recipient_id);

-- pass edges: (passer -> recipient) often filters by passer in events, recipient in passes
CREATE INDEX IF NOT EXISTS idx_pass_recipient_event
  ON eventstream_passes (recipient_id, event_id);

CREATE INDEX IF NOT EXISTS idx_ev_match_player_primary
  ON eventstream_events (match_id, player_id, type_primary);    

COMMIT;
