from pathlib import Path
import psycopg
import os

# DSN example:
# postgresql://user:password@host:5432/dbname
dsn = "postgresql://postgres:postgres@localhost:5432/postgres"
DSN = os.environ.get("PG_DSN", dsn)
if not DSN:
    raise RuntimeError("Set PG_DSN env var")

schema_path = "DB/multi_table_schema.sql"
schema_sql = Path(schema_path).read_text(encoding="utf-8")

with psycopg.connect(DSN) as conn:
    with conn.cursor() as cur:
        cur.execute(schema_sql)
    conn.commit()

print("✅ Schema applied successfully")
