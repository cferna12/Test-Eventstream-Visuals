"""
Apply a SQL schema file to a Postgres database.

Usage:
    python apply_schema.py path/to/schema.sql

Environment variables:
    PG_DSN - Postgres connection string
"""

from pathlib import Path
import argparse
import os
import psycopg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply a SQL schema file to a Postgres database"
    )
    parser.add_argument(
        "schema_path",
        type=Path,
        help="Path to the SQL schema file to apply",
    )
    args = parser.parse_args()

    # DSN example:
    # postgresql://user:password@host:5432/dbname
    default_dsn = "postgresql://postgres:postgres@localhost:5432/postgres"
    dsn = os.environ.get("PG_DSN", default_dsn)

    if not dsn:
        raise RuntimeError("Set PG_DSN environment variable")

    if not args.schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {args.schema_path}")

    schema_sql = args.schema_path.read_text(encoding="utf-8")

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(schema_sql)
        conn.commit()

    print(f"✅ Schema applied successfully: {args.schema_path}")


if __name__ == "__main__":
    main()
