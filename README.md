# Eventstream Database & API

This repository contains the database schemas, upsertion logic, and FastAPI application used to ingest, store, and serve eventstream-style soccer data.

The project is organized around three main components:
1. SQL schemas
2. Database initialization and upsertion scripts
3. A FastAPI backend for querying and visualization

---

## Repository Structure
├── sql/
│ ├── 01_bins_and_indexes.sql
│ └── ...
│
├── DB_multi/
│ ├── apply_schema.py
│ ├── upsert_events.py
│ ├── <initial_schema>.sql
│ └── ...
│
├── app/
│ ├── main.py
│ ├── routers/
│ ├── services/
│ └── ...
│
├── .gitignore
└── README.md


## `sql/`

Contains **raw SQL schemas and helper files**.

- Individual `.sql` files define tables, indexes, constraints, and views
- These files are intended to be reusable across environments
- Schemas here may be applied manually or via scripts in `DB_multi/`

---

## `DB_multi/`

Contains database setup and ingestion logic.

Key components:
- **Initial schema SQL**: baseline schema required to bootstrap the database
- **`apply_schema.py`**: applies schema files to a target database
- **Upsertion scripts**: handle inserting and updating event data across multiple tables

This folder is responsible for:
- Creating the database structure
- Ensuring schema consistency
- Performing idempotent upserts of eventstream data

---

## `app/`

FastAPI application exposing API endpoints on top of the database.

Responsibilities:
- Querying event, match, player, and team data
- Generating derived outputs (e.g. heatmaps, pass networks, possession chains)
- Serving images and structured JSON responses

Typical usage:
- Run locally during development
- Deploy as a backend service connected to a Postgres instance

- Run using: uvicorn app.main:app --reload --port 8000   

---

## Development Notes

- Certain local scripts, datasets, and database folders are intentionally ignored via `.gitignore`
- Database credentials and connection details should be provided via environment variables
- The backend assumes an existing, initialized database schema

---

## Getting Started (High-Level)

1. Initialize the database schema using scripts in `DB_multi/`
    - `python DB/apply_schema.py`
2. Ingest or upsert event data
    - `python DB/upsert_root_events.py`
3. Start the FastAPI application from `app/`
    - `uvicorn app.main:app --reload --port 8000`
4. Access API documentation via FastAPI’s `/docs` endpoint

---
