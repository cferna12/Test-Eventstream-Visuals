# # app/db.py
# import os
# from psycopg_pool import ConnectionPool
# from psycopg.rows import tuple_row

# PG_DSN = os.environ.get("PG_DSN", "postgresql://postgres:postgres@localhost:5432/eventstream")

# POOL = ConnectionPool(
#     conninfo=PG_DSN,
#     min_size=1,
#     max_size=10,
#     timeout=10,
#     open=False,  # <-- important: don't start worker threads at import time
#     kwargs={
#         "row_factory": tuple_row,
#         "autocommit": True,
#     },
# )

import os
from psycopg_pool import ConnectionPool
from psycopg.rows import tuple_row

PG_DSN = os.environ.get("PG_DSN", "postgresql://postgres:postgres@localhost:5432/eventstream")

POOL = ConnectionPool(
    conninfo=PG_DSN,
    min_size=1,
    max_size=10,
    timeout=30,  # bump timeout just for test server stability
    kwargs={
        "row_factory": tuple_row,
        "autocommit": True,
    },
)
