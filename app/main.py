# # app/main.py
# from fastapi import FastAPI
# from app.routers.touchmaps_v2 import router as touchmaps_router_v2
# from app.routers.heatmaps import router as heatmaps_router
# app = FastAPI(title="Eventstream Heatmaps", version="0.1.0")

# app.include_router(touchmaps_router_v2)
# app.include_router(heatmaps_router)


# @app.get("/health")
# def health():
#     return {"ok": True}


# app/main.py
# from contextlib import asynccontextmanager
# from fastapi import FastAPI
# import anyio

# from app.db import POOL

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     try:
#         POOL.open()  # start workers

#         # wait for min_size connections to be established
#         await anyio.to_thread.run_sync(lambda: POOL.wait(timeout=30))

#         # optional: lightweight health check AFTER wait
#         def _ping():
#             with POOL.connection() as conn:
#                 with conn.cursor() as cur:
#                     cur.execute("SELECT 1;")
#                     cur.fetchone()
#         await anyio.to_thread.run_sync(_ping)

#     except Exception as e:
#         POOL.close()
#         raise RuntimeError(f"DB pool failed at startup: {e}") from e

#     yield

#     POOL.close()

# app = FastAPI(title="Eventstream Heatmaps (POC)", version="0.1.0", lifespan=lifespan)

# from app.routers.touchmaps_v3 import router as touchmaps_router_v3
# from app.routers.heatmaps import router as heatmaps_router

# app.include_router(touchmaps_router_v3)
# app.include_router(heatmaps_router)


from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.db import POOL

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Don't touch the DB here. Let endpoints be the first to connect.
    yield
    try:
        POOL.close()
    except Exception:
        pass

app = FastAPI(title="Eventstream Heatmaps (POC)", version="0.1.0", lifespan=lifespan)

from app.routers.touchmaps_v3 import router as touchmaps_router_v3
from app.routers.heatmaps import router as heatmaps_router

app.include_router(touchmaps_router_v3)
app.include_router(heatmaps_router)

