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

from app.routers.touchmaps import router as touchmaps_router
from app.routers.heatmaps import router as heatmaps_router
from app.routers.possessions import router as possessions_router
from app.routers.passnetwork_image import router as passnetwork_router

app.include_router(touchmaps_router)
app.include_router(heatmaps_router)
app.include_router(possessions_router)
app.include_router(passnetwork_router)
