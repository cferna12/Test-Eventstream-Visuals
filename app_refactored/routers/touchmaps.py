from __future__ import annotations

from typing import List, Optional
from fastapi import APIRouter, Query

from app_refactored.db import POOL
from app_refactored.schemas.touchmaps import (
    PointsResponse,
    PassPointsResponse,
    CarryPointsResponse,
    ShotPointsResponse,
    GroundDuelPointsResponse,
    AerialDuelPointsResponse,
)
from app_refactored.utils.sql_filters import parse_bbox
from app_refactored.services.touchmaps import (
    get_points_service,
    get_pass_points_service,
    get_carry_points_service,
    get_shot_points_service,
    get_aerial_duel_points_service,
    get_ground_duel_points_service,
)

router = APIRouter(prefix="/v1/touchmaps", tags=["touchmaps"])


@router.get("/points", response_model=PointsResponse)
def get_points(
    player_id: int,
    match_id: Optional[int] = None,
    competition_id: Optional[int] = None,
    season_id: Optional[int] = None,
    primary: Optional[str] = Query(default=None, description="e.type_primary OR special token pass_reception"),
    secondary_any: Optional[List[str]] = Query(default=None),
    bbox: Optional[str] = None,
    limit: Optional[int] = None,
):
    bbox_t = parse_bbox(bbox)
    return get_points_service(
        pool=POOL,
        player_id=player_id,
        match_id=match_id,
        competition_id=competition_id,
        season_id=season_id,
        primary=primary,
        secondary_any=secondary_any,
        bbox=bbox_t,
        limit=limit,
    )


@router.get("/points/passes", response_model=PassPointsResponse)
def get_pass_points(
    player_id: int,
    match_id: Optional[int] = None,
    competition_id: Optional[int] = None,
    season_id: Optional[int] = None,
    secondary_any: Optional[List[str]] = Query(None),
    bbox: Optional[str] = None,
    limit: Optional[int] = None,
):
    bbox_t = parse_bbox(bbox)
    return get_pass_points_service(
        pool=POOL,
        player_id=player_id,
        match_id=match_id,
        competition_id=competition_id,
        season_id=season_id,
        secondary_any=secondary_any,
        bbox=bbox_t,
        limit=limit,
    )


@router.get("/points/carries", response_model=CarryPointsResponse)
def get_carry_points(
    player_id: int,
    match_id: Optional[int] = None,
    competition_id: Optional[int] = None,
    season_id: Optional[int] = None,
    bbox: Optional[str] = None,
    limit: Optional[int] = None,
):
    bbox_t = parse_bbox(bbox)
    return get_carry_points_service(
        pool=POOL,
        player_id=player_id,
        match_id=match_id,
        competition_id=competition_id,
        season_id=season_id,
        bbox=bbox_t,
        limit=limit,
    )


@router.get("/points/shots", response_model=ShotPointsResponse)
def get_shot_points(
    player_id: int,
    match_id: Optional[int] = None,
    competition_id: Optional[int] = None,
    season_id: Optional[int] = None,
    secondary_any: Optional[List[str]] = Query(None),
    bbox: Optional[str] = None,
    limit: Optional[int] = None,
):
    bbox_t = parse_bbox(bbox)
    return get_shot_points_service(
        pool=POOL,
        player_id=player_id,
        match_id=match_id,
        competition_id=competition_id,
        season_id=season_id,
        secondary_any=secondary_any,
        bbox=bbox_t,
        limit=limit,
    )


@router.get("/points/aerial-duels", response_model=AerialDuelPointsResponse)
def get_aerial_duel_points(
    player_id: int,
    match_id: Optional[int] = None,
    competition_id: Optional[int] = None,
    season_id: Optional[int] = None,
    bbox: Optional[str] = None,
    limit: Optional[int] = None,
):
    bbox_t = parse_bbox(bbox)
    return get_aerial_duel_points_service(
        pool=POOL,
        player_id=player_id,
        match_id=match_id,
        competition_id=competition_id,
        season_id=season_id,
        bbox=bbox_t,
        limit=limit,
    )


@router.get("/points/ground_duels", response_model=GroundDuelPointsResponse)
def get_ground_duel_points(
    player_id: int,
    match_id: Optional[int] = None,
    competition_id: Optional[int] = None,
    season_id: Optional[int] = None,
    secondary_any: Optional[List[str]] = Query(None),
    bbox: Optional[str] = None,
    limit: Optional[int] = None,
):
    bbox_t = parse_bbox(bbox)
    return get_ground_duel_points_service(
        pool=POOL,
        player_id=player_id,
        match_id=match_id,
        competition_id=competition_id,
        season_id=season_id,
        secondary_any=secondary_any,
        bbox=bbox_t,
        limit=limit,
    )
