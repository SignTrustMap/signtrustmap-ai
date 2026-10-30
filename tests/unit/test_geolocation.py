import math
from pathlib import Path
from typing import Dict, Any, List
import pytest
from loguru import logger
from src.evaluation.geolocation import (
    calculate_bearing,
    great_circle_destination,
    haversine,
    run_capstone_triangulation,
)


def test_ai_geo_01_haversine_distance():
    assert haversine(10.7769, 106.7009, 10.7769, 106.7009) == 0.0

    dist = haversine(0.0, 0.0, 1.0, 0.0)
    assert abs(dist - 111195.0) < 500.0
    logger.info(f"AI-GEO-01 verified: Haversine computes accurate ground distance ({dist:.1f}m).")


def test_ai_geo_02_calculate_bearing():
    bearing_north = calculate_bearing(0.0, 0.0, 1.0, 0.0)
    assert abs(bearing_north - 0.0) < 1e-4

    bearing_east = calculate_bearing(0.0, 0.0, 0.0, 1.0)
    assert abs(bearing_east - 90.0) < 1e-4

    bearing_south = calculate_bearing(1.0, 0.0, 0.0, 0.0)
    assert abs(bearing_south - 180.0) < 1e-4

    bearing_west = calculate_bearing(0.0, 1.0, 0.0, 0.0)
    assert abs(bearing_west - 270.0) < 1e-4
    logger.info("AI-GEO-02 verified: Cardinal bearings correctly calculated.")


def test_ai_geo_03_great_circle_destination():
    dest_lat, dest_lon = great_circle_destination(0.0, 0.0, 111195.0, 90.0)
    assert abs(dest_lat - 0.0) < 1e-3
    assert abs(dest_lon - 1.0) < 1e-2
    logger.info("AI-GEO-03 verified: Great-circle destination coordinates match geodesic projection.")


def test_ai_geo_04_triangulation_minimum_observations():
    insufficient = [{"c_lat": 10.0, "c_lon": 106.0, "heading": 0.0, "box_xyxy": [0, 0, 10, 10]}]
    lat, lon, dist, height = run_capstone_triangulation(insufficient, focal_px=800.0)
    assert lat is None
    assert lon is None
    assert dist is None
    assert height is None
    logger.info("AI-GEO-04 verified: Triangulation returns None when observations < 2.")

