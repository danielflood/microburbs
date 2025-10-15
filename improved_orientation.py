# v1: Bearing from house point to nearest road
# pip install requests shapely pyproj

import requests
from shapely.geometry import Point, LineString, shape
from pyproj import Transformer
from math import atan2, degrees

UA = {"User-Agent": "house-orientation-v1"}

def geocode(address: str):
    r = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": address, "format": "json", "limit": 1},
        headers=UA,
        timeout=20,
    )
    r.raise_for_status()
    j = r.json()
    if not j:
        raise ValueError("Address not found")
    return float(j[0]["lon"]), float(j[0]["lat"])

def overpass(query: str):
    r = requests.post(
        "https://overpass-api.de/api/interpreter",
        data={"data": query},
        headers=UA,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()

def fetch_nearby_roads(lon: float, lat: float, radius_m: int = 100):
    q = f"""
    [out:json][timeout:25];
    way(around:{radius_m},{lat},{lon})[highway];
    out geom;
    """
    data = overpass(q)
    roads = []
    for el in data.get("elements", []):
        if "geometry" not in el:
            continue
        coords = [(pt["lon"], pt["lat"]) for pt in el["geometry"]]
        if len(coords) >= 2:
            roads.append(LineString(coords))
    return roads

_transformer_fwd = Transformer.from_crs("EPSG:4326","EPSG:3857", always_xy=True)

def to_3857_xy(lon: float, lat: float):
    return _transformer_fwd.transform(lon, lat)

def bearing_from_point_to_point(px, py, qx, qy):
    # x = Easting, y = Northing (meters). Bearing: 0°=North, 90°=East.
    dx, dy = (qx - px), (qy - py)
    theta = degrees(atan2(dy, dx))      # angle from +x (east), CCW
    return (90 - theta) % 360

def bearing_to_compass(bearing_deg):
    dirs = ["N","NE","E","SE","S","SW","W","NW"]
    idx = int((bearing_deg + 22.5) // 45) % 8
    return dirs[idx]

def nearest_road_bearing_from_address(address: str, search_radius_m: int = 120):
    lon, lat = geocode(address)
    roads = fetch_nearby_roads(lon, lat, search_radius_m)
    if not roads:
        return {"lat": lat, "lon": lon, "bearing_deg": None, "note": "No roads found nearby"}

    # Project to meters for robust nearest-point math
    px, py = to_3857_xy(lon, lat)
    p = Point(px, py)

    # Find nearest road and nearest point on that road
    best = None
    best_dist = float("inf")
    for road_ll in roads:
        # project road to 3857
        road_xy = LineString([to_3857_xy(x, y) for (x, y) in road_ll.coords])
        d = p.distance(road_xy)
        if d < best_dist:
            best_dist = d
            best = road_xy

    q = best.interpolate(best.project(p))  # nearest point on road to the house point
    qx, qy = q.x, q.y

    bearing = bearing_from_point_to_point(px, py, qx, qy)
    return {
        "lat": lat,
        "lon": lon,
        "bearing_deg": round(bearing, 2),
        "compass": bearing_to_compass(bearing),
        "distance_to_road_m": round(best_dist, 2),
    }

# Run:
print(nearest_road_bearing_from_address("3 David St, St Kilda East VIC 3183"))
