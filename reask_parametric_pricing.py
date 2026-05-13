"""
reask_parametric_pricing.py

General-purpose parametric TC wind policy pricer using the Reask DeepCyc API.

Supports:
- Multiple shapes defined as GeoJSON geometry dicts, GeoJSON FeatureCollection
  files, or WKT strings
- Per-shape payout structures mapping hazard bands to payout percentages
- Aggregation of per-shape payouts per TC event as either MAX or SUM

Typical usage
-------------
See the ``__main__`` block at the bottom for a worked example with two
concentric squares where payout per event = max(inner_payout, outer_payout).
"""

from __future__ import annotations

import os
import numpy as np
import requests
import pandas as pd
import geojson
from shapely.geometry import shape as shapely_shape
from shapely import wkt as shapely_wkt
from typing import Literal


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def get_auth_token(username: str, password: str) -> str:
    """Obtain a Bearer token from the Reask auth endpoint."""
    resp = requests.post(
        "https://api.reask.earth/v2/token",
        data={"username": username, "password": password},
    )
    resp.raise_for_status()
    token = resp.json().get("access_token")
    if not token:
        raise RuntimeError("Authentication failed — check your credentials.")
    return token


def make_deepcyc_headers(token: str, version: str = "DeepCyc-2.0.7") -> dict:
    return {
        "accept": "application/json",
        "product-version": version,
        "Authorization": f"Bearer {token}",
    }


def make_metryc_headers(token: str, version: str = "Metryc-1.0.5") -> dict:
    return {
        "accept": "application/json",
        "product-version": version,
        "Authorization": f"Bearer {token}",
    }


# ---------------------------------------------------------------------------
# Shape utilities
# ---------------------------------------------------------------------------

def parse_geometry_to_coords(geometry) -> tuple[list[float], list[float]]:
    """
    Convert a GeoJSON geometry dict or a WKT string (Polygon) to
    (lat_list, lon_list) for the exterior ring.
    """
    if isinstance(geometry, str):
        geom = shapely_wkt.loads(geometry)
    elif isinstance(geometry, dict):
        geom = shapely_shape(geometry)
    else:
        raise TypeError(f"Unsupported geometry type: {type(geometry)}")

    if geom.geom_type != "Polygon":
        raise ValueError(f"Expected Polygon geometry, got {geom.geom_type}")

    coords = list(geom.exterior.coords)
    lon = [c[0] for c in coords]
    lat = [c[1] for c in coords]
    return lat, lon


def shapes_from_geojson_file(path: str, id_property: str = "shape_id") -> list[dict]:
    """
    Load shapes from a GeoJSON FeatureCollection file.

    Parameters
    ----------
    path : str
        Path to a GeoJSON file containing Polygon features.
    id_property : str
        Feature property to use as the shape identifier.

    Returns
    -------
    list of dicts, each with keys ``shape_id`` and ``geometry``
    (the GeoJSON geometry dict).
    """
    with open(path) as f:
        gj = geojson.load(f)

    shapes = []
    seen_ids: set = set()
    for feature in gj.get("features", []):
        if feature["geometry"]["type"] != "Polygon":
            continue
        sid = feature["properties"][id_property]
        if sid in seen_ids:
            raise ValueError(
                f"Duplicate shape_id '{sid}' found in {path}. "
                "All shape IDs must be unique."
            )
        seen_ids.add(sid)
        shapes.append({"shape_id": sid, "geometry": feature["geometry"]})
    return shapes


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_inputs(
    shapes: list[dict],
    payout_df: pd.DataFrame | None,
    peril: str = "wind_speed",
    units: str = "mph",
) -> list[dict]:
    """
    Run pre-flight checks on shapes and payout structures.

    Returns a list of issue dicts, each with keys:
        ``severity`` — "error" (blocks running) or "warning" (advisory only)
        ``check``    — short label used as a heading in the UI
        ``message``  — human-readable detail

    An empty list means no issues were found.
    """
    issues: list[dict] = []
    shape_ids = [s["shape_id"] for s in shapes]

    # ---- Duplicate shape IDs -------------------------------------------
    seen: set = set()
    dups: set = set()
    for sid in shape_ids:
        if sid in seen:
            dups.add(sid)
        else:
            seen.add(sid)
    if dups:
        issues.append({
            "severity": "error",
            "check": "Duplicate shape IDs",
            "message": (
                f"Shape IDs {sorted(dups)} appear more than once. "
                "Each shape must have a unique ID or results will be incorrect."
            ),
        })

    if payout_df is None or payout_df.empty:
        return issues

    # ---- Required payout columns ---------------------------------------
    required = {"shape_id", "from", "to", "payout_percentage"}
    missing_cols = required - set(payout_df.columns)
    if missing_cols:
        issues.append({
            "severity": "error",
            "check": "Missing payout columns",
            "message": f"Payout structure is missing required columns: {sorted(missing_cols)}.",
        })
        return issues  # can't safely run further checks

    payout_ids = set(payout_df["shape_id"].dropna().unique())
    shape_id_set = set(shape_ids)

    # ---- Shape ID / payout alignment -----------------------------------
    unmatched = payout_ids - shape_id_set
    if unmatched:
        issues.append({
            "severity": "warning",
            "check": "Unmatched payout shape IDs",
            "message": (
                f"Payout structure references shape IDs not in the loaded shapes: "
                f"{sorted(unmatched)}. These bands will never trigger."
            ),
        })

    no_payout = shape_id_set - payout_ids
    if no_payout:
        issues.append({
            "severity": "warning",
            "check": "Shapes without payout structure",
            "message": (
                f"Shape(s) {sorted(no_payout)} have no payout bands defined. "
                "They will always contribute zero to the EL."
            ),
        })

    # ---- Per-shape band checks -----------------------------------------
    _ws_ceilings = {"mph": 200.0, "kph": 320.0, "ms": 90.0, "knots": 174.0}

    df = payout_df.copy()
    df["from"] = pd.to_numeric(df["from"], errors="coerce")
    df["to"] = pd.to_numeric(df["to"], errors="coerce")

    for sid, group in df.groupby("shape_id"):
        g = group.dropna(subset=["from", "to"]).sort_values("from").reset_index(drop=True)
        if g.empty:
            continue

        froms = g["from"].values
        tos = g["to"].values

        for i in range(len(tos) - 1):
            if tos[i] > froms[i + 1]:
                issues.append({
                    "severity": "warning",
                    "check": "Overlapping bands",
                    "message": (
                        f"Shape '{sid}': band [{froms[i]:.1f}, {tos[i]:.1f}) overlaps "
                        f"[{froms[i+1]:.1f}, {tos[i+1]:.1f}). "
                        "Only the first matching band will fire."
                    ),
                })
            elif tos[i] < froms[i + 1]:
                issues.append({
                    "severity": "warning",
                    "check": "Gap in payout bands",
                    "message": (
                        f"Shape '{sid}': no band covers hazard values from "
                        f"{tos[i]:.1f} to {froms[i+1]:.1f} {units}. "
                        "Events in this range will trigger no payout."
                    ),
                })

        if peril == "wind_speed" and units in _ws_ceilings:
            threshold = _ws_ceilings[units]
            max_to = float(tos.max())
            if max_to < threshold:
                issues.append({
                    "severity": "warning",
                    "check": "Band ceiling too low",
                    "message": (
                        f"Shape '{sid}': top band ceiling is {max_to:.0f} {units}. "
                        f"Storms above this threshold will trigger no payout. "
                        f"Consider raising the ceiling to at least {threshold:.0f} {units}."
                    ),
                })

    return issues


# ---------------------------------------------------------------------------
# DeepCyc API calls
# ---------------------------------------------------------------------------

_DEEPCYC_ENDPOINTS = {
    "wind_speed":       "https://api.reask.earth/v2/deepcyc/tctrack/wind_speed/events",
    "central_pressure": "https://api.reask.earth/v2/deepcyc/tctrack/central_pressure/events",
}


def fetch_events_for_polygon(
    lat: list[float],
    lon: list[float],
    headers: dict,
    peril: Literal["wind_speed", "central_pressure"] = "wind_speed",
    scenario: str = "current_climate",
    time_horizon: str = "now",
    terrain_correction: str = "open_water",
    wind_speed_units: str = "mph",
    wind_speed_averaging_period: str = "1_minute",
    simulation_years: int = 41000,
) -> list[dict]:
    """
    Call the DeepCyc stochastic events endpoint for a single polygon.

    Returns a list of flat property dicts; the hazard value is stored under
    the key ``hazard`` regardless of peril.
    """
    if peril not in _DEEPCYC_ENDPOINTS:
        raise ValueError(f"peril must be one of {list(_DEEPCYC_ENDPOINTS)}")

    params = {
        "scenario": scenario,
        "time_horizon": time_horizon,
        "terrain_correction": terrain_correction,
        "lat": lat,
        "lon": lon,
        "geometry": "polygon",
        "radius_km": 0,
        "wind_speed_units": wind_speed_units,
        "accurate_flag": "true",
        "map_projection_used_for_geometric_calculations": "WGS84",
        "wind_speed_averaging_period": wind_speed_averaging_period,
        "simulation_years": simulation_years,
    }

    resp = requests.get(_DEEPCYC_ENDPOINTS[peril], params=params, headers=headers)
    print(f"    [{peril}] HTTP {resp.status_code}")
    resp.raise_for_status()
    data = resp.json()

    if "error" in data or "Error" in data:
        msg = data.get("error") or data.get("Error") or str(data)
        raise RuntimeError(f"DeepCyc API error (peril={peril}): {msg}")

    rows = []
    for event in data.get("features", []):
        props = event.get("properties", {}).copy()
        props["coordinates"] = event.get("geometry", {}).get("coordinates")
        props["peril"] = peril
        # Normalise to a single ``hazard`` key
        if peril == "wind_speed":
            props["hazard"] = props.pop("wind_speed", None)
        else:
            props["hazard"] = props.pop("central_pressure", None)
        rows.append(props)

    return rows


def fetch_events_for_all_shapes(
    shapes: list[dict],
    headers: dict,
    perils: list[str] = ("wind_speed",),
    **api_kwargs,
) -> pd.DataFrame:
    """
    Iterate over every shape and peril, call the DeepCyc API, and return a
    combined DataFrame with a ``shape_id`` column.

    Parameters
    ----------
    shapes : list of dicts
        Each dict must have keys ``shape_id`` and ``geometry``
        (GeoJSON geometry dict or WKT string).
    headers : dict
        DeepCyc authorisation headers from ``make_deepcyc_headers``.
    perils : list of str
        Which perils to query.  Defaults to ``["wind_speed"]``.
    **api_kwargs
        Forwarded to ``fetch_events_for_polygon`` (e.g. ``simulation_years``).
    """
    all_rows: list[dict] = []
    for shape_def in shapes:
        shape_id = shape_def["shape_id"]
        print(f"Fetching events for shape: {shape_id}")
        lat, lon = parse_geometry_to_coords(shape_def["geometry"])
        for peril in perils:
            rows = fetch_events_for_polygon(lat, lon, headers, peril=peril, **api_kwargs)
            for r in rows:
                r["shape_id"] = shape_id
            all_rows.extend(rows)

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# DeepCyc API calls — circle (point + radius) geometry
# ---------------------------------------------------------------------------

def fetch_events_for_circle(
    center_lat: float,
    center_lon: float,
    radius_km: float,
    headers: dict,
    peril: Literal["wind_speed", "central_pressure"] = "wind_speed",
    scenario: str = "current_climate",
    time_horizon: str = "now",
    terrain_correction: str = "open_water",
    wind_speed_units: str = "mph",
    wind_speed_averaging_period: str = "1_minute",
    simulation_years: int = 41000,
) -> list[dict]:
    """
    Call the DeepCyc stochastic events endpoint for a point + radius circle.

    Passes ``geometry=circle`` with a scalar centre lat/lon and ``radius_km``,
    which is the correct Reask format for circular exposure zones.
    Returns the same list-of-dicts format as ``fetch_events_for_polygon``,
    with a normalised ``hazard`` key.
    """
    if peril not in _DEEPCYC_ENDPOINTS:
        raise ValueError(f"peril must be one of {list(_DEEPCYC_ENDPOINTS)}")

    params = {
        "scenario": scenario,
        "time_horizon": time_horizon,
        "terrain_correction": terrain_correction,
        "lat": center_lat,
        "lon": center_lon,
        "geometry": "circle",
        "radius_km": radius_km,
        "wind_speed_units": wind_speed_units,
        "accurate_flag": "true",
        "map_projection_used_for_geometric_calculations": "WGS84",
        "wind_speed_averaging_period": wind_speed_averaging_period,
        "simulation_years": simulation_years,
    }

    resp = requests.get(_DEEPCYC_ENDPOINTS[peril], params=params, headers=headers)
    print(f"    [{peril}] HTTP {resp.status_code}")
    resp.raise_for_status()
    data = resp.json()

    if "error" in data or "Error" in data:
        msg = data.get("error") or data.get("Error") or str(data)
        raise RuntimeError(f"DeepCyc API error (peril={peril}): {msg}")

    rows = []
    for event in data.get("features", []):
        props = event.get("properties", {}).copy()
        props["coordinates"] = event.get("geometry", {}).get("coordinates")
        props["peril"] = peril
        if peril == "wind_speed":
            props["hazard"] = props.pop("wind_speed", None)
        else:
            props["hazard"] = props.pop("central_pressure", None)
        rows.append(props)

    return rows


def fetch_events_for_all_circles(
    circles: list[dict],
    headers: dict,
    perils: list[str] = ("wind_speed",),
    **api_kwargs,
) -> pd.DataFrame:
    """
    Iterate over every circle and peril, call the DeepCyc API, and return a
    combined DataFrame with a ``shape_id`` column.

    Parameters
    ----------
    circles : list of dicts
        Each dict must have keys ``shape_id``, ``lat``, ``lon``, ``radius_km``.
    headers : dict
        DeepCyc authorisation headers from ``make_deepcyc_headers``.
    perils : list of str
    **api_kwargs
        Forwarded to ``fetch_events_for_circle`` (e.g. ``simulation_years``).
    """
    all_rows: list[dict] = []
    for circle in circles:
        shape_id = circle["shape_id"]
        print(f"Fetching events for circle: {shape_id} ({circle['radius_km']} km radius)")
        for peril in perils:
            rows = fetch_events_for_circle(
                circle["lat"], circle["lon"], circle["radius_km"],
                headers, peril=peril, **api_kwargs,
            )
            for r in rows:
                r["shape_id"] = shape_id
            all_rows.extend(rows)

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Payout structure
# ---------------------------------------------------------------------------

def build_interval_lookup(payout_df: pd.DataFrame) -> dict:
    """
    Pre-build a per-shape interval index for fast hazard → payout lookup.

    Parameters
    ----------
    payout_df : DataFrame
        Required columns: ``shape_id``, ``from``, ``to``, ``payout_percentage``.
        Optional column: ``category`` (descriptive band label).
        Intervals are treated as left-closed, right-open: [from, to).

    Returns
    -------
    dict mapping shape_id → {intervals, payouts, categories}
    """
    df = payout_df.copy()
    df["from"] = df["from"].astype(float)
    df["to"] = df["to"].astype(float)
    df["payout_percentage"] = pd.to_numeric(df["payout_percentage"], errors="coerce")

    lookup: dict = {}
    for shape_id, group in df.groupby("shape_id"):
        lookup[shape_id] = {
            "intervals": pd.IntervalIndex.from_arrays(
                group["from"], group["to"], closed="left"
            ),
            "payouts": group["payout_percentage"].values,
            "categories": group["category"].values if "category" in group.columns else None,
        }
    return lookup


def lookup_payout(shape_id, hazard, interval_lookup: dict) -> tuple:
    """
    Return (category, payout_percentage) for a given shape and hazard value.
    Returns (None, None) if the shape is unknown or the hazard falls outside
    all defined bands.
    """
    if shape_id not in interval_lookup:
        return None, None
    entry = interval_lookup[shape_id]
    idx = entry["intervals"].get_indexer([hazard])[0]
    if idx == -1:
        return None, None
    category = entry["categories"][idx] if entry["categories"] is not None else None
    return category, entry["payouts"][idx]


# ---------------------------------------------------------------------------
# Expected loss calculation
# ---------------------------------------------------------------------------

def calculate_el(
    events_df: pd.DataFrame,
    interval_lookup: dict,
    aggregation: Literal["max", "sum"] = "max",
    simulation_years: int = 41000,
    event_limit: float = 1.0,
    annual_limit: float = 1.0,
) -> dict:
    """
    Calculate the Expected Loss (EL) for the policy.

    Aggregation logic
    -----------------
    For each stochastic TC event that intersects one or more shapes:

    1. Within each (shape, event): take the **maximum** payout across any
       duplicate hazard readings (preserves original behaviour).
    2. Across shapes within the same event:
       - ``"max"``: policy pays the highest single-shape payout
         (suitable for concentric / overlapping shapes).
       - ``"sum"``: policy pays the sum of all triggered shape payouts,
         capped at ``event_limit`` (suitable for independent non-overlapping
         shapes; cap prevents a single event from paying out more than 100 %).
    3. Within each year: sum all event payouts, then cap at ``annual_limit``.
    4. EL = sum of capped annual payouts / simulation_years.

    Parameters
    ----------
    events_df : DataFrame
        Output of ``fetch_events_for_all_shapes`` — must contain columns
        ``shape_id``, ``year_id``, ``event_id``, ``hazard``.
    interval_lookup : dict
        Output of ``build_interval_lookup``.
    aggregation : "max" or "sum"
    simulation_years : int
    event_limit : float
        Maximum payout per individual TC event as a fraction of insured value
        (default 1.0 = 100 %). Only applied when aggregation is ``"sum"``;
        with ``"max"`` the per-event payout is always ≤ the highest band value.
    annual_limit : float
        Maximum annual payout as a fraction of insured value (default 1.0 = 100 %).

    Returns
    -------
    dict with keys:
        ``el``       — scalar expected loss
        ``ylt``      — year loss table DataFrame (year_id, payout, payout_capped)
        ``event_lt`` — event loss table DataFrame (year_id, event_id, payout,
                       payout_capped)
        ``detail``   — full events DataFrame with band/payout columns populated
    """
    if aggregation not in ("max", "sum"):
        raise ValueError("aggregation must be 'max' or 'sum'")

    required_cols = {"shape_id", "year_id", "event_id", "hazard"}
    if events_df.empty or not required_cols.issubset(events_df.columns):
        return {
            "el": 0.0,
            "ylt": pd.DataFrame(columns=["year_id", "payout", "payout_capped"]),
            "event_lt": pd.DataFrame(columns=["year_id", "event_id", "payout", "payout_capped"]),
            "detail": events_df,
        }

    df = events_df.copy()
    df[["category", "payout"]] = df.apply(
        lambda row: pd.Series(
            lookup_payout(row["shape_id"], row["hazard"], interval_lookup)
        ),
        axis=1,
    )
    df = df.dropna(subset=["payout"])

    # Step 1 — within each (shape, event) keep the max payout
    per_shape_event = (
        df.groupby(["shape_id", "year_id", "event_id"])["payout"]
        .max()
        .reset_index()
    )

    # Step 2 — aggregate across shapes per event
    if aggregation == "max":
        event_lt = (
            per_shape_event.groupby(["year_id", "event_id"])["payout"]
            .max()
            .reset_index()
        )
        event_lt["payout_capped"] = event_lt["payout"]
    else:
        event_lt = (
            per_shape_event.groupby(["year_id", "event_id"])["payout"]
            .sum()
            .reset_index()
        )
        event_lt["payout_capped"] = event_lt["payout"].clip(upper=event_limit)

    # Step 3 — annual aggregation with cap
    ylt = event_lt.groupby("year_id")["payout_capped"].sum().reset_index()
    ylt = ylt.rename(columns={"payout_capped": "payout"})
    ylt["payout_capped"] = ylt["payout"].clip(upper=annual_limit)

    el = ylt["payout_capped"].sum() / simulation_years

    # Step 4 — add payout_effective: each event's real contribution after the
    # running year total is applied.  Unlike payout_capped (which only applies
    # the per-event cap), payout_effective answers "how much of the annual limit
    # did this event actually consume, given what had already been paid out
    # earlier in the same simulated year?"
    #
    # Example: annual_limit = 100%, two events in year 5000 both at 75%.
    #   event 1: cumulative before = 0%  → remaining = 100% → effective = 75%
    #   event 2: cumulative before = 75% → remaining = 25%  → effective = 25%
    #   year total = 100%  ✓
    #
    # Ordering within a year is by event_id (arbitrary but consistent).
    elt = event_lt.sort_values(["year_id", "event_id"]).copy()
    cumul_before = elt.groupby("year_id")["payout_capped"].cumsum() - elt["payout_capped"]
    remaining = (annual_limit - cumul_before).clip(lower=0)
    elt["payout_effective"] = np.minimum(elt["payout_capped"].values, remaining.values)

    return {
        "el": el,
        "ylt": ylt,
        "event_lt": elt,
        "detail": df,
    }


# ---------------------------------------------------------------------------
# Metryc — historic event data
# ---------------------------------------------------------------------------

_METRYC_BASE = "https://api.reask.earth/v2/metryc/historical"


def fetch_historic_events_for_polygon(
    lat: list[float],
    lon: list[float],
    headers: dict,
    wind_speed_units: str = "mph",
    terrain_correction: str = "open_water",
    wind_speed_averaging_period: str = "1_minute",
) -> list[dict]:
    """
    Call the Metryc historical tctrack wind speed events endpoint for a single
    polygon.  Returns a list of flat property dicts with a normalised ``hazard``
    key, mirroring the structure returned by ``fetch_events_for_polygon``.
    """
    params = {
        "lat": lat,
        "lon": lon,
        "geometry": "polygon",
        "radius_km": 0,
        "wind_speed_units": wind_speed_units,
        "terrain_correction": terrain_correction,
        "wind_speed_averaging_period": wind_speed_averaging_period,
        "accurate_flag": "true",
    }
    resp = requests.get(
        f"{_METRYC_BASE}/tctrack/wind_speed/events", params=params, headers=headers
    )
    print(f"    [metryc wind_speed] HTTP {resp.status_code}")
    resp.raise_for_status()
    data = resp.json()

    if "error" in data or "Error" in data:
        msg = data.get("error") or data.get("Error") or str(data)
        raise RuntimeError(f"Metryc API error: {msg}")

    rows = []
    for event in data.get("features", []):
        props = event.get("properties", {}).copy()
        props["coordinates"] = event.get("geometry", {}).get("coordinates")
        props["hazard"] = props.pop("wind_speed", None)
        rows.append(props)
    return rows


def fetch_historic_events_for_all_shapes(
    shapes: list[dict],
    headers: dict,
    wind_speed_units: str = "mph",
    terrain_correction: str = "open_water",
    wind_speed_averaging_period: str = "1_minute",
) -> pd.DataFrame:
    """
    Iterate over every shape, call the Metryc historical events API, and return
    a combined DataFrame with a ``shape_id`` column.

    The DataFrame contains ``storm_name``, ``storm_year``, ``storm_id``,
    ``event_id``, ``hazard``, and ``shape_id``.
    """
    all_rows: list[dict] = []
    for shape_def in shapes:
        shape_id = shape_def["shape_id"]
        print(f"Fetching historic events for shape: {shape_id}")
        lat, lon = parse_geometry_to_coords(shape_def["geometry"])
        rows = fetch_historic_events_for_polygon(
            lat, lon, headers,
            wind_speed_units=wind_speed_units,
            terrain_correction=terrain_correction,
            wind_speed_averaging_period=wind_speed_averaging_period,
        )
        print(f"  Found {len(rows)} historic events")
        for r in rows:
            r["shape_id"] = shape_id
        all_rows.extend(rows)

    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame(
        columns=["storm_name", "storm_year", "storm_id", "event_id",
                 "hazard", "shape_id", "coordinates"]
    )


def fetch_historic_events_for_circle(
    center_lat: float,
    center_lon: float,
    radius_km: float,
    headers: dict,
    wind_speed_units: str = "mph",
    terrain_correction: str = "open_water",
    wind_speed_averaging_period: str = "1_minute",
) -> list[dict]:
    """
    Call the Metryc historical events endpoint for a point + radius circle.
    Passes ``geometry=circle`` with a scalar centre lat/lon and ``radius_km``.
    Returns the same format as ``fetch_historic_events_for_polygon``.
    """
    params = {
        "lat": center_lat,
        "lon": center_lon,
        "geometry": "circle",
        "radius_km": radius_km,
        "wind_speed_units": wind_speed_units,
        "terrain_correction": terrain_correction,
        "wind_speed_averaging_period": wind_speed_averaging_period,
        "accurate_flag": "true",
    }
    resp = requests.get(
        f"{_METRYC_BASE}/tctrack/wind_speed/events", params=params, headers=headers
    )
    print(f"    [metryc wind_speed circle] HTTP {resp.status_code}")
    resp.raise_for_status()
    data = resp.json()

    if "error" in data or "Error" in data:
        msg = data.get("error") or data.get("Error") or str(data)
        raise RuntimeError(f"Metryc API error: {msg}")

    rows = []
    for event in data.get("features", []):
        props = event.get("properties", {}).copy()
        props["coordinates"] = event.get("geometry", {}).get("coordinates")
        props["hazard"] = props.pop("wind_speed", None)
        rows.append(props)
    return rows


def fetch_historic_events_for_all_circles(
    circles: list[dict],
    headers: dict,
    wind_speed_units: str = "mph",
    terrain_correction: str = "open_water",
    wind_speed_averaging_period: str = "1_minute",
) -> pd.DataFrame:
    """
    Iterate over every circle definition, call the Metryc historical events API,
    and return a combined DataFrame with a ``shape_id`` column.
    """
    all_rows: list[dict] = []
    for circle in circles:
        shape_id = circle["shape_id"]
        print(f"Fetching historic events for circle: {shape_id}")
        rows = fetch_historic_events_for_circle(
            circle["lat"], circle["lon"], circle["radius_km"],
            headers,
            wind_speed_units=wind_speed_units,
            terrain_correction=terrain_correction,
            wind_speed_averaging_period=wind_speed_averaging_period,
        )
        print(f"  Found {len(rows)} historic events")
        for r in rows:
            r["shape_id"] = shape_id
        all_rows.extend(rows)

    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame(
        columns=["storm_name", "storm_year", "storm_id", "event_id",
                 "hazard", "shape_id", "coordinates"]
    )


def fetch_storm_track(storm_id: str, headers: dict) -> list[dict]:
    """
    Fetch the full track (time-series of geographic points) for a given
    storm_id from the Metryc historical tctrack/points endpoint.

    Returns a list of dicts with keys:
        ``lon``, ``lat``, ``iso_time``, ``wind_speed``, ``category``,
        ``track_name``, ``storm_id``.
    """
    resp = requests.get(
        f"{_METRYC_BASE}/tctrack/points",
        params={"storm_id": storm_id},
        headers=headers,
    )
    resp.raise_for_status()
    data = resp.json()

    points = []
    for feat in data.get("features", []):
        props = feat.get("properties", {})
        coords = feat.get("geometry", {}).get("coordinates", [None, None])
        if coords and len(coords) >= 2 and None not in coords:
            points.append(
                {
                    "storm_id": storm_id,
                    "lon": coords[0],
                    "lat": coords[1],
                    "iso_time": props.get("iso_time"),
                    "wind_speed": props.get("wind_speed"),
                    "category": props.get("category"),
                    "track_name": props.get("track_name"),
                }
            )
    return points


def calculate_historic_payouts(
    historic_events_df: pd.DataFrame,
    interval_lookup: dict,
    aggregation: Literal["max", "sum"] = "max",
    event_limit: float = 1.0,
    annual_limit: float = 1.0,
) -> dict:
    """
    Apply payout structures to historic Metryc events and compute a historical
    expected loss rate.

    Parameters
    ----------
    historic_events_df : DataFrame
        Output of ``fetch_historic_events_for_all_shapes`` — must contain
        columns ``shape_id``, ``storm_year``, ``storm_id``, ``storm_name``,
        ``hazard``.
    interval_lookup : dict
        Output of ``build_interval_lookup``.
    aggregation : "max" or "sum"
    event_limit : float
        Per-event payout cap (applied for "sum" aggregation).
    annual_limit : float
        Annual payout cap.

    Returns
    -------
    dict with keys:
        ``el``           — historical EL rate (total payout / years covered)
        ``years_covered``— number of calendar years spanned by the data
        ``ylt``          — annual loss table (storm_year, payout, payout_capped)
        ``event_lt``     — event loss table (storm_year, storm_id, storm_name,
                           payout, payout_capped)
        ``detail``       — full events DataFrame with payout columns
    """
    _empty = pd.DataFrame

    if historic_events_df.empty:
        return {
            "el": 0.0,
            "years_covered": 0,
            "ylt": _empty(columns=["storm_year", "payout", "payout_capped"]),
            "event_lt": _empty(columns=["storm_year", "storm_id", "storm_name",
                                        "payout", "payout_capped"]),
            "detail": historic_events_df,
        }

    df = historic_events_df.copy()
    df[["category", "payout"]] = df.apply(
        lambda row: pd.Series(
            lookup_payout(row["shape_id"], row["hazard"], interval_lookup)
        ),
        axis=1,
    )
    df = df.dropna(subset=["payout"])

    years_covered = int(
        historic_events_df["storm_year"].max()
        - historic_events_df["storm_year"].min()
        + 1
    )

    if df.empty:
        return {
            "el": 0.0,
            "years_covered": years_covered,
            "ylt": _empty(columns=["storm_year", "payout", "payout_capped"]),
            "event_lt": _empty(columns=["storm_year", "storm_id", "storm_name",
                                        "payout", "payout_capped"]),
            "detail": df,
        }

    # Step 1 — per (shape, storm): max payout
    per_shape_storm = (
        df.groupby(["shape_id", "storm_year", "storm_id", "storm_name"])["payout"]
        .max()
        .reset_index()
    )

    # Step 2 — aggregate across shapes per storm event
    if aggregation == "max":
        event_lt = (
            per_shape_storm
            .groupby(["storm_year", "storm_id", "storm_name"])["payout"]
            .max()
            .reset_index()
        )
        event_lt["payout_capped"] = event_lt["payout"]
    else:
        event_lt = (
            per_shape_storm
            .groupby(["storm_year", "storm_id", "storm_name"])["payout"]
            .sum()
            .reset_index()
        )
        event_lt["payout_capped"] = event_lt["payout"].clip(upper=event_limit)

    # Step 3 — annual aggregation with cap
    ylt = (
        event_lt.groupby("storm_year")["payout_capped"]
        .sum()
        .reset_index()
        .rename(columns={"payout_capped": "payout"})
    )
    ylt["payout_capped"] = ylt["payout"].clip(upper=annual_limit)

    el = ylt["payout_capped"].sum() / years_covered if years_covered > 0 else 0.0

    # Step 4 — payout_effective: real contribution per event after running year cap.
    elt = event_lt.sort_values(["storm_year", "storm_id"]).copy()
    cumul_before = elt.groupby("storm_year")["payout_capped"].cumsum() - elt["payout_capped"]
    remaining = (annual_limit - cumul_before).clip(lower=0)
    elt["payout_effective"] = np.minimum(elt["payout_capped"].values, remaining.values)

    return {
        "el": el,
        "years_covered": years_covered,
        "ylt": ylt,
        "event_lt": elt,
        "detail": df,
    }


# ---------------------------------------------------------------------------
# Entry point — example: two concentric squares, payout = max across shapes
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ---- Credentials --------------------------------------------------------
    # Prefer environment variables to avoid committing secrets.
    USERNAME = os.environ.get("REASK_USERNAME", "ian@floodflash.com")
    PASSWORD = os.environ.get("REASK_PASSWORD", "")   # set via env or fill in

    # ---- Policy definition --------------------------------------------------
    # Shapes can be inline WKT/GeoJSON dicts (as below), or loaded from a file:
    #
    #   shapes = shapes_from_geojson_file("my_shapes.geojson", id_property="shape_id")
    #
    # Concentric squares centred roughly on Mexico City (illustrative coords):
    SHAPES = [
        {
            "shape_id": "inner_box",
            "geometry": "POLYGON((-99.5 19.2, -98.9 19.2, -98.9 19.8, -99.5 19.8, -99.5 19.2))",
        },
        {
            "shape_id": "outer_box",
            "geometry": "POLYGON((-100.0 18.8, -98.4 18.8, -98.4 20.2, -100.0 20.2, -100.0 18.8))",
        },
    ]

    # ---- Payout structures --------------------------------------------------
    # DataFrame with columns: shape_id, from, to, payout_percentage
    # (optionally add a 'category' column for band labels)
    # Can also be loaded from CSV: pd.read_csv("payout_structures.csv")
    PAYOUT_STRUCTURES = pd.DataFrame(
        [
            # inner box — triggers at lower wind speeds
            {"shape_id": "inner_box", "from": 0,   "to": 64,  "payout_percentage": 0.00},
            {"shape_id": "inner_box", "from": 64,  "to": 96,  "payout_percentage": 0.25},
            {"shape_id": "inner_box", "from": 96,  "to": 111, "payout_percentage": 0.50},
            {"shape_id": "inner_box", "from": 111, "to": 130, "payout_percentage": 0.75},
            {"shape_id": "inner_box", "from": 130, "to": 999, "payout_percentage": 1.00},
            # outer box — higher threshold (storm must be stronger to pay)
            {"shape_id": "outer_box", "from": 0,   "to": 96,  "payout_percentage": 0.00},
            {"shape_id": "outer_box", "from": 96,  "to": 111, "payout_percentage": 0.25},
            {"shape_id": "outer_box", "from": 111, "to": 130, "payout_percentage": 0.50},
            {"shape_id": "outer_box", "from": 130, "to": 999, "payout_percentage": 1.00},
        ]
    )

    # ---- Aggregation --------------------------------------------------------
    # "max"  → payout per event = highest payout across all triggered shapes
    #           (correct for concentric / overlapping shapes)
    # "sum"  → payout per event = sum of payouts across all triggered shapes
    #           (appropriate for independent non-overlapping shapes)
    AGGREGATION: Literal["max", "sum"] = "max"

    SIMULATION_YEARS = 41000
    OUTPUT_DIR = "."

    # ---- Run ----------------------------------------------------------------
    token = get_auth_token(USERNAME, PASSWORD)
    headers = make_deepcyc_headers(token)

    events_df = fetch_events_for_all_shapes(
        SHAPES,
        headers,
        perils=["wind_speed"],
        simulation_years=SIMULATION_YEARS,
    )

    eht_path = os.path.join(OUTPUT_DIR, "events_eht.csv")
    events_df.to_csv(eht_path, index=False)
    print(f"\nEvent hazard table → {eht_path}")

    interval_lookup = build_interval_lookup(PAYOUT_STRUCTURES)

    result = calculate_el(
        events_df,
        interval_lookup,
        aggregation=AGGREGATION,
        simulation_years=SIMULATION_YEARS,
    )

    print(f"Expected Loss (EL): {result['el']:.6f}")

    ylt_path = os.path.join(OUTPUT_DIR, "ylt.csv")
    result["ylt"].to_csv(ylt_path, index=False)
    print(f"YLT → {ylt_path}")

    elt_path = os.path.join(OUTPUT_DIR, "event_lt.csv")
    result["event_lt"].to_csv(elt_path, index=False)
    print(f"Event loss table → {elt_path}")
