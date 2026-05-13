"""
app.py — Streamlit front-end for reask_parametric_pricing.py

Run from the same directory as reask_parametric_pricing.py with:
    streamlit run app.py
"""

from __future__ import annotations

import io
import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import plotly.graph_objects as go

from reask_parametric_pricing import (
    get_auth_token,
    make_deepcyc_headers,
    make_metryc_headers,
    fetch_events_for_all_shapes,
    fetch_events_for_all_circles,
    fetch_historic_events_for_all_shapes,
    fetch_historic_events_for_all_circles,
    fetch_storm_track,
    build_interval_lookup,
    calculate_el,
    calculate_historic_payouts,
    parse_geometry_to_coords,
    validate_inputs,
)

# ---------------------------------------------------------------------------
# Export helper
# ---------------------------------------------------------------------------

def _ep_curve_df(sim_years: int, ylt: pd.DataFrame, event_lt: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of AEP and OEP payout at standard return periods."""
    aep_losses = np.sort(ylt["payout_capped"].values)[::-1]
    aep_rp = sim_years / np.arange(1, len(aep_losses) + 1)

    oep_annual = event_lt.groupby("year_id")["payout_capped"].max()
    oep_losses = np.sort(oep_annual.values)[::-1]
    oep_rp = sim_years / np.arange(1, len(oep_losses) + 1)

    def _interp(rp_arr, loss_arr, target):
        if len(loss_arr) == 0:
            return 0.0
        if target > rp_arr[0]:
            return float(loss_arr[0])
        if target < rp_arr[-1]:
            return 0.0
        idx = np.searchsorted(-rp_arr, -target)
        if idx == 0:
            return float(loss_arr[0])
        rp_hi, rp_lo = rp_arr[idx - 1], rp_arr[idx]
        l_hi, l_lo = loss_arr[idx - 1], loss_arr[idx]
        t = (target - rp_lo) / (rp_hi - rp_lo)
        return float(l_lo + t * (l_hi - l_lo))

    rp_targets = [2, 5, 10, 25, 50, 100, 200, 500, 1000]
    rows = []
    for rp in rp_targets:
        rows.append({
            "Return Period (years)": rp,
            "AEP Payout": _interp(aep_rp, aep_losses, rp),
            "OEP Payout": _interp(oep_rp, oep_losses, rp),
        })
    return pd.DataFrame(rows)


def build_export_excel(
    run_timestamp: str,
    peril: str,
    hazard_units: str,
    terrain_correction: str,
    simulation_years: int,
    aggregation: str,
    input_mode: str,
    valid_shapes: list,
    valid_circles: list,
    payout_df: pd.DataFrame,
    stochastic_el: float,
    stochastic_results: dict,
    historic_el: "float | None",
    historic_results: "dict | None",
    stochastic_weight: float,
    historic_weight: float,
    blended_el: float,
    target_nlr: float,
    total_comm: float,
    net_premium: float,
    gross_premium: float,
    policy_limit: float,
    currency: str,
) -> bytes:
    """
    Build a multi-sheet Excel workbook for audit and exposure management.

    Sheets
    ------
    Summary          — run parameters and pricing outputs
    Shapes           — location definitions
    Payout Structure — hazard band / payout table
    YELT (Stochastic)— Year Event Loss Table with payout_effective for reinsurance
    YLT (Stochastic) — Year Loss Table (annual aggregate)
    EP Curve         — AEP and OEP at standard return periods
    YELT (Historic)  — Historic triggered events (if available)
    YLT (Historic)   — Historic annual payouts (if available)
    """
    buf = io.BytesIO()

    with pd.ExcelWriter(buf, engine="openpyxl") as writer:

        # ---- Sheet 1: Summary ----------------------------------------------
        def _pct(v): return f"{v:.4%}"
        def _amt(v): return f"{v:,.0f}" if policy_limit > 0 else "—"

        el_blend_note = f"{stochastic_weight:.0%} stochastic"
        if historic_el is not None and historic_weight > 0:
            el_blend_note += f" + {historic_weight:.0%} historic"

        summary_rows = [
            ("Run timestamp", run_timestamp),
            ("", ""),
            ("--- Model Parameters ---", ""),
            ("Peril", peril),
            ("Hazard units", hazard_units),
            ("Terrain correction", terrain_correction),
            ("Simulation years", simulation_years),
            ("Aggregation", aggregation),
            ("", ""),
            ("--- Expected Loss ---", ""),
            ("Stochastic EL (DeepCyc)", _pct(stochastic_el)),
            ("Historic EL (Metryc)", _pct(historic_el) if historic_el is not None else "Not run"),
            ("Stochastic weight", f"{stochastic_weight:.0%}"),
            ("Historic weight", f"{historic_weight:.0%}"),
            ("Blended EL", _pct(blended_el)),
            ("Blend formula", el_blend_note),
            ("", ""),
            ("--- Pricing ---", ""),
            ("Target net loss ratio", _pct(target_nlr)),
            ("Total commissions", _pct(total_comm)),
            ("Net premium (rate)", _pct(net_premium)),
            ("Gross premium (rate)", _pct(gross_premium)),
            ("Currency", currency),
            ("Policy limit", _amt(policy_limit)),
            ("Expected annual loss", _amt(blended_el * policy_limit)),
            ("Net premium (monetary)", _amt(net_premium * policy_limit)),
            ("Gross premium (monetary)", _amt(gross_premium * policy_limit)),
        ]
        pd.DataFrame(summary_rows, columns=["Parameter", "Value"]).to_excel(
            writer, sheet_name="Summary", index=False
        )

        # ---- Sheet 2: Shapes -----------------------------------------------
        if input_mode == "circles":
            shapes_df = pd.DataFrame(valid_circles)
        else:
            shapes_df = pd.DataFrame([
                {"shape_id": s["shape_id"], "geometry": json.dumps(s["geometry"])}
                for s in valid_shapes
            ])
        shapes_df.to_excel(writer, sheet_name="Shapes", index=False)

        # ---- Sheet 3: Payout Structure -------------------------------------
        payout_df.to_excel(writer, sheet_name="Payout Structure", index=False)

        # ---- Sheet 4: YELT (Stochastic) ------------------------------------
        elt = stochastic_results["event_lt"].copy()
        for col in ["payout", "payout_capped", "payout_effective"]:
            if col in elt.columns:
                elt[col] = elt[col].mul(100).round(6)
        elt = elt.rename(columns={
            "payout": "payout_%",
            "payout_capped": "payout_capped_%",
            "payout_effective": "payout_effective_%",
        })
        elt.to_excel(writer, sheet_name="YELT (Stochastic)", index=False)

        # ---- Sheet 5: YLT (Stochastic) -------------------------------------
        ylt = stochastic_results["ylt"].copy()
        for col in ["payout", "payout_capped"]:
            ylt[col] = ylt[col].mul(100).round(6)
        ylt = ylt.rename(columns={"payout": "payout_%", "payout_capped": "payout_capped_%"})
        ylt.to_excel(writer, sheet_name="YLT (Stochastic)", index=False)

        # ---- Sheet 6: EP Curve ---------------------------------------------
        ep_df = _ep_curve_df(
            simulation_years,
            stochastic_results["ylt"],
            stochastic_results["event_lt"],
        )
        for col in ["AEP Payout", "OEP Payout"]:
            ep_df[col] = ep_df[col].mul(100).round(4)
        ep_df = ep_df.rename(columns={"AEP Payout": "AEP Payout (%)", "OEP Payout": "OEP Payout (%)"})
        ep_df.to_excel(writer, sheet_name="EP Curve", index=False)

        # ---- Sheet 7 & 8: Historic (if available) --------------------------
        if historic_results is not None:
            h_elt = historic_results["event_lt"].copy()
            for col in ["payout", "payout_capped", "payout_effective"]:
                if col in h_elt.columns:
                    h_elt[col] = h_elt[col].mul(100).round(6)
            h_rename = {}
            for c in h_elt.columns:
                if c in ("payout", "payout_capped", "payout_effective"):
                    h_rename[c] = c + "_%"
            h_elt = h_elt.rename(columns=h_rename)
            h_elt.to_excel(writer, sheet_name="YELT (Historic)", index=False)

            h_ylt = historic_results["ylt"].copy()
            for col in ["payout", "payout_capped"]:
                h_ylt[col] = h_ylt[col].mul(100).round(6)
            h_ylt = h_ylt.rename(columns={"payout": "payout_%", "payout_capped": "payout_capped_%"})
            h_ylt.to_excel(writer, sheet_name="YLT (Historic)", index=False)

    return buf.getvalue()


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Parametric TC Pricer",
    page_icon="🌀",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

_DEFAULTS: dict = {
    "events_df": None,
    "results": None,
    "cached_shapes_key": None,
    "run_simulation_years": 41000,
    "_payout_editor_df": None,
    "_payout_shape_ids": [],
    "historic_events_df": None,
    "historic_results": None,
    "storm_tracks": {},
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ---------------------------------------------------------------------------
# Sidebar — credentials & API settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Credentials")
    username = st.text_input(
        "Username",
        value=os.environ.get("REASK_USERNAME", ""),
        placeholder="your@email.com",
    )
    password = st.text_input(
        "Password",
        type="password",
        value=os.environ.get("REASK_PASSWORD", ""),
    )

    st.divider()
    st.header("API Settings")

    simulation_years = st.number_input(
        "Simulation years", value=41000, min_value=1000, step=1000
    )
    peril = st.selectbox(
        "Peril",
        options=["wind_speed", "central_pressure"],
        format_func=lambda x: "Wind Speed" if x == "wind_speed" else "Central Pressure",
    )
    if peril == "wind_speed":
        hazard_units = st.selectbox(
            "Wind speed units", options=["mph", "kph", "ms", "knots"]
        )
    else:
        hazard_units = "hPa"
        st.caption("Central pressure is always measured in hPa.")
    terrain_correction = st.selectbox(
        "Terrain correction", options=["open_water", "land"]
    )

    st.divider()
    st.header("Event Cache")
    if st.session_state.events_df is not None:
        st.success(f"{len(st.session_state.events_df):,} events cached")
        if st.button("Clear cached events", use_container_width=True):
            st.session_state.events_df = None
            st.session_state.results = None
            st.session_state.cached_shapes_key = None
            st.rerun()
    else:
        st.info("No events cached yet.")

# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

st.title("🌀 Parametric TC Pricer")
st.caption(
    "Price parametric tropical cyclone policies using the Reask DeepCyc stochastic catalogue."
)

tab_setup, tab_results, tab_historic, tab_pricing, tab_help = st.tabs([
    "📋 Policy Setup", "📊 Results", "📍 Historic Events", "💰 Pricing", "🔍 Validation & Help",
])

# ===========================================================================
# TAB 1 — Policy Setup
# ===========================================================================

with tab_setup:

    # -----------------------------------------------------------------------
    # Section 1 — Shapes
    # -----------------------------------------------------------------------
    st.subheader("1. Policy Shapes")

    shape_method = st.radio(
        "shape_input_method",
        ["Upload GeoJSON", "Enter manually (WKT)", "Define circles (lat/lon + radius)"],
        horizontal=True,
        label_visibility="collapsed",
    )

    raw_shapes: list[dict] = []
    valid_circles: list[dict] = []
    input_mode = "shapes"  # "shapes" or "circles"

    if shape_method == "Upload GeoJSON":
        col_upload, col_prop = st.columns([3, 1])
        with col_upload:
            uploaded_geojson = st.file_uploader(
                "GeoJSON FeatureCollection",
                type=["geojson", "json"],
                label_visibility="collapsed",
            )
        with col_prop:
            id_property = st.text_input("Shape ID property", value="shape_id")

        if uploaded_geojson:
            gj = json.load(uploaded_geojson)
            for feat in gj.get("features", []):
                if feat.get("geometry", {}).get("type") == "Polygon":
                    raw_shapes.append(
                        {
                            "shape_id": feat["properties"].get(
                                id_property, f"shape_{len(raw_shapes) + 1}"
                            ),
                            "geometry": feat["geometry"],
                        }
                    )
            if raw_shapes:
                st.success(
                    f"Loaded {len(raw_shapes)} polygon(s): "
                    + ", ".join(f"`{s['shape_id']}`" for s in raw_shapes)
                )
            else:
                st.warning(
                    "No Polygon features found — check the GeoJSON file and the "
                    "ID property name."
                )

    elif shape_method == "Enter manually (WKT)":
        st.caption(
            "Enter one shape per row. The geometry column must be a valid WKT Polygon string."
        )
        _default_wkt = pd.DataFrame(
            [
                {
                    "shape_id": "inner_box",
                    "geometry_wkt": "POLYGON((-99.5 19.2, -98.9 19.2, -98.9 19.8, -99.5 19.8, -99.5 19.2))",
                },
                {
                    "shape_id": "outer_box",
                    "geometry_wkt": "POLYGON((-100.0 18.8, -98.4 18.8, -98.4 20.2, -100.0 20.2, -100.0 18.8))",
                },
            ]
        )
        shapes_input = st.data_editor(
            _default_wkt, num_rows="dynamic", use_container_width=True
        )
        for _, row in shapes_input.dropna(
            subset=["shape_id", "geometry_wkt"]
        ).iterrows():
            if row["shape_id"] and row["geometry_wkt"]:
                raw_shapes.append(
                    {"shape_id": row["shape_id"], "geometry": row["geometry_wkt"]}
                )

    else:  # Define circles
        input_mode = "circles"
        st.caption(
            "Define up to 5 circular exposure zones. Each row is one shape: "
            "a centre point (lat/lon) plus a radius. Use the same `shape_id` across rows "
            "to create concentric circles around one location (pair with **Max** aggregation), "
            "or different IDs for independent locations (pair with **Sum**)."
        )

        _default_circles = pd.DataFrame([
            {"shape_id": "zone_1", "lat": 14.60, "lon": 121.10, "radius_km": 50.0},
            {"shape_id": "zone_2", "lat": 14.60, "lon": 121.10, "radius_km": 100.0},
        ])

        circles_input = st.data_editor(
            _default_circles,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "lat": st.column_config.NumberColumn(
                    "Latitude", min_value=-90.0, max_value=90.0, format="%.4f"
                ),
                "lon": st.column_config.NumberColumn(
                    "Longitude", min_value=-180.0, max_value=180.0, format="%.4f"
                ),
                "radius_km": st.column_config.NumberColumn(
                    "Radius (km)", min_value=1.0, format="%.1f"
                ),
            },
        )

        for _, row in circles_input.dropna(
            subset=["shape_id", "lat", "lon", "radius_km"]
        ).iterrows():
            if row["shape_id"] and row["radius_km"] > 0:
                valid_circles.append({
                    "shape_id": str(row["shape_id"]),
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                    "radius_km": float(row["radius_km"]),
                })

        if valid_circles:
            st.success(
                f"Defined {len(valid_circles)} circle(s): "
                + ", ".join(f"`{c['shape_id']}` ({c['radius_km']:.0f} km)" for c in valid_circles)
            )

    # Validate polygon geometries (only relevant in shapes mode)
    valid_shapes: list[dict] = []
    if input_mode == "shapes":
        for s in raw_shapes:
            try:
                parse_geometry_to_coords(s["geometry"])
                valid_shapes.append(s)
            except Exception as exc:
                st.error(f"Invalid geometry for `{s['shape_id']}`: {exc}")

    # Map preview
    if input_mode == "circles" and valid_circles:
        circle_map_data = [
            {
                "shape_id": c["shape_id"],
                "lat": c["lat"],
                "lon": c["lon"],
                "radius_m": c["radius_km"] * 1000,
            }
            for c in valid_circles
        ]
        map_center_lon = float(np.mean([c["lon"] for c in valid_circles]))
        map_center_lat = float(np.mean([c["lat"] for c in valid_circles]))
        max_radius_deg = max(c["radius_km"] for c in valid_circles) / 111.0
        zoom = max(1, min(10, int(8 - np.log2(max(max_radius_deg * 3, 0.01)))))

        st.pydeck_chart(
            pdk.Deck(
                layers=[
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=circle_map_data,
                        get_position="[lon, lat]",
                        get_radius="radius_m",
                        get_fill_color=[255, 140, 0, 55],
                        get_line_color=[255, 140, 0, 230],
                        line_width_min_pixels=2,
                        stroked=True,
                        filled=True,
                        pickable=True,
                        auto_highlight=True,
                    )
                ],
                initial_view_state=pdk.ViewState(
                    latitude=map_center_lat,
                    longitude=map_center_lon,
                    zoom=zoom,
                    pitch=0,
                ),
                tooltip={"text": "Shape: {shape_id}"},
                map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            ),
            use_container_width=True,
        )

    elif valid_shapes:
        poly_data = []
        for s in valid_shapes:
            lat, lon = parse_geometry_to_coords(s["geometry"])
            poly_data.append(
                {
                    "shape_id": s["shape_id"],
                    "coordinates": [[lo, la] for lo, la in zip(lon, lat)],
                }
            )

        center_lon = float(
            np.mean(
                [np.mean([c[0] for c in p["coordinates"]]) for p in poly_data]
            )
        )
        center_lat = float(
            np.mean(
                [np.mean([c[1] for c in p["coordinates"]]) for p in poly_data]
            )
        )
        span_lon = max(
            max(c[0] for c in p["coordinates"]) for p in poly_data
        ) - min(min(c[0] for c in p["coordinates"]) for p in poly_data)

        zoom = max(1, min(10, int(8 - np.log2(max(span_lon, 0.1)))))

        st.pydeck_chart(
            pdk.Deck(
                layers=[
                    pdk.Layer(
                        "PolygonLayer",
                        data=poly_data,
                        get_polygon="coordinates",
                        get_fill_color=[255, 140, 0, 55],
                        get_line_color=[255, 140, 0, 230],
                        line_width_min_pixels=2,
                        pickable=True,
                        auto_highlight=True,
                    )
                ],
                initial_view_state=pdk.ViewState(
                    latitude=center_lat,
                    longitude=center_lon,
                    zoom=zoom,
                    pitch=0,
                ),
                tooltip={"text": "Shape: {shape_id}"},
                map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            ),
            use_container_width=True,
        )

    st.divider()

    # -----------------------------------------------------------------------
    # Section 2 — Payout structures
    # -----------------------------------------------------------------------
    st.subheader("2. Payout Structures")
    st.caption(
        "Define hazard bands for each shape. `payout_percentage` is a fraction of the "
        "insured value (e.g. 0.25 = 25 %). Bands are **left-closed**: [from, to)."
    )

    payout_method = st.radio(
        "payout_input_method",
        ["Upload CSV", "Enter manually"],
        horizontal=True,
        label_visibility="collapsed",
    )

    payout_df: pd.DataFrame | None = None

    if payout_method == "Upload CSV":
        st.caption(
            "Required columns: `shape_id`, `from`, `to`, `payout_percentage`.  "
            "Optional: `category`."
        )
        uploaded_csv = st.file_uploader(
            "Payout structures CSV", type=["csv"], label_visibility="collapsed"
        )
        if uploaded_csv:
            payout_df = pd.read_csv(uploaded_csv)
            st.dataframe(payout_df, use_container_width=True, hide_index=True)

    else:  # manual editor
        if input_mode == "circles" and valid_circles:
            shape_ids = [c["shape_id"] for c in valid_circles]
        elif valid_shapes:
            shape_ids = [s["shape_id"] for s in valid_shapes]
        else:
            shape_ids = ["shape_1"]

        # Rebuild defaults only when the shape list changes
        if set(shape_ids) != set(st.session_state["_payout_shape_ids"]):
            rows = []
            for sid in shape_ids:
                rows += [
                    {"shape_id": sid, "category": "Cat 1", "from": 0.0,   "to": 64.0,  "payout_percentage": 0.00},
                    {"shape_id": sid, "category": "Cat 2", "from": 64.0,  "to": 96.0,  "payout_percentage": 0.25},
                    {"shape_id": sid, "category": "Cat 3", "from": 96.0,  "to": 111.0, "payout_percentage": 0.50},
                    {"shape_id": sid, "category": "Cat 4", "from": 111.0, "to": 130.0, "payout_percentage": 0.75},
                    {"shape_id": sid, "category": "Cat 5", "from": 130.0, "to": 999.0, "payout_percentage": 1.00},
                ]
            st.session_state["_payout_editor_df"] = pd.DataFrame(rows)
            st.session_state["_payout_shape_ids"] = shape_ids

        edited = st.data_editor(
            st.session_state["_payout_editor_df"],
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "payout_percentage": st.column_config.NumberColumn(
                    "payout_percentage",
                    min_value=0.0,
                    max_value=1.0,
                    format="%.2f",
                    step=0.05,
                ),
                "from": st.column_config.NumberColumn("from", format="%.1f"),
                "to":   st.column_config.NumberColumn("to",   format="%.1f"),
            },
        )
        st.session_state["_payout_editor_df"] = edited
        payout_df = edited

    st.divider()

    # -----------------------------------------------------------------------
    # Section 3 — Aggregation
    # -----------------------------------------------------------------------
    st.subheader("3. Multi-Shape Aggregation")
    aggregation = st.radio(
        "How should payouts from multiple triggered shapes be combined within a single TC event?",
        options=["max", "sum"],
        format_func=lambda x: (
            "Maximum — pay the highest single-shape payout  "
            "(use for concentric or overlapping shapes)"
            if x == "max"
            else "Sum — add payouts from all triggered shapes  "
            "(use for independent non-overlapping shapes)"
        ),
    )

    st.divider()

    # -----------------------------------------------------------------------
    # Section 4 — Run
    # -----------------------------------------------------------------------
    st.subheader("4. Run Analysis")

    issues = []
    if not username or not password:
        issues.append("credentials (sidebar)")
    if not valid_shapes and not valid_circles:
        issues.append("at least one valid shape or circle")
    if payout_df is None or payout_df.empty:
        issues.append("payout structures")

    # Run validation checks and surface errors/warnings before the run button
    # validate_inputs only needs shape_id keys, so circles work here too
    validation_issues = []
    _validate_items = valid_circles if input_mode == "circles" else valid_shapes
    if _validate_items and payout_df is not None and not payout_df.empty:
        validation_issues = validate_inputs(
            _validate_items, payout_df, peril=peril, units=hazard_units
        )

    val_errors = [vi for vi in validation_issues if vi["severity"] == "error"]
    val_warnings = [vi for vi in validation_issues if vi["severity"] == "warning"]

    for ve in val_errors:
        st.error(
            f"**{ve['check']}:** {ve['message']}  "
            "*(see the Validation & Help tab for details)*"
        )
    for vw in val_warnings:
        st.warning(
            f"**{vw['check']}:** {vw['message']}  "
            "*(see the Validation & Help tab for details)*"
        )

    if issues:
        st.warning(f"Before running, please provide: {', '.join(issues)}.")

    # Detect if cached events might be stale
    if input_mode == "circles":
        shapes_key = (
            str(sorted((c["shape_id"], c["lat"], c["lon"], c["radius_km"]) for c in valid_circles))
            + "circles"
            + peril
            + str(simulation_years)
            + terrain_correction
            + hazard_units
        )
    else:
        shapes_key = (
            str(sorted(s["shape_id"] for s in valid_shapes))
            + peril
            + str(simulation_years)
            + terrain_correction
            + hazard_units
        )
    events_stale = (
        st.session_state.events_df is not None
        and st.session_state.cached_shapes_key != shapes_key
    )
    if events_stale:
        st.warning(
            "The shapes, peril, or API settings have changed since the last fetch. "
            "Cached events may be stale — use **Re-fetch & Run** to refresh."
        )

    run_disabled = bool(issues) or bool(val_errors)

    col_run, col_refetch, _spacer = st.columns([1, 1, 3])

    with col_run:
        run_clicked = st.button(
            "▶ Run Analysis",
            disabled=run_disabled,
            type="primary",
            use_container_width=True,
            help="Use cached events if available, then recalculate EL.",
        )
    with col_refetch:
        refetch_clicked = st.button(
            "↺ Re-fetch & Run",
            disabled=run_disabled,
            use_container_width=True,
            help="Discard cached events, re-call the API, then recalculate EL.",
        )

    if refetch_clicked:
        st.session_state.events_df = None
        st.session_state.cached_shapes_key = None
        run_clicked = True  # fall through to the common run block

    if run_clicked and not run_disabled:
        with st.status("Running analysis…", expanded=True) as run_status:
            try:
                st.write("Authenticating with Reask API…")
                token = get_auth_token(username, password)
                headers = make_deepcyc_headers(token)

                if st.session_state.events_df is None:
                    if input_mode == "circles":
                        n_items = len(valid_circles)
                        item_label = "circle(s)"
                    else:
                        n_items = len(valid_shapes)
                        item_label = "shape(s)"
                    st.write(
                        f"Fetching events for {n_items} {item_label} — "
                        f"peril: **{peril}**, units: **{hazard_units}**.  "
                        f"*(This may take a minute for large catalogues.)*"
                    )
                    if input_mode == "circles":
                        events_df = fetch_events_for_all_circles(
                            valid_circles,
                            headers,
                            perils=[peril],
                            simulation_years=simulation_years,
                            terrain_correction=terrain_correction,
                            wind_speed_units=hazard_units,
                        )
                    else:
                        events_df = fetch_events_for_all_shapes(
                            valid_shapes,
                            headers,
                            perils=[peril],
                            simulation_years=simulation_years,
                            terrain_correction=terrain_correction,
                            wind_speed_units=hazard_units,
                        )
                    st.session_state.events_df = events_df
                    st.session_state.cached_shapes_key = shapes_key
                    st.write(f"Fetched **{len(events_df):,}** event records.")
                else:
                    st.write(
                        f"Using **{len(st.session_state.events_df):,}** cached event records."
                    )

                st.write("Applying payout structures…")
                interval_lookup = build_interval_lookup(payout_df)
                results = calculate_el(
                    st.session_state.events_df,
                    interval_lookup,
                    aggregation=aggregation,
                    simulation_years=simulation_years,
                )
                st.session_state.results = results
                st.session_state.run_simulation_years = simulation_years

                run_status.update(label="Analysis complete!", state="complete")
                st.success(f"Expected Loss: **{results['el']:.4%}**  — see the Results tab for details.")

            except Exception as exc:
                run_status.update(label=f"Error: {exc}", state="error")
                st.exception(exc)

# ===========================================================================
# TAB 2 — Results
# ===========================================================================

with tab_results:
    if st.session_state.results is None:
        st.info("Run the analysis on the **Policy Setup** tab to see results here.")
    else:
        res = st.session_state.results
        sim_years = st.session_state.run_simulation_years
        ylt = res["ylt"]
        elt = res["event_lt"]

        loss_years = int((ylt["payout_capped"] > 0).sum())

        # ---- EP curve computation ------------------------------------------
        # AEP: annual aggregate payout (sum of events per year, capped)
        aep_losses = np.sort(ylt["payout_capped"].values)[::-1]
        aep_rp = sim_years / np.arange(1, len(aep_losses) + 1)

        # OEP: largest single-event payout in each year
        oep_annual = elt.groupby("year_id")["payout_capped"].max()
        oep_losses = np.sort(oep_annual.values)[::-1]
        oep_rp = sim_years / np.arange(1, len(oep_losses) + 1)

        def loss_at_rp(rp_arr, loss_arr, target_rp):
            """Return the loss value at a given return period by linear interpolation."""
            if target_rp > rp_arr[0]:
                return loss_arr[0]
            if target_rp < rp_arr[-1]:
                return 0.0
            idx = np.searchsorted(-rp_arr, -target_rp)
            if idx == 0:
                return loss_arr[0]
            rp_hi, rp_lo = rp_arr[idx - 1], rp_arr[idx]
            l_hi, l_lo = loss_arr[idx - 1], loss_arr[idx]
            t = (target_rp - rp_lo) / (rp_hi - rp_lo)
            return float(l_lo + t * (l_hi - l_lo))

        aep_50  = loss_at_rp(aep_rp, aep_losses, 50)
        aep_100 = loss_at_rp(aep_rp, aep_losses, 100)

        # ---- Key metrics ---------------------------------------------------
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Expected Loss (EL)", f"{res['el']:.4%}")
        m2.metric(
            "Loss frequency",
            f"{loss_years / sim_years:.4%}",
            help="Fraction of simulated years with any payout",
        )
        m3.metric(
            "1-in-50 year AEP payout",
            f"{aep_50:.2%}",
            help="Annual aggregate payout exceeded once every 50 years on average",
        )
        m4.metric(
            "1-in-100 year AEP payout",
            f"{aep_100:.2%}",
            help="Annual aggregate payout exceeded once every 100 years on average",
        )

        st.divider()

        # ---- EP curve chart ------------------------------------------------
        st.subheader("Exceedance Probability Curve")
        st.caption(
            "**AEP** (Aggregate Exceedance Probability) — probability that the total "
            "annual payout exceeds the threshold.  "
            "**OEP** (Occurrence Exceedance Probability) — probability that the largest "
            "single-event payout in a year exceeds the threshold."
        )

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=aep_rp,
            y=aep_losses * 100,
            mode="lines",
            name="AEP",
            line=dict(color="#E8450A", width=2),
            hovertemplate="Return period: %{x:,.0f} yrs<br>AEP payout: %{y:.1f}%<extra></extra>",
        ))

        fig.add_trace(go.Scatter(
            x=oep_rp,
            y=oep_losses * 100,
            mode="lines",
            name="OEP",
            line=dict(color="#1A73E8", width=2, dash="dash"),
            hovertemplate="Return period: %{x:,.0f} yrs<br>OEP payout: %{y:.1f}%<extra></extra>",
        ))

        # Reference lines for 1-in-50 and 1-in-100
        for rp, label in [(50, "1-in-50"), (100, "1-in-100")]:
            fig.add_vline(
                x=rp,
                line=dict(color="grey", width=1, dash="dot"),
                annotation_text=label,
                annotation_position="top right",
                annotation_font_size=11,
            )

        fig.update_layout(
            xaxis=dict(
                title="Return Period (years)",
                type="log",
                range=[0, 3],  # log10(1) to log10(1000)
                tickvals=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
                ticktext=["1", "2", "5", "10", "20", "50", "100", "200", "500", "1,000"],
                showgrid=True,
                gridcolor="#eeeeee",
            ),
            yaxis=dict(
                title="Payout (% of insured value)",
                range=[0, 102],
                ticksuffix="%",
                showgrid=True,
                gridcolor="#eeeeee",
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(t=40, b=40, l=60, r=20),
            height=420,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # ---- Distribution histogram ----------------------------------------
        st.subheader("Annual Payout Distribution")
        loss_payouts = ylt.loc[ylt["payout_capped"] > 0, "payout_capped"]

        if not loss_payouts.empty:
            st.caption(
                f"{loss_years:,} loss years out of {sim_years:,} simulated "
                f"({loss_years / sim_years:.2%})"
            )
            n_bins = min(25, len(loss_payouts))
            counts, bin_edges = np.histogram(loss_payouts, bins=n_bins)
            bin_labels = [
                f"{e:.0%}–{bin_edges[i + 1]:.0%}"
                for i, e in enumerate(bin_edges[:-1])
            ]
            st.bar_chart(
                pd.DataFrame({"Years": counts}, index=bin_labels),
                use_container_width=True,
            )
        else:
            st.info("No loss years to display.")

        st.divider()

        # ---- Tables --------------------------------------------------------
        col_l, col_r = st.columns(2)

        with col_l:
            st.subheader("Year Loss Table")
            st.caption("Top 50 worst years")
            disp_ylt = ylt.sort_values("payout_capped", ascending=False).head(50).copy()
            disp_ylt["payout"] = disp_ylt["payout"].map("{:.2%}".format)
            disp_ylt["payout_capped"] = disp_ylt["payout_capped"].map("{:.2%}".format)
            st.dataframe(
                disp_ylt,
                use_container_width=True,
                hide_index=True,
            )

        with col_r:
            st.subheader("Largest Events")
            st.caption("Top 50 events by payout")
            disp_elt = elt.sort_values("payout_capped", ascending=False).head(50).copy()
            disp_elt["payout"] = disp_elt["payout"].map("{:.2%}".format)
            disp_elt["payout_capped"] = disp_elt["payout_capped"].map("{:.2%}".format)
            st.dataframe(
                disp_elt,
                use_container_width=True,
                hide_index=True,
            )

        st.divider()

        # ---- Downloads -----------------------------------------------------
        st.subheader("Export")
        dl1, dl2, dl3 = st.columns(3)

        with dl1:
            st.download_button(
                "⬇ Year Loss Table (CSV)",
                ylt.to_csv(index=False).encode(),
                "ylt.csv",
                "text/csv",
                use_container_width=True,
            )
        with dl2:
            st.download_button(
                "⬇ Event Loss Table (CSV)",
                elt.to_csv(index=False).encode(),
                "event_lt.csv",
                "text/csv",
                use_container_width=True,
            )
        with dl3:
            if st.session_state.events_df is not None:
                st.download_button(
                    "⬇ Event Hazard Table (CSV)",
                    st.session_state.events_df.to_csv(index=False).encode(),
                    "events_eht.csv",
                    "text/csv",
                    use_container_width=True,
                )

# ===========================================================================
# TAB 3 — Historic Events (Metryc)
# ===========================================================================

with tab_historic:
    if not valid_shapes and not valid_circles:
        st.info("Define shapes or circles on the **Policy Setup** tab first.")
    elif payout_df is None or payout_df.empty:
        st.info("Define payout structures on the **Policy Setup** tab first.")
    else:
        st.caption(
            "Fetch real historical tropical cyclone events from the Reask **Metryc** "
            "catalogue, apply your payout structures, and visualise storm tracks."
        )

        # ---- Controls ------------------------------------------------------
        col_fetch, col_clear, _sp = st.columns([1, 1, 3])
        with col_fetch:
            fetch_clicked = st.button(
                "▶ Fetch Historic Events",
                type="primary",
                use_container_width=True,
                disabled=not (username and password),
            )
        with col_clear:
            if st.button(
                "Clear",
                use_container_width=True,
                disabled=st.session_state.historic_events_df is None,
            ):
                st.session_state.historic_events_df = None
                st.session_state.historic_results = None
                st.session_state.storm_tracks = {}
                st.rerun()

        # ---- Fetch ---------------------------------------------------------
        if fetch_clicked:
            with st.status("Fetching historic events…", expanded=True) as h_status:
                try:
                    st.write("Authenticating…")
                    token = get_auth_token(username, password)
                    m_headers = make_metryc_headers(token)

                    if input_mode == "circles":
                        st.write(f"Querying Metryc for {len(valid_circles)} circle(s)…")
                        h_df = fetch_historic_events_for_all_circles(
                            valid_circles,
                            m_headers,
                            wind_speed_units=hazard_units,
                            terrain_correction=terrain_correction,
                        )
                    else:
                        st.write(f"Querying Metryc for {len(valid_shapes)} shape(s)…")
                        h_df = fetch_historic_events_for_all_shapes(
                            valid_shapes,
                            m_headers,
                            wind_speed_units=hazard_units,
                            terrain_correction=terrain_correction,
                        )
                    st.session_state.historic_events_df = h_df
                    st.write(f"Found **{len(h_df):,}** historic TC passages.")

                    st.write("Applying payout structures…")
                    interval_lookup = build_interval_lookup(payout_df)
                    h_res = calculate_historic_payouts(
                        h_df,
                        interval_lookup,
                        aggregation=aggregation,
                    )
                    st.session_state.historic_results = h_res
                    st.session_state.storm_tracks = {}  # reset tracks on re-fetch

                    h_status.update(label="Done!", state="complete")
                    st.success(
                        f"Historical EL rate: **{h_res['el']:.4%}** over "
                        f"{h_res['years_covered']} years"
                    )
                except Exception as exc:
                    h_status.update(label=f"Error: {exc}", state="error")
                    st.exception(exc)

        # ---- Results -------------------------------------------------------
        if st.session_state.historic_results is not None:
            hr = st.session_state.historic_results
            h_df = st.session_state.historic_events_df
            event_lt = hr["event_lt"]
            triggered = (
                event_lt[event_lt["payout"] > 0]
                .sort_values("payout_capped", ascending=False)
                .copy()
            )

            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Historical EL rate", f"{hr['el']:.4%}")
            m2.metric("Years covered", f"{hr['years_covered']}")
            m3.metric(
                "Triggered events",
                f"{len(triggered)}",
                help="Events where at least one shape paid out",
            )
            m4.metric(
                "Loss years",
                f"{(hr['ylt']['payout_capped'] > 0).sum()}",
            )

            st.divider()

            # ---- Annual payout bar chart -----------------------------------
            st.subheader("Annual Payout History")
            if not hr["ylt"].empty:
                ylt_h = (
                    hr["ylt"]
                    .set_index("storm_year")["payout_capped"]
                    .mul(100)
                    .rename("Payout (%)")
                )
                st.bar_chart(ylt_h, use_container_width=True)
            else:
                st.info("No payouts triggered in the historic record.")

            st.divider()

            # ---- Map with storm tracks -------------------------------------
            st.subheader("Storm Tracks")

            # Fetch tracks for triggered storms (up to 30, cache in session)
            triggered_ids = triggered["storm_id"].unique().tolist()[:30]
            to_fetch = [
                sid for sid in triggered_ids
                if sid not in st.session_state.storm_tracks
            ]

            if to_fetch:
                with st.spinner(f"Fetching {len(to_fetch)} storm track(s)…"):
                    try:
                        token = get_auth_token(username, password)
                        m_headers = make_metryc_headers(token)
                        for sid in to_fetch:
                            pts = fetch_storm_track(sid, m_headers)
                            if pts:
                                st.session_state.storm_tracks[sid] = pts
                    except Exception as exc:
                        st.warning(f"Could not fetch some tracks: {exc}")

            # Build payout colour lookup
            payout_by_storm = dict(
                zip(triggered["storm_id"], triggered["payout_capped"])
            )

            def _track_color(payout: float) -> list[int]:
                if payout >= 1.0:
                    return [200, 0, 0, 230]
                elif payout >= 0.5:
                    return [255, 110, 0, 230]
                else:
                    return [255, 210, 0, 230]

            track_layer_data = []
            for sid, pts in st.session_state.storm_tracks.items():
                if sid not in payout_by_storm:
                    continue
                payout = payout_by_storm[sid]
                path = [
                    [p["lon"], p["lat"]] for p in pts
                    if p["lon"] is not None and p["lat"] is not None
                ]
                if len(path) < 2:
                    continue
                name = pts[0].get("track_name", sid)
                year = triggered.loc[
                    triggered["storm_id"] == sid, "storm_year"
                ].values[0]
                track_layer_data.append(
                    {
                        "path": path,
                        "color": _track_color(payout),
                        "tooltip": f"{name} ({year}) — {payout:.0%} payout",
                    }
                )

            # Policy shape/circle base layer
            if input_mode == "circles":
                h_circle_data = [
                    {
                        "shape_id": c["shape_id"],
                        "lat": c["lat"],
                        "lon": c["lon"],
                        "radius_m": c["radius_km"] * 1000,
                    }
                    for c in valid_circles
                ]
                base_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=h_circle_data,
                    get_position="[lon, lat]",
                    get_radius="radius_m",
                    get_fill_color=[255, 140, 0, 50],
                    get_line_color=[255, 140, 0, 230],
                    line_width_min_pixels=2,
                    stroked=True,
                    filled=True,
                    pickable=True,
                )
                center_lon = float(np.mean([c["lon"] for c in valid_circles]))
                center_lat = float(np.mean([c["lat"] for c in valid_circles]))
            else:
                h_poly_data = []
                for s in valid_shapes:
                    lat_c, lon_c = parse_geometry_to_coords(s["geometry"])
                    h_poly_data.append(
                        {
                            "shape_id": s["shape_id"],
                            "coordinates": [[lo, la] for lo, la in zip(lon_c, lat_c)],
                        }
                    )
                base_layer = pdk.Layer(
                    "PolygonLayer",
                    data=h_poly_data,
                    get_polygon="coordinates",
                    get_fill_color=[255, 140, 0, 50],
                    get_line_color=[255, 140, 0, 230],
                    line_width_min_pixels=2,
                    pickable=True,
                )
                center_lon = float(
                    np.mean([np.mean([c[0] for c in p["coordinates"]]) for p in h_poly_data])
                )
                center_lat = float(
                    np.mean([np.mean([c[1] for c in p["coordinates"]]) for p in h_poly_data])
                )

            map_layers = [base_layer]
            if track_layer_data:
                map_layers.append(
                    pdk.Layer(
                        "PathLayer",
                        data=track_layer_data,
                        get_path="path",
                        get_color="color",
                        width_min_pixels=2,
                        width_scale=1,
                        pickable=True,
                    )
                )

            st.pydeck_chart(
                pdk.Deck(
                    layers=map_layers,
                    initial_view_state=pdk.ViewState(
                        latitude=center_lat, longitude=center_lon, zoom=4, pitch=0
                    ),
                    tooltip={"text": "{tooltip}\n{shape_id}"},
                    map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                ),
                use_container_width=True,
            )
            st.caption(
                "Track colours:  🔴 100% payout   🟠 50–99% payout   🟡 <50% payout  "
                f"(showing up to 30 triggered storms)"
            )

            st.divider()

            # ---- Triggered events table ------------------------------------
            st.subheader("Triggered Events")
            disp_triggered = triggered.copy()
            disp_triggered["payout"] = disp_triggered["payout"].map("{:.1%}".format)
            disp_triggered["payout_capped"] = disp_triggered["payout_capped"].map(
                "{:.1%}".format
            )
            st.dataframe(disp_triggered, use_container_width=True, hide_index=True)

            st.divider()

            # ---- Downloads -------------------------------------------------
            st.subheader("Export")
            dl_a, dl_b = st.columns(2)
            with dl_a:
                st.download_button(
                    "⬇ Triggered Events (CSV)",
                    event_lt.to_csv(index=False).encode(),
                    "historic_event_lt.csv",
                    "text/csv",
                    use_container_width=True,
                )
            with dl_b:
                st.download_button(
                    "⬇ Annual Payout History (CSV)",
                    hr["ylt"].to_csv(index=False).encode(),
                    "historic_ylt.csv",
                    "text/csv",
                    use_container_width=True,
                )

# ===========================================================================
# TAB 4 — Pricing
# ===========================================================================

with tab_pricing:

    # Pull EL values from session state
    stochastic_el = (
        st.session_state.results["el"]
        if st.session_state.results is not None
        else None
    )
    historic_el = (
        st.session_state.historic_results["el"]
        if st.session_state.historic_results is not None
        else None
    )
    both_available = stochastic_el is not None and historic_el is not None

    # ---- Section 1: Expected Loss Basis ------------------------------------
    st.subheader("1. Expected Loss Basis")

    col_s, col_h = st.columns(2)
    with col_s:
        if stochastic_el is not None:
            st.metric(
                "Stochastic EL (DeepCyc)",
                f"{stochastic_el:.4%}",
                help="From the DeepCyc stochastic analysis on the Results tab.",
            )
        else:
            st.info("Run the stochastic analysis on the **Policy Setup** tab first.")
    with col_h:
        if historic_el is not None:
            h_years = st.session_state.historic_results["years_covered"]
            st.metric(
                "Historic EL (Metryc)",
                f"{historic_el:.4%}",
                help=f"From {h_years} years of observed historical data (Metryc).",
            )
        else:
            st.info(
                "Optionally run the historic analysis on the **Historic Events** tab "
                "to enable blending."
            )

    if both_available:
        st.caption(
            "Both estimates are available. Use the slider to blend them. "
            "100% = use stochastic only; 0% = use historic only."
        )
        stochastic_pct = st.slider(
            "Stochastic weight",
            min_value=0,
            max_value=100,
            value=100,
            step=5,
            format="%d%%",
            help=(
                "Controls how much weight to place on the DeepCyc stochastic EL "
                "vs the Metryc historical EL. The historic EL weight is the remainder."
            ),
        )
        stochastic_weight = stochastic_pct / 100.0
        historic_weight = 1.0 - stochastic_weight
        blended_el = stochastic_weight * stochastic_el + historic_weight * historic_el
        if 0.0 < stochastic_weight < 1.0:
            st.caption(
                f"{stochastic_weight:.0%} × {stochastic_el:.4%} (stochastic)"
                f" + {historic_weight:.0%} × {historic_el:.4%} (historic)"
                f" = **{blended_el:.4%}** blended EL"
            )
    elif stochastic_el is not None:
        blended_el = stochastic_el
        stochastic_weight = 1.0
        historic_weight = 0.0
    else:
        blended_el = None
        stochastic_weight = 1.0
        historic_weight = 0.0

    st.divider()

    if blended_el is None:
        st.info(
            "Complete the stochastic analysis on the **Policy Setup** tab to generate pricing."
        )
    else:
        # Currency amount formatter — scales to M/K for readability
        def _fmt_amt(v: float, sym: str) -> str:
            if abs(v) >= 1_000_000:
                return f"{sym}{v / 1_000_000:,.3f}M"
            elif abs(v) >= 1_000:
                return f"{sym}{v / 1_000:,.2f}K"
            return f"{sym}{v:,.2f}"

        _CURRENCY_SYMBOLS = {
            "GBP": "£", "USD": "$", "EUR": "€", "PHP": "₱",
            "JPY": "¥", "AUD": "A$", "SGD": "S$", "MXN": "MX$",
        }

        # ---- Section 2: Pricing Parameters ---------------------------------
        st.subheader("2. Pricing Parameters")

        col_p1, col_p2, col_p3, col_p4 = st.columns([1, 1, 0.6, 1.4])
        with col_p1:
            target_nlr_pct = st.number_input(
                "Target net loss ratio (%)",
                min_value=1.0,
                max_value=200.0,
                value=60.0,
                step=1.0,
                format="%.1f",
                help=(
                    "The expected loss as a share of the net premium. "
                    "E.g. 60% means for every £1 of net premium charged, "
                    "£0.60 is expected to be paid out as claims on average."
                ),
            )
        with col_p2:
            total_comm_pct = st.number_input(
                "Total commissions (%)",
                min_value=0.0,
                max_value=99.0,
                value=32.5,
                step=0.5,
                format="%.1f",
                help=(
                    "Total distribution costs (broker commissions, fronting fees, etc.) "
                    "expressed as a percentage of the net premium."
                ),
            )
        with col_p3:
            currency = st.selectbox(
                "Currency",
                options=["GBP", "USD", "EUR", "PHP", "JPY", "AUD", "SGD", "MXN"],
                help="Currency used for displaying monetary values.",
            )
        with col_p4:
            policy_limit = st.number_input(
                "Policy limit (insured value)",
                min_value=0.0,
                value=1_000_000.0,
                step=100_000.0,
                format="%.0f",
                help=(
                    "The maximum policy payout — i.e. the insured value. "
                    "A 100% payout equals this amount. "
                    "Set to 0 to display percentages only."
                ),
            )

        target_nlr = target_nlr_pct / 100.0
        total_comm = total_comm_pct / 100.0
        currency_sym = _CURRENCY_SYMBOLS.get(currency, currency + " ")
        show_currency = policy_limit > 0

        # Core calculations
        net_premium = blended_el / target_nlr if target_nlr > 0 else 0.0
        gross_premium = net_premium / (1.0 - total_comm) if total_comm < 1.0 else 0.0

        # Monetary equivalents
        el_value = blended_el * policy_limit
        net_premium_value = net_premium * policy_limit
        gross_premium_value = gross_premium * policy_limit

        st.divider()

        # ---- Section 3: Pricing Output -------------------------------------
        st.subheader("3. Pricing Output")

        # Rate metrics (always shown)
        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Blended EL",
            f"{blended_el:.4%}",
            help=(
                f"{stochastic_weight:.0%} stochastic"
                + (f" / {historic_weight:.0%} historic" if both_available else "")
            ),
        )
        m2.metric(
            "Net Premium (rate)",
            f"{net_premium:.4%}",
            help=f"Blended EL ÷ target NLR  =  {blended_el:.4%} ÷ {target_nlr:.0%}",
        )
        m3.metric(
            "Gross Premium (rate)",
            f"{gross_premium:.4%}",
            help=(
                f"Net premium ÷ (1 − commissions)  =  "
                f"{net_premium:.4%} ÷ {1.0 - total_comm:.3f}"
            ),
        )

        # Monetary metrics (shown only when policy_limit > 0)
        if show_currency:
            c1, c2, c3 = st.columns(3)
            c1.metric(
                f"Expected Annual Loss ({currency})",
                _fmt_amt(el_value, currency_sym),
                help=f"Blended EL × policy limit  =  {blended_el:.4%} × {_fmt_amt(policy_limit, currency_sym)}",
            )
            c2.metric(
                f"Net Premium ({currency})",
                _fmt_amt(net_premium_value, currency_sym),
                help=f"Net premium rate × policy limit  =  {net_premium:.4%} × {_fmt_amt(policy_limit, currency_sym)}",
            )
            c3.metric(
                f"Gross Premium ({currency})",
                _fmt_amt(gross_premium_value, currency_sym),
                help=f"Gross premium rate × policy limit  =  {gross_premium:.4%} × {_fmt_amt(policy_limit, currency_sym)}",
            )

        # Full summary breakdown table
        el_note = f"{stochastic_weight:.0%} stochastic"
        if both_available and historic_weight > 0.0:
            el_note += f" + {historic_weight:.0%} historic"

        summary = {
            "Component": [
                "Blended Expected Loss",
                "Target Net Loss Ratio",
                "Net Premium",
                "Total Commissions",
                "Gross Premium",
            ],
            "Rate": [
                f"{blended_el:.4%}",
                f"{target_nlr:.1%}",
                f"{net_premium:.4%}",
                f"{total_comm:.1%}",
                f"{gross_premium:.4%}",
            ],
            "Formula": [
                el_note,
                "User-defined",
                "Blended EL ÷ NLR",
                "User-defined",
                "Net premium ÷ (1 − commissions)",
            ],
        }
        if show_currency:
            summary[f"Value ({currency})"] = [
                _fmt_amt(el_value, currency_sym),
                "—",
                _fmt_amt(net_premium_value, currency_sym),
                "—",
                _fmt_amt(gross_premium_value, currency_sym),
            ]
        st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

        st.divider()

        # ---- Section 4: Sensitivity Analysis --------------------------------
        st.subheader("4. Sensitivity: Net Loss Ratio")
        st.caption(
            f"Premiums across a range of target net loss ratios. "
            f"Commissions fixed at {total_comm:.1%}, blended EL = {blended_el:.4%}"
            + (f", policy limit = {_fmt_amt(policy_limit, currency_sym)}" if show_currency else "")
            + ". The row matching your current selection is marked ←."
        )

        nlr_steps = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90]
        sens_rows = []
        for nlr in nlr_steps:
            np_ = blended_el / nlr
            gp_ = np_ / (1.0 - total_comm) if total_comm < 1.0 else 0.0
            marker = " ←" if abs(nlr - target_nlr) < 0.005 else ""
            row = {
                "Target NLR": f"{nlr:.0%}{marker}",
                "Net Premium (rate)": f"{np_:.4%}",
                "Gross Premium (rate)": f"{gp_:.4%}",
            }
            if show_currency:
                row[f"Net Premium ({currency})"] = _fmt_amt(np_ * policy_limit, currency_sym)
                row[f"Gross Premium ({currency})"] = _fmt_amt(gp_ * policy_limit, currency_sym)
            sens_rows.append(row)
        st.dataframe(pd.DataFrame(sens_rows), use_container_width=True, hide_index=True)

        st.divider()

        # ---- Section 5: Export ---------------------------------------------
        st.subheader("5. Export")
        st.caption(
            "Download a complete audit-ready Excel workbook containing all inputs, "
            "results, and the Year Event Loss Table (YELT) for exposure management."
        )

        with st.expander("What's included in the export?"):
            st.markdown("""
**Summary** — all model parameters and pricing outputs in one place (peril, units, EL
basis, blended EL, premium rates and monetary values).

**Shapes** — the locations used (polygon geometry or circle lat/lon/radius).

**Payout Structure** — the hazard band table exactly as used in the analysis.

**YELT (Stochastic)** — Year Event Loss Table from the DeepCyc stochastic catalogue.
One row per triggered event, with three payout columns:

| Column | Meaning |
|--------|---------|
| `payout_%` | Raw event payout before any cap |
| `payout_capped_%` | Payout after the per-event cap (relevant for Sum aggregation) |
| `payout_effective_%` | **Net contribution after the running year total is applied.** If a prior event in the same year has already consumed part of the annual limit, this is reduced accordingly so the year total never exceeds 100%. |

**YLT (Stochastic)** — Year Loss Table: one row per loss year, showing the aggregate
annual payout before and after the annual cap.

**EP Curve** — AEP and OEP payout at standard return periods (2–1,000 years).

**YELT / YLT (Historic)** — same tables for the Metryc historical analysis, included
if you ran the Historic Events tab.

*Payout values are expressed as percentages of the insured value (e.g. 75.0 = 75%).*
""")

        export_bytes = build_export_excel(
            run_timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
            peril=peril,
            hazard_units=hazard_units,
            terrain_correction=terrain_correction,
            simulation_years=st.session_state.run_simulation_years,
            aggregation=aggregation,
            input_mode=input_mode,
            valid_shapes=valid_shapes,
            valid_circles=valid_circles,
            payout_df=payout_df if payout_df is not None else pd.DataFrame(),
            stochastic_el=stochastic_el,
            stochastic_results=st.session_state.results,
            historic_el=historic_el,
            historic_results=st.session_state.historic_results,
            stochastic_weight=stochastic_weight,
            historic_weight=historic_weight,
            blended_el=blended_el,
            target_nlr=target_nlr,
            total_comm=total_comm,
            net_premium=net_premium,
            gross_premium=gross_premium,
            policy_limit=policy_limit,
            currency=currency,
        )

        export_filename = (
            f"parametric_tc_pricing_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
        )
        st.download_button(
            "⬇ Download Full Export (Excel)",
            data=export_bytes,
            file_name=export_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            use_container_width=False,
        )


# ===========================================================================
# TAB 5 — Validation & Help
# ===========================================================================

with tab_help:

    # ---- Live validation ---------------------------------------------------
    st.subheader("Live Validation")

    _has_inputs = bool(valid_shapes) or bool(valid_circles)
    if _has_inputs and payout_df is not None and not payout_df.empty:
        if not validation_issues:
            st.success("All checks passed — no issues found with your current inputs.")
        else:
            errors = [vi for vi in validation_issues if vi["severity"] == "error"]
            warnings = [vi for vi in validation_issues if vi["severity"] == "warning"]
            if errors:
                st.markdown(f"**{len(errors)} error(s) — must be fixed before running:**")
                for ve in errors:
                    st.error(f"**{ve['check']}:** {ve['message']}")
            if warnings:
                st.markdown(f"**{len(warnings)} warning(s) — review before running:**")
                for vw in warnings:
                    st.warning(f"**{vw['check']}:** {vw['message']}")
    else:
        st.info(
            "Load shapes/circles and payout structures on the **Policy Setup** tab "
            "to run live validation checks here."
        )

    st.divider()

    # ---- Common issues guide -----------------------------------------------
    st.subheader("Common Issues Guide")

    with st.expander("My EL result is zero or suspiciously low"):
        st.markdown("""
**Possible causes:**

1. **Payout band ceiling too low.** If your top band ends at (say) 130 mph but some storm
   events in the catalogue exceed that value, those events fall outside all defined bands
   and contribute nothing to the EL. The Live Validation above will flag this. Fix: raise
   the `to` value of your highest band to something safely above the worst conceivable
   storm (e.g. 999 mph / 9999 hPa is a common convention).

2. **Gap between payout bands.** If your bands go `[0, 64)` then `[70, 96)` with nothing
   for 64–70, events with hazard values in that gap trigger no payout. Live Validation
   flags any gaps. Fix: make sure the `to` value of each band exactly equals the `from`
   value of the next.

3. **Shape ID mismatch.** The `shape_id` values in your payout CSV must exactly match
   the IDs in your shapes (case-sensitive, spaces matter). If they don't align, no payouts
   are ever found. Live Validation flags mismatches.

4. **API call returned no events.** If the API failed silently in a previous version of
   the app, the event table might be empty. This is now fixed — API errors raise
   immediately rather than returning zero events. Download the **Event Hazard Table**
   on the Results tab to inspect the raw events returned by the API.
""")

    with st.expander("The API call failed with an error"):
        st.markdown("""
**Common HTTP errors:**

- **401 Unauthorised** — wrong username/password, or your session token expired during a
  long run. Re-enter credentials and run again. Tokens typically expire after ~1 hour.
- **429 Too Many Requests** — you've hit a rate limit. Wait a few minutes, then retry.
- **500 Server Error** — a problem on Reask's side. Try again shortly; if it persists,
  contact Reask support.

**Note:** the app now raises an exception immediately on any failed API call, so errors
are always visible. Previously, a failed shape would silently contribute zero events and
the EL would be quietly underestimated with no warning shown.
""")

    with st.expander("DeepCyc (stochastic) and Metryc (historic) give very different EL rates"):
        st.markdown("""
This is expected and normal. A few things to understand:

- **DeepCyc** simulates 41,000 years of synthetic TC activity, including rare events that
  haven't occurred in the real historical record. This gives a more stable view of tail
  risk and is the primary pricing output.
- **Metryc** covers only the observed record — typically 40–60 years per basin. A single
  extreme historical event can dominate the historical rate, making it volatile.
- If Metryc EL >> DeepCyc EL, check whether one event is driving the historical rate
  (look at the Annual Payout History bar chart in the Historic Events tab).
- Differences within a factor of 2–3 are generally considered normal. Large divergences
  may indicate the payout structure triggers on historically unusual events.
""")

    with st.expander("The 'stale cache' warning keeps appearing"):
        st.markdown("""
Cached events are tied to your current **shapes, peril, simulation years, terrain
correction, and units**. If any of these change, the cache is flagged as stale.

- **▶ Run Analysis** — recalculates EL using the cached events (fast, no API call).
  Use this if you've only changed the payout structure, aggregation method, or limits.
- **↺ Re-fetch & Run** — discards the cache and re-calls the DeepCyc API (slower, uses
  API quota). Use this after changing shapes, peril, units, or terrain correction.

You can also clear the cache manually in the sidebar.
""")

    with st.expander("Peril: wind speed vs central pressure"):
        st.markdown("""
**Wind Speed** is the standard choice for most parametric TC policies. The bands
represent the maximum 1-minute averaged wind speed (open-water) at the worst point the
TC passes through the shape, in your chosen units (mph / kph / ms / knots).

**Central Pressure** represents the minimum central pressure of the TC as it passes
through the shape. Note that for central pressure:

- **Lower values = more intense storm** (880 hPa is more severe than 950 hPa).
- Your payout bands should be ordered accordingly, e.g.:
  - `[900, 1020)` → 0 % (sub-threshold — most tropical systems)
  - `[850, 900)` → 50 %
  - `[0, 850)` → 100 % (extreme events)
- Make sure the lowest `from` value (e.g. 0) extends far enough to capture the most
  intense possible storms.
- The band ceiling/gap checks in Live Validation are calibrated for wind speed only.
  When using central pressure, review your bands manually to ensure the full pressure
  range is covered.
""")

    with st.expander("Aggregation: max vs sum — which should I use?"):
        st.markdown("""
**Max** — the event payout equals the highest single-shape payout triggered.

Use this when your shapes are **concentric or overlapping** (e.g. an inner zone and
an outer zone around the same location). A TC passing through both zones should only
pay once — the worst-case zone.

**Sum** — the event payout equals the sum of all triggered shape payouts, capped at
the event limit.

Use this when your shapes are **independent locations** (e.g. two separate power
plants in different cities). A single TC can affect both simultaneously and both
should pay out. Set the event limit to 100 % (1.0) if you don't want the combined
payout to exceed full policy value in a single event.
""")

    with st.expander("Understanding simulation years"):
        st.markdown("""
The **Simulation years** setting in the sidebar controls two things simultaneously:

1. How many years of DeepCyc's stochastic catalogue are requested from the API.
2. The denominator used to convert total simulated losses into an annual EL rate.

**These two values are always kept in sync by the app** — the same number is sent to
the API and used in the EL formula. If you use the Python functions directly (outside
the app), make sure to pass the same value to both `fetch_events_for_all_shapes` and
`calculate_el`, otherwise the EL will be wrong.

The default of 41,000 years is Reask's full DeepCyc catalogue. Reducing this value
speeds up API calls but produces noisier estimates, particularly for rare tail events.
For final pricing, always use the full 41,000-year catalogue.
""")
