#!/usr/bin/env python3
"""Streamlit interface for comparing employee commutes to office locations."""

from __future__ import annotations

import datetime
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import folium
import googlemaps
import pandas as pd
import streamlit as st
from commute_analysis import (
    CommuteAnalyzer,
    InputValidationError,
    add_coordinate_pairs,
    apply_geocoded_coordinates,
    calculate_distance_matrix_in_chunks,
    combine_address_fields,
    element_duration_and_distance,
    ensure_coordinate_pairs,
    standardize_coordinates,
    unresolved_coordinate_mask,
)
from dateutil import tz
from streamlit_folium import st_folium
from timezonefinder import TimezoneFinder

ZIPCODE_DATA_PATH = Path(__file__).with_name("ZIP_Code_Population_Weighted_Centroids.csv")
GEOCODE_RESULT_COLUMNS = ["input_string", "formatted_address", "latitude", "longitude"]


def process_origins(df: pd.DataFrame) -> pd.DataFrame:
    """Assign origin identifiers and expand rows that represent multiple employees."""
    result = df.copy()
    if "Employee_ID" not in result.columns:
        result.insert(0, "Employee_ID", range(1, len(result) + 1))

    if "geoid" not in result.columns and "Geoid" not in result.columns:
        result.insert(1, "Geoid", range(1, len(result) + 1))
    elif "geoid" in result.columns and "Geoid" not in result.columns:
        result = result.rename(columns={"geoid": "Geoid"})

    if "count_employees" in result.columns:
        employee_counts = pd.to_numeric(result["count_employees"], errors="coerce")
        if employee_counts.isna().any() or (employee_counts < 0).any() or (employee_counts % 1 != 0).any():
            raise InputValidationError("count_employees must contain whole, non-negative numbers.")
        result = result.loc[result.index.repeat(employee_counts.astype(int))].reset_index(drop=True)
        result["Employee_ID"] = range(1, len(result) + 1)

    return result


@st.cache_data(show_spinner=False)
def geocode_single_address(address: str, api_key: str) -> dict[str, Any] | None:
    """Geocode one address and cache successful Google responses."""
    try:
        geocode_result = googlemaps.Client(key=api_key).geocode(address)
        time.sleep(0.1)
        return geocode_result[0] if geocode_result else None
    except Exception as error:  # noqa: BLE001 - Google client exceptions do not share a stable common base class.
        st.warning(f"Could not geocode an address: {error}")
        return None


def geocode_addresses(addresses: pd.Series, api_key: str) -> pd.DataFrame:
    """Geocode each distinct non-empty address once and return a stable result schema."""
    unique_addresses = pd.Series(addresses, dtype="string").dropna().str.strip()
    unique_addresses = unique_addresses[unique_addresses.ne("")].drop_duplicates().tolist()
    results: list[dict[str, Any]] = []
    progress_bar = st.progress(0)

    for index, address in enumerate(unique_addresses, start=1):
        result = geocode_single_address(address, api_key)
        if result:
            location = result.get("geometry", {}).get("location", {})
            results.append(
                {
                    "input_string": address,
                    "formatted_address": result.get("formatted_address"),
                    "latitude": location.get("lat"),
                    "longitude": location.get("lng"),
                }
            )
        progress_bar.progress(index / len(unique_addresses))
    progress_bar.empty()
    return pd.DataFrame(results, columns=GEOCODE_RESULT_COLUMNS)


def resolve_coordinates(df: pd.DataFrame, zipcode_data: pd.DataFrame, api_key: str, file_label: str) -> pd.DataFrame:
    """Resolve every row to valid coordinates using explicit values, ZIP centroids, then geocoding."""
    result = combine_address_fields(standardize_coordinates(df, zipcode_data))
    missing_coordinates = unresolved_coordinate_mask(result)

    if missing_coordinates.any():
        geocoded_addresses = geocode_addresses(result.loc[missing_coordinates, "ADDRESS_FULL"], api_key)
        result = apply_geocoded_coordinates(result, geocoded_addresses)

    result = ensure_coordinate_pairs(result)
    invalid_rows = unresolved_coordinate_mask(result)
    if invalid_rows.any():
        raise InputValidationError(
            f"{invalid_rows.sum()} {file_label} row(s) could not be resolved to valid coordinates. "
            "Provide latitude/longitude, a supported ZIP code, or a complete address."
        )

    if "Zipcode" not in result.columns:
        result["Zipcode"] = pd.NA
    return add_coordinate_pairs(result)


def get_timezone_info(df: pd.DataFrame) -> tuple[str, int]:
    """Return the first origin's timezone and the next Wednesday 8:00 AM departure timestamp."""
    first_origin = df.iloc[0]
    time_zone = TimezoneFinder().timezone_at(lng=float(first_origin["Longitude"]), lat=float(first_origin["Latitude"]))
    if not time_zone:
        raise InputValidationError("Could not determine a timezone for the first origin.")

    timezone = tz.gettz(time_zone)
    if timezone is None:
        raise InputValidationError(f"Could not load timezone '{time_zone}'.")

    current_time = datetime.datetime.now(timezone)
    days_ahead = (2 - current_time.weekday() + 7) % 7 or 7
    departure_time = datetime.datetime.combine(
        current_time.date() + datetime.timedelta(days=days_ahead),
        datetime.time(8, 0),
        tzinfo=timezone,
    )
    return time_zone, int(departure_time.timestamp())


def get_map_center(coords_list: list[tuple[float, float]]) -> list[float]:
    """Calculate the arithmetic center of the validated origin and destination coordinates."""
    return [
        sum(coordinate[0] for coordinate in coords_list) / len(coords_list),
        sum(coordinate[1] for coordinate in coords_list) / len(coords_list),
    ]


def get_api_key() -> str | None:
    """Read the Google Maps key and provide an actionable configuration error when it is absent."""
    try:
        return st.secrets["google_maps"]["api_key"]
    except KeyError:
        st.error("Google Maps API key is missing. Configure .streamlit/secrets.toml before running an analysis.")
        return None


def run_analysis(
    origins: pd.DataFrame,
    destinations: pd.DataFrame,
    method: str,
    api_key: str,
) -> dict[str, Any]:
    """Resolve input data and obtain a complete Google Distance Matrix result."""
    if origins.empty or destinations.empty:
        raise InputValidationError("Origins and destinations files must each contain at least one row.")
    if not ZIPCODE_DATA_PATH.is_file():
        raise InputValidationError(f"Missing ZIP code data file: {ZIPCODE_DATA_PATH.name}")

    zipcode_data = pd.read_csv(ZIPCODE_DATA_PATH)
    resolved_origins = resolve_coordinates(process_origins(origins), zipcode_data, api_key, "origin")
    resolved_destinations = resolve_coordinates(destinations, zipcode_data, api_key, "destination")
    _, departure_time = get_timezone_info(resolved_origins)

    client = googlemaps.Client(key=api_key)
    progress_bar = st.progress(0)

    def request_matrix(origin_chunk: Sequence[Any], destination_chunk: Sequence[Any]) -> dict[str, Any]:
        response = client.distance_matrix(
            origins=origin_chunk,
            destinations=destination_chunk,
            mode=method,
            units="imperial",
            departure_time=departure_time,
        )
        time.sleep(0.1)
        return response

    try:
        matrix_rows = calculate_distance_matrix_in_chunks(
            resolved_origins["Coords"].tolist(),
            resolved_destinations["Coords"].tolist(),
            request_matrix,
            progress_bar.progress,
        )
    finally:
        progress_bar.empty()

    results = [[element_duration_and_distance(element) for element in row] for row in matrix_rows]
    return {
        "origins": resolved_origins,
        "destinations": resolved_destinations,
        "durations": [[duration for duration, _ in row] for row in results],
        "distances": [[distance for _, distance in row] for row in results],
        "map_center": get_map_center(resolved_origins["Coords"].tolist() + resolved_destinations["Coords"].tolist()),
        "method": method,
    }


def display_results(results: dict[str, Any]) -> None:
    """Render and export the completed commute analysis."""
    st.header("Analysis Results")
    destinations = results["destinations"]
    durations_df = pd.DataFrame(results["durations"], columns=[f"Duration_to_{index}" for index in range(len(destinations))])
    distances_df = pd.DataFrame(results["distances"], columns=[f"Distance_to_{index + 1}" for index in range(len(destinations))])
    raw_results_df = pd.concat([results["origins"].reset_index(drop=True), durations_df, distances_df], axis=1)

    analyzer = CommuteAnalyzer({results["method"]: raw_results_df})
    final_df = analyzer.transform_for_visualization(analyzer.process_commute_data(), destinations)

    st.subheader("Categorized Commute Times")
    st.dataframe(final_df.head())
    st.download_button(
        "Download Categorized Data",
        final_df.to_csv(index=False).encode("utf-8"),
        f"CommuteAnalysis_{results['method']}_{datetime.datetime.now(datetime.UTC):%Y%m%d}.csv",
        "text/csv",
    )

    st.header("Map Visualization")
    map_object = folium.Map(location=results["map_center"], tiles="stadiaalidadesmooth", zoom_start=8)
    for _, row in results["origins"].iterrows():
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            popup=f"Origin: {row.get('ADDRESS_FULL', '')}",
            color="blue",
            radius=5,
        ).add_to(map_object)
    for _, row in destinations.iterrows():
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            popup=f"Destination: {row.get('ADDRESS_FULL', '')}",
            color="red",
            radius=7,
        ).add_to(map_object)
    st_folium(map_object, width=700, height=500)


def main() -> None:
    """Render the Streamlit application."""
    st.title("Commute Impact Analysis")
    st.caption("The first destination row is the current location used as the commute-change baseline.")
    if "results" not in st.session_state:
        st.session_state.results = None

    with st.sidebar.form(key="main_form"):
        st.header("Input Parameters")
        origins_file = st.file_uploader("Upload Origins CSV", type=["csv"])
        destinations_file = st.file_uploader("Upload Destinations CSV", type=["csv"])
        method = st.selectbox("Transit Method", ("driving", "transit"))
        submitted = st.form_submit_button("Run Analysis")

    if submitted:
        if origins_file is None or destinations_file is None:
            st.error("Upload both origins and destinations CSV files before running the analysis.")
        else:
            api_key = get_api_key()
            if api_key:
                try:
                    with st.spinner("Processing data..."):
                        st.session_state.results = run_analysis(
                            pd.read_csv(origins_file),
                            pd.read_csv(destinations_file),
                            method,
                            api_key,
                        )
                except InputValidationError as error:
                    st.error(str(error))
                except Exception as error:  # noqa: BLE001 - Surface service or CSV failures without retaining partial results.
                    st.error(f"Processing error: {error}")

    if st.session_state.results:
        display_results(st.session_state.results)


if __name__ == "__main__":
    main()
