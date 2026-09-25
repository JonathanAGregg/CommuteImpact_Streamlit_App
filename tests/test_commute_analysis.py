"""Regression tests for the deterministic commute-analysis layer."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "StreamlitApp"))

from commute_analysis import (
    CommuteAnalyzer,
    DistanceMatrixError,
    apply_geocoded_coordinates,
    calculate_distance_matrix_in_chunks,
    distance_matrix_request_plan,
    standardize_coordinates,
)


def _matrix_response(origins: list[str], destinations: list[str]) -> dict[str, list[dict[str, list[dict[str, object]]]]]:
    """Return a deterministic Google-shaped response for test coordinate labels."""
    return {
        "rows": [
            {
                "elements": [
                    {
                        "status": "OK",
                        "duration": {"value": int(origin[1:]) * 10_000 + int(destination[1:])},
                        "distance": {"value": 1_000},
                    }
                    for destination in destinations
                ]
            }
            for origin in origins
        ]
    }


@pytest.mark.parametrize(("origin_count", "destination_count"), [(2, 30), (26, 2), (26, 30)])
def test_chunked_matrix_preserves_every_origin_destination_pair(origin_count: int, destination_count: int) -> None:
    origins = [f"O{index}" for index in range(origin_count)]
    destinations = [f"D{index}" for index in range(destination_count)]
    progress: list[float] = []

    matrix = calculate_distance_matrix_in_chunks(origins, destinations, _matrix_response, progress.append)

    assert len(matrix) == origin_count
    assert [len(row) for row in matrix] == [destination_count] * origin_count
    assert matrix[0][0]["duration"]["value"] == 0
    assert matrix[-1][-1]["duration"]["value"] == (origin_count - 1) * 10_000 + destination_count - 1
    assert progress[-1] == 1


def test_matrix_request_plan_respects_google_element_limit() -> None:
    plan = distance_matrix_request_plan(list(range(26)), list(range(30)))

    assert all(len(origins) <= 25 for origins, _ in plan)
    assert all(len(destinations) <= 25 for _, destinations in plan)
    assert all(len(origins) * len(destinations) <= 100 for origins, destinations in plan)


def test_chunked_matrix_rejects_malformed_response() -> None:
    with pytest.raises(DistanceMatrixError, match="origin rows"):
        calculate_distance_matrix_in_chunks(["O0"], ["D0"], lambda _origins, _destinations: {"rows": []})


def test_standardize_coordinates_maps_gis_x_to_longitude_and_y_to_latitude() -> None:
    input_data = pd.DataFrame({"X": [-71.0589], "Y": [42.3601]})
    unused_centroids = pd.DataFrame({"STD_ZIP5": ["02108"], "LATITUDE": [42.3601], "LONGITUDE": [-71.0589]})

    result = standardize_coordinates(input_data, unused_centroids)

    assert result.loc[0, "Latitude"] == pytest.approx(42.3601)
    assert result.loc[0, "Longitude"] == pytest.approx(-71.0589)
    assert result.loc[0, "Coords"] == pytest.approx((42.3601, -71.0589))


def test_standardize_coordinates_resolves_numeric_zip_code_with_lost_leading_zero() -> None:
    input_data = pd.DataFrame({"Zipcode": [2108.0]})
    centroids = pd.DataFrame({"STD_ZIP5": ["02108"], "LATITUDE": [42.357], "LONGITUDE": [-71.063]})

    result = standardize_coordinates(input_data, centroids)

    assert result.loc[0, "Zipcode"] == "02108"
    assert result.loc[0, "Latitude"] == pytest.approx(42.357)
    assert result.loc[0, "Longitude"] == pytest.approx(-71.063)


def test_geocoded_coordinate_mapping_does_not_multiply_duplicate_addresses() -> None:
    input_data = pd.DataFrame({"ADDRESS_FULL": ["1 Main Street", "1 Main Street", "2 Main Street"]})
    geocoded_addresses = pd.DataFrame(
        {
            "input_string": ["1 Main Street", "2 Main Street"],
            "latitude": [42.36, 42.35],
            "longitude": [-71.06, -71.05],
        }
    )

    result = apply_geocoded_coordinates(input_data, geocoded_addresses)

    assert len(result) == 3
    assert result["Latitude"].tolist() == [42.36, 42.36, 42.35]
    assert result["Longitude"].tolist() == [-71.06, -71.06, -71.05]


def test_visualization_transform_accepts_coordinate_only_origins_and_unavailable_routes() -> None:
    raw_results = pd.DataFrame(
        {
            "Employee_ID": [1],
            "Latitude": [42.3601],
            "Longitude": [-71.0589],
            "Duration_to_0": [None],
            "Duration_to_1": [25.0],
        }
    )
    destinations = pd.DataFrame({"ADDRESS_FULL": ["Current Office", "Potential Office"]})

    result = CommuteAnalyzer({"transit": raw_results}).transform_for_visualization(
        CommuteAnalyzer({"transit": raw_results}).process_commute_data(), destinations
    )

    assert result["Zipcode"].isna().all()
    assert result.loc[result["variable"] == "Current_Commute_Time_Bucket", "value"].item() == "Unavailable"
    assert result.loc[result["variable"] == "Commute_Time_Category_Bucket_1", "value"].item() == "Unavailable"
