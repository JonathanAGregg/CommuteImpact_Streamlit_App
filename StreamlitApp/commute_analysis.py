"""Deterministic data preparation and response handling for the commute app."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

MAX_ORIGINS_PER_REQUEST = 25
MAX_DESTINATIONS_PER_REQUEST = 25
MAX_ELEMENTS_PER_REQUEST = 100

LATITUDE_ALIASES = ("latitude", "lat", "y")
LONGITUDE_ALIASES = ("longitude", "lon", "lng", "lgt", "x")
ZIPCODE_ALIASES = ("zipcode", "zip code", "zip", "postal code", "postal")


class InputValidationError(ValueError):
    """Raised when an uploaded file cannot produce valid routing inputs."""


class DistanceMatrixError(RuntimeError):
    """Raised when a Google Distance Matrix response has an unexpected shape."""


@dataclass(frozen=True)
class CommuteData:
    """The raw commute results grouped by transportation method."""

    dataframes_by_method: Mapping[str, pd.DataFrame]


def _find_column(columns: Iterable[str], aliases: Sequence[str]) -> str | None:
    """Return the first matching column using an ordered set of normalized aliases."""
    normalized_columns = {str(column).strip().casefold(): column for column in columns}
    return next((normalized_columns[alias] for alias in aliases if alias in normalized_columns), None)


def _normalize_zipcodes(values: pd.Series) -> pd.Series:
    """Normalize common numeric and ZIP+4 representations to five-character ZIP codes."""
    normalized = values.astype("string").str.strip().str.extract(r"^(\d{1,5})(?:\.0+)?(?:-\d{4})?$")[0]
    return normalized.str.zfill(5)


def valid_coordinate_mask(df: pd.DataFrame) -> pd.Series:
    """Return rows that have numeric, geographically valid latitude and longitude values."""
    if "Latitude" not in df.columns or "Longitude" not in df.columns:
        return pd.Series(False, index=df.index)

    latitude = pd.to_numeric(df["Latitude"], errors="coerce")
    longitude = pd.to_numeric(df["Longitude"], errors="coerce")
    return latitude.between(-90, 90) & longitude.between(-180, 180)


def add_coordinate_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``Coords`` pairs after coercing the canonical coordinate columns to numeric values."""
    result = df.copy()
    result["Latitude"] = pd.to_numeric(result["Latitude"], errors="coerce")
    result["Longitude"] = pd.to_numeric(result["Longitude"], errors="coerce")
    result["Coords"] = list(zip(result["Latitude"], result["Longitude"]))
    return result


def standardize_coordinates(df: pd.DataFrame, zipcode_data: pd.DataFrame) -> pd.DataFrame:
    """Use explicit coordinates or ZIP centroids to add canonical coordinate columns.

    Explicit coordinates take precedence. A partial coordinate pair is intentionally left unresolved,
    allowing the caller to fall back to address geocoding rather than routing with invalid data.
    """
    result = df.copy()
    latitude_column = _find_column(result.columns, LATITUDE_ALIASES)
    longitude_column = _find_column(result.columns, LONGITUDE_ALIASES)

    if latitude_column and longitude_column:
        result["Latitude"] = result[latitude_column]
        result["Longitude"] = result[longitude_column]
        return add_coordinate_pairs(result)

    zipcode_column = _find_column(result.columns, ZIPCODE_ALIASES)
    if not zipcode_column:
        return result

    centroid_zipcode_column = _find_column(zipcode_data.columns, ("std_zip5", *ZIPCODE_ALIASES))
    centroid_latitude_column = _find_column(zipcode_data.columns, LATITUDE_ALIASES)
    centroid_longitude_column = _find_column(zipcode_data.columns, LONGITUDE_ALIASES)
    if not all((centroid_zipcode_column, centroid_latitude_column, centroid_longitude_column)):
        raise InputValidationError("The ZIP centroid file is missing ZIP code or coordinate columns.")

    result["Zipcode"] = _normalize_zipcodes(result[zipcode_column])
    centroids = zipcode_data[
        [centroid_zipcode_column, centroid_latitude_column, centroid_longitude_column]
    ].copy()
    centroids.columns = ["_centroid_zipcode", "_centroid_latitude", "_centroid_longitude"]
    centroids["_centroid_zipcode"] = _normalize_zipcodes(centroids["_centroid_zipcode"])
    centroids = centroids.drop_duplicates("_centroid_zipcode")

    result = result.merge(
        centroids,
        how="left",
        left_on="Zipcode",
        right_on="_centroid_zipcode",
        validate="many_to_one",
    ).drop(columns=["_centroid_zipcode"])
    result["Latitude"] = result.pop("_centroid_latitude")
    result["Longitude"] = result.pop("_centroid_longitude")
    return add_coordinate_pairs(result)


def combine_address_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Create a consistently named full address from commonly supplied address fields."""
    result = df.copy()
    address_columns = [
        column
        for column in result.columns
        if str(column).strip().casefold() in {"city", "town", "state", "zip", "zipcode", "zip code", "postal", "postal code"}
        or str(column).strip().casefold().startswith("address")
    ]
    result["ADDRESS_FULL"] = (
        result[address_columns].apply(lambda row: ", ".join(row.dropna().astype(str)), axis=1)
        if address_columns
        else ""
    )
    return result


def unresolved_coordinate_mask(df: pd.DataFrame) -> pd.Series:
    """Return the rows that still need address geocoding."""
    return ~valid_coordinate_mask(df)


def apply_geocoded_coordinates(df: pd.DataFrame, geocoded_addresses: pd.DataFrame) -> pd.DataFrame:
    """Fill unresolved rows from one geocoded coordinate record per full address.

    Mapping rather than merging prevents duplicate uploaded addresses from multiplying origin or
    destination records.
    """
    required_geocode_columns = {"input_string", "latitude", "longitude"}
    if not required_geocode_columns.issubset(geocoded_addresses.columns):
        raise InputValidationError("Geocoding results are missing required coordinate columns.")
    if "ADDRESS_FULL" not in df.columns:
        raise InputValidationError("Address geocoding requires an ADDRESS_FULL column.")

    result = df.copy()
    unresolved_rows = unresolved_coordinate_mask(result)
    geocoded_by_address = geocoded_addresses.drop_duplicates("input_string").set_index("input_string")
    result.loc[unresolved_rows, "Latitude"] = result.loc[unresolved_rows, "ADDRESS_FULL"].map(
        geocoded_by_address["latitude"]
    )
    result.loc[unresolved_rows, "Longitude"] = result.loc[unresolved_rows, "ADDRESS_FULL"].map(
        geocoded_by_address["longitude"]
    )
    return result


def ensure_coordinate_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure normalized coordinate pairs are present once all fallbacks have been attempted."""
    if "Latitude" not in df.columns or "Longitude" not in df.columns:
        raise InputValidationError("Provide latitude/longitude, a supported ZIP code, or a geocodable address.")
    return add_coordinate_pairs(df)


def _chunked(values: Sequence[Any], size: int) -> list[Sequence[Any]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def distance_matrix_request_plan(
    origins: Sequence[Any], destinations: Sequence[Any]
) -> list[tuple[Sequence[Any], Sequence[Any]]]:
    """Build requests that honor Google origin, destination, and element limits."""
    if not origins or not destinations:
        raise InputValidationError("At least one origin and one destination are required.")

    plans: list[tuple[Sequence[Any], Sequence[Any]]] = []
    for origin_chunk in _chunked(origins, MAX_ORIGINS_PER_REQUEST):
        destination_chunk_size = min(
            MAX_DESTINATIONS_PER_REQUEST,
            MAX_ELEMENTS_PER_REQUEST // len(origin_chunk),
        )
        plans.extend((origin_chunk, destination_chunk) for destination_chunk in _chunked(destinations, destination_chunk_size))
    return plans


def calculate_distance_matrix_in_chunks(
    origins: Sequence[Any],
    destinations: Sequence[Any],
    request_matrix: Callable[[Sequence[Any], Sequence[Any]], Mapping[str, Any]],
    on_progress: Callable[[float], None] | None = None,
) -> list[list[Mapping[str, Any]]]:
    """Fetch a complete origin-destination matrix while preserving input row order.

    Google returns one row per origin for each destination chunk. Accumulating results per origin,
    rather than per response, prevents rows from different origins being combined when destinations
    exceed a single request.
    """
    request_plan = distance_matrix_request_plan(origins, destinations)
    assembled_rows: list[list[Mapping[str, Any]]] = [[] for _ in origins]
    completed_requests = 0
    origin_offset = 0

    for origin_chunk in _chunked(origins, MAX_ORIGINS_PER_REQUEST):
        destination_chunk_size = min(
            MAX_DESTINATIONS_PER_REQUEST,
            MAX_ELEMENTS_PER_REQUEST // len(origin_chunk),
        )
        for destination_chunk in _chunked(destinations, destination_chunk_size):
            response = request_matrix(origin_chunk, destination_chunk)
            response_rows = response.get("rows", [])
            if len(response_rows) != len(origin_chunk):
                raise DistanceMatrixError("Google returned an unexpected number of origin rows.")

            for row_index, response_row in enumerate(response_rows):
                elements = response_row.get("elements", [])
                if len(elements) != len(destination_chunk):
                    raise DistanceMatrixError("Google returned an unexpected number of destination elements.")
                assembled_rows[origin_offset + row_index].extend(elements)

            completed_requests += 1
            if on_progress:
                on_progress(completed_requests / len(request_plan))
        origin_offset += len(origin_chunk)

    return assembled_rows


def element_duration_and_distance(element: Mapping[str, Any]) -> tuple[float | None, float | None]:
    """Convert a Google matrix element to minutes and miles, preserving unavailable routes as null."""
    if element.get("status") != "OK":
        return None, None

    try:
        return element["duration"]["value"] / 60, element["distance"]["value"] * 0.000621371
    except (KeyError, TypeError):
        return None, None


class CommuteAnalyzer:
    """Transform matrix results into the established downloadable visualization format."""

    def __init__(self, dataframes_by_method: Mapping[str, pd.DataFrame]):
        self.commute_data = CommuteData(dataframes_by_method=dataframes_by_method)

    @staticmethod
    def _commute_time_bucket(commute_time: float | None) -> str:
        if pd.isna(commute_time):
            return "Unavailable"
        if commute_time == 0:
            return "0"
        if commute_time <= 15:
            return "0-15 Minutes"
        if commute_time <= 30:
            return "15-30 Minutes"
        if commute_time <= 45:
            return "30-45 Minutes"
        if commute_time <= 60:
            return "45-60 Minutes"
        return "60 Minutes +"

    @staticmethod
    def _calculate_time_change(time_diff: float | None) -> int | None:
        if pd.isna(time_diff):
            return None
        if time_diff <= -20:
            return -25
        if time_diff <= -15:
            return -20
        if time_diff <= -10:
            return -15
        if time_diff < 0:
            return -5
        if time_diff <= 5:
            return 5 if time_diff > 0 else 0
        if time_diff <= 10:
            return 10
        if time_diff <= 15:
            return 15
        if time_diff <= 20:
            return 20
        return 25

    @staticmethod
    def _determine_time_change(time_diff: float | None) -> str:
        if pd.isna(time_diff):
            return "Unavailable"
        if time_diff < 0:
            return "Time Reduced"
        if time_diff > 0:
            return "Time Added"
        return "No Change"

    def process_commute_data(self) -> pd.DataFrame:
        """Combine transportation-method dataframes and label every record with its method."""
        if not self.commute_data.dataframes_by_method:
            raise InputValidationError("No commute data was supplied.")
        return pd.concat(
            [df.assign(Method=method) for method, df in self.commute_data.dataframes_by_method.items()],
            ignore_index=True,
        )

    def transform_for_visualization(self, df: pd.DataFrame, destinations_df: pd.DataFrame) -> pd.DataFrame:
        """Convert wide matrix results to the existing long-form download schema."""
        duration_columns = [column for column in df.columns if column.startswith("Duration_to_")]
        if not duration_columns:
            raise InputValidationError("The distance matrix did not contain duration columns.")
        if len(duration_columns) != len(destinations_df):
            raise InputValidationError("The number of matrix duration columns does not match the destinations.")

        required_columns = {"Employee_ID", "Method", "Latitude", "Longitude"}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise InputValidationError(f"Missing required result columns: {', '.join(sorted(missing_columns))}.")

        destination_addresses = destinations_df.get("ADDRESS_FULL", pd.Series("", index=destinations_df.index)).fillna("").tolist()
        result_rows: list[dict[str, Any]] = []

        for _, row in df.iterrows():
            employee_base = {
                "Employee_ID": row["Employee_ID"],
                "Method": row["Method"],
                "Latitude": row["Latitude"],
                "Longitude": row["Longitude"],
                "Zipcode": row.get("Zipcode", pd.NA),
            }
            current_commute = row[duration_columns[0]]
            current_destination = destination_addresses[0]
            result_rows.extend(
                [
                    {
                        **employee_base,
                        "variable": "CurrentCommute_Time",
                        "value": "" if pd.isna(current_commute) else str(current_commute),
                        "names": current_destination,
                    },
                    {
                        **employee_base,
                        "variable": "Current_Commute_Time_Bucket",
                        "value": self._commute_time_bucket(current_commute),
                        "names": current_destination,
                    },
                ]
            )

            for index, duration_column in enumerate(duration_columns[1:], start=1):
                commute_time = row[duration_column]
                time_diff = None if pd.isna(current_commute) or pd.isna(commute_time) else commute_time - current_commute
                time_change_bucket = self._calculate_time_change(time_diff)
                destination_address = destination_addresses[index]
                result_rows.extend(
                    [
                        {
                            **employee_base,
                            "variable": f"Potential_Location_{index}",
                            "value": "" if pd.isna(commute_time) else str(commute_time),
                            "names": destination_address,
                        },
                        {
                            **employee_base,
                            "variable": f"Potential_Commute_Time_Reduced_Bucket_{index}",
                            "value": "" if time_change_bucket is None else str(time_change_bucket),
                            "names": destination_address,
                        },
                        {
                            **employee_base,
                            "variable": f"Change_Commute_{index}",
                            "value": "" if time_diff is None else str(time_diff),
                            "names": destination_address,
                        },
                        {
                            **employee_base,
                            "variable": f"Commute_Time_Reduced_Bucket_{index}",
                            "value": "" if time_change_bucket is None else str(time_change_bucket),
                            "names": destination_address,
                        },
                        {
                            **employee_base,
                            "variable": f"Commute_Time_Category_Bucket_{index}",
                            "value": self._determine_time_change(time_diff),
                            "names": destination_address,
                        },
                    ]
                )

        return pd.DataFrame(
            result_rows,
            columns=["Employee_ID", "Method", "Latitude", "Longitude", "Zipcode", "variable", "value", "names"],
        )
