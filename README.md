# Commute Impact Analysis

A Streamlit application that compares employee commute times to office locations using the Google Maps Distance Matrix API. It accepts origins and destinations as CSV files, calculates driving or transit durations, displays the locations on a map, and exports the established long-form commute-impact CSV.

## Requirements

- Python 3.11 (the configured development-container version)
- A Google Maps API key with Distance Matrix and Geocoding access enabled

> Google Maps requests can incur charges. Use a restricted API key and configure quotas appropriate for the expected upload size.

## Local setup

```bash
git clone https://github.com/JonathanAGregg/CommuteImpact_Streamlit_App.git
cd CommuteImpact_Streamlit_App/StreamlitApp
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Set the API key in `.streamlit/secrets.toml`:

```toml
[google_maps]
api_key = "YOUR_GOOGLE_MAPS_API_KEY"
```

The real secrets file is ignored by Git. The bundled `ZIP_Code_Population_Weighted_Centroids.csv` is used when an uploaded row supplies a supported US ZIP code rather than coordinates.

Run the application from the `StreamlitApp` directory:

```bash
streamlit run StreamlitApp_PublicTransit.py
```

## CSV input contract

Upload one origins CSV and one destinations CSV.

For each row, provide one of the following location forms:

- both latitude and longitude (`Latitude`/`Longitude`, `lat`/`lon`, or GIS `Y`/`X`);
- a US ZIP code (`Zipcode`, `Zip Code`, `Zip`, `Postal Code`, or `Postal`), resolved through the bundled centroid file; or
- a complete address assembled from address, city/town, state, and ZIP/postal columns, which is geocoded through Google.

Origins may include `Employee_ID`; otherwise sequential IDs are generated. A `count_employees` column expands a row into that many employees and must contain whole, non-negative values.

The **first row in the destinations CSV is the current location**. Every subsequent destination is compared to it. Reordering destination rows changes the meaning of the commute deltas.

The app uses the first valid origin's timezone to send Google a departure time of 8:00 AM on the next Wednesday. For geographically distributed workforces, this is a single shared departure-time policy rather than per-employee local 8:00 AM routing.

## Development and validation

The deterministic processing logic is in `StreamlitApp/commute_analysis.py`; the Streamlit and Google-client integration remains in `StreamlitApp/StreamlitApp_PublicTransit.py`.

Run the regression suite from the repository root:

```bash
python -m pytest tests -q
python -m compileall -q StreamlitApp tests
```

The tests exercise coordinate normalization, ZIP lookup, Google request chunking, response-shape validation, unavailable routes, and coordinate-only inputs. They make no Google API calls.

## License

Distributed under the MIT License. See `LICENSE.txt`.
