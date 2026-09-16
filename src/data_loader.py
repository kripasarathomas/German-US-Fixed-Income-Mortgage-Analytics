import requests
import json
import pandas as pd

BASE_URL = "https://api.statistiken.bundesbank.de/rest/data"

MORTGAGE_RATE_SERIES = {
    "total": "M.DE.B.A2C.A.C.A.2250.EUR.N",
    "up_to_1_year": "M.DE.B.A2C.F.R.A.2250.EUR.N",
    "1_to_5_years": "M.DE.B.A2C.I.R.A.2250.EUR.N",
    "5_to_10_years": "M.DE.B.A2C.O.R.A.2250.EUR.N",
    "over_10_years": "M.DE.B.A2C.P.R.A.2250.EUR.N",
}


def fetch_bundesbank_data(
    series_key: str,
    start_period: str = "2019-01",
    end_period: str = "2026-09",
):
    """
    Fetch a single monthly time series from the
    Deutsche Bundesbank SDMX API.
    """

    flow_ref = "BBIM1"
    url = f"{BASE_URL}/{flow_ref}/{series_key}"

    params = {
        "startPeriod": start_period,
        "endPeriod": end_period,
        "format": "sdmx_json",
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    return response.json()


def extract_series(data, column_name):
    """
    Extract observations from Bundesbank SDMX JSON
    and convert them into a DataFrame.
    """

    series = data["data"]["dataSets"][0]["series"]

    series_data = next(iter(series.values()))

    observations = series_data["observations"]

    rows = []

    for period_index, values in observations.items():
        rows.append({
            "period_index": int(period_index),
            column_name: float(values[0])
        })

    df = pd.DataFrame(rows)

    df["date"] = pd.date_range(
        start="2019-01-01",
        periods=len(df),
        freq="MS"
    )

    return df[["date", column_name]]


if __name__ == "__main__":

    all_data = None

    for name, series_key in MORTGAGE_RATE_SERIES.items():

        print(f"\nDownloading: {name}")

        data = fetch_bundesbank_data(series_key)

        print(f"Downloaded: {name}")

        # Save raw JSON
        with open(
            f"bundesbank_{name}.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(data, f, indent=2)

        # Extract clean data
        df = extract_series(data, name)

        print(f"Observations: {len(df)}")

        # Combine with previous series
        if all_data is None:
            all_data = df
        else:
            all_data = all_data.merge(
                df,
                on="date",
                how="outer"
            )

    # Sort by date
    all_data = all_data.sort_values("date")

    # Save combined dataset
    all_data.to_csv(
        "germany_mortgage_rates.csv",
        index=False
    )

    print("\n--------------------------------")
    print("German mortgage dataset created")
    print("--------------------------------")

    print("\nColumns:")
    print(all_data.columns.tolist())

    print("\nShape:")
    print(all_data.shape)

    print("\nFirst rows:")
    print(all_data.head())

    print("\nLast rows:")
    print(all_data.tail())

    print("\nSaved to:")
    print("germany_mortgage_rates.csv")