import pandas as pd


FRED_URL = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv"
    "?id=MORTGAGE30US"
    "&cosd=2019-01-01"
    "&coed=2026-09-30"
)


def load_us_mortgage_data():

    print("Downloading US mortgage rate data...")

    df = pd.read_csv(FRED_URL)

    # Rename columns
    df.columns = ["date", "us_30yr_mortgage_rate"]

    # Convert date
    df["date"] = pd.to_datetime(df["date"])

    # Convert rate to numeric
    df["us_30yr_mortgage_rate"] = pd.to_numeric(
        df["us_30yr_mortgage_rate"],
        errors="coerce"
    )

    # Remove missing values
    df = df.dropna()

    # Convert weekly data to monthly average
    df["month"] = df["date"].dt.to_period("M")

    monthly_df = (
        df.groupby("month")["us_30yr_mortgage_rate"]
        .mean()
        .reset_index()
    )

    # Convert month to actual date
    monthly_df["date"] = monthly_df["month"].dt.to_timestamp()

    monthly_df = monthly_df[
        ["date", "us_30yr_mortgage_rate"]
    ]

    return monthly_df


if __name__ == "__main__":

    df = load_us_mortgage_data()

    print("\nUS mortgage data:")
    print(df.head())

    print("\nLast rows:")
    print(df.tail())

    print("\nNumber of observations:")
    print(len(df))

    df.to_csv(
        "us_mortgage_rates.csv",
        index=False
    )

    print("\nSaved to:")
    print("us_mortgage_rates.csv")