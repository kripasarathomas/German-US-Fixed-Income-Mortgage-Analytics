import pandas as pd


GERMANY_10Y_URL = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv"
    "?id=IRLTLT01DEM156N"
    "&cosd=2019-01-01"
    "&coed=2026-09-30"
)

US_10Y_URL = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv"
    "?id=DGS10"
    "&cosd=2019-01-01"
    "&coed=2026-09-30"
)


def load_germany_10y():

    print("Downloading German 10Y Bund yield...")

    df = pd.read_csv(GERMANY_10Y_URL)

    df.columns = ["date", "germany_10y_bund_yield"]

    df["date"] = pd.to_datetime(df["date"])

    df["germany_10y_bund_yield"] = pd.to_numeric(
        df["germany_10y_bund_yield"],
        errors="coerce"
    )

    df = df.dropna()

    return df


def load_us_10y():

    print("Downloading US 10Y Treasury yield...")

    df = pd.read_csv(US_10Y_URL)

    df.columns = ["date", "us_10y_treasury_yield"]

    df["date"] = pd.to_datetime(df["date"])

    df["us_10y_treasury_yield"] = pd.to_numeric(
        df["us_10y_treasury_yield"],
        errors="coerce"
    )

    df = df.dropna()

    # Convert daily observations to monthly average
    df["month"] = df["date"].dt.to_period("M")

    monthly_df = (
        df.groupby("month")["us_10y_treasury_yield"]
        .mean()
        .reset_index()
    )

    monthly_df["date"] = monthly_df["month"].dt.to_timestamp()

    monthly_df = monthly_df[
        ["date", "us_10y_treasury_yield"]
    ]

    return monthly_df


if __name__ == "__main__":

    germany_df = load_germany_10y()

    us_df = load_us_10y()

    print("\nGerman 10Y:")
    print(germany_df.head())

    print("\nUS 10Y:")
    print(us_df.head())

    # Merge both datasets
    bond_df = germany_df.merge(
        us_df,
        on="date",
        how="inner"
    )

    # Sort by date
    bond_df = bond_df.sort_values("date")

    print("\nCombined bond yield data:")
    print(bond_df.head())

    print("\nLast rows:")
    print(bond_df.tail())

    print("\nNumber of observations:")
    print(len(bond_df))

    # Save
    bond_df.to_csv(
        "bond_yields.csv",
        index=False
    )

    print("\nSaved to:")
    print("bond_yields.csv")