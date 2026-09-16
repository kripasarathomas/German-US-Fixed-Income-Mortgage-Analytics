import pandas as pd

# ---------------------------------------
# Load master dataset
# ---------------------------------------

df = pd.read_csv(
    "master_dataset.csv",
    parse_dates=["date"]
)


# ---------------------------------------
# Basic statistics
# ---------------------------------------

metrics = [
    "total",
    "us_30yr_mortgage_rate",
    "germany_10y_bund_yield",
    "us_10y_treasury_yield",
    "germany_mortgage_spread",
    "us_mortgage_spread",
    "mortgage_rate_gap_us_minus_germany",
]


summary = df[metrics].agg(
    ["mean", "min", "max", "std"]
).T


summary.columns = [
    "average",
    "minimum",
    "maximum",
    "standard_deviation"
]


print("\n===================================")
print("FIXED-INCOME SUMMARY")
print("===================================")

print(
    summary.round(3)
)


# ---------------------------------------
# Germany vs US
# ---------------------------------------

print("\n===================================")
print("GERMANY VS USA")
print("===================================")

print(
    "\nAverage German mortgage rate:",
    round(df["total"].mean(), 3)
)

print(
    "Average US mortgage rate:",
    round(
        df["us_30yr_mortgage_rate"].mean(),
        3
    )
)

print(
    "Average German mortgage spread:",
    round(
        df["germany_mortgage_spread"].mean(),
        3
    )
)

print(
    "Average US mortgage spread:",
    round(
        df["us_mortgage_spread"].mean(),
        3
    )
)


# ---------------------------------------
# Largest mortgage-rate gap
# ---------------------------------------

largest_gap = df.loc[
    df["mortgage_rate_gap_us_minus_germany"].idxmax()
]

smallest_gap = df.loc[
    df["mortgage_rate_gap_us_minus_germany"].idxmin()
]


print("\n===================================")
print("MORTGAGE RATE GAP")
print("===================================")

print("\nLargest US-Germany gap:")

print(
    largest_gap[
        [
            "date",
            "total",
            "us_30yr_mortgage_rate",
            "mortgage_rate_gap_us_minus_germany"
        ]
    ]
)


print("\nSmallest US-Germany gap:")

print(
    smallest_gap[
        [
            "date",
            "total",
            "us_30yr_mortgage_rate",
            "mortgage_rate_gap_us_minus_germany"
        ]
    ]
)


# ---------------------------------------
# Largest mortgage spreads
# ---------------------------------------

largest_germany_spread = df.loc[
    df["germany_mortgage_spread"].idxmax()
]

largest_us_spread = df.loc[
    df["us_mortgage_spread"].idxmax()
]


print("\n===================================")
print("LARGEST MORTGAGE SPREADS")
print("===================================")

print("\nGermany:")

print(
    largest_germany_spread[
        [
            "date",
            "total",
            "germany_10y_bund_yield",
            "germany_mortgage_spread"
        ]
    ]
)


print("\nUSA:")

print(
    largest_us_spread[
        [
            "date",
            "us_30yr_mortgage_rate",
            "us_10y_treasury_yield",
            "us_mortgage_spread"
        ]
    ]
)


# ---------------------------------------
# Save summary
# ---------------------------------------

summary.to_csv(
    "fixed_income_summary.csv"
)

print("\nSaved summary to:")
print("fixed_income_summary.csv")