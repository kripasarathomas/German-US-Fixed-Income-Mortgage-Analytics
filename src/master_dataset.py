import pandas as pd

germany = pd.read_csv("germany_mortgage_rates.csv", parse_dates=["date"])
us = pd.read_csv("us_mortgage_rates.csv", parse_dates=["date"])
bonds = pd.read_csv("bond_yields.csv", parse_dates=["date"])
policy = pd.read_csv("policy_rates.csv", parse_dates=["date"])

master = germany.merge(us, on="date", how="inner")
master = master.merge(bonds, on="date", how="inner")
master = master.merge(policy, on="date", how="inner")

master["germany_mortgage_spread"] = (
    master["total"] - master["germany_10y_bund_yield"]
)

master["us_mortgage_spread"] = (
    master["us_30yr_mortgage_rate"] - master["us_10y_treasury_yield"]
)

master["mortgage_rate_gap_us_minus_germany"] = (
    master["us_30yr_mortgage_rate"] - master["total"]
)

master = master.sort_values("date").reset_index(drop=True)

master.to_csv("master_dataset.csv", index=False)

print()
print("===================================")
print("MASTER FIXED-INCOME DATASET")
print("===================================")

print()
print("Shape:")
print(master.shape)

print()
print("Columns:")
print(master.columns.tolist())

print()
print("First rows:")
print(master.head())

print()
print("Last rows:")
print(master.tail())

print()
print("Missing values:")
print(master.isna().sum())

print()
print("Saved as:")
print("master_dataset.csv")