import pandas as pd
import requests
from io import StringIO


# ============================================================
# DATA SOURCES
# ============================================================

# ECB Deposit Facility Rate
# Source: European Central Bank via FRED
ECB_URL = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv"
    "?id=ECBDFR"
    "&cosd=2019-01-01"
    "&coed=2026-09-30"
)

# US Federal Funds Effective Rate
# Source: Federal Reserve via FRED
FED_URL = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv"
    "?id=FEDFUNDS"
    "&cosd=2019-01-01"
    "&coed=2026-09-30"
)


# ============================================================
# LOAD ECB DATA
# ============================================================

print("Downloading ECB Deposit Facility Rate...")

ecb_response = requests.get(ECB_URL)

print("ECB/FRED status:", ecb_response.status_code)

ecb_response.raise_for_status()

ecb = pd.read_csv(
    StringIO(ecb_response.text)
)

print("ECB columns:", ecb.columns.tolist())

print("ECB data downloaded successfully.")


# ============================================================
# PROCESS ECB DATA
# ============================================================

ecb["observation_date"] = pd.to_datetime(
    ecb["observation_date"]
)

ecb["ECBDFR"] = pd.to_numeric(
    ecb["ECBDFR"],
    errors="coerce"
)

ecb = ecb.rename(
    columns={
        "observation_date": "date",
        "ECBDFR": "ecb_policy_rate"
    }
)

ecb = ecb[
    ["date", "ecb_policy_rate"]
].copy()


# Convert daily ECB data to monthly.
# Use the last available observation in each month.
ecb["date"] = (
    ecb["date"]
    .dt.to_period("M")
    .dt.to_timestamp()
)

ecb = (
    ecb.groupby("date", as_index=False)["ecb_policy_rate"]
    .last()
)


# ============================================================
# LOAD US FEDERAL FUNDS RATE
# ============================================================

print("\nDownloading US Federal Funds Rate...")

fed_response = requests.get(FED_URL)

print("Fed status:", fed_response.status_code)

fed_response.raise_for_status()

fed = pd.read_csv(
    StringIO(fed_response.text)
)

print("Fed columns:", fed.columns.tolist())

print("Federal Funds Rate data downloaded successfully.")


# ============================================================
# PROCESS FED DATA
# ============================================================

fed["observation_date"] = pd.to_datetime(
    fed["observation_date"]
)

fed["FEDFUNDS"] = pd.to_numeric(
    fed["FEDFUNDS"],
    errors="coerce"
)

fed = fed.rename(
    columns={
        "observation_date": "date",
        "FEDFUNDS": "fed_funds_rate"
    }
)

fed = fed[
    ["date", "fed_funds_rate"]
].copy()


# FEDFUNDS is already monthly
fed["date"] = (
    fed["date"]
    .dt.to_period("M")
    .dt.to_timestamp()
)

fed = (
    fed.groupby("date", as_index=False)["fed_funds_rate"]
    .last()
)


# ============================================================
# MERGE ECB + FED
# ============================================================

policy_rates = pd.merge(
    ecb,
    fed,
    on="date",
    how="outer"
)

policy_rates = policy_rates.sort_values(
    "date"
)


# ============================================================
# SAVE DATASET
# ============================================================

policy_rates.to_csv(
    "policy_rates.csv",
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print()
print("==============================================")
print("POLICY RATE DATASET CREATED")
print("==============================================")

print()
print("Shape:")
print(policy_rates.shape)

print()
print("Columns:")
print(policy_rates.columns.tolist())

print()
print("First 10 rows:")
print(policy_rates.head(10))

print()
print("Last 10 rows:")
print(policy_rates.tail(10))

print()
print("Missing values:")
print(policy_rates.isna().sum())

print()
print("Saved as:")
print("policy_rates.csv")