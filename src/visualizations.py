import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# LOAD MASTER DATASET
# ============================================================

df = pd.read_csv(
    "master_dataset.csv",
    parse_dates=["date"]
)

print("Master dataset loaded successfully.")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# CHART 1
# Germany vs USA Mortgage Rates
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    df["date"],
    df["total"],
    label="Germany mortgage"
)

plt.plot(
    df["date"],
    df["us_30yr_mortgage_rate"],
    label="US 30Y mortgage"
)

plt.title("Germany vs USA Mortgage Rates")
plt.xlabel("Date")
plt.ylabel("Mortgage rate (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "germany_vs_us_mortgage_rates.png",
    dpi=300
)

plt.close()


# ============================================================
# CHART 2
# Germany vs USA Mortgage Spreads
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    df["date"],
    df["germany_mortgage_spread"],
    label="Germany mortgage spread"
)

plt.plot(
    df["date"],
    df["us_mortgage_spread"],
    label="US mortgage spread"
)

plt.title("Mortgage Spread: Germany vs USA")
plt.xlabel("Date")
plt.ylabel("Spread (percentage points)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "mortgage_spreads.png",
    dpi=300
)

plt.close()


# ============================================================
# CHART 3
# Germany 10Y Bund vs US 10Y Treasury
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    df["date"],
    df["germany_10y_bund_yield"],
    label="Germany 10Y Bund"
)

plt.plot(
    df["date"],
    df["us_10y_treasury_yield"],
    label="US 10Y Treasury"
)

plt.title("10-Year Government Bond Yields")
plt.xlabel("Date")
plt.ylabel("Yield (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "10y_government_yields.png",
    dpi=300
)

plt.close()


# ============================================================
# CHART 4
# German Mortgage Rates by Maturity
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    df["date"],
    df["up_to_1_year"],
    label="Up to 1 year"
)

plt.plot(
    df["date"],
    df["1_to_5_years"],
    label="1–5 years"
)

plt.plot(
    df["date"],
    df["5_to_10_years"],
    label="5–10 years"
)

plt.plot(
    df["date"],
    df["over_10_years"],
    label="Over 10 years"
)

plt.title("German Mortgage Rates by Maturity")
plt.xlabel("Date")
plt.ylabel("Mortgage rate (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "german_mortgage_maturity.png",
    dpi=300
)

plt.close()


# ============================================================
# FINISHED
# ============================================================

print()
print("==============================================")
print("ALL CHARTS CREATED SUCCESSFULLY")
print("==============================================")
print()
print("1. germany_vs_us_mortgage_rates.png")
print("2. mortgage_spreads.png")
print("3. 10y_government_yields.png")
print("4. german_mortgage_maturity.png")
print()