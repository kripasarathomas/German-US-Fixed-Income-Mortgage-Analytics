import pandas as pd

# ============================================================
# LOAD MASTER DATASET
# ============================================================

df = pd.read_csv("master_dataset.csv", parse_dates=["date"])

print("==============================================")
print("POLICY RATE ANALYSIS")
print("==============================================")

print("\nDataset shape:")
print(df.shape)

print("\nDate range:")
print(df["date"].min(), "to", df["date"].max())


# ============================================================
# 1. CORRELATION ANALYSIS
# ============================================================

correlations = pd.DataFrame({
    "relationship": [
        "ECB policy rate vs Germany mortgage rate",
        "Fed funds rate vs US mortgage rate",
        "ECB policy rate vs Germany mortgage spread",
        "Fed funds rate vs US mortgage spread",
        "Germany mortgage rate vs Bund 10Y",
        "US mortgage rate vs Treasury 10Y",
    ],
    "correlation": [
        df["ecb_policy_rate"].corr(df["total"]),
        df["fed_funds_rate"].corr(df["us_30yr_mortgage_rate"]),
        df["ecb_policy_rate"].corr(df["germany_mortgage_spread"]),
        df["fed_funds_rate"].corr(df["us_mortgage_spread"]),
        df["total"].corr(df["germany_10y_bund_yield"]),
        df["us_30yr_mortgage_rate"].corr(df["us_10y_treasury_yield"]),
    ]
})

print("\n==============================================")
print("CORRELATION ANALYSIS")
print("==============================================")

print(correlations.to_string(index=False))

correlations.to_csv(
    "policy_correlations.csv",
    index=False
)


# ============================================================
# 2. LAGGED CORRELATION ANALYSIS
# ============================================================

lags = [0, 1, 2, 3, 6]

lag_results = []

for lag in lags:

    # Policy rate from earlier month
    ecb_lagged = df["ecb_policy_rate"].shift(lag)
    fed_lagged = df["fed_funds_rate"].shift(lag)

    # Correlation with current mortgage rate
    ecb_corr = ecb_lagged.corr(df["total"])
    fed_corr = fed_lagged.corr(df["us_30yr_mortgage_rate"])

    lag_results.append({
        "lag_months": lag,
        "ecb_vs_germany_mortgage": ecb_corr,
        "fed_vs_us_mortgage": fed_corr
    })


lag_correlations = pd.DataFrame(lag_results)

print("\n==============================================")
print("LAGGED CORRELATION ANALYSIS")
print("==============================================")

print(lag_correlations.to_string(index=False))

lag_correlations.to_csv(
    "policy_lag_correlations.csv",
    index=False
)


# ============================================================
# 3. MONTHLY CHANGES
# ============================================================

df["ecb_policy_change"] = df["ecb_policy_rate"].diff()
df["fed_funds_change"] = df["fed_funds_rate"].diff()

df["germany_mortgage_change"] = df["total"].diff()
df["us_mortgage_change"] = df["us_30yr_mortgage_rate"].diff()

df["germany_bund_change"] = df["germany_10y_bund_yield"].diff()
df["us_treasury_change"] = df["us_10y_treasury_yield"].diff()


# ============================================================
# 4. CORRELATION OF MONTHLY CHANGES
# ============================================================

change_correlations = pd.DataFrame({
    "relationship": [
        "ECB policy change vs Germany mortgage change",
        "Fed funds change vs US mortgage change",
        "Bund 10Y change vs Germany mortgage change",
        "Treasury 10Y change vs US mortgage change",
    ],
    "correlation": [
        df["ecb_policy_change"].corr(
            df["germany_mortgage_change"]
        ),
        df["fed_funds_change"].corr(
            df["us_mortgage_change"]
        ),
        df["germany_bund_change"].corr(
            df["germany_mortgage_change"]
        ),
        df["us_treasury_change"].corr(
            df["us_mortgage_change"]
        ),
    ]
})

print("\n==============================================")
print("MONTHLY CHANGE CORRELATIONS")
print("==============================================")

print(change_correlations.to_string(index=False))

change_correlations.to_csv(
    "policy_change_correlations.csv",
    index=False
)


# ============================================================
# 5. SAVE EXTENDED ANALYSIS DATASET
# ============================================================

df.to_csv(
    "master_dataset_with_changes.csv",
    index=False
)


# ============================================================
# FINISHED
# ============================================================

print("\n==============================================")
print("ANALYSIS FILES CREATED")
print("==============================================")

print("1. policy_correlations.csv")
print("2. policy_lag_correlations.csv")
print("3. policy_change_correlations.csv")
print("4. master_dataset_with_changes.csv")

print("\nPolicy analysis completed successfully.")