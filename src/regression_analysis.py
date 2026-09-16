import pandas as pd
import statsmodels.api as sm


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    "master_dataset.csv",
    parse_dates=["date"]
)

print("==============================================")
print("REGRESSION ANALYSIS")
print("==============================================")

print("\nDataset shape:")
print(df.shape)


# ============================================================
# 1. GERMANY MORTGAGE RATE REGRESSION
# ============================================================
#
# Dependent variable:
#   Germany mortgage rate
#
# Explanatory variables:
#   ECB policy rate
#   Germany 10Y Bund yield
#
# Model:
#   Mortgage = intercept
#            + ECB policy rate
#            + Bund 10Y yield
#

X_germany = df[
    [
        "ecb_policy_rate",
        "germany_10y_bund_yield"
    ]
]

y_germany = df["total"]

X_germany = sm.add_constant(X_germany)

germany_model = sm.OLS(
    y_germany,
    X_germany
).fit()

print("\n==============================================")
print("GERMANY MORTGAGE RATE REGRESSION")
print("==============================================")

print(germany_model.summary())


# ============================================================
# 2. USA MORTGAGE RATE REGRESSION
# ============================================================
#
# Dependent variable:
#   US 30Y mortgage rate
#
# Explanatory variables:
#   Federal Funds Rate
#   US 10Y Treasury Yield
#

X_us = df[
    [
        "fed_funds_rate",
        "us_10y_treasury_yield"
    ]
]

y_us = df["us_30yr_mortgage_rate"]

X_us = sm.add_constant(X_us)

us_model = sm.OLS(
    y_us,
    X_us
).fit()

print("\n==============================================")
print("US MORTGAGE RATE REGRESSION")
print("==============================================")

print(us_model.summary())


# ============================================================
# 3. SAVE REGRESSION COEFFICIENTS
# ============================================================

germany_coefficients = pd.DataFrame({
    "model": "Germany mortgage rate",
    "variable": germany_model.params.index,
    "coefficient": germany_model.params.values,
    "p_value": germany_model.pvalues.values,
    "r_squared": germany_model.rsquared
})

us_coefficients = pd.DataFrame({
    "model": "US 30Y mortgage rate",
    "variable": us_model.params.index,
    "coefficient": us_model.params.values,
    "p_value": us_model.pvalues.values,
    "r_squared": us_model.rsquared
})

regression_results = pd.concat(
    [
        germany_coefficients,
        us_coefficients
    ],
    ignore_index=True
)

regression_results.to_csv(
    "regression_results.csv",
    index=False
)


# ============================================================
# 4. MODEL PREDICTIONS
# ============================================================

df["germany_mortgage_predicted"] = (
    germany_model.predict(X_germany)
)

df["us_mortgage_predicted"] = (
    us_model.predict(X_us)
)

df["germany_model_error"] = (
    df["total"] -
    df["germany_mortgage_predicted"]
)

df["us_model_error"] = (
    df["us_30yr_mortgage_rate"] -
    df["us_mortgage_predicted"]
)

df.to_csv(
    "regression_predictions.csv",
    index=False
)


# ============================================================
# 5. MODEL SUMMARY
# ============================================================

model_summary = pd.DataFrame({
    "model": [
        "Germany mortgage rate",
        "US 30Y mortgage rate"
    ],
    "r_squared": [
        germany_model.rsquared,
        us_model.rsquared
    ],
    "adjusted_r_squared": [
        germany_model.rsquared_adj,
        us_model.rsquared_adj
    ],
    "observations": [
        int(germany_model.nobs),
        int(us_model.nobs)
    ]
})

model_summary.to_csv(
    "regression_model_summary.csv",
    index=False
)


print("\n==============================================")
print("REGRESSION FILES CREATED")
print("==============================================")

print("1. regression_results.csv")
print("2. regression_predictions.csv")
print("3. regression_model_summary.csv")

print("\nRegression analysis completed successfully.")