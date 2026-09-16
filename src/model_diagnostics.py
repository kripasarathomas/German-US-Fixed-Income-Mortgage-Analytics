import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    "master_dataset.csv",
    parse_dates=["date"]
)


# ============================================================
# GERMANY MODEL
# ============================================================

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

df["germany_predicted"] = germany_model.predict(X_germany)


# ============================================================
# US MODEL
# ============================================================

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

df["us_predicted"] = us_model.predict(X_us)


# ============================================================
# MODEL ERROR
# ============================================================

df["germany_error"] = (
    df["total"] - df["germany_predicted"]
)

df["us_error"] = (
    df["us_30yr_mortgage_rate"] - df["us_predicted"]
)


# ============================================================
# RMSE
# ============================================================

germany_rmse = (
    (df["germany_error"] ** 2).mean()
) ** 0.5

us_rmse = (
    (df["us_error"] ** 2).mean()
) ** 0.5


# ============================================================
# MAE
# ============================================================

germany_mae = (
    df["germany_error"].abs().mean()
)

us_mae = (
    df["us_error"].abs().mean()
)


# ============================================================
# PRINT MODEL METRICS
# ============================================================

print("==============================================")
print("MODEL DIAGNOSTICS")
print("==============================================")

print("\nGermany mortgage model")
print("----------------------------------------------")
print(f"R-squared:       {germany_model.rsquared:.4f}")
print(f"Adjusted R²:     {germany_model.rsquared_adj:.4f}")
print(f"RMSE:            {germany_rmse:.4f}")
print(f"MAE:             {germany_mae:.4f}")
print(f"Durbin-Watson:   {sm.stats.stattools.durbin_watson(germany_model.resid):.4f}")


print("\nUS mortgage model")
print("----------------------------------------------")
print(f"R-squared:       {us_model.rsquared:.4f}")
print(f"Adjusted R²:     {us_model.rsquared_adj:.4f}")
print(f"RMSE:            {us_rmse:.4f}")
print(f"MAE:             {us_mae:.4f}")
print(f"Durbin-Watson:   {sm.stats.stattools.durbin_watson(us_model.resid):.4f}")


# ============================================================
# SAVE MODEL PREDICTIONS
# ============================================================

df.to_csv(
    "model_diagnostics.csv",
    index=False
)


# ============================================================
# ACTUAL VS PREDICTED — GERMANY
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    df["date"],
    df["total"],
    label="Actual Germany mortgage"
)

plt.plot(
    df["date"],
    df["germany_predicted"],
    label="Predicted Germany mortgage"
)

plt.title("Germany Mortgage Rate: Actual vs Predicted")
plt.xlabel("Date")
plt.ylabel("Mortgage rate (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "germany_actual_vs_predicted.png",
    dpi=300
)

plt.close()


# ============================================================
# ACTUAL VS PREDICTED — USA
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    df["date"],
    df["us_30yr_mortgage_rate"],
    label="Actual US 30Y mortgage"
)

plt.plot(
    df["date"],
    df["us_predicted"],
    label="Predicted US mortgage"
)

plt.title("US Mortgage Rate: Actual vs Predicted")
plt.xlabel("Date")
plt.ylabel("Mortgage rate (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "us_actual_vs_predicted.png",
    dpi=300
)

plt.close()


# ============================================================
# ERROR DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    df["germany_error"],
    bins=15,
    alpha=0.7,
    label="Germany"
)

plt.hist(
    df["us_error"],
    bins=15,
    alpha=0.7,
    label="USA"
)

plt.title("Regression Model Errors")
plt.xlabel("Prediction error (percentage points)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "model_error_distribution.png",
    dpi=300
)

plt.close()


print("\n==============================================")
print("DIAGNOSTIC FILES CREATED")
print("==============================================")

print("1. model_diagnostics.csv")
print("2. germany_actual_vs_predicted.png")
print("3. us_actual_vs_predicted.png")
print("4. model_error_distribution.png")

print("\nModel diagnostics completed successfully.")