import pandas as pd
import statsmodels.api as sm


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    "master_dataset.csv",
    parse_dates=["date"]
)


# ============================================================
# FIT GERMANY MODEL
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


# ============================================================
# FIT US MODEL
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


# ============================================================
# SCENARIO DEFINITIONS
# ============================================================

germany_scenarios = pd.DataFrame({
    "scenario": [
        "Lower rates",
        "Baseline",
        "Higher rates"
    ],
    "ecb_policy_rate": [
        1.50,
        2.25,
        3.00
    ],
    "germany_10y_bund_yield": [
        1.50,
        2.50,
        3.50
    ]
})


us_scenarios = pd.DataFrame({
    "scenario": [
        "Lower rates",
        "Baseline",
        "Higher rates"
    ],
    "fed_funds_rate": [
        3.00,
        3.63,
        5.00
    ],
    "us_10y_treasury_yield": [
        3.00,
        4.00,
        5.00
    ]
})


# ============================================================
# GERMANY SCENARIO PREDICTIONS
# ============================================================

germany_X = sm.add_constant(
    germany_scenarios[
        [
            "ecb_policy_rate",
            "germany_10y_bund_yield"
        ]
    ],
    has_constant="add"
)

germany_scenarios["estimated_mortgage_rate"] = (
    germany_model.predict(germany_X)
)


# ============================================================
# US SCENARIO PREDICTIONS
# ============================================================

us_X = sm.add_constant(
    us_scenarios[
        [
            "fed_funds_rate",
            "us_10y_treasury_yield"
        ]
    ],
    has_constant="add"
)

us_scenarios["estimated_mortgage_rate"] = (
    us_model.predict(us_X)
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("==============================================")
print("SCENARIO ANALYSIS")
print("==============================================")


print("\nGERMANY SCENARIOS")
print("----------------------------------------------")

print(
    germany_scenarios.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)


print("\nUSA SCENARIOS")
print("----------------------------------------------")

print(
    us_scenarios.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

germany_scenarios.to_csv(
    "germany_mortgage_scenarios.csv",
    index=False
)

us_scenarios.to_csv(
    "us_mortgage_scenarios.csv",
    index=False
)


print("\n==============================================")
print("SCENARIO FILES CREATED")
print("==============================================")

print("1. germany_mortgage_scenarios.csv")
print("2. us_mortgage_scenarios.csv")

print("\nScenario analysis completed successfully.")