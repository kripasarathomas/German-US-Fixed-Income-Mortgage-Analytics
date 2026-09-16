import pandas as pd

from src.analytics import (
    calculate_changes,
    calculate_summary,
)


def test_calculate_changes():

    df = pd.DataFrame(
        {
            "date": pd.date_range(
                "2025-01-01",
                periods=15,
                freq="MS",
            ),
            "value": range(15),
        }
    )

    result = calculate_changes(df)

    assert "monthly_change" in result.columns
    assert "yearly_change" in result.columns
    assert "rolling_3m_average" in result.columns
    assert "rolling_12m_average" in result.columns


def test_calculate_summary():

    df = pd.DataFrame(
        {
            "date": pd.date_range(
                "2025-01-01",
                periods=3,
                freq="MS",
            ),
            "value": [2.5, 3.0, 3.5],
        }
    )

    summary = calculate_summary(df)

    assert summary["latest_value"] == 3.5
    assert summary["minimum"] == 2.5
    assert summary["maximum"] == 3.5