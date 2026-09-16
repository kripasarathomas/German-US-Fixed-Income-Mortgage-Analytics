from src.data_loader import (
    fetch_bundesbank_data,
    MORTGAGE_RATE_SERIES,
)

import sys
from pathlib import Path

import streamlit as st
import plotly.express as px

# Allow imports from the src folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_loader import (
    fetch_bundesbank_data,
    MORTGAGE_RATE_SERIES,
)

from src.analytics import (
    parse_bundesbank_json,
    calculate_changes,
    calculate_summary,
)


st.set_page_config(
    page_title="German Mortgage Desk Analytics",
    page_icon="📊",
    layout="wide",
)


st.title("German Mortgage Desk Analytics")

st.markdown(
    """
    **Deutsche Bundesbank market-data analytics prototype**

    This dashboard analyses German housing-loan interest rates
    using publicly available Bundesbank data.
    """
)


@st.cache_data
def load_data():
    payload = fetch_bundesbank_data(
        MORTGAGE_RATE_SERIES["total"]
    )

    df = parse_bundesbank_json(payload)

    df = calculate_changes(df)

    return df


try:

    df = load_data()

    summary = calculate_summary(df)

    # ---------------------------------------------------------
    # KPI SECTION
    # ---------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Latest Mortgage Rate",
        f"{summary['latest_value']:.2f}%"
    )

    col2.metric(
        "12M Change",
        f"{df['yearly_change'].iloc[-1]:+.2f} pp"
    )

    col3.metric(
        "Minimum",
        f"{summary['minimum']:.2f}%"
    )

    col4.metric(
        "Maximum",
        f"{summary['maximum']:.2f}%"
    )

    st.divider()

    # ---------------------------------------------------------
    # MAIN CHART
    # ---------------------------------------------------------

    st.subheader("German Mortgage Rate Trend")

    fig = px.line(
        df,
        x="date",
        y="value",
        title="Effective Interest Rate — New Housing Loans",
        labels={
            "date": "Date",
            "value": "Interest Rate (%)",
        },
    )

    fig.update_layout(
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ---------------------------------------------------------
    # ROLLING AVERAGE
    # ---------------------------------------------------------

    st.subheader("Mortgage Rate vs Rolling Average")

    chart_df = df[
        [
            "date",
            "value",
            "rolling_3m_average",
            "rolling_12m_average",
        ]
    ].copy()

    chart_df = chart_df.rename(
        columns={
            "value": "Mortgage Rate",
            "rolling_3m_average": "3M Average",
            "rolling_12m_average": "12M Average",
        }
    )

    fig2 = px.line(
        chart_df,
        x="date",
        y=[
            "Mortgage Rate",
            "3M Average",
            "12M Average",
        ],
        title="Mortgage Rate Trend and Rolling Averages",
    )

    fig2.update_layout(
        hovermode="x unified"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    # ---------------------------------------------------------
    # DATA TABLE
    # ---------------------------------------------------------

    st.subheader("Market Data")

    display_df = df.copy()

    display_df["date"] = (
        display_df["date"]
        .dt.strftime("%Y-%m-%d")
    )

    st.dataframe(
        display_df,
        use_container_width=True,
    )

    st.caption(
        "Source: Deutsche Bundesbank. "
        "Data retrieved through the Bundesbank SDMX API."
    )


except Exception as exc:

    st.error(
        "Unable to load Bundesbank data."
    )

    st.exception(exc)