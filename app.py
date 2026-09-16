import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import statsmodels.api as sm

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Fixed-Income Mortgage Desk Analytics",
    page_icon="📊",
    layout="wide",
)

# ============================================================
# TERMINAL THEME — CSS injection + Plotly template
# ============================================================

def inject_terminal_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap');

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }

        div[data-testid="stMetric"] {
            background-color: #161b22;
            border: 1px solid #262b33;
            border-radius: 6px;
            padding: 14px 18px 10px 18px;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #8b949e;
        }
        div[data-testid="stMetricValue"] {
            font-family: 'IBM Plex Mono', monospace;
            font-weight: 600;
            font-size: 1.6rem;
        }
        div[data-testid="stMetricDelta"] {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.85rem;
        }

        button[data-baseweb="tab"] {
            font-size: 0.9rem;
        }
        div[data-baseweb="tab-highlight"] {
            background-color: #00d4ff !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #00d4ff !important;
        }

        h2, h3 {
            margin-top: 0.6rem;
            margin-bottom: 0.4rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


TERMINAL_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(family="IBM Plex Mono, monospace", size=12, color="#c9d1d9"),
        colorway=["#00d4ff", "#f0883e", "#3fb950", "#f85149", "#a371f7"],
        xaxis=dict(showgrid=False, zeroline=False, linecolor="#30363d", tickfont=dict(size=11)),
        yaxis=dict(showgrid=True, gridcolor="#1c2128", gridwidth=1, zeroline=False, linecolor="#30363d", tickfont=dict(size=11)),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)", font=dict(size=11)),
        margin=dict(l=50, r=20, t=30, b=40),
        hovermode="x unified",
    )
)
pio.templates["terminal"] = TERMINAL_TEMPLATE
pio.templates.default = "terminal"

inject_terminal_css()

# ============================================================
# LABELS
# ============================================================

LABELS = {
    "total": "Germany Mortgage Rate",
    "us_30yr_mortgage_rate": "US 30Y Mortgage Rate",
    "ecb_policy_rate": "ECB Deposit Facility Rate",
    "fed_funds_rate": "US Federal Funds Rate",
    "germany_10y_bund_yield": "Germany 10Y Bund Yield",
    "us_10y_treasury_yield": "US 10Y Treasury Yield",
    "germany_mortgage_spread": "Germany Mortgage Spread",
    "us_mortgage_spread": "US Mortgage Spread",
    "mortgage_rate_gap_us_minus_germany": "US − Germany Mortgage Rate Gap",
    "up_to_1_year": "Up to 1 Year",
    "1_to_5_years": "1–5 Years",
    "5_to_10_years": "5–10 Years",
    "over_10_years": "Over 10 Years",
}

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv(
        "master_dataset.csv",
        parse_dates=["date"],
    )
    return df.sort_values("date").reset_index(drop=True)


df = load_data()

# ============================================================
# MODELS
# ============================================================

X_de = df[["ecb_policy_rate", "germany_10y_bund_yield"]]
y_de = df["total"]
X_de = sm.add_constant(X_de)
model_de = sm.OLS(y_de, X_de).fit()

X_us = df[["fed_funds_rate", "us_10y_treasury_yield"]]
y_us = df["us_30yr_mortgage_rate"]
X_us = sm.add_constant(X_us)
model_us = sm.OLS(y_us, X_us).fit()

# Predictions
df["germany_predicted"] = model_de.predict(X_de)
df["us_predicted"] = model_us.predict(X_us)

df["germany_error"] = df["total"] - df["germany_predicted"]
df["us_error"] = df["us_30yr_mortgage_rate"] - df["us_predicted"]

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Dashboard Controls")

start_date = st.sidebar.date_input(
    "Start date",
    df["date"].min().date(),
)

end_date = st.sidebar.date_input(
    "End date",
    df["date"].max().date(),
)

if start_date > end_date:
    st.error("Start date must be before the end date.")
    st.stop()

filtered_df = df[
    (df["date"].dt.date >= start_date)
    & (df["date"].dt.date <= end_date)
].copy()

if filtered_df.empty:
    st.warning("No observations exist for the selected date range.")
    st.stop()

latest = filtered_df.iloc[-1]

# ============================================================
# HEADER
# ============================================================

st.title("📊 Fixed-Income Mortgage Desk Analytics")

st.markdown(
    """
    **Germany vs USA | Mortgage Rates • Government Yields • Policy Rates • Spreads**

    An interactive fixed-income analytics dashboard examining the relationship
    between mortgage rates, central-bank policy rates and 10-year government
    bond yields.
    """
)

st.caption(
    f"Selected period: {start_date.strftime('%b %Y')} – "
    f"{end_date.strftime('%b %Y')}  |  "
    f"{len(filtered_df)} monthly observations"
)

st.divider()

# ============================================================
# KPI CARDS
# ============================================================

st.subheader("Market Snapshot")

previous = filtered_df.iloc[-2] if len(filtered_df) >= 2 else None

k1, k2, k3, k4 = st.columns(4)

def metric_delta(current, previous_value):
    if previous_value is None:
        return None
    return f"{current - previous_value:+.2f} pp"

with k1:
    st.metric(
        "Germany Mortgage",
        f"{latest['total']:.2f}%",
        metric_delta(
            latest["total"],
            previous["total"] if previous is not None else None,
        ),
    )

with k2:
    st.metric(
        "US 30Y Mortgage",
        f"{latest['us_30yr_mortgage_rate']:.2f}%",
        metric_delta(
            latest["us_30yr_mortgage_rate"],
            previous["us_30yr_mortgage_rate"]
            if previous is not None else None,
        ),
    )

with k3:
    st.metric(
        "Germany Mortgage Spread",
        f"{latest['germany_mortgage_spread']:.2f} pp",
        metric_delta(
            latest["germany_mortgage_spread"],
            previous["germany_mortgage_spread"]
            if previous is not None else None,
        ),
    )

with k4:
    st.metric(
        "US Mortgage Spread",
        f"{latest['us_mortgage_spread']:.2f} pp",
        metric_delta(
            latest["us_mortgage_spread"],
            previous["us_mortgage_spread"]
            if previous is not None else None,
        ),
    )

# ============================================================
# TABS
# ============================================================

tab_overview, tab_relationships, tab_models, tab_scenarios, tab_data = st.tabs(
    [
        "📈 Market Overview",
        "🔗 Rate Relationships",
        "📐 Regression Models",
        "🎚️ Scenario Analysis",
        "🗂️ Data",
    ]
)

# ============================================================
# TAB 1 — MARKET OVERVIEW
# ============================================================

with tab_overview:

    st.subheader("Mortgage Rate Comparison")

    mortgage_fig = px.line(
        filtered_df,
        x="date",
        y=["total", "us_30yr_mortgage_rate"],
        labels={
            "date": "Date",
            "value": "Mortgage Rate (%)",
            "variable": "Series",
        },
    )

    mortgage_fig.update_traces(line_width=2.5)
    mortgage_fig.for_each_trace(
        lambda trace: trace.update(name=LABELS.get(trace.name, trace.name))
    )
    mortgage_fig.update_layout(
        legend_title_text="",
        hovermode="x unified",
        yaxis_title="Mortgage Rate (%)",
        xaxis_title="",
    )

    st.plotly_chart(
        mortgage_fig,
        use_container_width=True,
    )

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Central Bank Policy Rates")

        policy_fig = px.line(
            filtered_df,
            x="date",
            y=["ecb_policy_rate", "fed_funds_rate"],
            labels={
                "date": "Date",
                "value": "Policy Rate (%)",
                "variable": "Series",
            },
        )

        policy_fig.update_traces(line_width=2.5)
        policy_fig.for_each_trace(
            lambda trace: trace.update(name=LABELS.get(trace.name, trace.name))
        )
        policy_fig.update_layout(
            legend_title_text="",
            hovermode="x unified",
            yaxis_title="Policy Rate (%)",
            xaxis_title="",
        )

        st.plotly_chart(
            policy_fig,
            use_container_width=True,
        )

    with c2:
        st.subheader("10-Year Government Bond Yields")

        bond_fig = px.line(
            filtered_df,
            x="date",
            y=[
                "germany_10y_bund_yield",
                "us_10y_treasury_yield",
            ],
            labels={
                "date": "Date",
                "value": "Yield (%)",
                "variable": "Series",
            },
        )

        bond_fig.update_traces(line_width=2.5)
        bond_fig.for_each_trace(
            lambda trace: trace.update(name=LABELS.get(trace.name, trace.name))
        )
        bond_fig.update_layout(
            legend_title_text="",
            hovermode="x unified",
            yaxis_title="Yield (%)",
            xaxis_title="",
        )

        st.plotly_chart(
            bond_fig,
            use_container_width=True,
        )

    st.subheader("Mortgage Spreads")

    spread_fig = px.line(
        filtered_df,
        x="date",
        y=[
            "germany_mortgage_spread",
            "us_mortgage_spread",
        ],
        labels={
            "date": "Date",
            "value": "Spread (percentage points)",
            "variable": "Series",
        },
    )

    spread_fig.update_traces(line_width=2.5)
    spread_fig.for_each_trace(
        lambda trace: trace.update(name=LABELS.get(trace.name, trace.name))
    )
    spread_fig.update_layout(
        legend_title_text="",
        hovermode="x unified",
        yaxis_title="Spread (pp)",
        xaxis_title="",
    )

    st.plotly_chart(
        spread_fig,
        use_container_width=True,
    )

    st.subheader("German Mortgage Rates by Maturity")

    maturity_fig = px.line(
        filtered_df,
        x="date",
        y=[
            "up_to_1_year",
            "1_to_5_years",
            "5_to_10_years",
            "over_10_years",
        ],
        labels={
            "date": "Date",
            "value": "Mortgage Rate (%)",
            "variable": "Maturity",
        },
    )

    maturity_fig.update_traces(line_width=2.5)
    maturity_fig.for_each_trace(
        lambda trace: trace.update(name=LABELS.get(trace.name, trace.name))
    )
    maturity_fig.update_layout(
        legend_title_text="",
        hovermode="x unified",
        yaxis_title="Mortgage Rate (%)",
        xaxis_title="",
    )

    st.plotly_chart(
        maturity_fig,
        use_container_width=True,
    )

# ============================================================
# TAB 2 — RATE RELATIONSHIPS
# ============================================================

with tab_relationships:

    st.subheader("Correlation Analysis")

    correlation_columns = [
        "total",
        "us_30yr_mortgage_rate",
        "ecb_policy_rate",
        "fed_funds_rate",
        "germany_10y_bund_yield",
        "us_10y_treasury_yield",
        "germany_mortgage_spread",
        "us_mortgage_spread",
    ]

    corr = filtered_df[correlation_columns].corr()
    corr_display = corr.rename(
        index=LABELS,
        columns=LABELS,
    )

    heatmap = go.Figure(
        data=go.Heatmap(
            z=corr_display.values,
            x=corr_display.columns,
            y=corr_display.index,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            text=corr_display.round(2).values,
            texttemplate="%{text}",
            hovertemplate="%{y}<br>%{x}<br>Correlation: %{z:.3f}<extra></extra>",
        )
    )

    heatmap.update_layout(
        height=650,
        xaxis_title="",
        yaxis_title="",
    )

    st.plotly_chart(
        heatmap,
        use_container_width=True,
    )

    st.caption(
        "Correlation measures co-movement, not causation."
    )

    st.divider()

    st.subheader("Monthly Change Relationships")

    change_pairs = {
        "ECB policy rate → Germany mortgage": (
            filtered_df["ecb_policy_rate"].diff(),
            filtered_df["total"].diff(),
        ),
        "Bund 10Y → Germany mortgage": (
            filtered_df["germany_10y_bund_yield"].diff(),
            filtered_df["total"].diff(),
        ),
        "Fed Funds → US mortgage": (
            filtered_df["fed_funds_rate"].diff(),
            filtered_df["us_30yr_mortgage_rate"].diff(),
        ),
        "Treasury 10Y → US mortgage": (
            filtered_df["us_10y_treasury_yield"].diff(),
            filtered_df["us_30yr_mortgage_rate"].diff(),
        ),
    }

    change_results = []

    for name, (x, y) in change_pairs.items():
        change_results.append(
            {
                "Relationship": name,
                "Correlation": x.corr(y),
            }
        )

    change_df = pd.DataFrame(change_results)

    st.dataframe(
        change_df.style.format({"Correlation": "{:.3f}"}),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.subheader("Lag Analysis")

    lag_rows = []

    for lag in [0, 1, 2, 3, 6]:
        germany_corr = (
            filtered_df["ecb_policy_rate"]
            .shift(lag)
            .corr(filtered_df["total"])
        )

        us_corr = (
            filtered_df["fed_funds_rate"]
            .shift(lag)
            .corr(filtered_df["us_30yr_mortgage_rate"])
        )

        lag_rows.append(
            {
                "Lag (months)": lag,
                "ECB → Germany Mortgage": germany_corr,
                "Fed Funds → US Mortgage": us_corr,
            }
        )

    lag_df = pd.DataFrame(lag_rows)

    st.dataframe(
        lag_df.style.format(
            {
                "ECB → Germany Mortgage": "{:.3f}",
                "Fed Funds → US Mortgage": "{:.3f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "A lag correlation shows statistical association after shifting one "
        "series relative to another. It does not establish a causal or "
        "forecasting relationship."
    )

# ============================================================
# TAB 3 — REGRESSION MODELS
# ============================================================

with tab_models:

    st.subheader("Mortgage Rate Regression Models")

    st.markdown(
        """
        **Germany:** Mortgage Rate = ECB Deposit Facility Rate + Bund 10Y Yield

        **USA:** Mortgage Rate = Federal Funds Rate + Treasury 10Y Yield
        """
    )

    m1, m2 = st.columns(2)

    with m1:
        st.markdown(
            '<h3><img src="https://flagcdn.com/24x18/de.png" '
            'style="vertical-align:middle;margin-right:8px;border-radius:2px;"> '
            'Germany</h3>',
            unsafe_allow_html=True,
        )

        a, b, c = st.columns(3)

        with a:
            st.metric("R²", f"{model_de.rsquared:.3f}")

        with b:
            st.metric(
                "Adj. R²",
                f"{model_de.rsquared_adj:.3f}",
            )

        with c:
            rmse_de = (df["germany_error"] ** 2).mean() ** 0.5
            st.metric("RMSE", f"{rmse_de:.3f} pp")

        st.write(
            f"ECB coefficient: **{model_de.params['ecb_policy_rate']:.3f}**"
        )

        st.write(
            f"Bund 10Y coefficient: "
            f"**{model_de.params['germany_10y_bund_yield']:.3f}**"
        )

        st.write(
            f"Durbin-Watson: **{sm.stats.stattools.durbin_watson(model_de.resid):.3f}**"
        )

    with m2:
        st.markdown(
            '<h3><img src="https://flagcdn.com/24x18/us.png" '
            'style="vertical-align:middle;margin-right:8px;border-radius:2px;"> '
            'USA</h3>',
            unsafe_allow_html=True,
        )

        a, b, c = st.columns(3)

        with a:
            st.metric("R²", f"{model_us.rsquared:.3f}")

        with b:
            st.metric(
                "Adj. R²",
                f"{model_us.rsquared_adj:.3f}",
            )

        with c:
            rmse_us = (df["us_error"] ** 2).mean() ** 0.5
            st.metric("RMSE", f"{rmse_us:.3f} pp")

        st.write(
            f"Fed Funds coefficient: **{model_us.params['fed_funds_rate']:.3f}**"
        )

        st.write(
            f"Treasury 10Y coefficient: "
            f"**{model_us.params['us_10y_treasury_yield']:.3f}**"
        )

        st.write(
            f"Durbin-Watson: **{sm.stats.stattools.durbin_watson(model_us.resid):.3f}**"
        )

    st.divider()

    st.subheader("Actual vs Model-Implied Mortgage Rates")

    actual_pred_option = st.selectbox(
        "Select market",
        ["Germany", "USA"],
    )

    if actual_pred_option == "Germany":

        pred_fig = px.line(
            filtered_df,
            x="date",
            y=["total", "germany_predicted"],
            labels={
                "date": "Date",
                "value": "Mortgage Rate (%)",
                "variable": "Series",
            },
        )

    else:

        pred_fig = px.line(
            filtered_df,
            x="date",
            y=[
                "us_30yr_mortgage_rate",
                "us_predicted",
            ],
            labels={
                "date": "Date",
                "value": "Mortgage Rate (%)",
                "variable": "Series",
            },
        )

    pred_fig.update_traces(line_width=2.5)
    pred_fig.update_layout(
        legend_title_text="",
        hovermode="x unified",
        yaxis_title="Mortgage Rate (%)",
        xaxis_title="",
    )

    pred_fig.for_each_trace(
        lambda trace: trace.update(
            name={
                "total": "Actual Germany Mortgage",
                "germany_predicted": "Model-Implied Germany Mortgage",
                "us_30yr_mortgage_rate": "Actual US 30Y Mortgage",
                "us_predicted": "Model-Implied US Mortgage",
            }.get(trace.name, trace.name)
        )
    )

    st.plotly_chart(
        pred_fig,
        use_container_width=True,
    )

    st.subheader("Model Diagnostics")

    diagnostic_rows = [
        {
            "Market": "Germany",
            "R²": model_de.rsquared,
            "Adjusted R²": model_de.rsquared_adj,
            "RMSE (pp)": rmse_de,
            "MAE (pp)": df["germany_error"].abs().mean(),
            "Durbin-Watson": sm.stats.stattools.durbin_watson(model_de.resid),
        },
        {
            "Market": "USA",
            "R²": model_us.rsquared,
            "Adjusted R²": model_us.rsquared_adj,
            "RMSE (pp)": rmse_us,
            "MAE (pp)": df["us_error"].abs().mean(),
            "Durbin-Watson": sm.stats.stattools.durbin_watson(model_us.resid),
        },
    ]

    diagnostics_df = pd.DataFrame(diagnostic_rows)

    st.dataframe(
        diagnostics_df.style.format(
            {
                "R²": "{:.4f}",
                "Adjusted R²": "{:.4f}",
                "RMSE (pp)": "{:.4f}",
                "MAE (pp)": "{:.4f}",
                "Durbin-Watson": "{:.4f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.warning(
        "Model limitation: the low Durbin-Watson statistics indicate "
        "substantial residual autocorrelation. These OLS models should "
        "therefore be treated as illustrative analytical models rather "
        "than causal or forecasting models."
    )

    st.divider()

    st.subheader("Key Analytical Observations")

    germany_corr = filtered_df["ecb_policy_rate"].corr(
        filtered_df["total"]
    )

    us_corr = filtered_df["fed_funds_rate"].corr(
        filtered_df["us_30yr_mortgage_rate"]
    )

    germany_bond_corr = filtered_df["germany_10y_bund_yield"].corr(
        filtered_df["total"]
    )

    us_bond_corr = filtered_df["us_10y_treasury_yield"].corr(
        filtered_df["us_30yr_mortgage_rate"]
    )

    st.markdown(
        f"""
        - Germany mortgage rates and the ECB Deposit Facility Rate have a
          **{germany_corr:.3f}** contemporaneous correlation in the selected sample.
        - US 30Y mortgage rates and the Federal Funds Rate have a
          **{us_corr:.3f}** contemporaneous correlation.
        - Germany mortgage rates and the Bund 10Y yield have a
          **{germany_bond_corr:.3f}** correlation.
        - US 30Y mortgage rates and the Treasury 10Y yield have a
          **{us_bond_corr:.3f}** correlation.
        - These are **statistical associations within the selected sample**;
          they should not be interpreted as proof of causality.
        """
    )

# ============================================================
# TAB 4 — SCENARIO ANALYSIS
# ============================================================

with tab_scenarios:

    st.subheader("Interactive Mortgage Rate Scenario Analysis")

    st.markdown(
        """
        Adjust the central-bank policy rate and 10-year government bond yield
        to calculate the mortgage rate implied by the regression model.

        **These are model-implied scenarios, not forecasts.**
        """
    )

    s1, s2 = st.columns(2)

    with s1:

        st.markdown(
            '<h2><img src="https://flagcdn.com/28x21/de.png" '
            'style="vertical-align:middle;margin-right:10px;border-radius:2px;"> '
            'Germany</h2>',
            unsafe_allow_html=True,
        )

        current_ecb = float(latest["ecb_policy_rate"])
        current_bund = float(latest["germany_10y_bund_yield"])

        ecb_input = st.slider(
            "ECB Deposit Facility Rate (%)",
            min_value=-0.50,
            max_value=6.00,
            value=max(-0.50, min(6.00, round(current_ecb * 4) / 4)),
            step=0.25,
        )

        bund_input = st.slider(
            "Bund 10Y Yield (%)",
            min_value=-1.00,
            max_value=6.00,
            value=max(-1.00, min(6.00, round(current_bund * 4) / 4)),
            step=0.25,
        )

        germany_prediction = model_de.predict(
            pd.DataFrame(
                {
                    "const": [1],
                    "ecb_policy_rate": [ecb_input],
                    "germany_10y_bund_yield": [bund_input],
                }
            )
        ).iloc[0]

        st.metric(
            "Model-Implied German Mortgage Rate",
            f"{germany_prediction:.2f}%",
        )

        germany_change = germany_prediction - latest["total"]

        st.metric(
            "Change vs Latest Observed Rate",
            f"{germany_change:+.2f} pp",
        )

    with s2:

        st.markdown(
            '<h2><img src="https://flagcdn.com/28x21/us.png" '
            'style="vertical-align:middle;margin-right:10px;border-radius:2px;"> '
            'USA</h2>',
            unsafe_allow_html=True,
        )

        current_fed = float(latest["fed_funds_rate"])
        current_treasury = float(latest["us_10y_treasury_yield"])

        fed_input = st.slider(
            "US Federal Funds Rate (%)",
            min_value=0.00,
            max_value=7.00,
            value=max(0.00, min(7.00, round(current_fed * 4) / 4)),
            step=0.25,
        )

        treasury_input = st.slider(
            "Treasury 10Y Yield (%)",
            min_value=0.00,
            max_value=7.00,
            value=max(0.00, min(7.00, round(current_treasury * 4) / 4)),
            step=0.25,
        )

        us_prediction = model_us.predict(
            pd.DataFrame(
                {
                    "const": [1],
                    "fed_funds_rate": [fed_input],
                    "us_10y_treasury_yield": [treasury_input],
                }
            )
        ).iloc[0]

        st.metric(
            "Model-Implied US 30Y Mortgage Rate",
            f"{us_prediction:.2f}%",
        )

        us_change = us_prediction - latest["us_30yr_mortgage_rate"]

        st.metric(
            "Change vs Latest Observed Rate",
            f"{us_change:+.2f} pp",
        )

    st.divider()

    st.subheader("Standardized Scenarios")

    scenario_names = ["Lower Rate", "Baseline", "Higher Rate"]

    scenario_inputs_de = {
        "Lower Rate": (1.50, 1.50),
        "Baseline": (2.25, 2.50),
        "Higher Rate": (3.00, 3.50),
    }

    scenario_inputs_us = {
        "Lower Rate": (3.00, 3.00),
        "Baseline": (3.63, 4.00),
        "Higher Rate": (5.00, 5.00),
    }

    scenario_rows = []

    for name in scenario_names:

        ecb_value, bund_value = scenario_inputs_de[name]

        de_prediction = model_de.predict(
            pd.DataFrame(
                {
                    "const": [1],
                    "ecb_policy_rate": [ecb_value],
                    "germany_10y_bund_yield": [bund_value],
                }
            )
        ).iloc[0]

        fed_value, treasury_value = scenario_inputs_us[name]

        us_prediction_scenario = model_us.predict(
            pd.DataFrame(
                {
                    "const": [1],
                    "fed_funds_rate": [fed_value],
                    "us_10y_treasury_yield": [treasury_value],
                }
            )
        ).iloc[0]

        scenario_rows.append(
            {
                "Scenario": name,
                "Germany Mortgage (%)": de_prediction,
                "US 30Y Mortgage (%)": us_prediction_scenario,
            }
        )

    scenario_df = pd.DataFrame(scenario_rows)

    st.dataframe(
        scenario_df.style.format(
            {
                "Germany Mortgage (%)": "{:.2f}",
                "US 30Y Mortgage (%)": "{:.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

# ============================================================
# TAB 5 — DATA
# ============================================================

with tab_data:

    st.subheader("Master Dataset")

    st.write(
        f"Showing {len(filtered_df)} observations from "
        f"{start_date} to {end_date}."
    )

    display_df = filtered_df.copy()

    display_df.columns = [
        LABELS.get(column, column)
        for column in display_df.columns
    ]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "⬇️ Download Selected Data as CSV",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="fixed_income_selected_data.csv",
        mime="text/csv",
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Data period: January 2019 – June 2026. "
    "Mortgage rates, government yields and policy rates are analyzed on a "
    "monthly basis. Regression outputs are model-implied analytical estimates, "
    "not forecasts. Correlation does not imply causation. "
    "For educational and portfolio purposes."
)