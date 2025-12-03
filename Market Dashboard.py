import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Market Regime Dashboard",
    layout="wide",
)

st.title("📈 Market Regime Dashboard")
st.write(
    """
    This dashboard explores the relationship between **stock performance, interest rate changes, 
    and macroeconomic conditions**.  
    You can choose different stocks, time ranges, and examine returns, correlation, and drawdowns
    under different interest-rate regimes.
    """
)

# Data loading functions (with caching)
@st.cache_data
def load_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily stock data and compute returns and cumulative returns."""
    data = yf.download(ticker, start=start, end=end, progress=False)
    if data.empty:
        return data
    # Ensure timezone-naive index for merging
    data.index = data.index.tz_localize(None)
    data["Return"] = data["Adj Close"].pct_change()
    data["CumReturn"] = (1 + data["Return"]).cumprod() - 1
    return data


@st.cache_data
def load_fred_series(series_id: str) -> pd.DataFrame:
    """
    Load a single time series from FRED via CSV (no API key required).
    Examples: FEDFUNDS (policy rate), CPIAUCSL (CPI), DGS10 (10y Treasury yield).
    """
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url)
    df["DATE"] = pd.to_datetime(df["DATE"])
    df.set_index("DATE", inplace=True)
    df.replace(".", np.nan, inplace=True)
    df[series_id] = df[series_id].astype(float)
    return df


@st.cache_data
def load_macro_data() -> pd.DataFrame:
    """Combine several FRED series into a single macro dataframe."""
    ffr = load_fred_series("FEDFUNDS")     # Federal funds rate
    cpi = load_fred_series("CPIAUCSL")     # CPI index
    dgs10 = load_fred_series("DGS10")      # 10-year Treasury yield

    macro = ffr.join([cpi, dgs10], how="outer")
    macro.columns = ["FFR", "CPI", "DGS10"]
    # Sort, interpolate, and forward-fill missing values
    macro = macro.sort_index().interpolate().ffill()
    return macro


@st.cache_data
def compute_regimes(macro: pd.DataFrame) -> pd.DataFrame:
    """
    Define simple interest-rate regimes based on changes in the federal funds rate:
    - Tightening: FFR increases more than 0.02
    - Easing:    FFR decreases less than -0.02
    - Flat:      small changes in between
    """
    df = macro.copy()
    df["FFR_change"] = df["FFR"].diff()
    conds = [
        df["FFR_change"] > 0.02,
        df["FFR_change"] < -0.02,
    ]
    choices = ["Tightening", "Easing"]
    df["Regime"] = np.select(conds, choices, default="Flat")
    return df


@st.cache_data
def align_stock_macro(stock: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """
    Align daily stock data with lower-frequency macro data by forward-filling
    macro values to each trading day.
    """
    df = stock.join(macro, how="left")
    df[["FFR", "CPI", "DGS10", "Regime"]] = df[["FFR", "CPI", "DGS10", "Regime"]].ffill()
    return df


# Sidebar controls

with st.sidebar:
    st.header("⚙️ Controls")

    default_start = datetime.today() - timedelta(days=365 * 5)
    default_end = datetime.today()

    ticker = st.selectbox(
        "Select stock / index",
        options=["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "SPY", "QQQ"],
        index=7,  # default: SPY
    )

    start_date = st.date_input("Start date", default_start).strftime("%Y-%m-%d")
    end_date = st.date_input("End date", default_end).strftime("%Y-%m-%d")

    benchmark = st.selectbox(
        "Benchmark for correlation",
        options=["SPY", "QQQ"],
        index=0,
    )

    window = st.slider(
        "Rolling correlation window (days)",
        min_value=30,
        max_value=252,
        value=90,
        step=10,
    )


# Load data

stock = load_stock_data(ticker, start_date, end_date)
bench = load_stock_data(benchmark, start_date, end_date)
macro = load_macro_data()
macro_reg = compute_regimes(macro)

if stock.empty or bench.empty:
    st.error("No data available for the selected period. Please adjust the dates or ticker.")
    st.stop()

# Align macro data to stock date range
macro_reg = macro_reg.loc[stock.index.min(): stock.index.max()]
data = align_stock_macro(stock, macro_reg)

# Rolling correlation with benchmark
merged = stock[["Return"]].join(
    bench["Return"], how="inner", lsuffix="_stock", rsuffix="_bench"
)
merged["RollingCorr"] = merged["Return_stock"].rolling(window).corr(
    merged["Return_bench"]
)


# Drawdown calculation
def compute_drawdown(series: pd.Series) -> pd.Series:
    """
    Compute percentage drawdown from running maximum of cumulative wealth.
    """
    cummax = np.maximum.accumulate(series.fillna(0) + 1)
    dd = (series + 1) / cummax - 1
    return dd


data["Drawdown"] = compute_drawdown(data["CumReturn"].fillna(0))


# Layout: tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Overview", "🏛 Market Regimes", "🔗 Rolling Correlation", "📉 Drawdown", "📖 Documentation"]
)

# Tab 1: Overview

with tab1:
    st.subheader(f"Price and Return Overview: {ticker}")

    col1, col2 = st.columns(2)

    with col1:
        fig_price = go.Figure()
        fig_price.add_trace(
            go.Scatter(x=data.index, y=data["Adj Close"], mode="lines", name=ticker)
        )
        fig_price.update_layout(
            title=f"{ticker} Adjusted Close Price",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
        )
        st.plotly_chart(fig_price, use_container_width=True)

    with col2:
        fig_cum = go.Figure()
        fig_cum.add_trace(
            go.Scatter(x=data.index, y=data["CumReturn"], mode="lines", name=ticker)
        )
        fig_cum.update_layout(
            title=f"{ticker} Cumulative Return",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
        )
        st.plotly_chart(fig_cum, use_container_width=True)

    st.markdown("### Summary statistics")
    latest = data.dropna().iloc[-1]
    st.write(
        {
            "Total Return (%)": round(latest["CumReturn"] * 100, 2),
            "Mean Daily Return (%)": round(data["Return"].mean() * 100, 3),
            "Volatility (Daily std, %)": round(data["Return"].std() * 100, 3),
        }
    )

# Tab 2: Market Regimes 
with tab2:
    st.subheader("Interest Rate Regimes and Stock Returns")

    regime_colors = {
        "Tightening": "red",
        "Easing": "green",
        "Flat": "gray",
    }

    fig_reg = go.Figure()

    for regime, group in data.groupby("Regime"):
        fig_reg.add_trace(
            go.Scatter(
                x=group.index,
                y=group["CumReturn"],
                mode="lines",
                name=regime,
                line=dict(color=regime_colors.get(regime, "blue")),
            )
        )

    fig_reg.update_layout(
        title=f"{ticker} Cumulative Return by FFR Regime",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
    )
    st.plotly_chart(fig_reg, use_container_width=True)

    st.markdown(
        """
        **Interpretation hints:**  
        - Red:   interest-rate **tightening** periods  
        - Green: interest-rate **easing** periods  
        - Gray:  relatively **flat** periods  

        You can visually inspect whether this stock behaves differently across regimes.
        """
    )

    st.markdown("### Mean daily return by regime")
    st.write(
        data.groupby("Regime")["Return"]
        .mean()
        .mul(100)
        .round(3)
        .rename("Mean Daily Return (%)")
    )

# Tab 3: Rolling Correlation

with tab3:
    st.subheader(f"{ticker} vs {benchmark}: Rolling Correlation")

    fig_corr = go.Figure()
    fig_corr.add_trace(
        go.Scatter(
            x=merged.index,
            y=merged["RollingCorr"],
            mode="lines",
            name="Rolling Correlation",
        )
    )
    fig_corr.update_layout(
        title=f"{window}-day Rolling Correlation: {ticker} vs {benchmark}",
        xaxis_title="Date",
        yaxis_title="Correlation",
        yaxis_range=[-1, 1],
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown(
        """
        **How to read this:**  
        - Correlation close to **1** → the stock moves very similarly to the benchmark  
        - Correlation near **0**   → more independent behavior  
        - Correlation **negative** → potential hedging properties  
        """
    )

# Tab 4: Drawdown

with tab4:
    st.subheader(f"{ticker} Drawdown Analysis")

    fig_dd = go.Figure()
    fig_dd.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Drawdown"],
            mode="lines",
            name="Drawdown",
        )
    )
    fig_dd.update_layout(
        title=f"{ticker} Drawdown over Time",
        xaxis_title="Date",
        yaxis_title="Drawdown",
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    max_dd = data["Drawdown"].min()
    st.write(f"**Max drawdown:** {round(max_dd * 100, 2)} %")

# Tab 5: Documentation 

with tab5:
    st.subheader("Project Documentation")

    st.markdown(
        """
        ### 1. Project goals  
        - Explore how stock returns relate to interest-rate regimes and macroeconomic conditions.  
        - Provide an interactive visualization tool to examine performance of different stocks 
          under different market environments.  

        ### 2. Data sources  
        - **Prices & returns**: Yahoo Finance via the `yfinance` Python package  
        - **Federal funds rate (FEDFUNDS)**: FRED  
        - **CPI index (CPIAUCSL)**: FRED  
        - **10-year Treasury yield (DGS10)**: FRED  

        ### 3. Methods  
        - Compute daily returns, cumulative returns, and drawdowns.  
        - Define interest-rate regimes (Tightening / Easing / Flat) based on changes in the 
          federal funds rate.  
        - Compute rolling correlation with a benchmark index (SPY / QQQ).  
        - Build an interactive dashboard using Streamlit and Plotly.  

        ### 4. Alignment with CMSE 830 learning goals  
        - Multi-source data integration (equity prices + several macroeconomic series).  
        - Data cleaning and interpolation of missing values.  
        - Feature engineering (returns, cumulative returns, drawdown, regime labels, 
          rolling correlation).  
        - Multiple interactive visualizations (prices, returns, regime-segmented performance, 
          rolling correlation, drawdown).  
        - Deployed as a Streamlit web app and version-controlled via a public GitHub repository.  
        """
    )