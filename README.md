# market-regime-dashboard
# Market Regime Dashboard

This project studies how the S&P 500 reacts to key macroeconomic variables over the past ~20 years.  
The focus is on three main drivers:

- Consumer Price Index (CPI)
- Federal Funds Rate (FFR)
- 10-Year Treasury Yield (DGS10)

The project combines data cleaning, exploratory analysis, economic regime classification, lagged correlations, and a simple predictive model. The results are presented in an interactive Streamlit web app.

## 1. Research Question

How does the S&P 500 behave under different macro environments?

More specifically:

- How do inflation (CPI), policy rates (FFR), and the yield curve (10Y – FFR) co-move with the S&P 500?
- Are there meaningful lag relationships (for example, macro moves first, equities react later)?
- How do S&P 500 returns differ across macro “regimes” such as:
  - High vs low inflation
  - Tightening vs easing policy
  - Normal vs inverted yield curve
- Can a simple regression model quantify the direction and strength of these relationships?

## 2. Data Sources

All data are publicly available:

- **S&P 500 index**: Yahoo Finance, ticker `^GSPC` (via `yfinance`)
- **CPI (CPIAUCSL)**: FRED
- **Federal Funds Rate (FEDFUNDS)**: FRED
- **10-Year Treasury Yield (DGS10)**: FRED

FRED series are loaded via `pandas.read_csv` from the official FRED CSV endpoints.

## 3. Methods (High-level)

### 3.1 Data Cleaning and Missingness

- Align all time series on a common **daily** calendar.
- S&P 500: forward-fill missing non-trading days.
- CPI and FFR: originally monthly; reindexed to daily and interpolated.
- DGS10: daily but with many holiday gaps; we use time-based interpolation and forward/backward filling.
- Visualized missingness before and after interpolation using `missingno`.

### 3.2 Exploratory Analysis

- Plot raw levels of S&P 500, CPI, FFR, DGS10.
- Normalize series to start at 1.0 to compare co-movements.
- Compute a monthly correlation matrix between:
  - S&P 500 monthly returns
  - CPI YoY
  - FFR YoY
  - Yield curve (10Y – FFR)

### 3.3 Economic Regimes (Step D)

At monthly frequency we define:

- **Inflation regimes** (high / normal / low) based on CPI YoY thresholds.
- **Policy regimes** (tightening / easing / flat) based on 12-month change in FFR.
- **Yield curve regimes** (normal / inverted) based on whether 10Y – FFR is positive or negative.
- **Combined regimes** by joining inflation + policy.

For each regime we compute:

- Average S&P 500 monthly return
- Volatility (standard deviation)
- Simple Sharpe ratio

Results are shown as tables and heatmaps.

### 3.4 Lagged Correlation (Step C)

- Use S&P 500 returns and macro variables at monthly frequency.
- Compute correlations for lags from 0 to 12 months.
- Visualize lag–correlation patterns to see whether macro variables tend to lead equity returns.

### 3.5 Predictive Modeling (Step E)

Simple OLS regression at monthly frequency:

\[
\text{SP500\_ret} = \beta_0 + \beta_1 \cdot \text{CPI\_YoY} + \beta_2 \cdot \text{FFR\_YoY} + \beta_3 \cdot \text{YieldCurve} + \epsilon
\]

- We use `statsmodels` to estimate coefficients.
- The goal is **interpretation**, not forecasting performance:
  - Sign and magnitude of coefficients
  - R-squared is expected to be low, because macro only explains part of market variation.

### 3.6 Recession-style Stress Score (Step E)

- Construct a simple macro stress index based on:
  - Yield curve inversion depth
  - High inflation flag
  - Tightening policy flag
  - Negative 6-month equity momentum
- Normalize components to [0, 1] and combine into a “recession probability proxy”.
- Plot the time series with a threshold line.

## 4. Streamlit Web App
The web app presents all results in an interactive way.

### Tabs

1. **Overview**  
   - Project description  
   - Key KPI cards (latest S&P 500, CPI YoY, FFR, 10Y yield, yield curve, macro regime, recession probability)  
   - Price chart with date range selector  

2. **Data Cleaning & Missingness**  
   - Missing counts before/after interpolation  
   - Bar chart comparing missingness  
   - Sample of cleaned daily data  

3. **Exploratory Analysis**  
   - Normalized series plot (SP500, CPI, FFR, DGS10)  
   - Monthly correlation matrix  

4. **Economic Regimes**  
   - Regime counts  
   - Box plot of S&P 500 returns by regime  
   - Policy regime timeline  

5. **Lag Correlation**  
   - Interactive lag–correlation bar chart (user chooses macro variable and max lag)  

6. **Predictive Modeling**  
   - Regression model description  
   - Clean summary table of coefficients and p-values  
   - Actual vs predicted return plot  

7. **Conclusion**  
   - Key findings from regimes, lag analysis and regression  
   - Discussion of limitations and possible extensions  


## 5. Project Structure

Example structure:
