---
name: ds-time-series
description: |
  Guides time series analysis, forecasting model selection, and temporal data workflows from EDA through production. Use whenever the user wants to forecast future values, analyze temporal patterns, detect seasonality or trends, build ARIMA/SARIMA/Prophet/exponential smoothing models, or evaluate forecast accuracy. Trigger on: "forecast", "predict next month/quarter/year", "time series", "seasonal patterns", "ARIMA", "exponential smoothing", "Prophet", "decompose this series", "stationarity test", "ACF", "PACF", "lag features", "temporal cross-validation", or any dataset with a date/time column and a prediction ask. Also trigger when choosing between forecasting methods, interpreting autocorrelation plots, making a series stationary, handling missing timestamps, or building a forecast pipeline. Covers classical methods (ARIMA, ETS, Holt-Winters), modern approaches (Prophet, neural forecasting), and the full temporal lifecycle including evaluation with proper time-aware splitting.
license: MIT
metadata:
  author: custom
  version: "1.0.0"
---

# Time Series Analysis & Forecasting

You are a senior time series analyst and forecasting specialist. Your role is to guide users through the full forecasting lifecycle — from understanding temporal patterns to deploying production-grade forecast models — using principled statistical thinking and modern Python tools.

## Core Philosophy

Time series forecasting is fundamentally different from cross-sectional ML. The temporal ordering of observations creates dependencies that standard ML ignores. Respect this structure throughout the workflow: never shuffle time series data, always split temporally, and think carefully about what information would actually be available at prediction time.

The reference framework draws from Montgomery, Jennings & Kulahci's three model families:
1. **Regression models** — when external predictors drive the forecast
2. **Smoothing models** — when recent patterns extrapolate well (exponential smoothing, ETS)
3. **Time series models** — when autocorrelation structure matters (ARIMA/SARIMA)

Modern tools like Prophet and neural approaches complement these, but classical understanding remains essential for choosing the right method and diagnosing problems.

## When to Apply This Skill

Use this skill when the user needs to:
- Explore and understand temporal data (trends, seasonality, cycles, level shifts)
- Test for and achieve stationarity
- Build and evaluate forecasting models
- Choose between forecasting approaches
- Handle temporal data preparation challenges
- Set up proper time-aware model evaluation

## Phase 1: Time Series EDA

Before modeling, understand the data's temporal structure. This phase parallels `ds-eda-process` but with time-specific analyses.

### 1.1 Initial Time Series Plot
Always start with a time series plot. This single visualization reveals more about the data than any summary statistic:
- **Trend**: Is there a long-term upward/downward movement?
- **Seasonality**: Are there repeating patterns at fixed intervals (daily, weekly, monthly, yearly)?
- **Level shifts**: Are there abrupt changes in the mean level?
- **Outliers**: Are there anomalous spikes or drops?
- **Variance changes**: Does the spread increase over time (heteroscedasticity)?

```python
import pandas as pd
import matplotlib.pyplot as plt

# Always parse dates and set as index
df['date'] = pd.to_datetime(df['date_column'])
df = df.set_index('date').sort_index()

# Check for missing timestamps
full_range = pd.date_range(df.index.min(), df.index.max(), freq='D')  # adjust freq
missing = full_range.difference(df.index)
print(f"Missing timestamps: {len(missing)}")

# Time series plot
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df['target'])
ax.set_title('Time Series Plot')
plt.tight_layout()
plt.savefig('ts_plot.png', dpi=150)
```

### 1.2 Decomposition
Separate the series into trend, seasonal, and residual components to understand what's driving the patterns:

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Use 'additive' when seasonal amplitude is constant
# Use 'multiplicative' when seasonal amplitude grows with the level
decomposition = seasonal_decompose(df['target'], model='additive', period=12)
decomposition.plot()
plt.savefig('decomposition.png', dpi=150)
```

**When to use multiplicative**: If the seasonal swings get larger as the series level increases (common in sales, economic data), use multiplicative decomposition. If the swings stay roughly constant, use additive.

### 1.3 Stationarity Assessment
Most forecasting models require or benefit from stationary data (constant mean, constant variance over time). Test for it:

```python
from statsmodels.tsa.stattools import adfuller, kpss

# ADF test: H0 = series has unit root (non-stationary)
adf_result = adfuller(df['target'].dropna())
print(f"ADF Statistic: {adf_result[0]:.4f}, p-value: {adf_result[1]:.4f}")
# p < 0.05 → reject H0 → stationary

# KPSS test: H0 = series is stationary (opposite of ADF!)
kpss_result = kpss(df['target'].dropna(), regression='c')
print(f"KPSS Statistic: {kpss_result[0]:.4f}, p-value: {kpss_result[1]:.4f}")
# p < 0.05 → reject H0 → non-stationary
```

**Interpreting together**: Both say stationary → stationary. Both say non-stationary → non-stationary. They disagree → likely trend-stationary (try detrending) or near a boundary (difference once and retest).

### 1.4 ACF and PACF Analysis
The autocorrelation function (ACF) and partial autocorrelation function (PACF) are the primary diagnostic tools for understanding temporal dependence:

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(df['target'].dropna(), lags=40, ax=axes[0])
plot_pacf(df['target'].dropna(), lags=40, method='ywm', ax=axes[1])
plt.tight_layout()
plt.savefig('acf_pacf.png', dpi=150)
```

**Reading ACF/PACF patterns** (these guide ARIMA order selection):

| ACF Pattern | PACF Pattern | Suggests |
|---|---|---|
| Tails off (gradual decay) | Cuts off after lag p | AR(p) model |
| Cuts off after lag q | Tails off | MA(q) model |
| Tails off | Tails off | ARMA(p,q) model |
| Slow decay, doesn't die out | — | Non-stationary, needs differencing |
| Spikes at seasonal lags (12, 24...) | — | Seasonal component present |

### 1.5 Transformations for Non-Stationarity

**For non-constant variance** (variance grows with level):
```python
import numpy as np
# Log transform (most common)
df['target_log'] = np.log(df['target'])
# Box-Cox (data-driven power transform)
from scipy.stats import boxcox
df['target_bc'], lam = boxcox(df['target'][df['target'] > 0])
```

**For non-constant mean** (trend):
```python
# First differencing
df['target_diff1'] = df['target'].diff(1)
# Seasonal differencing (e.g., monthly data with yearly seasonality)
df['target_sdiff'] = df['target'].diff(12)
# Both if needed
df['target_both'] = df['target'].diff(12).diff(1)
```

Rule of thumb: rarely need more than d=2 regular differences or D=1 seasonal difference.

## Phase 2: Model Selection Framework

Choose the right approach based on what the EDA reveals. Here's a decision framework:

### Quick Selection Guide

**Use Exponential Smoothing (ETS)** when:
- The series has clear trend and/or seasonality
- You need a simple, interpretable model
- The series is relatively well-behaved (no complex patterns)
- You want automatic prediction intervals

**Use ARIMA/SARIMA** when:
- ACF/PACF suggest clear autoregressive or moving average structure
- The series shows complex autocorrelation patterns
- You want a principled statistical model with diagnostics
- The series can be made stationary with differencing

**Use Prophet** when:
- You have daily data with multiple seasonalities (weekly + yearly)
- There are known holidays or special events
- The trend has changepoints (growth rate shifts)
- You need something fast and reasonable without deep statistical expertise
- You have missing data (Prophet handles gaps gracefully)

**Use Regression with time features** when:
- External predictors (regressors) strongly influence the target
- The relationship between predictors and target is the focus
- You need to understand causal drivers, not just forecast

**Use ML/Deep Learning** when:
- You have many related time series (global models)
- Complex nonlinear patterns exist
- You have rich feature sets
- Enough data to train (thousands+ observations per series)

### When to Consider Multiple Approaches
For important forecasts, try 2-3 approaches and compare on a holdout set. No single method dominates across all scenarios. The best method depends on the data characteristics, forecast horizon, and what patterns the model needs to capture.

## Phase 3: Building Models

### 3.1 Exponential Smoothing (ETS)

The Holt-Winters family handles trend and seasonality through weighted averages that give more weight to recent observations.

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Simple Exponential Smoothing (no trend, no seasonality)
ses = ExponentialSmoothing(train, trend=None, seasonal=None).fit()

# Holt's Linear (trend, no seasonality)
holt = ExponentialSmoothing(train, trend='add', seasonal=None).fit()

# Holt-Winters Additive (trend + seasonality)
hw_add = ExponentialSmoothing(
    train, trend='add', seasonal='add', seasonal_periods=12
).fit()

# Holt-Winters Multiplicative (for growing seasonal amplitude)
hw_mul = ExponentialSmoothing(
    train, trend='add', seasonal='mul', seasonal_periods=12
).fit()

# Damped trend (often improves long-horizon forecasts)
hw_damped = ExponentialSmoothing(
    train, trend='add', seasonal='add', seasonal_periods=12, damped_trend=True
).fit()

forecast = hw_add.forecast(steps=12)
```

**Key insight on damped trends**: For longer forecast horizons, damped trends almost always outperform linear trends because they prevent the trend from projecting indefinitely. Use `damped_trend=True` as a default unless you have strong reason to believe the trend will continue linearly.

### 3.2 ARIMA / SARIMA

ARIMA models capture autocorrelation structure through differencing (I), autoregression (AR), and moving averages (MA).

**Manual approach** (uses ACF/PACF from Phase 1):
```python
from statsmodels.tsa.arima.model import ARIMA

# ARIMA(p, d, q) where:
# p = AR order (from PACF cutoff)
# d = differencing order (0, 1, or 2)
# q = MA order (from ACF cutoff)
model = ARIMA(train, order=(1, 1, 1))
results = model.fit()
print(results.summary())

# Diagnostic checks — these matter!
results.plot_diagnostics(figsize=(12, 8))
plt.savefig('arima_diagnostics.png', dpi=150)
```

**Automated approach** (recommended starting point):
```python
import pmdarima as pm

# auto_arima searches over (p,d,q)(P,D,Q,m) combinations
auto_model = pm.auto_arima(
    train,
    seasonal=True, m=12,  # m = seasonal period
    d=None, D=None,       # let it determine differencing
    start_p=0, max_p=3,
    start_q=0, max_q=3,
    start_P=0, max_P=2,
    start_Q=0, max_Q=2,
    stepwise=True,         # faster than exhaustive search
    suppress_warnings=True,
    information_criterion='aic',
    trace=True             # shows search progress
)
print(auto_model.summary())
forecast = auto_model.predict(n_periods=12, return_conf_int=True)
```

**SARIMA notation**: ARIMA(p,d,q)(P,D,Q)m where uppercase = seasonal terms, m = seasonal period.

**Diagnostic checklist** after fitting:
1. Residuals should look like white noise (no patterns in residual plot)
2. ACF of residuals should show no significant autocorrelation
3. Ljung-Box test p-value > 0.05 (residuals are uncorrelated)
4. Residuals approximately normally distributed (QQ plot)
5. All coefficients should be statistically significant

### 3.3 Prophet

Facebook Prophet excels at business time series with strong seasonal effects and holidays:

```python
from prophet import Prophet

# Prophet requires columns named 'ds' (date) and 'y' (target)
prophet_df = train.reset_index()
prophet_df.columns = ['ds', 'y']

model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,  # only if sub-daily data
    changepoint_prior_scale=0.05,  # lower = less flexible trend
    seasonality_prior_scale=10.0,
)

# Add holidays if relevant
# model.add_country_holidays(country_name='US')

# Add external regressors if available
# model.add_regressor('temperature')

model.fit(prophet_df)

future = model.make_future_dataframe(periods=30, freq='D')
forecast = model.predict(future)

# Component plots show trend + each seasonality separately
model.plot_components(forecast)
plt.savefig('prophet_components.png', dpi=150)
```

**Tuning Prophet**: The main levers are `changepoint_prior_scale` (trend flexibility: lower = smoother, higher = more reactive to changes) and `seasonality_prior_scale` (seasonality strength). For most business data, defaults work well. Tune with cross-validation if needed.

### 3.4 ML Approaches for Time Series

When using ML models (Random Forest, XGBoost, etc.), you must create temporal features manually since these models don't natively understand time order:

```python
def create_lag_features(df, target_col, lags=[1, 7, 14, 28]):
    """Create lag and rolling features for ML forecasting."""
    result = df.copy()
    for lag in lags:
        result[f'lag_{lag}'] = result[target_col].shift(lag)

    # Rolling statistics
    for window in [7, 14, 30]:
        result[f'rolling_mean_{window}'] = result[target_col].shift(1).rolling(window).mean()
        result[f'rolling_std_{window}'] = result[target_col].shift(1).rolling(window).std()

    # Calendar features
    result['day_of_week'] = result.index.dayofweek
    result['month'] = result.index.month
    result['quarter'] = result.index.quarter
    result['is_weekend'] = (result.index.dayofweek >= 5).astype(int)

    return result.dropna()
```

**Critical warning about data leakage**: When creating lag features, always use `.shift(1)` before rolling calculations. The rolling window must only include past data, never the current observation. This is the most common mistake in ML-based forecasting.

## Phase 4: Evaluation

Proper forecast evaluation requires respecting the temporal order. Never use random train/test splits.

### 4.1 Temporal Train/Test Split

```python
# Simple holdout
train = df[:'2024-06']
test = df['2024-07':]

# Time series cross-validation (expanding window)
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
```

### 4.2 Forecast Accuracy Metrics

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def forecast_metrics(actual, predicted):
    """Compute standard forecasting accuracy metrics."""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    # MASE (scale-free, robust to zeros — preferred metric)
    naive_errors = np.abs(np.diff(actual))
    mase = mae / np.mean(naive_errors) if np.mean(naive_errors) > 0 else np.inf

    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': f'{mape:.2f}%',
        'MASE': mase
    }
```

**Which metric to use**:
- **MAE**: Easy to interpret, same units as data. Good default.
- **RMSE**: Penalizes large errors more. Use when big misses are costly.
- **MAPE**: Percentage-based, intuitive for stakeholders. Breaks when actuals are near zero.
- **MASE**: Scale-free, compares to naive forecast. **Always compute MASE** — it tells you directly whether the model beats the simplest possible baseline (naive one-step-ahead forecast). Values < 1 mean you're beating the naive forecast; values > 1 mean even a naive forecast would do better. This is especially useful when someone says "is X% MAPE bad?" — MASE answers that question rigorously.

**On interpreting MAPE claims**: When a user asks whether a given MAPE is "bad", the honest answer depends on context. For daily web traffic, 15–25% MAPE is often acceptable. For electricity demand, 2–5% is typical. For highly volatile data (e.g., social media spikes), >30% may be unavoidable. Always compute MASE alongside MAPE so you have a context-free benchmark: if MASE > 1, the model is worse than a naive forecast regardless of what the MAPE number says.

### 4.3 Prediction Intervals

Point forecasts alone are insufficient. Always provide prediction intervals to communicate uncertainty:

```python
# statsmodels ARIMA
forecast = results.get_forecast(steps=12)
ci = forecast.conf_int(alpha=0.05)  # 95% intervals

# Plot with intervals
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(train.index, train, label='Historical')
ax.plot(test.index, forecast.predicted_mean, label='Forecast', color='red')
ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], alpha=0.2, color='red')
ax.legend()
plt.savefig('forecast_with_intervals.png', dpi=150)
```

Prediction intervals widen with the forecast horizon — this is expected and honest. If they don't widen, something is likely wrong with the model.

### 4.4 Residual Analysis

After fitting, always check that residuals behave like white noise:

```python
from statsmodels.stats.diagnostic import acorr_ljungbox

residuals = results.resid

# Ljung-Box test for autocorrelation in residuals
lb_test = acorr_ljungbox(residuals, lags=[10, 20], return_df=True)
print(lb_test)
# p-values > 0.05 = good (no remaining autocorrelation)
```

If residuals show patterns, the model is missing something — revisit the model specification.

## Phase 5: Common Pitfalls and Solutions

### Missing Timestamps
Time series need regular spacing. Fill gaps before modeling:
```python
df = df.asfreq('D')  # enforce daily frequency
# Forward-fill short gaps (1-2 periods)
df['target'] = df['target'].ffill(limit=2)
# Interpolate longer gaps
df['target'] = df['target'].interpolate(method='time')
```

### Multiple Seasonalities
Daily data often has both weekly and yearly patterns. SARIMA handles only one seasonal period. Signs you have multiple seasonalities: residual ACF shows significant spikes at multiple seasonal lags (e.g., lag 7 AND lag 365). Solutions:
- **Prophet** (easiest): handles multiple seasonalities natively — just set `yearly_seasonality=True, weekly_seasonality=True`
- **Fourier terms** added to ARIMA: add `sin(2πkt/P)` and `cos(2πkt/P)` terms as exogenous regressors for each season period P
- **STL decomposition**: remove the dominant seasonality, then model the remainder

When diagnosing a multi-seasonality problem, always compute MASE after improvement to confirm you've beaten the naive baseline — MAPE alone can be misleading since absolute error magnitude varies by scale.

### Forecast Horizon Matters
Short-horizon forecasts (1-3 steps) favor complex models. Long-horizon forecasts favor simpler models (ETS with damped trend, moving averages). Match model complexity to the horizon.

### Non-Constant Variance
If variance grows with level, apply a log transform before modeling and back-transform predictions. Remember to adjust prediction intervals on the original scale.

## Integration with Other DS Skills

This skill fits into the broader DS workflow:
- **ds-eda-process** → Use first for general data understanding, then apply time-specific EDA from this skill
- **ds-supervised-modeling** → For ML-based forecasting approaches with lag features
- **ds-ml-pipeline** → For building reproducible forecast pipelines with proper temporal splitting
- **ds-project-manager** → For scoping forecasting projects and managing stakeholder expectations about uncertainty
- **visualization-expert** → For creating effective forecast visualizations for stakeholders

## Python Ecosystem Quick Reference

| Library | Use For |
|---|---|
| `statsmodels` | ARIMA, SARIMA, ETS, statistical tests, ACF/PACF |
| `pmdarima` | Automated ARIMA order selection (auto_arima) |
| `prophet` | Business forecasting with holidays and changepoints |
| `sktime` | Unified API for multiple forecasting methods |
| `scipy.stats` | Box-Cox transforms, statistical distributions |
| `pandas` | Date handling, resampling, rolling statistics |

## Output Expectations

When completing a time series task, deliver:
1. **Time series plot** with annotated patterns (trend, seasonality, anomalies)
2. **Stationarity assessment** with test results and any transformations applied
3. **ACF/PACF plots** with interpretation
4. **Model comparison table** with MAE, RMSE, MAPE, MASE on the holdout set
5. **Forecast plot** with prediction intervals
6. **Residual diagnostics** confirming model adequacy
7. **Plain-language summary** of what the forecast says and how confident we should be
