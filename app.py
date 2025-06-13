# fast_forecast_app.py  —  Seasonal-Naïve, Holt-Winters, VAR, Random-Forest

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import VAR
from sklearn.ensemble import RandomForestRegressor
import os, warnings
warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(page_title="Groundwater Fast Forecast", layout="wide")
st.title("Groundwater Forecasting — Fast Table View")

DATA_PATH, HORIZON_M = "GW data (missing filled).csv", 60
SUMMARY_CSV = "yearly_summaries.csv"

# ---------- helpers ----------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_raw(path: str):
    if not Path(path).exists():
        return None
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Year"].astype(str) + "-" +
                                df["Months"].astype(str).str.zfill(2) + "-01")
    return df.sort_values("Date").reset_index(drop=True)

def clean_series(df, well):
    s = df[well].copy()
    q1, q3 = s.quantile([0.25, 0.75]); iqr = q3 - q1
    s = s.where(s.between(q1 - 3*iqr, q3 + 3*iqr)).interpolate(limit_direction="both")
    return pd.Series(s.values, index=df["Date"])

def clean_multivariate(df, wells):
    """Apply the same cleaning to every well and return a Date-indexed dataframe."""
    out = {}
    for w in wells:
        out[w] = clean_series(df, w)
    return pd.DataFrame(out).dropna()

# ---------- forecasting back-ends ---------------------------------------------------
def seasonal_naive_forecast(series, horizon):
    y_pred = series.shift(12).dropna()
    rmse   = round(np.sqrt(mean_squared_error(series.loc[y_pred.index], y_pred)), 4)
    idx    = pd.date_range(series.index[-1] + pd.DateOffset(months=1), periods=horizon, freq="MS")
    vals   = np.tile(series.iloc[-12:], int(np.ceil(horizon/12)))[:horizon]
    return {"RMSE test": rmse}, pd.Series(vals.round(2), index=idx)

def holt_winters_forecast(series, horizon):
    mdl   = ExponentialSmoothing(series, trend="add", seasonal="add",
                                 seasonal_periods=12,
                                 initialization_method="estimated").fit()
    rmse  = round(np.sqrt(mean_squared_error(series, mdl.fittedvalues)), 4)
    idx   = pd.date_range(series.index[-1] + pd.DateOffset(months=1), periods=horizon, freq="MS")
    fc    = pd.Series(mdl.forecast(horizon).round(2), index=idx)
    return {"AIC": round(mdl.aic,1), "RMSE train": rmse}, fc

def var_forecast(df_wells, target_well, horizon, lag_order=12):
    """Fit VAR on all wells, forecast `horizon` months, return target well only."""
    train = df_wells.iloc[:-horizon] if len(df_wells) > horizon+lag_order else df_wells
    model = VAR(train)
    res   = model.fit(lag_order, ic=None, trend="c")
    # simple test RMSE on last horizon points (if we withheld them)
    if len(df_wells) > horizon+lag_order:
        test_pred = res.forecast(train.values[-lag_order:], horizon)
        rmse = round(np.sqrt(mean_squared_error(
            df_wells[target_well].iloc[-horizon:], test_pred[:, df_wells.columns.get_loc(target_well)])), 4)
    else:
        rmse = np.nan
    future_vals = res.forecast(df_wells.values[-lag_order:], horizon)[:, df_wells.columns.get_loc(target_well)]
    idx = pd.date_range(df_wells.index[-1] + pd.DateOffset(months=1), periods=horizon, freq="MS")
    return {"Lag order": lag_order, "RMSE test": rmse}, pd.Series(future_vals.round(2), index=idx)

def rf_forecast(series, horizon, n_lags=12, n_estimators=300, random_state=42):
    """Lag-feature Random‐Forest, iterative multi-step forecast."""
    # build lag matrix
    df_lag = pd.concat({f"lag{k}": series.shift(k) for k in range(1, n_lags+1)}, axis=1).dropna()
    X, y   = df_lag.values, series.loc[df_lag.index].values
    split  = int(len(X)*0.8)
    rf     = RandomForestRegressor(n_estimators=n_estimators,
                                   random_state=random_state).fit(X[:split], y[:split])
    rmse   = round(np.sqrt(mean_squared_error(y[split:], rf.predict(X[split:]))), 4)

    last_vals = list(series.iloc[-n_lags:])
    fc_vals   = []
    for _ in range(horizon):
        x_pred   = np.array(last_vals[-n_lags:][::-1])    # lag1 = latest
        next_val = rf.predict(x_pred.reshape(1, -1))[0]
        fc_vals.append(round(next_val,2))
        last_vals.append(next_val)

    idx = pd.date_range(series.index[-1] + pd.DateOffset(months=1), periods=horizon, freq="MS")
    return {"Lags": n_lags, "RMSE test": rmse}, pd.Series(fc_vals, index=idx)

# ---------- UI ---------------------------------------------------------------------
raw = load_raw(DATA_PATH)
if raw is None:
    st.error("CSV not found. Upload a groundwater file.")
    if up := st.sidebar.file_uploader("Upload CSV", type="csv"):
        Path(DATA_PATH).write_bytes(up.read()); st.experimental_rerun()
    st.stop()

wells = [c for c in raw.columns if c.startswith("W")]
well  = st.sidebar.selectbox("Well", wells)
model_choice = st.sidebar.radio(
    "Model",
    ["Seasonal Naïve (super-fast)",
     "Holt-Winters (fast)",
     "Vector Auto-Regression (VAR)",
     "Random-Forest (lags)"]
)

series = clean_series(raw, well)
if len(series) < 24:
    st.warning("Need ≥24 points for forecasting."); st.stop()

# extra setting for RF
if model_choice.startswith("Random"):
    n_lags = st.sidebar.slider("RF: number of lags", 6, 24, 12, step=2)

with st.spinner("Calculating forecast…"):
    if model_choice.startswith("Seasonal"):
        metrics, future = seasonal_naive_forecast(series, HORIZON_M)

    elif model_choice.startswith("Holt"):
        metrics, future = holt_winters_forecast(series, HORIZON_M)

    elif model_choice.startswith("Vector"):
        df_all = clean_multivariate(raw, wells).loc[series.index]  # align dates
        metrics, future = var_forecast(df_all, well, HORIZON_M)

    else:  # Random-Forest
        metrics, future = rf_forecast(series, HORIZON_M, n_lags=n_lags)

# ---------- display ----------------------------------------------------------------
st.subheader("Model & accuracy")
st.table(pd.DataFrame(metrics, index=["Value"]))

st.subheader("5-year monthly forecast")
st.dataframe(future.to_frame("Depth"), use_container_width=True)

# ---------- save yearly summary ----------------------------------------------------
if st.button("💾 Save yearly summary"):
    row = {"Well": well}
    yearly = future.resample("A").mean()
    for yr in range(2025, 2030):
        val = yearly.get(str(yr))
        row[str(yr)] = val.values[0] if val is not None else np.nan
    row.update(metrics)

    new_df = pd.DataFrame([row])
    if os.path.exists(SUMMARY_CSV):
        pd.concat([pd.read_csv(SUMMARY_CSV), new_df], ignore_index=True).to_csv(SUMMARY_CSV, index=False)
    else:
        new_df.to_csv(SUMMARY_CSV, index=False)

    st.session_state.setdefault("summary_rows", []).append(new_df)
    st.success(f"Saved to '{SUMMARY_CSV}' — total rows: {len(st.session_state['summary_rows'])}")

# ---------- download area ----------------------------------------------------------
n_rows = len(st.session_state.get("summary_rows", []))
st.sidebar.markdown(f"**Saved summaries in session:** {n_rows}")
if n_rows:
    combined = pd.concat(st.session_state["summary_rows"]).reset_index(drop=True)
    st.sidebar.download_button("⬇️ Download CSV from session",
                               combined.to_csv(index=False).encode(),
                               f"well_summaries_{datetime.today().date()}.csv",
                               "text/csv")
if os.path.exists(SUMMARY_CSV):
    with open(SUMMARY_CSV, "rb") as f:
        st.sidebar.download_button("⬇️ Download saved CSV file",
                                   f.read(), SUMMARY_CSV, "text/csv")


