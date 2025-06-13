# fast_forecast_app.py  —  SARIMA, LSTM, CNN-LSTM

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
import os, warnings, tensorflow as tf
warnings.filterwarnings("ignore", category=UserWarning)
tf.get_logger().setLevel("ERROR")          # silence TF startup spam

st.set_page_config(page_title="Groundwater Deep-&-Classic Forecast", layout="wide")
st.title("Groundwater Forecasting — SARIMA & Deep Learning")

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

# ----------------------------------------------------------------- SARIMA ----------
def sarima_forecast(series, horizon, seasonal=True):
    order = (1,1,1)
    s_order = (1,1,1,12) if seasonal else (0,0,0,0)
    train_end = int(len(series)*0.8)
    train, test = series.iloc[:train_end], series.iloc[train_end:]
    mdl = SARIMAX(train, order=order, seasonal_order=s_order,
                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    rmse = round(np.sqrt(mean_squared_error(test, mdl.forecast(len(test)))), 4)
    mdl_full = SARIMAX(series, order=order, seasonal_order=s_order,
                       enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    idx = pd.date_range(series.index[-1] + pd.DateOffset(months=1),
                        periods=horizon, freq="MS")
    fc = pd.Series(mdl_full.forecast(horizon).round(2), index=idx)
    return {"AIC": round(mdl_full.aic,1), "BIC": round(mdl_full.bic,1), "RMSE test": rmse}, fc

# ------------------------------------------------------------- LSTM helpers --------
def build_lstm(input_shape, units=64):
    model = Sequential([
        LSTM(units, activation="tanh", input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def build_cnn_lstm(input_shape, filters=32, kernel=3, units=32):
    model = Sequential([
        Conv1D(filters, kernel_size=kernel, activation="relu", input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(units, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def make_supervised(s: np.ndarray, n_lags: int):
    X, y = [], []
    for i in range(n_lags, len(s)):
        X.append(s[i-n_lags:i])
        y.append(s[i])
    return np.array(X), np.array(y)

def deep_forecast(series, horizon, n_lags=12, epochs=30, batch=16, model_type="lstm"):
    # scale 0-1 for neural nets
    scaler = MinMaxScaler()
    s_scaled = scaler.fit_transform(series.values.reshape(-1,1)).flatten()
    X, y = make_supervised(s_scaled, n_lags)
    split = int(len(X)*0.8)
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]

    # reshape for keras [samples, timesteps, features]
    Xtr = Xtr.reshape((Xtr.shape[0], Xtr.shape[1], 1))
    Xte = Xte.reshape((Xte.shape[0], Xte.shape[1], 1))

    if model_type == "lstm":
        net = build_lstm((n_lags,1))
    else:
        net = build_cnn_lstm((n_lags,1))

    net.fit(Xtr, ytr, validation_data=(Xte, yte),
            epochs=epochs, batch_size=batch,
            verbose=0, callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])

    rmse = round(np.sqrt(mean_squared_error(yte, net.predict(Xte, verbose=0).flatten())), 4)

    # iterative forecast
    history = list(s_scaled[-n_lags:])
    fc_vals = []
    for _ in range(horizon):
        x_input = np.array(history[-n_lags:]).reshape((1,n_lags,1))
        yhat = net.predict(x_input, verbose=0)[0][0]
        fc_vals.append(yhat)
        history.append(yhat)

    fc_vals = scaler.inverse_transform(np.array(fc_vals).reshape(-1,1)).flatten().round(2)
    idx = pd.date_range(series.index[-1] + pd.DateOffset(months=1),
                        periods=horizon, freq="MS")
    return {"RMSE test": rmse, "Lags": n_lags, "Epochs": epochs}, pd.Series(fc_vals, index=idx)

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
    "Select model",
    ["SARIMA / SARIMAX (classic)",
     "LSTM (deep learning)",
     "CNN-LSTM (hybrid deep)"])

series = clean_series(raw, well)
if len(series) < 36:
    st.warning("Need ≥36 points for these models."); st.stop()

# extra sliders for deep nets
if model_choice.startswith("LSTM") or model_choice.startswith("CNN"):
    n_lags = st.sidebar.slider("Number of lags (timesteps)", 6, 24, 12, step=2)
    epochs = st.sidebar.slider("Training epochs", 10, 100, 30, step=10)

with st.spinner("Fitting / forecasting…"):
    if model_choice.startswith("SARIMA"):
        metrics, future = sarima_forecast(series, HORIZON_M)

    elif model_choice.startswith("LSTM"):
        metrics, future = deep_forecast(series, HORIZON_M,
                                        n_lags=n_lags, epochs=epochs,
                                        model_type="lstm")

    else:  # CNN-LSTM
        metrics, future = deep_forecast(series, HORIZON_M,
                                        n_lags=n_lags, epochs=epochs,
                                        model_type="cnn")

# ---------- display ----------------------------------------------------------------
st.subheader("Model & accuracy")
st.table(pd.DataFrame(metrics, index=["Value"]))

st.subheader("5-year monthly forecast")
st.dataframe(future.to_frame("Depth"), use_container_width=True)

# ---------- yearly summary save ----------------------------------------------------
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
