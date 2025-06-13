# app.py — Groundwater Forecasting
# Models: SARIMA · Random-Forest (lags) · LSTM · CNN-LSTM   (TF optional)
# ----------------------------------------------------------------------
# • Upload groundwater CSV (Year, Months, W1…Wn)
# • Choose model + “Scope”:
#       – Single well   → detailed metrics + monthly forecast + yearly table
#       – All wells     → one-click yearly-average table for every well
# • “Save yearly summary” works in single-well scope
#
# runtime.txt   : python-3.11.8
# requirements  : streamlit==1.45.1
#                 pandas==2.3.0
#                 numpy==1.26.4
#                 plotly==6.1.2
#                 scikit-learn==1.4.2
#                 statsmodels==0.14.4
#                 tensorflow-cpu==2.15.0   # optional – deep models disabled if absent
# ----------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os, warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ───────── Optional TensorFlow import ─────────
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten
    from tensorflow.keras.callbacks import EarlyStopping
    tf.get_logger().setLevel("ERROR")
    _TF_AVAILABLE = True
except ModuleNotFoundError:
    _TF_AVAILABLE = False

# ───────── Page config ─────────
st.set_page_config(page_title="Groundwater Forecasts", layout="wide")
st.title("Groundwater Forecasting — Classic, ML & Deep Learning")

DATA_PATH   = "GW data (missing filled).csv"
HORIZON_M   = 60             # 5-year horizon (months)
SUMMARY_CSV = "yearly_summaries.csv"

# ───────── Helper functions ─────────
@st.cache_data(show_spinner=False)
def load_raw(path: str):
    if not Path(path).exists():
        return None
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Months"].astype(str).str.zfill(2) + "-01"
    )
    return df.sort_values("Date").reset_index(drop=True)

def clean_series(df: pd.DataFrame, well: str) -> pd.Series:
    s = df[well].copy()
    q1, q3 = s.quantile([0.25, 0.75]); iqr = q3 - q1
    s = s.where(s.between(q1 - 3*iqr, q3 + 3*iqr)).interpolate(limit_direction="both")
    return pd.Series(s.values, index=df["Date"])

# ───────── SARIMA ─────────
def sarima_forecast(series, horizon):
    order, s_order = (1,1,1), (1,1,1,12)
    split = int(len(series)*0.8)
    train, test = series.iloc[:split], series.iloc[split:]

    mdl  = SARIMAX(train, order=order, seasonal_order=s_order,
                   enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    rmse = round(np.sqrt(mean_squared_error(test, mdl.forecast(len(test)))), 4)

    full = SARIMAX(series, order=order, seasonal_order=s_order,
                   enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    idx  = pd.date_range(series.index[-1]+pd.DateOffset(months=1), periods=horizon, freq="MS")
    fc   = pd.Series(full.forecast(horizon).round(2), index=idx)
    return {"RMSE test": rmse, "AIC": round(full.aic,1), "BIC": round(full.bic,1)}, fc

# ───────── Random-Forest (lag features) ─────────
def rf_forecast(series, horizon, n_lags=12):
    df_lag = pd.concat({f"lag{k}": series.shift(k) for k in range(1, n_lags+1)}, axis=1).dropna()
    X, y = df_lag.values, series.loc[df_lag.index].values
    split = int(len(X)*0.8)
    rf = RandomForestRegressor(n_estimators=300, random_state=42).fit(X[:split], y[:split])
    rmse = round(np.sqrt(mean_squared_error(y[split:], rf.predict(X[split:]))), 4)

    history = list(series.values[-n_lags:])
    fc_vals = []
    for _ in range(horizon):
        x = np.array(history[-n_lags:][::-1]).reshape(1, -1)
        y_hat = rf.predict(x)[0]
        fc_vals.append(round(y_hat,2)); history.append(y_hat)

    idx = pd.date_range(series.index[-1]+pd.DateOffset(months=1), periods=horizon, freq="MS")
    return {"RMSE test": rmse, "Lags": n_lags, "Trees": 300}, pd.Series(fc_vals, index=idx)

# ───────── Deep-learning helpers (only if TF) ─────────
if _TF_AVAILABLE:
    def build_lstm(input_shape, units=64):
        net = Sequential([LSTM(units, activation="tanh", input_shape=input_shape),
                          Dense(1)])
        net.compile(optimizer="adam", loss="mse")
        return net

    def build_cnn_lstm(input_shape, filters=32, kernel=3, units=32):
        net = Sequential([
            Conv1D(filters, kernel_size=kernel, activation="relu", input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(units, activation="relu"),
            Dense(1)
        ])
        net.compile(optimizer="adam", loss="mse")
        return net

    def to_supervised(arr, n_lags):
        X, y = [], []
        for i in range(n_lags, len(arr)):
            X.append(arr[i-n_lags:i]); y.append(arr[i])
        return np.array(X), np.array(y)

    def deep_forecast(series, horizon, n_lags=12, epochs=30, model_type="lstm"):
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(series.values.reshape(-1,1)).flatten()
        X, y   = to_supervised(scaled, n_lags)
        split  = int(len(X)*0.8)
        Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]
        Xtr = Xtr.reshape((Xtr.shape[0], Xtr.shape[1], 1))
        Xte = Xte.reshape((Xte.shape[0], Xte.shape[1], 1))

        net = build_lstm((n_lags,1)) if model_type=="lstm" else build_cnn_lstm((n_lags,1))
        net.fit(Xtr, ytr, validation_data=(Xte,yte),
                epochs=epochs, batch_size=16, verbose=0,
                callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])

        rmse = round(np.sqrt(mean_squared_error(
            yte, net.predict(Xte, verbose=0).flatten())), 4)

        history = list(scaled[-n_lags:]); fc=[]
        for _ in range(horizon):
            yhat = net.predict(np.array(history[-n_lags:]).reshape(1,n_lags,1), verbose=0)[0][0]
            fc.append(yhat); history.append(yhat)

        fc_vals = scaler.inverse_transform(np.array(fc).reshape(-1,1)).flatten().round(2)
        idx = pd.date_range(series.index[-1]+pd.DateOffset(months=1), periods=horizon, freq="MS")
        return {"RMSE test": rmse, "Lags": n_lags, "Epochs": epochs}, pd.Series(fc_vals, index=idx)

# ──────────── UI setup ────────────
raw = load_raw(DATA_PATH)
if raw is None:
    st.error("CSV not found. Upload below.")
    if up := st.sidebar.file_uploader("Upload CSV", type="csv"):
        Path(DATA_PATH).write_bytes(up.read()); st.experimental_rerun()
    st.stop()

wells = [c for c in raw.columns if c.startswith("W")]

run_scope = st.sidebar.radio("Scope", ["Single well", "All wells"])

if run_scope == "Single well":
    well = st.sidebar.selectbox("Well", wells)

model_labels = [
    "SARIMA / SARIMAX (classic)",
    "Random-Forest (lags)",
    "LSTM (deep learning)"   + ("" if _TF_AVAILABLE else "  —  ❌ TF not installed"),
    "CNN-LSTM (hybrid deep)" + ("" if _TF_AVAILABLE else "  —  ❌ TF not installed"),
]
model_choice = st.sidebar.radio("Model", model_labels)

# extra sliders
if model_choice.startswith("Random"):
    n_lags = st.sidebar.slider("RF: number of lags", 6, 24, 12, step=2)
elif model_choice.startswith(("LSTM", "CNN")):
    if not _TF_AVAILABLE:
        st.error("TensorFlow isn’t installed; deep models unavailable.")
        st.stop()
    n_lags = st.sidebar.slider("DL: number of lags", 6, 24, 12, step=2)
    epochs = st.sidebar.slider("Epochs", 10, 100, 30, step=10)

# ──────────── Forecast helper ────────────
def run_forecast(well_id: str):
    s = clean_series(raw, well_id)
    if model_choice.startswith("SARIMA"):
        return sarima_forecast(s, HORIZON_M)
    if model_choice.startswith("Random"):
        return rf_forecast(s, HORIZON_M, n_lags)
    if model_choice.startswith("LSTM"):
        return deep_forecast(s, HORIZON_M, n_lags, epochs, "lstm")
    return deep_forecast(s, HORIZON_M, n_lags, epochs, "cnn")   # CNN-LSTM

# ──────────── Run forecasts ────────────
with st.spinner("Training / forecasting…"):
    targets = wells if run_scope == "All wells" else [well]
    summary_rows, detail_future, detail_metrics = [], None, None

    for w in targets:
        metrics, future = run_forecast(w)
        yearly = future.resample("A").mean()
        row = {
            "Well ID": w,
            "2025": yearly.get("2025"),
            "2026": yearly.get("2026"),
            "2027": yearly.get("2027"),
            "2028": yearly.get("2028"),
            "2029": yearly.get("2029"),
        }
        row.update(metrics); summary_rows.append(row)

        if run_scope == "Single well":   # keep for detailed table
            detail_future, detail_metrics = future, metrics

summary_df = pd.DataFrame(summary_rows)

# ──────────── Display ────────────
st.subheader("Yearly forecast summary (average depth)")
st.dataframe(summary_df, use_container_width=True)

if run_scope == "Single well":
    st.subheader("Model metrics (selected well)")
    st.table(pd.DataFrame(detail_metrics, index=["Value"]))
    st.subheader("5-year monthly forecast")
    st.dataframe(detail_future.to_frame("Depth"), use_container_width=True)

# ──────────── Save yearly summary (only single well) ────────────
if run_scope == "Single well":
    if st.button("💾 Save yearly summary"):
        df_row = summary_df  # single row
        if os.path.exists(SUMMARY_CSV):
            pd.concat([pd.read_csv(SUMMARY_CSV), df_row],
                      ignore_index=True).to_csv(SUMMARY_CSV, index=False)
        else:
            df_row.to_csv(SUMMARY_CSV, index=False)

        st.session_state.setdefault("summary_rows", []).append(df_row)
        st.success(f"Saved to '{SUMMARY_CSV}' – total rows: {len(st.session_state['summary_rows'])}")

# ──────────── Download area ────────────
n_rows = len(st.session_state.get("summary_rows", []))
st.sidebar.markdown(f"**Saved summaries in session:** {n_rows}")
if n_rows:
    combo = pd.concat(st.session_state["summary_rows"]).reset_index(drop=True)
    st.sidebar.download_button("⬇️ Download CSV from session",
                               combo.to_csv(index=False).encode(),
                               f"well_summaries_{datetime.today().date()}.csv",
                               "text/csv")
if os.path.exists(SUMMARY_CSV):
    with open(SUMMARY_CSV, "rb") as f:
        st.sidebar.download_button("⬇️ Download saved CSV",
                                   f.read(), SUMMARY_CSV, "text/csv")
