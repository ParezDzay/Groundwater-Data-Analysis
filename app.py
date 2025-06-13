# app.py — Groundwater Forecasting
# Models: SARIMA · Random-Forest (lags) · LSTM · CNN-LSTM (TF optional)
# -------------------------------------------------------------------
# ▸ Upload CSV (Year Months W1…Wn)
# ▸ Choose model and scope:
#       • Single well → detailed metrics + monthly forecast + yearly table
#       • All wells   → one-click yearly table for every well
# ▸ “Save yearly summary” (single-well scope) writes/updates yearly_summaries.csv
#
# runtime.txt    python-3.11.8
# requirements   streamlit==1.45.1
#                pandas==2.3.0    numpy==1.26.4
#                plotly==6.1.2    scikit-learn==1.4.2
#                statsmodels==0.14.4
#                tensorflow-cpu==2.15.0   # optional
# -------------------------------------------------------------------

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

# ───────── optional TensorFlow import ─────────
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten
    from tensorflow.keras.callbacks import EarlyStopping
    tf.get_logger().setLevel("ERROR")
    _TF_AVAILABLE = True
except ModuleNotFoundError:
    _TF_AVAILABLE = False

# ───────── Streamlit page ─────────
st.set_page_config(page_title="Groundwater Forecasts", layout="wide")
st.title("Groundwater Forecasting — Classic, ML & Deep")

DATA_PATH, HORIZON_M = "GW data (missing filled).csv", 60       # 5-year horizon
SUMMARY_CSV = "yearly_summaries.csv"

# ───────── helpers ─────────
@st.cache_data(show_spinner=False)
def load_raw(path: str):
    if not Path(path).exists():
        return None
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Year"].astype(str) + "-" +
                                df["Months"].astype(str).str.zfill(2) + "-01")
    return df.sort_values("Date").reset_index(drop=True)

def clean_series(df: pd.DataFrame, well: str):
    s = df[well].copy()
    q1, q3 = s.quantile([0.25, 0.75]); iqr = q3 - q1
    s = s.where(s.between(q1-3*iqr, q3+3*iqr)).interpolate(limit_direction="both")
    return pd.Series(s.values, index=df["Date"])

# ───────── SARIMA ─────────
def sarima_forecast(series, horizon):
    order, s_order = (1,1,1), (1,1,1,12)
    split = int(len(series)*0.8)
    train, test = series.iloc[:split], series.iloc[split:]
    mdl = SARIMAX(train, order=order, seasonal_order=s_order,
                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    rmse = round(np.sqrt(mean_squared_error(test, mdl.forecast(len(test)))), 4)

    full = SARIMAX(series, order=order, seasonal_order=s_order,
                   enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    idx = pd.date_range(series.index[-1] + pd.DateOffset(months=1),
                        periods=horizon, freq="MS")
    fc = pd.Series(full.forecast(horizon).round(2), index=idx)
    return {"RMSE": rmse, "AIC": round(full.aic,1), "BIC": round(full.bic,1)}, fc

# ───────── Random-Forest (lags) ─────────
def rf_forecast(series, horizon, n_lags=12):
    df_lag = pd.concat({f"lag{k}": series.shift(k) for k in range(1, n_lags+1)}, axis=1).dropna()
    X, y = df_lag.values, series.loc[df_lag.index].values
    split = int(len(X)*0.8)
    rf = RandomForestRegressor(n_estimators=300, random_state=42).fit(X[:split], y[:split])
    rmse = round(np.sqrt(mean_squared_error(y[split:], rf.predict(X[split:]))), 4)

    history = list(series.values[-n_lags:]); fc=[]
    for _ in range(horizon):
        x = np.array(history[-n_lags:][::-1]).reshape(1,-1)
        nxt = rf.predict(x)[0]; fc.append(round(nxt,2)); history.append(nxt)

    idx = pd.date_range(series.index[-1]+pd.DateOffset(months=1), periods=horizon, freq="MS")
    return {"RMSE": rmse, "Lags": n_lags, "Trees": 300}, pd.Series(fc, index=idx)

# ───────── Deep helpers (only if TF) ─────────
if _TF_AVAILABLE:
    def build_lstm(shape, units=64):
        net = Sequential([LSTM(units, activation="tanh", input_shape=shape), Dense(1)])
        net.compile(optimizer="adam", loss="mse"); return net

    def build_cnn_lstm(shape, f=32, k=3, u=32):
        net = Sequential([
            Conv1D(f, k, activation="relu", input_shape=shape),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(u, activation="relu"),
            Dense(1)
        ])
        net.compile(optimizer="adam", loss="mse"); return net

    def to_supervised(a, n):
        X,y = [],[]
        for i in range(n, len(a)):
            X.append(a[i-n:i]); y.append(a[i])
        return np.array(X), np.array(y)

    def deep_forecast(series, horizon, n_lags=12, epochs=30, model="lstm"):
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(series.values.reshape(-1,1)).flatten()
        X,y = to_supervised(scaled, n_lags)
        split = int(len(X)*0.8)
        Xtr,Xte,ytr,yte = X[:split],X[split:],y[:split],y[split:]
        Xtr = Xtr.reshape((Xtr.shape[0], Xtr.shape[1],1))
        Xte = Xte.reshape((Xte.shape[0], Xte.shape[1],1))

        net = build_lstm((n_lags,1)) if model=="lstm" else build_cnn_lstm((n_lags,1))
        net.fit(Xtr, ytr, validation_data=(Xte,yte),
                epochs=epochs, batch_size=16, verbose=0,
                callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])

        rmse = round(np.sqrt(mean_squared_error(
            yte, net.predict(Xte, verbose=0).flatten())),4)

        hist = list(scaled[-n_lags:]); fc=[]
        for _ in range(horizon):
            yhat = net.predict(np.array(hist[-n_lags:]).reshape(1,n_lags,1), verbose=0)[0][0]
            fc.append(yhat); hist.append(yhat)

        fc = scaler.inverse_transform(np.array(fc).reshape(-1,1)).flatten().round(2)
        idx = pd.date_range(series.index[-1]+pd.DateOffset(months=1), periods=horizon, freq="MS")
        return {"RMSE": rmse, "Lags": n_lags, "Epochs": epochs}, pd.Series(fc, index=idx)

# ───────── UI controls ─────────
raw = load_raw(DATA_PATH)
if raw is None:
    st.error("CSV not found. Upload below.")
    if up := st.sidebar.file_uploader("Upload CSV",type="csv"):
        Path(DATA_PATH).write_bytes(up.read()); st.experimental_rerun()
    st.stop()

wells = [c for c in raw.columns if c.startswith("W")]
scope = st.sidebar.radio("Scope", ["Single well","All wells"])

if scope=="Single well":
    well = st.sidebar.selectbox("Well", wells)

models = [
    "SARIMA / SARIMAX",
    "Random-Forest (lags)",
    "LSTM (deep)"   + ("" if _TF_AVAILABLE else "  ❌ TF not installed"),
    "CNN-LSTM"      + ("" if _TF_AVAILABLE else "  ❌ TF not installed"),
]
model_choice = st.sidebar.radio("Model", models)

if model_choice.startswith("Random"):
    n_lags = st.sidebar.slider("RF: lags",6,24,12,2)
elif model_choice.startswith(("LSTM","CNN")):
    if not _TF_AVAILABLE:
        st.error("TensorFlow missing → deep models disabled."); st.stop()
    n_lags = st.sidebar.slider("DL: lags",6,24,12,2)
    epochs = st.sidebar.slider("Epochs",10,100,30,10)

# ───────── forecast wrapper ─────────
def run_model(well_id):
    s = clean_series(raw, well_id)
    if model_choice.startswith("SARIMA"):
        return sarima_forecast(s, HORIZON_M)
    if model_choice.startswith("Random"):
        return rf_forecast(s, HORIZON_M, n_lags)
    if model_choice.startswith("LSTM"):
        return deep_forecast(s, HORIZON_M, n_lags, epochs, "lstm")
    return deep_forecast(s, HORIZON_M, n_lags, epochs, "cnn")

# ───────── run forecasts ─────────
with st.spinner("Training / forecasting…"):
    targets = wells if scope=="All wells" else [well]
    summary_rows=[]; future_one=None; metrics_one=None

    for w in targets:
        metrics, future = run_model(w)
        yearly = future.resample("A").mean()

        row = {"Well": w}
        for yr in range(2025,2030):
            sel = yearly[yearly.index.year==yr]
            row[str(yr)] = round(sel.iloc[0],2) if not sel.empty else np.nan
        row.update(metrics)
        summary_rows.append(row)

        if scope=="Single well":
            future_one, metrics_one = future, metrics

summary_df = pd.DataFrame(summary_rows)

# ───────── display ─────────
st.subheader("Yearly forecast (average depth)")
st.dataframe(summary_df, use_container_width=True)

if scope=="Single well":
    st.subheader("Model metrics")
    st.table(pd.DataFrame(metrics_one, index=["Value"]))
    st.subheader("5-year monthly forecast")
    st.dataframe(future_one.to_frame("Depth"), use_container_width=True)

# ───────── save single-well summary ─────────
if scope=="Single well":
    if st.button("💾 Save yearly summary"):
        if os.path.exists(SUMMARY_CSV):
            pd.concat([pd.read_csv(SUMMARY_CSV), summary_df],
                      ignore_index=True).to_csv(SUMMARY_CSV,index=False)
        else:
            summary_df.to_csv(SUMMARY_CSV,index=False)

        st.session_state.setdefault("summary_rows",[]).append(summary_df)
        st.success(f"Saved to '{SUMMARY_CSV}'")

# ───────── downloads ─────────
n_rows = len(st.session_state.get("summary_rows",[]))
st.sidebar.markdown(f"**Saved summaries in session:** {n_rows}")
if n_rows:
    combo = pd.concat(st.session_state["summary_rows"]).reset_index(drop=True)
    st.sidebar.download_button("⬇ Download session CSV",
                               combo.to_csv(index=False).encode(),
                               f"well_summaries_{datetime.today().date()}.csv",
                               "text/csv")
if os.path.exists(SUMMARY_CSV):
    with open(SUMMARY_CSV,"rb") as f:
        st.sidebar.download_button("⬇ Download saved CSV",
                                   f.read(), SUMMARY_CSV,"text/csv")
