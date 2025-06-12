# app.py — Groundwater forecasting (ANN + ARIMA)
# ----------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from datetime import datetime

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics         import r2_score, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="Groundwater Forecasting (ANN & ARIMA)", layout="wide")
st.title("Groundwater Forecasting – ANN vs ARIMA (Depth View)")

DATA_PATH     = "GW data (missing filled).csv"
FORE_HORIZON  = 60   # 5 years

# ───────── helpers ─────────
@st.cache_data(show_spinner=False)
def load_data(path):
    if not Path(path).exists():
        return None
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Year"].astype(str) + "-" +
                                df["Months"].astype(str) + "-01")
    df["month_sin"] = np.sin(2*np.pi*df["Months"]/12)
    df["month_cos"] = np.cos(2*np.pi*df["Months"]/12)
    return df.sort_values("Date").reset_index(drop=True)

def add_lags(df, well, n):
    out = df.copy()
    for k in range(1, n+1):
        out[f"{well}_lag{k}"] = out[well].shift(k)
    return out.dropna().reset_index(drop=True)

# ─── ANN training & forecast ───
def train_ann(df_feat, well, hidden_layers, lag_steps):
    X = df_feat.drop(columns=[well,"Date"]);  y = df_feat[well]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    Xtr_s, Xte_s = scaler.fit_transform(Xtr), scaler.transform(Xte)

    mdl = MLPRegressor(hidden_layer_sizes=hidden_layers,
                       activation="relu", solver="adam",
                       max_iter=2000, random_state=42,
                       early_stopping=True).fit(Xtr_s, ytr)

    df_feat.loc[Xtr.index, "pred"] = mdl.predict(Xtr_s)
    df_feat.loc[Xte.index, "pred"] = mdl.predict(Xte_s)

    metrics = {
        "R² train":  round(r2_score(ytr, df_feat.loc[Xtr.index, "pred"]), 4),
        "RMSE train": round(np.sqrt(mean_squared_error(ytr, df_feat.loc[Xtr.index, "pred"])), 4),
        "R² test":   round(r2_score(yte, df_feat.loc[Xte.index, "pred"]), 4),
        "RMSE test": round(np.sqrt(mean_squared_error(yte, df_feat.loc[Xte.index, "pred"])), 4)
    }

    # recursive forecast
    feats = scaler.feature_names_in_
    r = df_feat.tail(1).iloc[0].copy()
    rows = []
    for _ in range(FORE_HORIZON):
        for k in range(lag_steps, 1, -1):
            r[f"{well}_lag{k}"] = r[f"{well}_lag{k-1}"]
        r[f"{well}_lag1"] = r["pred"]

        nxt = r["Date"] + pd.DateOffset(months=1)
        r.update({
            "Date": nxt,
            "Months": nxt.month,
            "month_sin": np.sin(2*np.pi*nxt.month/12),
            "month_cos": np.cos(2*np.pi*nxt.month/12)
        })

        val = mdl.predict(scaler.transform(r[feats].to_frame().T))[0]
        r[well] = r["pred"] = val
        rows.append({"Date": nxt, "Depth": val})

    return metrics, df_feat, pd.DataFrame(rows)

# ─── ARIMA training & forecast ───
def train_arima(series, seasonal):
    order = (1,1,1)
    sorder = (1,1,1,12) if seasonal else (0,0,0,0)
    model = ARIMA(series, order=order, seasonal_order=sorder).fit()
    pred  = model.get_forecast(FORE_HORIZON)

    future = pd.DataFrame({
        "Date": pd.date_range(series.index[-1] + pd.DateOffset(months=1),
                              periods=FORE_HORIZON, freq="MS"),
        "Depth": pred.predicted_mean.values,
        "lower": pred.conf_int(alpha=0.05)["lower Depth"].values,
        "upper": pred.conf_int(alpha=0.05)["upper Depth"].values
    })
    metrics = {"AIC": round(model.aic, 1), "BIC": round(model.bic, 1)}
    return metrics, pred, future

# ───────── UI ─────────
df = load_data(DATA_PATH)
if df is None:
    st.error(f"CSV '{DATA_PATH}' not found. Upload to continue.")
    if up := st.file_uploader("Upload CSV", type="csv"):
        Path(DATA_PATH).write_bytes(up.getvalue())
        st.experimental_rerun()
    st.stop()

wells = [c for c in df.columns if c.startswith("W")]
well  = st.sidebar.selectbox("Well", wells)
model_type = st.sidebar.radio("Choose model", ["🔮 ANN", "📈 ARIMA"])

if model_type == "🔮 ANN":
    lags = st.sidebar.slider("Lag steps", 1, 24, 12)
    layers = tuple(int(x) for x in st.sidebar.text_input("Hidden layers", "64,32").split(",") if x.strip())

    feat = add_lags(df[["Date","Months","month_sin","month_cos",well]], well, lags)
    metrics, hist_df, future = train_ann(feat, well, layers, lags)

    st.subheader("ANN metrics"); st.json(metrics)

    df_actual = hist_df[["Date", well]].rename(columns={well:"Depth"}).assign(Type="Actual")
    df_fit    = hist_df[["Date","pred"]].rename(columns={"pred":"Depth"}).assign(Type="Predicted")
    df_fore   = future.assign(Type="Forecast")
    plot_df   = pd.concat([df_actual, df_fit, df_fore])

    fig = px.line(plot_df, x="Date", y="Depth", color="Type",
                  labels={"Depth":"Water-table depth (m)"},
                  title=f"{well} — ANN fit & 5-year forecast")
    fig.update_yaxes(autorange="reversed")
    for tr in fig.data:
        if tr.name == "Forecast":
            tr.update(line=dict(dash="dash"))
    fig.add_vline(x=hist_df["Date"].max(), line_dash="dot", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)

else:  # ARIMA
    seasonal = st.sidebar.checkbox("Include 12-month seasonality", True)
    series = pd.Series(df[well].values, index=df["Date"])
    metrics, pred_obj, future = train_arima(series, seasonal)

    st.subheader("ARIMA metrics"); st.json(metrics)

    df_fit    = pd.DataFrame({"Date": series.index, "Depth": pred_obj.fittedvalues, "Type": "Predicted"})
    df_actual = pd.DataFrame({"Date": series.index, "Depth": series.values,        "Type": "Actual"})
    df_fore   = future.assign(Type="Forecast")
    plot_df   = pd.concat([df_actual, df_fit, df_fore])

    fig = px.line(plot_df, x="Date", y="Depth", color="Type",
                  labels={"Depth":"Water-table depth (m)"},
                  title=f"{well} — ARIMA fit & 5-year forecast")
    fig.update_yaxes(autorange="reversed")
    for tr in fig.data:
        if tr.name == "Forecast":
            tr.update(line=dict(dash="dash"))
    fig.add_vline(x=series.index[-1], line_dash="dot", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)

# ---- table & download ----
st.subheader("🗒️ 5-Year Forecast Table")
st.dataframe(future[["Date","Depth"]].style.format({"Depth":"{:.2f}"}), use_container_width=True)

st.download_button("Download forecast CSV",
                   future[["Date","Depth"]].to_csv(index=False).encode(),
                   file_name=f"{well}_{model_type.strip()}_forecast_{datetime.today().date()}.csv",
                   mime="text/csv")
