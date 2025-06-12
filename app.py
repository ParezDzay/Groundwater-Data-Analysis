# app.py — ANN-only groundwater forecaster (depth view)
# -----------------------------------------------------
# • Loads “GW data (missing filled).csv”
# • User chooses well, lags, test split, hidden layers, horizon
# • Trains an MLPRegressor, shows depth-inverted plots, downloads CSV
# • Fixed: forecast step now drops extra “pred” column → no ValueError

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

st.set_page_config(page_title="ANN Groundwater Prediction", layout="wide")
st.title("🔮 ANN Groundwater Prediction (Depth View)")

DATA_PATH = "GW data (missing filled).csv"

# ───────── helpers ─────────
@st.cache_data(show_spinner=False)
def load_data(path):
    if not Path(path).exists():
        return None
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Year"].astype(str) + "-" +
                                df["Months"].astype(str) + "-01")
    df["month_sin"] = np.sin(2 * np.pi * df["Months"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Months"] / 12)
    return df.sort_values("Date").reset_index(drop=True)

def add_lags(df, well, n):
    out = df.copy()
    for k in range(1, n + 1):
        out[f"{well}_lag{k}"] = out[well].shift(k)
    return out.dropna().reset_index(drop=True)

def fit_ann(df_feat, well, test_size, layers):
    X = df_feat.drop(columns=[well, "Date"])
    y = df_feat[well]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    model = MLPRegressor(hidden_layer_sizes=layers,
                         activation="relu",
                         solver="adam",
                         max_iter=2000,
                         random_state=42,
                         early_stopping=True)
    model.fit(X_tr_s, y_tr)

    df_feat.loc[X_tr.index, "pred"] = model.predict(X_tr_s)
    df_feat.loc[X_te.index, "pred"] = model.predict(X_te_s)

    metrics = {
        "R² train":  round(r2_score(y_tr, df_feat.loc[X_tr.index, "pred"]), 4),
        "RMSE train": round(np.sqrt(mean_squared_error(y_tr, df_feat.loc[X_tr.index, "pred"])), 4),
        "R² test":   round(r2_score(y_te, df_feat.loc[X_te.index, "pred"]), 4),
        "RMSE test":  round(np.sqrt(mean_squared_error(y_te, df_feat.loc[X_te.index, "pred"])), 4)
    }
    return model, scaler, df_feat, metrics

def recursive_forecast(model, scaler, last_row, well, lags, horizon):
    """Iterative forecast—drops columns not seen by scaler (e.g., 'pred')."""
    features = list(scaler.feature_names_in_)  # columns used during training
    rows, r = [], last_row.copy()

    for _ in range(horizon):
        # shift lag columns
        for k in range(lags, 1, -1):
            r[f"{well}_lag{k}"] = r[f"{well}_lag{k-1}"]
        r[f"{well}_lag1"] = r["pred"]

        # advance date + cyclical encodings
        nxt = r["Date"] + pd.DateOffset(months=1)
        r.update({
            "Date": nxt,
            "Months": nxt.month,
            "month_sin": np.sin(2 * np.pi * nxt.month / 12),
            "month_cos": np.cos(2 * np.pi * nxt.month / 12)
        })

        # prepare feature vector exactly in training order
        X_next = r[features].to_frame().T.values
        val = model.predict(X_next)[0]

        r[well] = r["pred"] = val
        rows.append({"Date": nxt, "Depth": val})

    return pd.DataFrame(rows)

# ───────── UI ─────────
df = load_data(DATA_PATH)
if df is None:
    st.error(f"CSV '{DATA_PATH}' not found. Upload it to continue.")
    uploaded = st.file_uploader("Upload groundwater CSV", type="csv")
    if uploaded:
        Path(DATA_PATH).write_bytes(uploaded.getvalue())
        st.experimental_rerun()
    st.stop()

wells = [c for c in df.columns if c.startswith("W")]
well   = st.sidebar.selectbox("Well", wells)
lags   = st.sidebar.slider("Lag steps", 1, 24, 12)
test_f = st.sidebar.slider("Test size", 0.1, 0.5, 0.2, step=0.05)
layers = tuple(int(x.strip()) for x in
               st.sidebar.text_input("Hidden layers (comma-sep)", "64,32").split(",")
               if x.strip())
horiz  = st.sidebar.number_input("Forecast horizon (months)", 1, 60, 12)

feat   = add_lags(df[["Date", "Months", "month_sin", "month_cos", well]], well, lags)
model, scaler, df_pred, metrics = fit_ann(feat, well, test_f, layers)

st.subheader("Model performance")
st.json(metrics)

fig_hist = px.line(df_pred, x="Date", y=[well, "pred"],
                   labels={"value": "Water-table depth (m)",
                           "variable": "Legend"},
                   title=f"{well} — actual vs predicted")
fig_hist.update_yaxes(autorange="reversed")
st.plotly_chart(fig_hist, use_container_width=True)

future = recursive_forecast(model, scaler, df_pred.tail(1).iloc[0],
                            well, lags, horiz)

fig_fore = px.line(
    pd.concat([df_pred[["Date", well]].rename(columns={well: "Depth"}),
               future]),
    x="Date", y="Depth",
    title=f"{well} — {horiz}-month forecast"
)
fig_fore.update_yaxes(autorange="reversed")
st.plotly_chart(fig_fore, use_container_width=True)

st.download_button(
    "Download forecast CSV",
    future.to_csv(index=False).encode(),
    file_name=f"{well}_forecast_{datetime.today().date()}.csv",
    mime="text/csv"
)
