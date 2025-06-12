# app.py — ANN groundwater forecaster with a clarified combined plot
# ------------------------------------------------------------------
# • Reads “GW data (missing filled).csv”
# • Lets you pick well, lags, test-split, hidden layers, forecast horizon
# • Trains an MLPRegressor
# • Shows one depth-inverted chart containing:
#     Actual  (blue)
#     In-sample predictions (orange)
#     Forecast (green dashed)
#   plus a dotted vertical line marking the data/forecast boundary

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics         import r2_score, mean_squared_error

st.set_page_config(page_title="ANN Groundwater Prediction", layout="wide")
st.title("🔮 ANN Groundwater Prediction (Depth View)")

DATA_PATH = "GW data (missing filled).csv"

# ───────────────── helper functions ─────────────────
@st.cache_data(show_spinner=False)
def load_data(path):
    if not Path(path).exists(): return None
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

def train_ann(df_feat, well, test_size, layers):
    X = df_feat.drop(columns=[well,"Date"]); y = df_feat[well]
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=test_size, shuffle=False)

    sc = StandardScaler()
    Xtrain_s, Xtest_s = sc.fit_transform(Xtrain), sc.transform(Xtest)

    mdl = MLPRegressor(hidden_layer_sizes=layers, max_iter=2000,
                       activation="relu", solver="adam",
                       random_state=42, early_stopping=True)
    mdl.fit(Xtrain_s, ytrain)

    df_feat.loc[Xtrain.index,"pred"] = mdl.predict(Xtrain_s)
    df_feat.loc[Xtest.index ,"pred"] = mdl.predict(Xtest_s)

    met = {
        "R² train" : round(r2_score(ytrain, df_feat.loc[Xtrain.index,"pred"]),4),
        "RMSE train": round(np.sqrt(mean_squared_error(ytrain, df_feat.loc[Xtrain.index,"pred"])),4),
        "R² test"  : round(r2_score(ytest , df_feat.loc[Xtest.index ,"pred"])),4,
        "RMSE test": round(np.sqrt(mean_squared_error(ytest , df_feat.loc[Xtest.index ,"pred"])),4)
    }
    return mdl, sc, df_feat, met

def forecast(mdl, sc, last_row, well, lags, horizon):
    feats = list(sc.feature_names_in_)
    rows, r = [], last_row.copy()
    for _ in range(horizon):
        for k in range(lags,1,-1):
            r[f"{well}_lag{k}"] = r[f"{well}_lag{k-1}"]
        r[f"{well}_lag1"] = r["pred"]
        nxt = r["Date"] + pd.DateOffset(months=1)
        r.update({"Date":nxt,
                  "Months":nxt.month,
                  "month_sin":np.sin(2*np.pi*nxt.month/12),
                  "month_cos":np.cos(2*np.pi*nxt.month/12)})
        val = mdl.predict(sc.transform(r[feats].to_frame().T))[0]
        r[well] = r["pred"] = val
        rows.append({"Date":nxt,"Depth":val})
    return pd.DataFrame(rows)

# ───────────────── UI ─────────────────
df = load_data(DATA_PATH)
if df is None:
    st.error(f"CSV '{DATA_PATH}' not found. Upload to continue.")
    up = st.file_uploader("Upload groundwater CSV", type="csv")
    if up: Path(DATA_PATH).write_bytes(up.getvalue()); st.experimental_rerun()
    st.stop()

wells = [c for c in df.columns if c.startswith("W")]
well = st.sidebar.selectbox("Well", wells)
lags = st.sidebar.slider("Lag steps", 1, 24, 12)
test_frac = st.sidebar.slider("Test size", 0.1, 0.5, 0.2, step=0.05)
layers = tuple(int(x) for x in st.sidebar.text_input("Hidden layers", "64,32").split(",") if x.strip())
horizon = st.sidebar.number_input("Forecast horizon (months)", 1, 60, 12)

feat = add_lags(df[["Date","Months","month_sin","month_cos",well]], well, lags)
mdl, sc, df_pred, metrics = train_ann(feat, well, test_frac, layers)

st.subheader("Model performance")
st.json(metrics)

future = forecast(mdl, sc, df_pred.tail(1).iloc[0], well, lags, horizon)

# Merge for unified plot
df_actual = df_pred[["Date",well]].rename(columns={well:"Depth"}).assign(Type="Actual")
df_pred_hist = df_pred[["Date","pred"]].rename(columns={"pred":"Depth"}).assign(Type="Predicted")
df_fore = future.assign(Type="Forecast")
plot_df = pd.concat([df_actual, df_pred_hist, df_fore])

fig = px.line(plot_df, x="Date", y="Depth", color="Type",
              labels={"Depth":"Water-table depth (m)"},
              title=f"{well} — fit & {horizon}-month forecast")
fig.update_yaxes(autorange="reversed")
fig.update_traces(mode="lines")  # hide markers
# make forecast dashed
for t in fig.data:
    if t.name == "Forecast":
        t.update(line=dict(dash="dash"))
# dotted vertical line at split
boundary = df_pred["Date"].max()
fig.add_vline(x=boundary, line_dash="dot", line_width=1, line_color="gray")

st.plotly_chart(fig, use_container_width=True)

st.download_button("Download forecast CSV",
                   df_fore.to_csv(index=False).encode(),
                   file_name=f"{well}_forecast_{datetime.today().date()}.csv",
                   mime="text/csv")
