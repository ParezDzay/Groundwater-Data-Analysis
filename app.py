# ann_prediction.py
# ────────────────────────────────────────────────────────────────
# Streamlit module for ANN-based groundwater-level forecasting
# --------------------------------------------------------------
# • Loads “GW data (missing filled).csv”   (20 wells, 2004-2024)
# • Lets the user pick: well, # lags, test-size, hidden-layer
# • Builds an MLPRegressor with standard-scaled inputs
# • Shows train/test metrics & interactive plots
# • Generates multi-step forecasts and offers CSV download
#
# You can drop this file into the same GitHub repo as app.py and
# either `streamlit run ann_prediction.py` directly **or** import
# its `groundwater_ann_page()` function inside the Prediction tab
# of your main app.  No other tweaks required.
# ————————————————————————————————————————————————————————

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px
import io
from datetime import datetime

# ─────────────── CONFIG ───────────────
DATA_PATH = r"C:\Parez\GW data (missing filled).csv"
st.set_page_config(page_title="ANN Groundwater Forecast", layout="wide")

# 📥 ─────────── DATA LOAD & PREP ───────────
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure correct dtypes
    df["Year"] = df["Year"].astype(int)
    df["Months"] = df["Months"].astype(int)
    # Parse date if not already
    if "Date" not in df.columns:
        df["Date"] = pd.to_datetime(df["Year"].astype(str) + "-" + df["Months"].astype(str) + "-01")
    else:
        df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df

def add_cyclical_month(df: pd.DataFrame) -> pd.DataFrame:
    # encode seasonality (helps neural net)
    df = df.copy()
    df["month_sin"] = np.sin(2 * np.pi * df["Months"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Months"] / 12)
    return df

def create_lag_features(df: pd.DataFrame, well: str, n_lags: int) -> pd.DataFrame:
    df = df.copy()
    for lag in range(1, n_lags + 1):
        df[f"{well}_lag{lag}"] = df[well].shift(lag)
    return df.dropna().reset_index(drop=True)

# 📊 ─────────── MODEL & FORECAST ───────────
def train_ann(df_feat: pd.DataFrame, well: str, test_size: float,
              hidden_layer: tuple[int, ...]):
    X = df_feat.drop(columns=[well, "Date"])
    y = df_feat[well]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    model = MLPRegressor(hidden_layer_sizes=hidden_layer,
                         activation="relu",
                         solver="adam",
                         random_state=42,
                         max_iter=2000,
                         early_stopping=True)
    model.fit(X_train_std, y_train)

    y_pred_train = model.predict(X_train_std)
    y_pred_test = model.predict(X_test_std)

    metrics = {
        "R² (train)": r2_score(y_train, y_pred_train),
        "RMSE (train)": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "R² (test)": r2_score(y_test, y_pred_test),
        "RMSE (test)": np.sqrt(mean_squared_error(y_test, y_pred_test)),
    }

    # Attach predictions to full dataframe (NaNs before split)
    df_feat.loc[X_train.index, "pred"] = y_pred_train
    df_feat.loc[X_test.index, "pred"] = y_pred_test

    return model, scaler, df_feat, metrics

def recursive_forecast(model, scaler, history: pd.DataFrame,
                       well: str, horizon: int) -> pd.DataFrame:
    """
    history = last (n_lags) rows of df_feat with latest actual value
    Returns df with future dates & forecasts
    """
    last_row = history.iloc[-1].copy()
    n_lags = sum(col.startswith(f"{well}_lag") for col in history.columns)
    future_rows = []
    for step in range(1, horizon + 1):
        # Shift lag columns
        for lag in range(n_lags, 1, -1):
            last_row[f"{well}_lag{lag}"] = last_row[f"{well}_lag{lag-1}"]
        last_row[f"{well}_lag1"] = last_row["pred"] if "pred" in last_row else last_row[well]
        # Advance month/year
        next_date = last_row["Date"] + pd.DateOffset(months=1)
        next_month = next_date.month
        last_row["Months"] = next_month
        last_row["month_sin"] = np.sin(2 * np.pi * next_month / 12)
        last_row["month_cos"] = np.cos(2 * np.pi * next_month / 12)
        last_row["Date"] = next_date

        X_next = scaler.transform(last_row.drop(labels=[well, "Date"]).to_frame().T)
        next_pred = model.predict(X_next)[0]
        last_row[well] = next_pred
        last_row["pred"] = next_pred
        future_rows.append(last_row.copy())

    future_df = pd.DataFrame(future_rows)
    return future_df[["Date", "pred"]].rename(columns={"pred": "Forecast"})

# 🖥️ ─────────── UI ───────────
def groundwater_ann_page():
    st.title("🔮 ANN Groundwater Prediction")

    df = load_data(DATA_PATH)
    df = add_cyclical_month(df)

    well_cols = [c for c in df.columns if c.startswith("W")]
    well = st.sidebar.selectbox("Select Well to Model", well_cols, index=0)

    n_lags = st.sidebar.slider("Number of lag steps", 1, 24, 12)
    test_size = st.sidebar.slider("Test size (fraction)", 0.1, 0.5, 0.2, step=0.05)
    hidden_layer_sz = st.sidebar.text_input(
        "Hidden-layer sizes (comma-sep)", "64,32"
    )
    hidden_layer = tuple(int(x) for x in hidden_layer_sz.split(",") if x.strip())

    horizon = st.sidebar.number_input("Forecast horizon (months)", 1, 60, 12)

    # Build feature set
    df_feat = create_lag_features(df[["Date", "Months", "month_sin", "month_cos", well]], well, n_lags)

    # Train & predict
    model, scaler, df_pred, metrics = train_ann(df_feat, well, test_size, hidden_layer)

    # ────── METRICS ──────
    st.subheader("Model performance")
    st.json({k: round(v, 4) for k, v in metrics.items()})

    # ────── PLOT: actual vs pred ──────
    fig_hist = px.line(
        df_pred, x="Date", y=[well, "pred"],
        labels={"value": "Groundwater Level (m)", "variable": "Legend"},
        title=f"Actual vs ANN prediction – {well}"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # ────── FORECAST ──────
    future_df = recursive_forecast(model, scaler, df_pred.tail(1), well, horizon)
    st.subheader(f"{horizon}-month forecast")
    fig_future = px.line(
        pd.concat([df_pred[["Date", well]].rename(columns={well: "Level"}),
                   future_df.rename(columns={"Forecast": "Level"})]),
        x="Date", y="Level", title=f"Forecast horizon ({horizon} months) – {well}"
    )
    st.plotly_chart(fig_future, use_container_width=True)

    # Download button
    csv_buffer = io.StringIO()
    future_df.to_csv(csv_buffer, index=False)
    st.download_button(
        "Download forecast CSV",
        data=csv_buffer.getvalue(),
        file_name=f"{well}_forecast_{datetime.today().date()}.csv",
        mime="text/csv"
    )

# If running standalone:
if __name__ == "__main__":
    groundwater_ann_page()
