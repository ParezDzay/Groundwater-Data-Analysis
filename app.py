# app.py  – full Streamlit app
# ──────────────────────────────────────────────────────────────────
# Pages:
#   • Home
#   • Well Map Viewer
#   • 📈 Groundwater Data  (clean / edit / outlier handling)
#   • 📉 Groundwater Level Trends for Wells   (placeholder)
#   • 🔮 ANN Groundwater Prediction           (NEW – full MLP page)
#   • 📊 ARIMA Prediction                     (kept as placeholder)
# ------------------------------------------------------------------

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
import pymannkendall as mk
import base64
import requests
from datetime import datetime
# ANN-specific
from pathlib import Path
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import io

# ──────────────── CONFIG ────────────────
st.set_page_config(page_title="Well Data App", layout="wide")
st.sidebar.title("Navigation")

# GitHub info from secrets
token    = st.secrets["github"]["token"]
username = st.secrets["github"]["username"]
repo     = st.secrets["github"]["repo"]
branch   = st.secrets["github"]["branch"]

# Paths
file_path            = "Wells detailed data.csv"
gw_file_path         = "GW data.csv"
output_path          = "GW data (missing filled).csv"
cleaned_outlier_path = "GW data (missing filled).csv"
DATA_PATH            = r"C:\Parez\GW data (missing filled).csv"   # for ANN

# ──────────────── GitHub Upload Function ────────────────
def push_to_github(file_path, commit_message):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    encoded_content = base64.b64encode(content.encode()).decode()
    filename = os.path.basename(file_path)
    url = f"https://api.github.com/repos/{username}/{repo}/contents/{filename}"

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json"
    }

    # Check if file exists on GitHub
    response = requests.get(url, headers=headers)
    sha = response.json()["sha"] if response.status_code == 200 else None

    payload = {
        "message": commit_message,
        "content": encoded_content,
        "branch": branch
    }
    if sha:
        payload["sha"] = sha

    res = requests.put(url, headers=headers, json=payload)
    return res.status_code in [200, 201]

# ───────────────── ANN HELPERS ─────────────────
@st.cache_data(show_spinner=False)
def load_gw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Year"]   = df["Year"].astype(int)
    df["Months"] = df["Months"].astype(int)
    if "Date" not in df.columns:
        df["Date"] = pd.to_datetime(df["Year"].astype(str) + "-" + df["Months"].astype(str) + "-01")
    else:
        df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)

def add_cyclical_month(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month_sin"] = np.sin(2 * np.pi * df["Months"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Months"] / 12)
    return df

def lag_features(df: pd.DataFrame, well: str, n_lags: int) -> pd.DataFrame:
    df = df.copy()
    for lag in range(1, n_lags + 1):
        df[f"{well}_lag{lag}"] = df[well].shift(lag)
    return df.dropna().reset_index(drop=True)

def train_ann(df_feat: pd.DataFrame, well: str, test_size: float, hidden_layers: tuple[int, ...]):
    X = df_feat.drop(columns=[well, "Date"])
    y = df_feat[well]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std  = scaler.transform(X_test)

    model = MLPRegressor(hidden_layer_sizes=hidden_layers,
                         activation="relu",
                         solver="adam",
                         max_iter=2000,
                         random_state=42,
                         early_stopping=True)
    model.fit(X_train_std, y_train)

    y_pred_train = model.predict(X_train_std)
    y_pred_test  = model.predict(X_test_std)

    metrics = {
        "R² (train)"  : round(r2_score(y_train, y_pred_train), 4),
        "RMSE (train)": round(np.sqrt(mean_squared_error(y_train, y_pred_train)), 4),
        "R² (test)"   : round(r2_score(y_test, y_pred_test), 4),
        "RMSE (test)" : round(np.sqrt(mean_squared_error(y_test, y_pred_test)), 4),
    }

    df_feat.loc[X_train.index, "pred"] = y_pred_train
    df_feat.loc[X_test.index , "pred"] = y_pred_test
    return model, scaler, df_feat, metrics

def recursive_forecast(model, scaler, last_row: pd.Series, well: str, n_lags: int, horizon: int):
    # last_row: most recent row WITH 'pred' column
    forecasts = []
    row = last_row.copy()

    for step in range(1, horizon + 1):
        # shift lag columns
        for lag in range(n_lags, 1, -1):
            row[f"{well}_lag{lag}"] = row[f"{well}_lag{lag-1}"]
        row[f"{well}_lag1"] = row["pred"]

        # advance date
        next_date  = row["Date"] + pd.DateOffset(months=1)
        next_month = next_date.month
        row["Date"]      = next_date
        row["Months"]    = next_month
        row["month_sin"] = np.sin(2 * np.pi * next_month / 12)
        row["month_cos"] = np.cos(2 * np.pi * next_month / 12)

        X_next = scaler.transform(row.drop(labels=[well, "Date"]).to_frame().T)
        next_val = model.predict(X_next)[0]

        row[well] = next_val
        row["pred"] = next_val
        forecasts.append({"Date": next_date, "Forecast": next_val})

    return pd.DataFrame(forecasts)

def ann_page():
    st.title("🔮 ANN Groundwater Prediction")

    df = add_cyclical_month(load_gw(DATA_PATH))
    well_cols = [c for c in df.columns if c.startswith('W')]

    well      = st.sidebar.selectbox("Select well", well_cols, index=0)
    n_lags    = st.sidebar.slider("Lag steps", 1, 24, 12)
    test_size = st.sidebar.slider("Test fraction", 0.1, 0.5, 0.2, step=0.05)
    hidden_in = st.sidebar.text_input("Hidden layer sizes (comma-sep)", "64,32")
    horizon   = st.sidebar.number_input("Forecast horizon (months)", 1, 60, 12)

    hidden_layers = tuple(int(x) for x in hidden_in.split(",") if x.strip())

    df_feat = lag_features(df[["Date", "Months", "month_sin", "month_cos", well]], well, n_lags)

    model, scaler, df_pred, metrics = train_ann(df_feat, well, test_size, hidden_layers)

    st.subheader("Performance")
    st.json(metrics)

    st.subheader("Actual vs. Predicted")
    fig_hist = px.line(df_pred, x="Date", y=[well, "pred"],
                       labels={"value": "Groundwater Level (m)", "variable": "Legend"},
                       title=f"{well} | Train/Test fit")
    st.plotly_chart(fig_hist, use_container_width=True)

    future_df = recursive_forecast(model, scaler, df_pred.tail(1).iloc[0], well, n_lags, horizon)

    st.subheader(f"{horizon}-month forecast")
    fig_fut = px.line(
        pd.concat([df_pred[["Date", well]].rename(columns={well: "Level"}),
                   future_df.rename(columns={"Forecast": "Level"})]),
        x="Date", y="Level"
    )
    st.plotly_chart(fig_fut, use_container_width=True)

    csv = future_df.to_csv(index=False).encode()
    st.download_button("Download forecast CSV", csv,
                       file_name=f"{well}_forecast_{datetime.today().date()}.csv",
                       mime="text/csv")

# ──────────── PAGE ROUTING ────────────
page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Well Map Viewer",
        "📈 Groundwater Data",
        "📉 Groundwater Level Trends for Wells",
        "🔮 ANN Groundwater Prediction",   # new standalone page
        "📊 ARIMA Prediction"              # old prediction placeholder
    ]
)

# ──────────── PAGE LOGIC ────────────
if page == "Home":
    st.title("Erbil Central Sub-Basin CSB Groundwater Data Analysis")
    st.markdown("Explore well data, visualize maps, and analyze groundwater trends and forecasts.")

elif page == "Well Map Viewer":
    # … (UNCHANGED code) …
    # ------------- existing Well Map Viewer block -----------------
    if not os.path.exists(file_path):
        st.error("Well CSV file not found.")
        st.stop()
    df = pd.read_csv(file_path, on_bad_lines='skip')
    df.columns = [col.strip().replace('\n', ' ').replace('\r', '') for col in df.columns]
    df["Coordinate X"] = pd.to_numeric(df.get("Coordinate X"), errors="coerce")
    df["Coordinate Y"] = pd.to_numeric(df.get("Coordinate Y"), errors="coerce")
    df["Depth (m)"]    = pd.to_numeric(df.get("Depth (m)"), errors="coerce")
    df.rename(columns={"Coordinate X": "lat", "Coordinate Y": "lon"}, inplace=True)
    df = df.dropna(subset=["lat", "lon"])

    st.title("Well Map Viewer")
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Table", "🗺️ Map View", "🔍 Filters", "⬆️ Upload CSV"])

    with tab3:
        selected_basin    = st.multiselect("Select Basin(s):", df["Basin"].unique(), default=df["Basin"].unique())
        selected_district = st.multiselect("Select Sub-District(s):", df["sub district"].unique(),
                                           default=df["sub district"].unique())
        filtered_df = df[df["Basin"].isin(selected_basin) & df["sub district"].isin(selected_district)]
        st.success("Filters applied.")

    with tab1:
        st.dataframe(filtered_df)

    with tab2:
        fig = px.scatter_mapbox(filtered_df, lat="lat", lon="lon", color="Basin",
                                hover_name="Well Name", hover_data={"Depth (m)": True},
                                zoom=10, height=600)
        fig.update_layout(mapbox_style="open-street-map",
                          margin=dict(r=0, l=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        uploaded_file = st.file_uploader("Upload a CSV", type="csv")
        if uploaded_file:
            new_data = pd.read_csv(uploaded_file, on_bad_lines='skip')
            new_data.columns = [col.strip().replace('\n', ' ').replace('\r', '') for col in new_data.columns]
            st.dataframe(new_data)
            if st.button("Append Uploaded Data to Dataset"):
                new_data.to_csv(file_path, mode='a', header=False, index=False)
                push_to_github(file_path, "Appended new well data")
                st.success("Data uploaded and pushed to GitHub.")

elif page == "📈 Groundwater Data":
    # … (UNCHANGED Groundwater Data cleaning block) …
    if not os.path.exists(output_path):
        st.error("Processed groundwater data not found.")
        st.stop()

    gw_df = pd.read_csv(output_path)
    gw_df["Year"]   = gw_df["Year"].astype(int)
    gw_df["Months"] = gw_df["Months"].astype(int)
    gw_df["Date"]   = pd.to_datetime(gw_df["Year"].astype(str) + "-" + gw_df["Months"].astype(str) + "-01")

    st.title("📈 Groundwater Data Over 20 Years")
    tabs = st.tabs(["📉 Data with Missing", "✏️ Edit Raw Data", "✅ Cleaned Data",
                    "⚠️ Outlier %", "🧹 Clean & Save"])

    well_cols = [col for col in gw_df.columns if col not in ["Year", "Months", "Date"]]

    with tabs[0]:
        st.dataframe(gw_df)

    with tabs[1]:
        edited_df = st.data_editor(gw_df, num_rows="dynamic", use_container_width=True)
        if st.button("Save Edited Data and Push to GitHub"):
            edited_df.to_csv(gw_file_path, index=False)
            if push_to_github(gw_file_path, f"Edited and saved on {datetime.now()}"):
                st.success("Data saved and pushed to GitHub.")
            else:
                st.error("Failed to push to GitHub.")

    with tabs[2]:
        st.dataframe(gw_df)

    with tabs[3]:
        def outlier_pct(df_in):
            pct = {}
            for w in well_cols:
                s = df_in[w].dropna()
                Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
                IQR = Q3 - Q1
                lb, ub = Q1 - 1.5*IQR, Q3 + 1.5*IQR
                pct[w] = round(((s < lb) | (s > ub)).mean()*100, 2)
            return pct

        before = outlier_pct(gw_df)
        cleaned = gw_df.copy()
        for w in well_cols:
            s = cleaned[w]
            Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
            IQR = Q3 - Q1
            lb, ub = Q1 - 1.5*IQR, Q3 + 1.5*IQR
            cleaned.loc[(s < lb) | (s > ub), w] = np.nan
            cleaned[w] = cleaned[w].interpolate(method='linear', limit_direction='both')
        after = outlier_pct(cleaned)

        st.dataframe(pd.DataFrame({"Outlier % Before": before, "Outlier % After": after}))

    with tabs[4]:
        st.dataframe(cleaned)
        if st.button("Save Cleaned Data to GitHub"):
            cleaned.to_csv(cleaned_outlier_path, index=False)
            if push_to_github(cleaned_outlier_path, "Cleaned and saved groundwater data"):
                st.success("Cleaned data pushed to GitHub.")
            else:
                st.error("Push to GitHub failed.")

elif page == "📉 Groundwater Level Trends for Wells":
    st.title("📉 Groundwater Level Trends for Wells")
    st.info("Trend analysis page is under construction.")

elif page == "🔮 ANN Groundwater Prediction":
    ann_page()   # ← full MLP forecasting interface

elif page == "📊 ARIMA Prediction":
    st.title("📊 ARIMA Prediction Models")
    st.markdown("Placeholder for future ARIMA forecasts.")
