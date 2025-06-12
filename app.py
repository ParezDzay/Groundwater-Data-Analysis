# app.py  – Groundwater-data dashboard with ANN forecasting
# ──────────────────────────────────────────────────────────────
# Pages
#   • Home
#   • Well Map Viewer
#   • 📈 Groundwater Data     (clean / edit / outliers)
#   • 📉 Groundwater Level Trends for Wells   (placeholder)
#   • 🔮 ANN Groundwater Prediction           (full MLP page)
#   • 📊 ARIMA Prediction                     (placeholder)
#
# NOTE:  only the ANN page + its helpers were changed; every
#        other function is identical to your last working build
#        so you can paste this whole file over the old one.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os, base64, requests, io
from datetime import datetime
from pathlib import Path

# ───────── external libs for ANN ─────────
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics         import r2_score, mean_squared_error

# ───────────── CONFIG ─────────────
st.set_page_config(page_title="Well Data App", layout="wide")
st.sidebar.title("Navigation")

# GitHub secrets (Streamlit-Cloud style)
token    = st.secrets["github"]["token"]
username = st.secrets["github"]["username"]
repo     = st.secrets["github"]["repo"]
branch   = st.secrets["github"]["branch"]

# File paths (all relative so they work in the cloud)
file_path            = "Wells detailed data.csv"
gw_file_path         = "GW data.csv"
output_path          = "GW data (missing filled).csv"      # already processed
cleaned_outlier_path = "GW data (missing filled).csv"
GW_CSV               = "GW data (missing filled).csv"      # used by ANN

# ───────── GitHub push helper ─────────
def push_to_github(fp, msg):
    with open(fp, "r", encoding="utf-8") as f:
        content = f.read()
    payload = {
        "message": msg,
        "content": base64.b64encode(content.encode()).decode(),
        "branch":  branch
    }
    url  = f"https://api.github.com/repos/{username}/{repo}/contents/{os.path.basename(fp)}"
    hdrs = {"Authorization": f"token {token}",
            "Accept":        "application/vnd.github+json"}
    sha  = requests.get(url, headers=hdrs).json().get("sha")
    if sha:
        payload["sha"] = sha
    r = requests.put(url, headers=hdrs, json=payload)
    return r.status_code in (200, 201)

# ───────────────────────────────── ANN HELPERS ─────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_gw_csv(path: str) -> pd.DataFrame | None:
    """Return groundwater dataframe or None if file not found."""
    if not Path(path).exists():
        return None
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Year"].astype(str) + "-" +
                                df["Months"].astype(str) + "-01")
    df["month_sin"] = np.sin(2 * np.pi * df["Months"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Months"] / 12)
    return df.sort_values("Date").reset_index(drop=True)

def _add_lags(df: pd.DataFrame, well: str, n_lags: int) -> pd.DataFrame:
    with_lags = df.copy()
    for k in range(1, n_lags + 1):
        with_lags[f"{well}_lag{k}"] = with_lags[well].shift(k)
    return with_lags.dropna().reset_index(drop=True)

def _fit_ann(df_feat: pd.DataFrame, well: str,
             test_frac: float, hidden_layers: tuple[int, ...]):
    X = df_feat.drop(columns=[well, "Date"])
    y = df_feat[well]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_frac, shuffle=False
    )

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    model = MLPRegressor(hidden_layer_sizes=hidden_layers,
                         activation="relu",
                         solver="adam",
                         max_iter=2000,
                         random_state=42,
                         early_stopping=True)
    model.fit(X_tr_s, y_tr)

    y_tr_pred = model.predict(X_tr_s)
    y_te_pred = model.predict(X_te_s)

    df_feat.loc[X_tr.index, "pred"] = y_tr_pred
    df_feat.loc[X_te.index, "pred"] = y_te_pred

    metrics = {
        "R² train" : round(r2_score(y_tr, y_tr_pred), 4),
        "RMSE train": round(np.sqrt(mean_squared_error(y_tr, y_tr_pred)), 4),
        "R² test"  : round(r2_score(y_te, y_te_pred), 4),
        "RMSE test": round(np.sqrt(mean_squared_error(y_te, y_te_pred)), 4)
    }
    return model, scaler, df_feat, metrics

def _recursive_forecast(model, scaler, last_row: pd.Series,
                        well: str, n_lags: int, horizon: int):
    """Generate horizon-step forecast from last known row."""
    rows = []
    r = last_row.copy()
    for _ in range(horizon):
        # shift lags
        for k in range(n_lags, 1, -1):
            r[f"{well}_lag{k}"] = r[f"{well}_lag{k-1}"]
        r[f"{well}_lag1"] = r["pred"]

        next_date = r["Date"] + pd.DateOffset(months=1)
        r.update({
            "Date": next_date,
            "Months": next_date.month,
            "month_sin": np.sin(2 * np.pi * next_date.month / 12),
            "month_cos": np.cos(2 * np.pi * next_date.month / 12)
        })

        X_next = scaler.transform(r.drop(labels=[well, "Date"]).to_frame().T)
        next_val = model.predict(X_next)[0]
        r[well] = r["pred"] = next_val
        rows.append({"Date": next_date, "Forecast": next_val})

    return pd.DataFrame(rows)

# ───────────────────────────────── ANN PAGE ─────────────────────────────────
def ann_page():
    st.title("🔮 ANN Groundwater Prediction")

    df = _load_gw_csv(GW_CSV)
    if df is None:
        st.error(f"CSV “{GW_CSV}” not found in repo.  Upload it below ⬇️")
        up = st.file_uploader("Upload groundwater CSV", type="csv")
        if up:
            Path(GW_CSV).write_bytes(up.getvalue())
            st.success("File saved — reload the page.")
        return

    wells = [c for c in df.columns if c.startswith("W")]
    well   = st.sidebar.selectbox("Well", wells)
    lags   = st.sidebar.slider("Lag steps", 1, 24, 12)
    t_frac = st.sidebar.slider("Test size", 0.1, 0.5, 0.2, step=0.05)
    h_text = st.sidebar.text_input("Hidden layers (comma-sep)", "64,32")
    h_layers = tuple(int(x) for x in h_text.split(",") if x.strip())
    horizon  = st.sidebar.number_input("Forecast horizon (months)", 1, 60, 12)

    df_feat = _add_lags(df[["Date", "Months", "month_sin", "month_cos", well]], well, lags)
    model, scaler, df_pred, metrics = _fit_ann(df_feat, well, t_frac, h_layers)

    st.subheader("Model Performance")
    st.json(metrics)

    st.subheader("Actual vs Predicted")
    st.plotly_chart(
        px.line(df_pred, x="Date", y=[well, "pred"],
                labels={"value": "Groundwater Level (m)", "variable": "Legend"}),
        use_container_width=True
    )

    future = _recursive_forecast(model, scaler, df_pred.tail(1).iloc[0],
                                 well, lags, horizon)
    st.subheader(f"{horizon}-month Forecast")
    st.plotly_chart(
        px.line(pd.concat([
            df_pred[["Date", well]].rename(columns={well: "Level"}),
            future.rename(columns={"Forecast": "Level"})
        ]), x="Date", y="Level"),
        use_container_width=True
    )

    st.download_button(
        "Download forecast CSV",
        future.to_csv(index=False).encode(),
        file_name=f"{well}_forecast_{datetime.today().date()}.csv",
        mime="text/csv"
    )

# ───────────────────────────────── OTHER PAGES (unchanged) ─────────────────────────────────
# Home, Well Map Viewer, Groundwater Data cleaner, Trends placeholder, ARIMA placeholder

def well_map_viewer():
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
    t_data, t_map, t_filters, t_upload = st.tabs(
        ["📊 Data Table", "🗺️ Map View", "🔍 Filters", "⬆️ Upload CSV"]
    )

    with t_filters:
        basins   = st.multiselect("Select Basin(s):", df["Basin"].unique(),
                                  default=df["Basin"].unique())
        districts = st.multiselect("Select Sub-District(s):",
                                   df["sub district"].unique(),
                                   default=df["sub district"].unique())
        filtered = df[df["Basin"].isin(basins) & df["sub district"].isin(districts)]
        st.success("Filters applied.")

    with t_data:
        st.dataframe(filtered)

    with t_map:
        fig = px.scatter_mapbox(
            filtered, lat="lat", lon="lon", color="Basin",
            hover_name="Well Name", hover_data={"Depth (m)": True},
            zoom=10, height=600
        )
        fig.update_layout(mapbox_style="open-street-map",
                          margin=dict(r=0, l=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with t_upload:
        up = st.file_uploader("Upload a CSV", type="csv")
        if up:
            new = pd.read_csv(up, on_bad_lines='skip')
            new.columns = [c.strip().replace('\n', ' ').replace('\r', '') for c in new.columns]
            st.dataframe(new)
            if st.button("Append Uploaded Data to Dataset"):
                new.to_csv(file_path, mode='a', header=False, index=False)
                push_to_github(file_path, "Appended new well data")
                st.success("Data uploaded and pushed to GitHub.")

def groundwater_data_page():
    if not os.path.exists(output_path):
        st.error("Processed groundwater data not found.")
        st.stop()

    gw_df = pd.read_csv(output_path)
    gw_df["Year"]   = gw_df["Year"].astype(int)
    gw_df["Months"] = gw_df["Months"].astype(int)
    gw_df["Date"]   = pd.to_datetime(gw_df["Year"].astype(str) + "-" +
                                     gw_df["Months"].astype(str) + "-01")

    st.title("📈 Groundwater Data Over 20 Years")
    t_raw, t_edit, t_cleaned, t_out, t_save = st.tabs(
        ["📉 Data with Missing", "✏️ Edit Raw Data", "✅ Cleaned Data",
         "⚠️ Outlier %", "🧹 Clean & Save"]
    )

    wells = [c for c in gw_df.columns if c not in ("Year", "Months", "Date")]

    with t_raw:
        st.dataframe(gw_df)

    with t_edit:
        edited = st.data_editor(gw_df, num_rows="dynamic", use_container_width=True)
        if st.button("Save Edited Data and Push to GitHub"):
            edited.to_csv(gw_file_path, index=False)
            if push_to_github(gw_file_path, f"Edited and saved on {datetime.now()}"):
                st.success("Data saved and pushed to GitHub.")
            else:
                st.error("Failed to push to GitHub.")

    with t_cleaned:
        st.dataframe(gw_df)

    with t_out:
        def _pct(df_in):
            pct = {}
            for w in wells:
                s = df_in[w].dropna()
                Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
                IQR = Q3 - Q1
                lb, ub = Q1 - 1.5*IQR, Q3 + 1.5*IQR
                pct[w] = round(((s < lb) | (s > ub)).mean()*100, 2)
            return pct

        before = _pct(gw_df)
        cleaned = gw_df.copy()
        for w in wells:
            s = cleaned[w]
            Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
            IQR = Q3 - Q1
            lb, ub = Q1 - 1.5*IQR, Q3 + 1.5*IQR
            cleaned.loc[(s < lb) | (s > ub), w] = np.nan
            cleaned[w] = cleaned[w].interpolate(method='linear', limit_direction='both')
        after = _pct(cleaned)

        st.dataframe(pd.DataFrame({"Before %": before, "After %": after}))

    with t_save:
        st.dataframe(cleaned)
        if st.button("Save Cleaned Data to GitHub"):
            cleaned.to_csv(cleaned_outlier_path, index=False)
            if push_to_github(cleaned_outlier_path, "Cleaned and saved groundwater data"):
                st.success("Cleaned data pushed to GitHub.")
            else:
                st.error("Push to GitHub failed.")

# ───────────────────── ROUTER ─────────────────────
page = st.sidebar.radio(
    "Go to",
    ["Home", "Well Map Viewer", "📈 Groundwater Data",
     "📉 Groundwater Level Trends for Wells",
     "🔮 ANN Groundwater Prediction", "📊 ARIMA Prediction"]
)

if page == "Home":
    st.title("Erbil Central Sub-Basin Groundwater Dashboard")
    st.markdown("Explore well locations, clean datasets, and generate forecasts.")

elif page == "Well Map Viewer":
    well_map_viewer()

elif page == "📈 Groundwater Data":
    groundwater_data_page()

elif page == "📉 Groundwater Level Trends for Wells":
    st.title("📉 Groundwater Level Trends for Wells")
    st.info("Trend analysis page is under construction.")

elif page == "🔮 ANN Groundwater Prediction":
    ann_page()

elif page == "📊 ARIMA Prediction":
    st.title("📊 ARIMA Prediction Models")
    st.markdown("Placeholder for future ARIMA forecasts.")
