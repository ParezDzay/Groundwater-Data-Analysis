import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
import pymannkendall as mk
import base64
import requests
from datetime import datetime

# ──────────────── CONFIG ────────────────
st.set_page_config(page_title="Well Data App", layout="wide")
st.sidebar.title("Navigation")

# GitHub info from secrets
token = st.secrets["github"]["token"]
username = st.secrets["github"]["username"]
repo = st.secrets["github"]["repo"]
branch = st.secrets["github"]["branch"]

# Define paths
file_path = "Wells detailed data.csv"
gw_file_path = "GW data.csv"
output_path = "GW data (missing filled).csv"
cleaned_outlier_path = "GW data (missing filled).csv"

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

# ──────────────── PAGE ROUTING ────────────────
page = st.sidebar.radio(
    "Go to",
    ["Home", "Well Map Viewer", "📈 Groundwater Data", "📉 Groundwater Level Trends for Wells", "📊 Groundwater Prediction"]
)

# ──────────────── PAGE LOGIC ────────────────
if page == "Home":
    st.title("Erbil Central Sub-Basin CSB Groundwater Data Analysis")
    st.markdown("Explore well data, visualize maps, and analyze groundwater trends and forecasts.")

elif page == "Well Map Viewer":
    if not os.path.exists(file_path):
        st.error("Well CSV file not found.")
        st.stop()
    df = pd.read_csv(file_path, on_bad_lines='skip')
    df.columns = [col.strip().replace('\n', ' ').replace('\r', '') for col in df.columns]
    df["Coordinate X"] = pd.to_numeric(df.get("Coordinate X"), errors="coerce")
    df["Coordinate Y"] = pd.to_numeric(df.get("Coordinate Y"), errors="coerce")
    df["Depth (m)"] = pd.to_numeric(df.get("Depth (m)"), errors="coerce")
    df.rename(columns={"Coordinate X": "lat", "Coordinate Y": "lon"}, inplace=True)
    df = df.dropna(subset=["lat", "lon"])

    st.title("Well Map Viewer")
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Table", "🗺️ Map View", "🔍 Filters", "⬆️ Upload CSV"])

    with tab3:
        selected_basin = st.multiselect("Select Basin(s):", df["Basin"].unique(), default=df["Basin"].unique())
        selected_district = st.multiselect("Select Sub-District(s):", df["sub district"].unique(), default=df["sub district"].unique())
        filtered_df = df[df["Basin"].isin(selected_basin) & df["sub district"].isin(selected_district)]
        st.success("Filters applied.")

    with tab1:
        st.dataframe(filtered_df)

    with tab2:
        fig = px.scatter_mapbox(
            filtered_df, lat="lat", lon="lon", color="Basin",
            hover_name="Well Name", hover_data={"Depth (m)": True},
            zoom=10, height=600
        )
        fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
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
    if not os.path.exists(output_path):
        st.error("Processed groundwater data not found.")
        st.stop()

    gw_df = pd.read_csv(output_path)
    gw_df["Year"] = gw_df["Year"].astype(int)
    gw_df["Months"] = gw_df["Months"].astype(int)
    gw_df["Date"] = pd.to_datetime(gw_df["Year"].astype(str) + "-" + gw_df["Months"].astype(str) + "-01")

    st.title("📈 Groundwater Data Over 20 Years")
    tabs = st.tabs(["📉 Data with Missing", "✏️ Edit Raw Data", "✅ Cleaned Data", "⚠️ Outlier %", "🧹 Clean & Save"])

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
        def calculate_outlier_percent(df_in):
            outlier_pct = {}
            for well in well_cols:
                series = df_in[well].dropna()
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = series[(series < lower_bound) | (series > upper_bound)]
                percent = (len(outliers) / len(series)) * 100 if len(series) > 0 else 0
                outlier_pct[well] = round(percent, 2)
            return outlier_pct

        before_cleaning = calculate_outlier_percent(gw_df)

        cleaned_df = gw_df.copy()
        for well in well_cols:
            series = cleaned_df[well]
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            cleaned_df.loc[(series < lower_bound) | (series > upper_bound), well] = np.nan
        for well in well_cols:
            cleaned_df[well] = cleaned_df[well].interpolate(method='linear', limit_direction='both')

        after_cleaning = calculate_outlier_percent(cleaned_df)

        combined_df = pd.DataFrame({
            "Outlier % Before": before_cleaning,
            "Outlier % After": after_cleaning
        })
        st.dataframe(combined_df)

    with tabs[4]:
        st.dataframe(cleaned_df)
        if st.button("Save Cleaned Data to GitHub"):
            cleaned_df.to_csv(cleaned_outlier_path, index=False)
            if push_to_github(cleaned_outlier_path, "Cleaned and saved groundwater data"):
                st.success("Cleaned data pushed to GitHub.")
            else:
                st.error("Push to GitHub failed.")

elif page == "📉 Groundwater Level Trends for Wells":
    st.title("📉 Groundwater Level Trends for Wells")
    # Add trend analysis logic here if needed
    st.info("Trend analysis page is under construction.")

elif page == "📊 Groundwater Prediction":
    st.title("📊 Groundwater Prediction Models")
    tab1, tab2 = st.tabs(["🔮 ANN Prediction", "📉 ARIMA"])

    with tab1:
        st.markdown("Placeholder for ANN model predictions.")
    with tab2:
        st.markdown("Placeholder for ARIMA model predictions.")
