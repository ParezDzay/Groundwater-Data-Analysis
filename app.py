import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np
import pymannkendall as mk  # Assuming this package for Mann-Kendall tests

# ──────────────── CONFIG ────────────────
st.set_page_config(page_title="Well Data App", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Well Map Viewer", "📈 Groundwater Data", "📉 Groundwater Level Trends for Wells", "📊 Groundwater Prediction"]
)

file_path = r"C:\Parez\Wells detailed data.csv"
gw_file_path = r"C:\Parez\GW data.csv"
output_path = r"C:\Parez\GW data (missing filled).csv"
cleaned_outlier_path = r"C:\Parez\GW data (missing filled).csv"

# ──────────────── LOAD WELL DATA ────────────────
if page not in ["📈 Groundwater Data", "📉 Groundwater Level Trends for Wells"]:
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

# ──────────────── PAGE ROUTING ────────────────
if page == "Home":
    st.title("Erbil Central Sub-Basin CSB Groundwater Data Analysis")
    st.markdown("""
        **Features**:
        - Explore 20 well data
        - Visualize well locations on a map
        - Groundwater Analysis for 20 wells in CSB in Erbil City
    """)

elif page == "Well Map Viewer":
    st.title("Well Map Viewer")
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Table", "🗺️ Map View", "🔍 Filters", "⬆️ Upload CSV"])

    with tab3:
        st.subheader("Filter Options")
        col1, col2 = st.columns(2)
        with col1:
            selected_basin = st.multiselect("Select Basin(s):", df["Basin"].unique(), default=df["Basin"].unique())
        with col2:
            selected_district = st.multiselect("Select Sub-District(s):", df["sub district"].unique(), default=df["sub district"].unique())
        filtered_df = df[df["Basin"].isin(selected_basin) & df["sub district"].isin(selected_district)]
        st.success("Filters applied.")

    with tab1:
        st.subheader("Filtered Well Data")
        st.dataframe(filtered_df)

    with tab2:
        st.subheader("Well Locations on Map")
        fig = px.scatter_mapbox(
            filtered_df,
            lat="lat",
            lon="lon",
            color="Basin",
            hover_name="Well Name",
            hover_data={"Depth (m)": True, "Geological Formation": True, "lat": False, "lon": False},
            zoom=10,
            height=600
        )
        fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Upload Additional Well Data (CSV)")
        uploaded_file = st.file_uploader("Upload a CSV file with matching column format", type="csv")
        if uploaded_file:
            try:
                new_data = pd.read_csv(uploaded_file, on_bad_lines='skip')
                new_data.columns = [col.strip().replace('\n', ' ').replace('\r', '') for col in new_data.columns]
                st.write("Preview of uploaded data:")
                st.dataframe(new_data)

                if st.button("Append Uploaded Data to Dataset"):
                    new_data.to_csv(file_path, mode='a', header=False, index=False)
                    st.success("Data uploaded and appended successfully.")
            except Exception as e:
                st.error(f"Failed to upload: {e}")

elif page == "📈 Groundwater Data":
    st.title("Groundwater Data Over 20 Years")

    if not os.path.exists(output_path):
        st.error("Groundwater CSV file (missing filled) not found.")
        st.stop()

    gw_df = pd.read_csv(output_path)
    try:
        gw_df["Year"] = gw_df["Year"].astype(int)
        gw_df["Months"] = gw_df["Months"].astype(int)
        gw_df["Date"] = pd.to_datetime(gw_df["Year"].astype(str) + "-" + gw_df["Months"].astype(str) + "-01", format="%Y-%m-%d")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📉 Data with Missing",
            "✏️ Edit Raw Data",
            "✅ Data without Missing",
            "⚠️ Outlier % per Well",
            "🧹 Clean Data from Outliers"
        ])

        well_cols = [col for col in gw_df.columns if col not in ["Year", "Months", "Date"]]

        # Tab 1: Raw data with missing
        with tab1:
            st.subheader("Raw Groundwater Table (with missing)")
            st.dataframe(gw_df, use_container_width=True)

        # Tab 2: Edit and save raw data
        with tab2:
            st.subheader("Edit and Save Raw Groundwater Data")
            edited_df = st.data_editor(gw_df, num_rows="dynamic", use_container_width=True)
            if st.button("Save Edited Data"):
                edited_df.to_csv(gw_file_path, index=False)
                st.success("Groundwater data saved successfully.")

        # Tab 3: Data without missing (filled)
        with tab3:
            st.subheader("Groundwater Table (Missing Data Filled)")
            st.dataframe(gw_df, use_container_width=True)

            st.subheader("📊 Groundwater Trends (Depth Plot)")
            wells = well_cols
            selected_wells = st.multiselect("Select wells to display:", wells, default=wells[:3])

            if selected_wells:
                melted = gw_df.melt(id_vars=["Date"], value_vars=selected_wells, var_name="Well", value_name="GW_Level")
                melted["GW_Level"] = -melted["GW_Level"]
                fig = px.line(
                    melted,
                    x="Date",
                    y="GW_Level",
                    color="Well",
                    title="Monthly Groundwater Depth Over Time",
                    markers=True
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select at least one well.")

        # Tab 4: Outlier percentage before and after cleaning
        with tab4:
            st.subheader("Outlier Percentage per Well (Before and After Cleaning)")

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

            # Outlier % before cleaning
            before_cleaning = calculate_outlier_percent(gw_df)

            # Prepare cleaned data for after cleaning calculation
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

            # Combine into dataframe
            combined_df = pd.DataFrame({
                "Outlier % Before Cleaning": before_cleaning,
                "Outlier % After Cleaning": after_cleaning
            })

            st.dataframe(combined_df, use_container_width=True)

        # Tab 5: Clean data from outliers and interpolate
        with tab5:
            st.subheader("Clean Data by Removing Outliers and Interpolating")

            st.dataframe(cleaned_df, use_container_width=True)

            if st.button("Save Cleaned Data to CSV"):
                try:
                    cleaned_df.to_csv(cleaned_outlier_path, index=False)
                    st.success(f"Cleaned data saved successfully to:\n{cleaned_outlier_path}")
                except PermissionError:
                    st.error("Permission denied: Please close the file if it is open and try again.")

    except Exception as e:
        st.error(f"Error loading groundwater data: {e}")

elif page == "📉 Groundwater Level Trends for Wells":
    st.title("📉 Groundwater Level Trends for Wells")

    if not os.path.exists(output_path):
        st.error("Processed groundwater data not found.")
        st.stop()

    gw_df = pd.read_csv(output_path)
    gw_df["Date"] = pd.to_datetime(gw_df["Year"].astype(str) + "-" + gw_df["Months"].astype(str) + "-01", format="%Y-%m-%d")

    wells = [col for col in gw_df.columns if col not in ["Year", "Months", "Date"]]

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Plots", "📊 MK & Sen’s Slope", "🧪 MMK Analysis", "💡 ITA Analysis"])

    with tab1:
        st.subheader("Groundwater Level Trends")

        color_sequence = px.colors.qualitative.D3  # Good distinct colors

        for i, well in enumerate(wells):
            fig = px.line(
                gw_df,
                x="Date",
                y=well,
                title=f"{well} Groundwater Level Trend",
                markers=True,
                color_discrete_sequence=[color_sequence[i % len(color_sequence)]],
                labels={"Date": "Date", well: "Groundwater Level (m)"}
            )
            fig.update_traces(
                line=dict(width=1, shape='spline', smoothing=1.3),
                marker=dict(size=3, symbol='circle')
            )
            fig.update_layout(
                yaxis_title="Groundwater Level (m)",
                xaxis_title="Years",
                yaxis_autorange='reversed',
                plot_bgcolor='rgba(245,245,245,1)',
                hovermode='x unified',
                margin=dict(l=40, r=20, t=40, b=40),
                font=dict(size=13),
                xaxis=dict(showgrid=True, gridcolor='lightgray', zeroline=False),
                yaxis=dict(showgrid=True, gridcolor='lightgray', zeroline=False)
            )
            st.plotly_chart(fig, use_container_width=True)

    def run_mann_kendall(series):
        try:
            result = mk.original_test(series.dropna())
            return result
        except Exception:
            return None

    def run_modified_mk(series):
        try:
            result = mk.hamed_rao_modification_test(series.dropna())
            return result
        except Exception:
            return None

    with tab2:
        st.subheader("Mann-Kendall & Sen’s Slope Analysis")
        mk_results = {}
        for well in wells:
            res = run_mann_kendall(gw_df[well])
            if res:
                mk_results[well] = {
                    "Trend": res.trend,
                    "p-value": round(res.p, 4),
                    "Sen’s slope": round(res.slope, 4)
                }
            else:
                mk_results[well] = {
                    "Trend": "N/A",
                    "p-value": "N/A",
                    "Sen’s slope": "N/A"
                }
        mk_df = pd.DataFrame(mk_results).T
        st.dataframe(mk_df.style.format({"p-value": "{:.4f}", "Sen’s slope": "{:.4f}"}), use_container_width=True)

    with tab3:
        st.subheader("Modified Mann-Kendall Test Results")
        mmk_results = {}
        for well in wells:
            res = run_modified_mk(gw_df[well])
            if res:
                mmk_results[well] = {
                    "Trend": res.trend,
                    "p-value": round(res.p, 4),
                    "Sen’s slope": round(res.slope, 4)
                }
            else:
                mmk_results[well] = {
                    "Trend": "N/A",
                    "p-value": "N/A",
                    "Sen’s slope": "N/A"
                }
        mmk_df = pd.DataFrame(mmk_results).T
        st.dataframe(mmk_df.style.format({"p-value": "{:.4f}", "Sen’s slope": "{:.4f}"}), use_container_width=True)
    with tab4:
         st.subheader("ITA Analysis – Trend Metrics")

    ita_results = []

    for well in wells:
        series = gw_df[well].dropna()
        if len(series) < 2:
            continue  # Skip wells with insufficient data

        x = np.arange(len(series))
        y = series.values

        # Linear regression
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        r_squared = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

        # Calculate Sand and Scrit
        std_dev = np.std(y)
        sand = 0.5 * std_dev
        scrit = 0.95 * std_dev  # Adjust factor as needed

        ita_results.append({
            "Well": well,
            "Slope": round(slope, 4),
            "S": round(sand, 4),
            "Scrit": round(scrit, 4),
            "R²": round(r_squared, 4)
        })

    ita_df = pd.DataFrame(ita_results)
    st.dataframe(ita_df, use_container_width=True)

# ──────────────── GROUNDWATER PREDICTION ────────────────
elif page == "📊 Groundwater Prediction":
    st.title("📊 Groundwater Prediction Models")
    tab1, tab2 = st.tabs(["🔮 ANN Prediction", "📉 ARIMA"])

    with tab1:
        st.subheader("Artificial Neural Network (ANN) Groundwater Prediction")
        st.markdown("""
        - This section will use ANN models to predict groundwater levels based on climate variables.
        - Configure model inputs and view prediction plots here.
        """)
        st.info("ANN prediction logic not implemented yet.")

    with tab2:
        st.subheader("ARIMA Time Series Groundwater Prediction")
        st.markdown("""
        - This section will use ARIMA models to forecast groundwater levels based on time series trends.
        - ARIMA configuration and visual outputs will be displayed here.
        """)
        st.info("ARIMA prediction logic not implemented yet.")
