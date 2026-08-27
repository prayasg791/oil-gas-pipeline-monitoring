import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Oil & Gas Pipeline Dashboard", layout="wide")
st.title("🛢️ Oil & Gas Pipeline Incident Monitoring (2010–2026)")

# 1. Load Dataset
@st.cache_data
def load_data():
    path = "Dataset/database.txt"
    try:
        df = pd.read_csv(path, sep=None, engine="python", encoding="latin1")
    except Exception:
        df = pd.read_csv("Dataset/database.csv", sep=None, engine="python", encoding="latin1")

    # Clean column names
    df.columns = [c.strip().upper() for c in df.columns]

    # Map Year / Date
    if "IYEAR" in df.columns:
        df["Year"] = pd.to_numeric(df["IYEAR"], errors="coerce")
    elif "ACCIDENT DATE/TIME" in df.columns:
        df["Accident Date/Time"] = pd.to_datetime(df["ACCIDENT DATE/TIME"], errors="coerce")
        df["Year"] = df["Accident Date/Time"].dt.year
    else:
        date_col = next((c for c in df.columns if "DATE" in c or "TIME" in c), None)
        df["Year"] = pd.to_datetime(df[date_col], errors="coerce").dt.year if date_col else 2026

    # Map Costs
    cost_col = next((c for c in ["TOTAL_COST_CURRENT", "TOTAL_COST_INFLATION", "ALL COSTS", "TOTAL_COST"] if c in df.columns), None)
    df["Cost_Clean"] = pd.to_numeric(df[cost_col], errors="coerce").fillna(0) if cost_col else 0.0

    # Map Net Loss / Release volume
    loss_col = next((c for c in ["UNINTENTIONAL_RELEASE_BBLS", "UNINTENTIONAL_RELEASE_MCF", "NET LOSS (BARRELS)", "TOTAL_RELEASE_BBLS"] if c in df.columns), None)
    df["Loss_Clean"] = pd.to_numeric(df[loss_col], errors="coerce").fillna(0) if loss_col else 0.0

    # Map Cause, Commodity, and Operator
    df["Cause_Clean"] = df[next((c for c in ["CAUSE", "CAUSE CATEGORY", "GENERAL_CAUSE"] if c in df.columns), "")] if any(c in df.columns for c in ["CAUSE", "CAUSE CATEGORY", "GENERAL_CAUSE"]) else "Unspecified"
    df["Commodity_Clean"] = df[next((c for c in ["COMMODITY_RELEASED_TYPE", "LIQUID NAME", "COMMODITY_SUB_TYPE"] if c in df.columns), "")] if any(c in df.columns for c in ["COMMODITY_RELEASED_TYPE", "LIQUID NAME", "COMMODITY_SUB_TYPE"]) else "Gas/Liquid"
    df["Operator_Clean"] = df[next((c for c in ["NAME", "OPERATOR NAME", "OPERATOR_NAME"] if c in df.columns), "")] if any(c in df.columns for c in ["NAME", "OPERATOR NAME", "OPERATOR_NAME"]) else "Operator"

    # Map Coordinates
    lat_col = next((c for c in ["LOCATION_LATITUDE", "DECIMAL_LATITUDE", "ACCIDENT LATITUDE"] if c in df.columns), None)
    lon_col = next((c for c in ["LOCATION_LONGITUDE", "DECIMAL_LONGITUDE", "ACCIDENT LONGITUDE"] if c in df.columns), None)
    df["Latitude"] = pd.to_numeric(df[lat_col], errors="coerce") if lat_col else np.nan
    df["Longitude"] = pd.to_numeric(df[lon_col], errors="coerce") if lon_col else np.nan

    return df

# Execute loading
df = load_data()

# 2. Sidebar Filters
st.sidebar.header("Filter Incidents")
available_years = sorted([int(y) for y in df["Year"].dropna().unique() if 2000 <= y <= 2030])
selected_year = st.sidebar.selectbox("Select Year", options=["All"] + available_years)

filtered_df = df if selected_year == "All" else df[df["Year"] == selected_year]

# 3. KPI Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Incidents", f"{len(filtered_df):,}")
col2.metric("Total Cost ($M)", f"${filtered_df['Cost_Clean'].sum() / 1e6:,.2f}M")
col3.metric("Volume Loss Reported", f"{filtered_df['Loss_Clean'].sum():,.0f} units")

# 4. Visualizations
c1, c2 = st.columns(2)

with c1:
    st.subheader("Incidents by Commodity / Liquid Type")
    top_comm = filtered_df["Commodity_Clean"].value_counts().head(8).reset_index()
    top_comm.columns = ["Commodity", "Incidents"]
    fig_comm = px.bar(top_comm, x="Commodity", y="Incidents", color="Incidents", color_continuous_scale="Blues")
    st.plotly_chart(fig_comm, use_container_width=True)

with c2:
    st.subheader("Top Causes of Incidents")
    fig_cause = px.pie(filtered_df, names="Cause_Clean", hole=0.4, title="Incident Distribution by Primary Cause")
    st.plotly_chart(fig_cause, use_container_width=True)

# 5. Geographical Incident Map
st.subheader("Geographical Incident Map")
geo_df = filtered_df.dropna(subset=["Latitude", "Longitude"]).copy()

# Valid geographical bounds filter
geo_df = geo_df[(geo_df["Latitude"].between(20.0, 55.0)) & (geo_df["Longitude"].between(-130.0, -65.0))]

if not geo_df.empty:
    try:
        fig_map = px.scatter_map(
            geo_df,
            lat="Latitude",
            lon="Longitude",
            color="Commodity_Clean",
            hover_name="Operator_Clean",
            hover_data={"Cost_Clean": ":$,.0f", "Latitude": False, "Longitude": False},
            zoom=3,
            map_style="carto-positron"
        )
    except AttributeError:
        fig_map = px.scatter_geo(
            geo_df,
            lat="Latitude",
            lon="Longitude",
            color="Commodity_Clean",
            hover_name="Operator_Clean",
            scope="usa"
        )

    # Clean styling aur fixed radius markers
    fig_map.update_traces(marker=dict(size=8, opacity=0.75))
    fig_map.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=550)
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.info("No valid geographical coordinates found for current filter.")

# 6. Predictive Machine Learning Section
st.markdown("---")
st.subheader("🤖 Predictive Analytics: Incident Cost Estimator")

# Helper to clean numeric columns with commas, dollar signs, etc.
def parse_numeric(series):
    return pd.to_numeric(series.astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.strip(), errors='coerce')

# Auto-detect available volume/cost columns in dataset
volume_candidates = [c for c in df.columns if any(k in c for k in ["RELEASE", "VOLUME", "LOSS", "UNINTENTIONAL", "MCF", "BBLS"])]
cost_candidates = [c for c in df.columns if any(k in c for k in ["TOTAL_COST", "COST", "PROPERTY_DAMAGE", "EMERGENCY_COST"])]

vol_col = volume_candidates[0] if volume_candidates else "Loss_Clean"
c_col = cost_candidates[0] if cost_candidates else "Cost_Clean"

ml_df = pd.DataFrame()
ml_df["Volume"] = parse_numeric(df[vol_col])
ml_df["Cost"] = parse_numeric(df[c_col])

# Fallback: if selected volume column is empty, try all numeric columns with variance
if ml_df["Volume"].dropna().nunique() <= 1:
    for candidate in volume_candidates:
        test_vol = parse_numeric(df[candidate])
        if test_vol.dropna().nunique() > 5:
            ml_df["Volume"] = test_vol
            vol_col = candidate
            break

# Drop NaNs and zeros
train_data = ml_df.dropna().loc[(ml_df["Volume"] > 0) & (ml_df["Cost"] > 100)].copy()

if len(train_data) >= 10:
    X = train_data[["Volume"]]
    y = np.log1p(train_data["Cost"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    p_col1, p_col2 = st.columns(2)
    with p_col1:
        st.write(f"**Simulate Incident Cost** *(Feature: `{vol_col}`)*")
        default_val = float(np.round(train_data["Volume"].median(), 1)) if len(train_data) > 0 else 100.0
        input_volume = st.number_input("Estimated Release Volume", min_value=1.0, value=max(1.0, default_val), step=25.0)
        
        if st.button("Predict Incident Cost", key="btn_predict_cost"):
            pred_log = rf.predict(pd.DataFrame([[input_volume]], columns=["Volume"]))
            pred_cost = np.expm1(pred_log)[0]
            st.success(f"Estimated Total Incident Cost: **${pred_cost:,.2f}**")

    with p_col2:
        st.write("**Cost vs. Volume Correlation**")
        sample_size = min(500, len(train_data))
        fig_scatter = px.scatter(
            train_data.sample(sample_size, random_state=42),
            x="Volume",
            y="Cost",
            log_x=True,
            log_y=True,
            labels={"Volume": f"Release Volume ({vol_col})", "Cost": "Total Cost ($)"},
            title="Log Scale: Volume Loss vs Total Cost"
        )
        fig_scatter.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.info("Training with fallback incident feature parameters...")
    # Secondary fallback using incident index / year if direct release volumes are null
    df["Synthetic_Vol"] = parse_numeric(df.get("COMMODITY_RELEASED_VOLUME", pd.Series(range(1, len(df)+1))))
    df["Cost_Clean"] = parse_numeric(df.get("Cost_Clean", 10000))
    st.warning("Volume fields in this specific file are non-numeric or zero. Try selecting 'All' in the year filter or check the raw column names.")