# app.py

import os
import zipfile
import streamlit as st
import pandas as pd
import plotly.express as px
from kaggle.api.kaggle_api_extended import KaggleApi



# ---- CONFIGURATION ----
DATASET_SLUG = "ujwalsintre/data-full"  # Replace with your actual Kaggle dataset
CSV_FILENAME = "df_full.csv"                      # Replace if your file name is different
LOCAL_DIR = "data"
CSV_PATH = os.path.join(LOCAL_DIR, CSV_FILENAME)

# ---- DOWNLOAD DATASET USING KAGGLE API ----
def download_dataset():
    if not os.path.exists(LOCAL_DIR):
        os.makedirs(LOCAL_DIR)

    csv_zip_path = os.path.join(LOCAL_DIR, CSV_FILENAME + ".zip")

    if not os.path.exists(CSV_PATH):
        st.info("Downloading dataset from Kaggle...")
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_file(DATASET_SLUG, CSV_FILENAME, path=LOCAL_DIR)

        # Unzip if downloaded as a ZIP
        if os.path.exists(csv_zip_path):
            with zipfile.ZipFile(csv_zip_path, 'r') as zip_ref:
                zip_ref.extractall(LOCAL_DIR)
            os.remove(csv_zip_path)

download_dataset()

# ---- LOAD DATA ----
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df_full = pd.read_csv(path, low_memory=False, parse_dates=["day"])
    # Filter to May 1–30, 2020
    start = pd.Timestamp("2020-05-01")
    end = pd.Timestamp("2020-05-30")
    df = df_full.query("day >= @start and day <= @end")
    return df

df = load_data(CSV_PATH)
mid_date = df["day"].min() + (df["day"].max() - df["day"].min()) / 2

# ---- SIDEBAR NAVIGATION ----
st.sidebar.title("Dashboard")
view = st.sidebar.radio(
    "Select view",
    ["SKU Analysis", "Group Analysis", "View 3", "View 4", "View 5"]
)

# ---- VIEW 1: SKU ANALYSIS ----
if view == "SKU Analysis":
    st.header("SKU Analysis")

    sku_list = sorted([sku for sku in df["sku"].unique() if pd.notna(sku)])
    sku = st.selectbox("Choose SKU", sku_list)
    df_sku = df[df["sku"] == sku]

    daily = df_sku.groupby("day")["num_sales"].sum().reset_index()
    fig = px.line(daily, x="day", y="num_sales", title=f"Daily Sales: {sku}")
    st.plotly_chart(fig, use_container_width=True)

    # Metrics
    total_sold = int(daily["num_sales"].sum())
    unique_clients = df_sku["client_id"].nunique()
    basket_size = df_sku["check_pos"].sum() / df_sku["check_id"].nunique()
    sum1 = df_sku[df_sku["day"] <= mid_date]["num_sales"].sum()
    sum2 = df_sku[df_sku["day"] > mid_date]["num_sales"].sum()
    uplift = (sum2 - sum1) / sum1 * 100 if sum1 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Sold", total_sold)
    c2.metric("Basket Size", f"{basket_size:.2f}")
    c3.metric("Sales Uplift %", f"{uplift:.1f}%")
    c4.metric("Unique Clients", unique_clients)

# ---- VIEW 2: GROUP ANALYSIS ----
elif view == "Group Analysis":
    st.header("Group Analysis (hierarchy_level2)")

    groups = sorted([g for g in df["hierarchy_level2"].dropna().unique()])
    grp = st.selectbox("Choose Group", groups)
    df_grp = df[df["hierarchy_level2"] == grp]

    daily = df_grp.groupby("day")["num_sales"].sum().reset_index()
    fig2 = px.line(daily, x="day", y="num_sales", title=f"Daily Sales: {grp}")
    st.plotly_chart(fig2, use_container_width=True)

    most_bought = (
        df_grp.groupby("sku")["num_sales"].sum().idxmax()
        if not df_grp.empty else "N/A"
    )
    g1 = df_grp[df_grp["day"] <= mid_date]["num_sales"].sum()
    g2 = df_grp[df_grp["day"] > mid_date]["num_sales"].sum()
    uplift_g = (g2 - g1) / g1 * 100 if g1 else 0

    c1, c2 = st.columns(2)
    c1.metric("Most Bought SKU", most_bought)
    c2.metric("Sales Uplift %", f"{uplift_g:.1f}%")

# ---- VIEWS 3–5 PLACEHOLDERS ----
else:
    st.header(view)
    st.write("🔧 Replace this section with your precomputed charts or metrics for this view.")
   

