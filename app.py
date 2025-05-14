# app.py

import os
import zipfile
import streamlit as st
import pandas as pd
import plotly.express as px
from kaggle.api.kaggle_api_extended import KaggleApi


st.set_page_config(page_title="Promotion Analysis Dashboard", layout="wide")
st.title("📊 Promotion Analysis Dashboard")



# ---- CONFIGURATION ----
DATASET_SLUG = "ujwalsintre/dataset-filtered"  # Replace with your actual Kaggle dataset
CSV_FILENAME = "filtered_data1.csv"                      # Replace if your file name is different
LOCAL_DIR = "data"
CSV_PATH = os.path.join(LOCAL_DIR, CSV_FILENAME)
 
@st.cache_data
def calculate_promotion_metrics(df_full, target_skus, start_date, end_date):
    import numpy as np

    # Filter for SKUs & date range
    grp = df_full.loc[
        (df_full["sku"].isin(target_skus)) &
        (df_full["day"] >= start_date) &
        (df_full["day"] <= end_date),
        ["sku","day","promo_flag","num_sales","selling_price"]
    ]
    if grp.empty:
        return pd.DataFrame(), "No data for selected SKUs in that date range."

    # Identify contiguous promotion periods by sku
    pp = (
        grp.loc[grp["promo_flag"]=="Promoted", ["sku","day"]]
        .drop_duplicates()
        .sort_values(["sku","day"])
    )
    pp["day_diff"]   = pp.groupby("sku")["day"].diff().dt.days.fillna(1)
    pp["period_id"]  = pp.groupby("sku")["day_diff"].transform(lambda x: (x!=1).cumsum())
    valid = pp.groupby("sku")["period_id"].nunique()
    valid_skus = valid[valid>=3].index.tolist()
    if not valid_skus:
        return pd.DataFrame(), "No SKUs with ≥3 promo periods."

    records = []
    for sku in target_skus:
        if sku not in valid_skus:
            continue
        sku_df = grp[grp["sku"]==sku]
        daily = sku_df.groupby(["day","promo_flag"], as_index=False).agg({
            "num_sales":"mean","selling_price":"mean"
        })
        # baseline = mean non‐promo sales & price
        base_sales = daily.loc[daily["promo_flag"]=="NonPromoted","num_sales"].mean() or 0
        base_price = daily.loc[daily["promo_flag"]=="NonPromoted","selling_price"].mean() or np.nan

        promo_days = daily.loc[daily["promo_flag"]=="Promoted",["day","num_sales","selling_price"]]
        promo_days = promo_days.sort_values("day")
        promo_days["day_diff"]  = promo_days["day"].diff().dt.days.fillna(1)
        promo_days["period_id"] = (promo_days["day_diff"]!=1).cumsum()

        for pid, dfp in promo_days.groupby("period_id"):
            start_p, end_p = dfp["day"].min(), dfp["day"].max()
            mean_sales = dfp["num_sales"].mean()
            # uplift %
            uplift = ((mean_sales - base_sales)/base_sales*100) if base_sales>0 else (100.0 if mean_sales>0 else 0)
            if uplift<=0:
                continue
            # discount %
            promo_price = dfp["selling_price"].mean()
            discount = ((base_price - promo_price)/base_price*100) if base_price>0 else np.nan
            if pd.isna(discount) or discount<=0:
                continue
            records.append({
                "SKU": sku,
                "Promotion Start": start_p.strftime("%Y-%m-%d"),
                "Promotion End":   end_p.strftime("%Y-%m-%d"),
                "Discount (%)":    round(discount,2),
                "Uplift (%)":      round(uplift,2)
            })

    if not records:
        return pd.DataFrame(), "No valid promo periods with positive discount & uplift."
    res = pd.DataFrame(records)
    # keep only SKUs with ≥3 periods
    keep = res["SKU"].value_counts()[lambda x: x>=3].index
    res = res[res["SKU"].isin(keep)].sort_values("Promotion Start")
    if res.empty:
        return pd.DataFrame(), "No SKUs left with ≥3 valid periods."
    return res, None


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
st.sidebar.title("Content Views")
view = st.sidebar.radio(
    "Select view",
    ["SKU Analysis", "Group Analysis", "Compliment Analysis"]
) 

# ---- VIEW 1: SKU ANALYSIS ----
if view == "SKU Analysis":
    st.markdown("##  SKU Analysis")

    allowed_skus = [
        "0035c505fea20479301f4eefbbeb9dfe",
        "073a6e04a132fd757d391c5daca6daa1",
        "0630c162a89ae0a9e4aa379a574a3158",
        "14eca079b09854ffd5b1c8b9b9328ed8",
        "2dd483b26d19cb16ace7fd4b7055eea2"
    ]

    sku_list = sorted([sku for sku in df["sku"].unique() if sku in allowed_skus])
    
    sku = st.selectbox("Choose SKU to Analyze", sku_list)
    df_sku = df[df["sku"] == sku]

    # Line chart for daily sales
    daily = df_sku.groupby("day")["num_sales"].sum().reset_index()
    fig = px.line(
        daily,
        x="day",
        y="num_sales",
        title=f" Daily Sales for SKU: `{sku}`",
        template="plotly_white",
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)

    # Metrics
    total_sold = int(daily["num_sales"].sum())
    unique_clients = df_sku["client_id"].nunique()
    basket_size = df_sku["check_pos"].sum() / df_sku["check_id"].nunique()
    sum1 = df_sku[df_sku["day"] <= mid_date]["num_sales"].sum()
    sum2 = df_sku[df_sku["day"] > mid_date]["num_sales"].sum()
    uplift = (sum2 - sum1) / sum1 * 100 if sum1 else 0

    st.markdown("### Key Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🛒 Total Units Sold", total_sold)
    c2.metric("📦 Basket Size Increase", f"{basket_size:.2f}%")
    c3.metric("📈 Sales Uplift", f"{uplift:.1f}%")
    c4.metric("👥 Unique Clients", unique_clients)

    st.markdown("---")
# ---- VIEW 2: GROUP ANALYSIS ----
elif view == "Group Analysis":
    st.header("Group Analysis")

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



# - View # 


elif view == "Compliment Analysis":
    st.markdown("## Compliment Analysis")

    # SKU selection
    skus = sorted(df["sku"].dropna().unique())
    selected_sku = st.selectbox("Select SKU to Analyze Compliments", skus)

    # Date range selection
    min_date = pd.to_datetime("2019-05-01")
    max_date = pd.to_datetime("2020-09-30")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", min_value=min_date, max_value=max_date, value=min_date)
    with col2:
        end_date = st.date_input("End Date", min_value=min_date, max_value=max_date, value=max_date)

    if start_date > end_date:
        st.error("Start date cannot be after end date.")
    else:
        # Filter data by date range
        df_range = df[(df["day"] >= pd.Timestamp(start_date)) & (df["day"] <= pd.Timestamp(end_date))]

        # Filter for baskets with selected SKU
        df_sku = df_range[df_range["sku"] == selected_sku]
        promotional_check_ids = df_sku["check_id"].unique()
        total_promo_baskets = len(promotional_check_ids)

        # Filter for other SKUs in those baskets
        df_compliments = df_range[
            (df_range["check_id"].isin(promotional_check_ids)) &
            (df_range["sku"] != selected_sku)
        ]

        # Count top 5 co-occurring SKUs
        compliment_counts = (
            df_compliments.groupby("sku")["check_id"]
            .nunique()
            .sort_values(ascending=False)
            .head(5)
            .reset_index()
            .rename(columns={"check_id": "baskets_with_both"})
        )

        compliment_counts["total_promo_baskets"] = total_promo_baskets
        compliment_counts["compliment_rate"] = (
            compliment_counts["baskets_with_both"] / total_promo_baskets * 100
        ).round(2)

        if not compliment_counts.empty:
            fig3 = px.pie(
                compliment_counts,
                names="sku",
                values="baskets_with_both",
                title=f"Top 5 SKUs Bought with '{selected_sku}'",
                hole=0.4,
                template="plotly_white",
                custom_data=["baskets_with_both", "total_promo_baskets", "compliment_rate"]
            )


            st.plotly_chart(fig3, use_container_width=True)

            st.subheader("Compliment Rate Table")
            st.dataframe(compliment_counts, use_container_width=True)

        else:
            st.info("No compliment products found for the selected SKU in this date range.")

