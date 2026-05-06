import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import plotly.graph_objects as go
import re

# --- Page Layout ---

st.set_page_config(page_title="Salary Comparison", layout="wide")

st.title("Agency vs Direct Employer Salary Comparison")
#st.caption("How well are you paid? Let's find out!")

st.header("Dashboard Overview")
#st.subheader("What this app will show")
#st.markdown("""- <ENTER LATER>""")

# --- Data Loading and Cleaning ---

# Load the dataset

import os

# DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'SGJobData.csv')
DATA_PATH = 'data/SGJobData.csv'  # Adjust this path as needed  
df = pd.read_csv(DATA_PATH)

# Display the first few rows of the dataset
# st.header("Data Overview")
# st.dataframe(df.head()) 

# Date Cleaning 

df['metadata_originalPostingDate'] = pd.to_datetime(df['metadata_originalPostingDate'])
df['metadata_newPostingDate']       = pd.to_datetime(df['metadata_newPostingDate'])
df['metadata_expiryDate']           = pd.to_datetime(df['metadata_expiryDate'])

# Salary Cleaning 

## Remove obvious outliers using the 99th percentile
p99 = np.percentile(df['average_salary'].dropna(), 99)
df_clean = df[(df['average_salary'] > 500) & (df['average_salary'] <= p99)].copy()

## Use NumPy to compute stats on the cleaned array
sal = df_clean['average_salary'].to_numpy()
print(f"Mean: {np.mean(sal):,.0f}  Median: {np.median(sal):,.0f}  Std: {np.std(sal):,.0f}")
print(f"25th pct: {np.percentile(sal, 25):,.0f}  75th pct: {np.percentile(sal, 75):,.0f}")

# Category Cleaning

def extract_first_category(cat_str):
    """Extract the first category label from the JSON-like string."""
    if pd.isna(cat_str):
        return np.nan
    match = re.search(r'"category"\s*:\s*"([^"]+)"', str(cat_str))
    return match.group(1) if match else np.nan

df_clean['primary_category'] = df_clean['categories'].apply(extract_first_category)

# Identify agency vs direct job postings

agency_keywords = ['RECRUIT', 'HR ADVISORY', 'MANPOWER', 'STAFFING', 'CONSULT', 'TALENT']
pattern = '|'.join(agency_keywords) # create a single string pattern for regex search

agencies_only = df_clean[
    (df_clean['postedCompany_name'].str.upper().str.contains(pattern, na=False)) & # Exclude rows where company name contains any of the agency keywords
    (df_clean['metadata_isPostedOnBehalf'] == True) # Exclude rows where the job is posted on behalf of another company (likely agencies)
]

direct_only = df_clean[
    (~df_clean['postedCompany_name'].str.upper().str.contains(pattern, na=False)) & # Exclude rows where company name contains any of the agency keywords
    (df_clean['metadata_isPostedOnBehalf'] == False) # Exclude rows where the job is posted on behalf of another company (likely agencies)
]

# --- Part 1: Salary Comparison with Filters and Bar Chart ---

## Assuming agencies_only and direct_only are already loaded
## If CSV:
## agencies_only = pd.read_csv("agencies_only.csv")
## direct_only = pd.read_csv("direct_only.csv")

filter_cols = ["primary_category", "employmentTypes", "positionLevels"]

combined = pd.concat([
    agencies_only[filter_cols],
    direct_only[filter_cols]
])

st.sidebar.header("Filters")

## Initialize first
if "category" not in st.session_state:
    st.session_state.category = "All"
if "employment" not in st.session_state:
    st.session_state.employment = "All"
if "position" not in st.session_state:
    st.session_state.position = "All"

## Clear button should be near the top
if st.sidebar.button("🧹 Clear Filters"):
    st.session_state.category = "All"
    st.session_state.employment = "All"
    st.session_state.position = "All"
    st.rerun()

# Cascading Filters Logic

filtered_combined = combined.copy()

## 1️⃣ Category filter (no dependency)
category_options = ["All"] + sorted(combined["primary_category"].dropna().unique())

selected_category = st.sidebar.selectbox(
    "Primary Category",
    category_options,
    key="category"
)

## 2️⃣ Filter based on category
if selected_category != "All":
    filtered_combined = filtered_combined[
        filtered_combined["primary_category"] == selected_category
    ]

## 3️⃣ Employment options depend on category
employment_options = ["All"] + sorted(
    filtered_combined["employmentTypes"].dropna().unique()
)

selected_employment = st.sidebar.selectbox(
    "Employment Type",
    employment_options,
    key="employment"
)

## 4️⃣ Filter further
if selected_employment != "All":
    filtered_combined = filtered_combined[
        filtered_combined["employmentTypes"] == selected_employment
    ]

## 5️⃣ Position options depend on BOTH above
position_options = ["All"] + sorted(
    filtered_combined["positionLevels"].dropna().unique()
)

selected_position = st.sidebar.selectbox(
    "Position Level",
    position_options,
    key="position"
)

# Apply Filter Logic 

def apply_filters(df):
    filtered = df.copy()

    if selected_category != "All":
        filtered = filtered[filtered["primary_category"] == selected_category]

    if selected_employment != "All":
        filtered = filtered[filtered["employmentTypes"] == selected_employment]

    if selected_position != "All":
        filtered = filtered[filtered["positionLevels"] == selected_position]

    return filtered

agency_filtered = apply_filters(agencies_only)
direct_filtered = apply_filters(direct_only)

agency_mean = agency_filtered["average_salary"].mean()
direct_mean = direct_filtered["average_salary"].mean()

# Display side-by-side tables and metrics

col1, col2 = st.columns(2)

with col1:
    st.subheader("Agency Listings")
    st.metric("Mean Salary", f"${agency_mean:,.0f}" if pd.notna(agency_mean) else "No data")
    st.write(f"Rows: {len(agency_filtered)}")
    st.dataframe(agency_filtered.head(50))

with col2:
    st.subheader("Direct Employer Listings")
    st.metric("Mean Salary", f"${direct_mean:,.0f}" if pd.notna(direct_mean) else "No data")
    st.write(f"Rows: {len(direct_filtered)}")
    st.dataframe(direct_filtered.head(50))

st.divider()

# Summary Table and Bar Chart

summary = pd.DataFrame({
    "Dataset": ["Agency", "Direct Employer"],
    "Mean Salary": [agency_mean, direct_mean],
    "Number of Listings": [len(agency_filtered), len(direct_filtered)]
})

st.subheader("Comparison Summary")
st.dataframe(summary)

st.bar_chart(summary.set_index("Dataset")["Mean Salary"])

# --- Part 2: Most Common Employment Types and Position Levels by Industry --- 

st.divider()
st.subheader("Most Common Employment Types and Position Levels For this Indsutry")

# Create combined source dataset for trend chart
agency_trend = agencies_only.copy()
agency_trend["Source"] = "Agencies"

direct_trend = direct_only.copy()
direct_trend["Source"] = "Direct"

df_clean_all = pd.concat([agency_trend, direct_trend], ignore_index=True)

# Helper function for count + % chart

def make_count_pct_chart(plot_df, x_col, count_col, pct_col, title):
    fig = go.Figure()

    fig.add_bar(
        x=plot_df[x_col],
        y=plot_df[count_col],
        name="Count",
        yaxis="y"
    )

    fig.add_bar(
        x=plot_df[x_col],
        y=plot_df[pct_col],
        name="% of Total",
        yaxis="y2",
        opacity=0.45
    )

    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis=dict(title="Job Count"),
        yaxis2=dict(
            title="% of Total",
            overlaying="y",
            side="right"
        ),
        barmode="group",
        hovermode="x unified",
        height=500
    )

    return fig

# --- Part 2A: Employment Type Distribution ---

st.divider()
st.subheader("Employment Type Distribution by Source")

distribution_df = df_clean_all.copy()

if selected_category != "All":
    distribution_df = distribution_df[distribution_df["primary_category"] == selected_category]

if selected_position != "All":
    distribution_df = distribution_df[distribution_df["positionLevels"] == selected_position]

pivot_count_ET = distribution_df.pivot_table(
    index="employmentTypes",
    columns="Source",
    values="metadata_jobPostId",
    aggfunc="count"
).fillna(0).reset_index()

for col in ["Agencies", "Direct"]:
    if col not in pivot_count_ET.columns:
        pivot_count_ET[col] = 0

pivot_count_ET = pivot_count_ET.rename(columns={
    "Agencies": "Agencies (Count)",
    "Direct": "Direct (Count)"
})

count_cols = ["Agencies (Count)", "Direct (Count)"]

pivot_pct_ET = (
    pivot_count_ET.set_index("employmentTypes")[count_cols]
    .div(pivot_count_ET[count_cols].sum(axis=0).replace(0, np.nan), axis=1)
    * 100
).fillna(0).reset_index()

pivot_pct_ET.columns = [
    "employmentTypes",
    "Agencies (% of Total)",
    "Direct (% of Total)"
]

pivot_ET = pivot_count_ET.merge(pivot_pct_ET, on="employmentTypes", how="left")
pivot_ET = pivot_ET.sort_values(by="Direct (Count)", ascending=False)

st.dataframe(pivot_ET.round(2), use_container_width=True)

agency_ET_chart = make_count_pct_chart(
    pivot_ET.fillna(0),
    "employmentTypes",
    "Agencies (Count)",
    "Agencies (% of Total)",
    "Agency Employment Types"
)

direct_ET_chart = make_count_pct_chart(
    pivot_ET.fillna(0),
    "employmentTypes",
    "Direct (Count)",
    "Direct (% of Total)",
    "Direct Employment Types"
)

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(agency_ET_chart, use_container_width=True)

with col2:
    st.plotly_chart(direct_ET_chart, use_container_width=True)

# --- Part 2B: Position Level Distribution ---

st.divider()
st.subheader("Position Level Distribution by Source")

distribution_df = df_clean_all.copy()

if selected_category != "All":
    distribution_df = distribution_df[distribution_df["primary_category"] == selected_category]

if selected_employment != "All":
    distribution_df = distribution_df[distribution_df["employmentTypes"] == selected_employment]

pivot_count_PL = distribution_df.pivot_table(
    index="positionLevels",
    columns="Source",
    values="metadata_jobPostId",
    aggfunc="count"
).fillna(0).reset_index()

for col in ["Agencies", "Direct"]:
    if col not in pivot_count_PL.columns:
        pivot_count_PL[col] = 0

pivot_count_PL = pivot_count_PL.rename(columns={
    "Agencies": "Agencies (Count)",
    "Direct": "Direct (Count)"
})

count_cols = ["Agencies (Count)", "Direct (Count)"]

pivot_pct_PL = (
    pivot_count_PL.set_index("positionLevels")[count_cols]
    .div(pivot_count_PL[count_cols].sum(axis=0).replace(0, np.nan), axis=1)
    * 100
).fillna(0).reset_index()

pivot_pct_PL.columns = [
    "positionLevels",
    "Agencies (% of Total)",
    "Direct (% of Total)"
]

pivot_PL = pivot_count_PL.merge(pivot_pct_PL, on="positionLevels", how="left")
pivot_PL = pivot_PL.sort_values(by="Direct (Count)", ascending=False)

st.dataframe(pivot_PL.round(2), use_container_width=True)

agency_PL_chart = make_count_pct_chart(
    pivot_PL.fillna(0),
    "positionLevels",
    "Agencies (Count)",
    "Agencies (% of Total)",
    "Agency Position Levels"
)

direct_PL_chart = make_count_pct_chart(
    pivot_PL.fillna(0),
    "positionLevels",
    "Direct (Count)",
    "Direct (% of Total)",
    "Direct Position Levels"
)

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(agency_PL_chart, use_container_width=True)

with col2:
    st.plotly_chart(direct_PL_chart, use_container_width=True)

# --- Part 3: Trend Chart for Categories, Employment Types, and Position Levels ---

st.divider()
st.subheader("Average Salary Trend Over Time")

# Create year-month column
df_clean_all["year_month"] = (
    df_clean_all["metadata_originalPostingDate"]
    .dt.to_period("M")
    .astype(str)
)

# Apply same selected filters
trend_filtered = apply_filters(df_clean_all)
st.write("Trend rows after filtering:", len(trend_filtered))

# Create pivot table
pivot = trend_filtered.pivot_table(
    index="year_month",
    columns="Source",
    values="average_salary",
    aggfunc="mean"
).reset_index()

# Sort by time
pivot = pivot.sort_values("year_month")

st.write("Trend pivot preview:")
st.dataframe(pivot.head())

available_sources = [col for col in ["Agencies", "Direct"] if col in pivot.columns]

if pivot.empty:
    st.warning("No trend data available for the selected filters.")
elif len(available_sources) == 0:
    st.warning("No Agencies or Direct salary data available for this filter.")
else:
    st.line_chart(
        pivot.set_index("year_month")[available_sources]
    )
