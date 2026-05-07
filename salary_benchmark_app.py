"""
Singapore Salary Benchmarking Dashboard
Based on MyCareersFuture job posting data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SG Salary Benchmark",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px; border-radius: 12px; color: white;
        text-align: center; margin: 4px;
    }
    .metric-card h2 { font-size: 2rem; margin: 0; }
    .metric-card p  { margin: 0; opacity: 0.85; font-size: 0.9rem; }
    .section-header {
        border-left: 4px solid #667eea;
        padding-left: 10px;
        margin-bottom: 1rem;
    }
    [data-testid="stMetricValue"] { font-size: 1.6rem !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
LEVEL_ORDER = [
    "Fresh/entry level", "Non-executive", "Junior Executive",
    "Executive", "Professional", "Senior Executive",
    "Manager", "Middle Management", "Senior Management",
]

LEVEL_COLORS = {
    "Fresh/entry level": "#a8e6cf",
    "Non-executive": "#88d8b0",
    "Junior Executive": "#ffeaa7",
    "Executive": "#fdcb6e",
    "Professional": "#74b9ff",
    "Senior Executive": "#fd79a8",
    "Manager": "#e17055",
    "Middle Management": "#d63031",
    "Senior Management": "#6c5ce7",
}
def hex_to_rgba(hex_color: str, alpha: float = 0.08) -> str:
    """Convert #rrggbb to rgba(r,g,b,alpha)"""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# ─────────────────────────────────────────────
# Data loading & caching
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data...")
def load_data(path: str) -> pd.DataFrame:
    # Load the pre-cleaned Parquet file — no cleaning needed here,
    # all transformations were done in clean_data.ipynb
    df = pd.read_parquet(path)

    # Restore ordered Categorical for positionLevels
    # (Parquet preserves it, but we re-cast to be safe)
    df["positionLevels"] = pd.Categorical(
        df["positionLevels"], categories=LEVEL_ORDER, ordered=True
    )

    return df


# ─────────────────────────────────────────────
# Sidebar – data & filters
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/48/Flag_of_Singapore.svg/320px-Flag_of_Singapore.svg.png", width=60)
    st.title("⚙️ Controls")

    data_path = st.text_input(
        "Parquet file path",
        value="sg_jobs_salary.parquet",
        help="Path to the cleaned Parquet file (output of clean_data.ipynb)",
    )

    try:
        df_full = load_data(data_path)
        st.success(f"✅ {len(df_full):,} records loaded")
    except FileNotFoundError:
        st.error("File not found. Please check the path.")
        st.stop()

    st.divider()

    all_cats   = sorted(df_full["category"].dropna().unique())
    all_levels = [l for l in LEVEL_ORDER if l in df_full["positionLevels"].cat.categories]

    sel_cats   = st.multiselect("Job Categories", all_cats, default=all_cats[:6])
    sel_levels = st.multiselect("Position Levels", all_levels, default=all_levels)

    sal_min, sal_max = st.slider(
        "Salary Range (SGD / month)",
        500, 50_000, (500, 30_000), step=500,
        format="$%d",
    )

    # Company analysis input
    st.divider()
    st.subheader("🏢 Company Analysis")
    company_input = st.text_input(
        "Company name (partial match)",
        placeholder="e.g. GRAB, DBS, GOOGLE",
    ).upper()

    st.divider()
    st.caption("Data: MyCareersFuture (Singapore)")

# ─────────────────────────────────────────────
# Apply filters
# ─────────────────────────────────────────────
df = df_full.copy()
if sel_cats:
    df = df[df["category"].isin(sel_cats)]
if sel_levels:
    df = df[df["positionLevels"].isin(sel_levels)]
df = df[(df["average_salary"] >= sal_min) & (df["average_salary"] <= sal_max)]

# ─────────────────────────────────────────────
# Navigation tabs
# ─────────────────────────────────────────────
tabs = st.tabs([
    "📊 Overview",
    "📈 Salary Distribution",
    "🏆 Market Benchmarks",
    "🔗 Experience vs Salary",
    "🏢 Company Positioning",
    "🎯 Salary Recommender",
])

# ══════════════════════════════════════════════
# TAB 1 – Overview
# ══════════════════════════════════════════════
with tabs[0]:
    st.markdown("## 📊 Market Overview")
    st.markdown("Real-time salary intelligence from Singapore job postings (MyCareersFuture).")

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Postings",  f"{len(df):,}")
    c2.metric("Unique Companies", f"{df['postedCompany_name'].nunique():,}")
    c3.metric("Job Categories",   f"{df['category'].nunique()}")
    c4.metric("Median Salary",    f"${df['average_salary'].median():,.0f}")
    c5.metric("P75 Salary",       f"${df['average_salary'].quantile(0.75):,.0f}")

    st.divider()
    col1, col2 = st.columns(2)

    # Postings by category bar
    with col1:
        st.markdown("#### Postings by Category")
        cat_cnt = df["category"].value_counts().reset_index()
        cat_cnt.columns = ["Category", "Count"]
        fig = px.bar(
            cat_cnt.head(15), x="Count", y="Category", orientation="h",
            color="Count", color_continuous_scale="Purples",
            template="plotly_white",
        )
        fig.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=20, b=0), height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Postings by level
    with col2:
        st.markdown("#### Postings by Position Level")
        lvl_cnt = df["positionLevels"].value_counts().reset_index()
        lvl_cnt.columns = ["Level", "Count"]
        fig = px.pie(
            lvl_cnt, values="Count", names="Level",
            color="Level",
            color_discrete_map=LEVEL_COLORS,
            template="plotly_white",
            hole=0.4,
        )
        fig.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Salary heatmap: Category × Level
    st.markdown("#### Median Salary Heatmap (Category × Level)")
    heat = (
        df.groupby(["category", "positionLevels"], observed=True)["average_salary"]
        .median()
        .reset_index()
    )
    heat_pivot = heat.pivot(index="category", columns="positionLevels", values="average_salary")
    # Keep level order
    ordered_cols = [l for l in LEVEL_ORDER if l in heat_pivot.columns]
    heat_pivot = heat_pivot[ordered_cols]

    fig = px.imshow(
        heat_pivot,
        color_continuous_scale="RdYlGn",
        aspect="auto",
        labels={"color": "Median Salary (SGD)"},
        template="plotly_white",
    )
    fig.update_layout(margin=dict(l=0, r=10, t=10, b=0), height=500)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 2 – Salary Distribution
# ══════════════════════════════════════════════
with tabs[1]:
    st.markdown("## 📈 Salary Distribution")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Distribution by Position Level (Box Plot)")
        fig = px.box(
            df[df["positionLevels"].notna()],
            x="positionLevels", y="average_salary",
            color="positionLevels",
            color_discrete_map=LEVEL_COLORS,
            category_orders={"positionLevels": LEVEL_ORDER},
            labels={"average_salary": "Monthly Salary (SGD)", "positionLevels": "Level"},
            template="plotly_white",
        )
        fig.update_layout(showlegend=False, height=420, margin=dict(l=0, r=0, t=20, b=0))
        fig.update_xaxes(tickangle=30)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Salary Density Curves (KDE) by Level")
        fig = go.Figure()
        for level in LEVEL_ORDER:
            sub = df[df["positionLevels"] == level]["average_salary"].dropna()
            if len(sub) < 30:
                continue
            kde = gaussian_kde(sub, bw_method=0.3)
            x_range = np.linspace(sub.quantile(0.01), sub.quantile(0.99), 300)
            y_vals  = kde(x_range)
            fig.add_trace(go.Scatter(
                x=x_range, y=y_vals, mode="lines",
                name=level,
                line=dict(color=LEVEL_COLORS.get(level, "#888"), width=2),
                fill="tozeroy",
                fillcolor=hex_to_rgba(LEVEL_COLORS.get(level, "#888888"), alpha=0.08),
            ))
        fig.update_layout(
            xaxis_title="Monthly Salary (SGD)", yaxis_title="Density",
            template="plotly_white", height=420,
            margin=dict(l=0, r=0, t=20, b=0), legend_title="Level",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("#### Salary Distribution by Job Category (Violin)")
    top_cats = df["category"].value_counts().head(12).index.tolist()
    fig = px.violin(
        df[df["category"].isin(top_cats)],
        x="category", y="average_salary",
        color="category", box=True, points=False,
        labels={"average_salary": "Monthly Salary (SGD)", "category": "Category"},
        template="plotly_white",
    )
    fig.update_layout(showlegend=False, height=450, margin=dict(l=0, r=0, t=20, b=0))
    fig.update_xaxes(tickangle=35)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 3 – Market Benchmarks (P25/P50/P75/P90)
# ══════════════════════════════════════════════
with tabs[2]:
    st.markdown("## 🏆 Market Benchmarks")
    st.caption("Salary percentiles computed from real job postings — your defensible benchmark baseline.")

    bench_cat = st.selectbox("Select Category for Detailed Benchmark", sorted(df["category"].dropna().unique()))
    df_bench  = df[df["category"] == bench_cat]

    # Percentile table
    pct_tbl = (
        df_bench.groupby("positionLevels", observed=True)["average_salary"]
        .agg(
            N="count",
            P25=lambda x: x.quantile(0.25),
            P50=lambda x: x.quantile(0.50),
            P75=lambda x: x.quantile(0.75),
            P90=lambda x: x.quantile(0.90),
        )
        .reset_index()
    )
    pct_tbl = pct_tbl[pct_tbl["N"] >= 5]  # require ≥5 samples
    for col in ["P25", "P50", "P75", "P90"]:
        pct_tbl[col] = pct_tbl[col].round(0).astype(int)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(f"#### Percentile Table — {bench_cat}")
        st.dataframe(
            pct_tbl.rename(columns={"positionLevels": "Level", "N": "# Postings"}),
            hide_index=True, use_container_width=True,
        )

    with col2:
        st.markdown("#### Percentile Bands by Level")
        fig = go.Figure()
        levels_in_bench = pct_tbl["positionLevels"].tolist()
        fig.add_trace(go.Bar(
            name="P25", x=levels_in_bench, y=pct_tbl["P25"],
            marker_color="#a8e6cf",
        ))
        fig.add_trace(go.Bar(
            name="P50 (Median)", x=levels_in_bench, y=pct_tbl["P50"],
            marker_color="#fdcb6e",
        ))
        fig.add_trace(go.Bar(
            name="P75", x=levels_in_bench, y=pct_tbl["P75"],
            marker_color="#fd79a8",
        ))
        fig.add_trace(go.Bar(
            name="P90", x=levels_in_bench, y=pct_tbl["P90"],
            marker_color="#6c5ce7",
        ))
        fig.update_layout(
            barmode="group", template="plotly_white",
            xaxis_title="Position Level", yaxis_title="Monthly Salary (SGD)",
            height=380, margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Multi-category P50 comparison
    st.divider()
    st.markdown("#### Cross-Category P50 Comparison by Level")
    p50_multi = (
        df.groupby(["category", "positionLevels"], observed=True)["average_salary"]
        .median().reset_index()
    )
    p50_multi.columns = ["Category", "Level", "P50_Salary"]
    top10 = df["category"].value_counts().head(10).index.tolist()
    p50_multi = p50_multi[p50_multi["Category"].isin(top10)]
    fig = px.line(
        p50_multi, x="Level", y="P50_Salary", color="Category",
        markers=True,
        category_orders={"Level": LEVEL_ORDER},
        labels={"P50_Salary": "Median Salary (SGD)", "Level": "Position Level"},
        template="plotly_white",
    )
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=20, b=0))
    fig.update_xaxes(tickangle=30)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 4 – Experience vs Salary
# ══════════════════════════════════════════════
with tabs[3]:
    st.markdown("## 🔗 Experience vs Salary Trends")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Salary Progression by Experience Band")
        exp_pct = (
            df.groupby(["exp_band", "positionLevels"], observed=True)["average_salary"]
            .median().reset_index()
        )
        exp_pct.columns = ["Experience Band", "Level", "Median Salary"]
        fig = px.bar(
            exp_pct, x="Experience Band", y="Median Salary",
            color="Level", barmode="group",
            color_discrete_map=LEVEL_COLORS,
            category_orders={"Level": LEVEL_ORDER},
            template="plotly_white",
            labels={"Median Salary": "Median Monthly Salary (SGD)"},
        )
        fig.update_layout(height=420, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Experience vs Salary Scatter (sampled 5 000 pts)")
        sample = df.dropna(subset=["positionLevels"]).sample(min(5000, len(df)), random_state=42)
        fig = px.scatter(
            sample, x="minimumYearsExperience", y="average_salary",
            color="positionLevels",
            color_discrete_map=LEVEL_COLORS,
            category_orders={"positionLevels": LEVEL_ORDER},
            opacity=0.5, trendline=None,
            labels={
                "minimumYearsExperience": "Min. Years Experience",
                "average_salary": "Average Salary (SGD)",
                "positionLevels": "Level",
            },
            template="plotly_white",
        )
        fig.update_layout(height=420, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # Salary growth curve by category
    st.divider()
    st.markdown("#### Salary Growth Curve by Category (Median across experience years 0-15)")
    sel_cats_exp = st.multiselect(
        "Pick categories", all_cats,
        default=["Information Technology", "Banking and Finance", "Engineering", "Healthcare / Pharmaceutical"],
        key="exp_cats",
    )
    exp_line = (
        df[df["category"].isin(sel_cats_exp)]
        .groupby(["category", "minimumYearsExperience"], observed=True)["average_salary"]
        .median().reset_index()
    )
    exp_line = exp_line[exp_line["minimumYearsExperience"] <= 15]
    fig = px.line(
        exp_line, x="minimumYearsExperience", y="average_salary",
        color="category", markers=True,
        labels={
            "minimumYearsExperience": "Min. Years Experience",
            "average_salary": "Median Salary (SGD)",
            "category": "Category",
        },
        template="plotly_white",
    )
    fig.update_layout(height=400, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 5 – Company Positioning
# ══════════════════════════════════════════════
with tabs[4]:
    st.markdown("## 🏢 Company Competitive Positioning")
    st.caption("See where a company's postings sit relative to market benchmarks.")

    if not company_input:
        st.info("👈 Enter a company name in the sidebar to begin the analysis.")
    else:
        matches = df[df["postedCompany_name"].str.contains(company_input, na=False)]
        if len(matches) == 0:
            st.warning(f"No postings found matching '{company_input}'. Try a shorter keyword.")
        else:
            company_display = matches["postedCompany_name"].value_counts().index[0]
            st.success(f"Found **{len(matches):,}** postings for: **{company_display}**")

            # ── Compa-ratio computation ──────────────────────
            market_p50 = (
                df.groupby(["category", "positionLevels"], observed=True)["average_salary"]
                .median().reset_index()
                .rename(columns={"average_salary": "market_p50"})
            )
            comp_df = matches.merge(market_p50, on=["category", "positionLevels"], how="left")
            comp_df["compa_ratio"] = comp_df["average_salary"] / comp_df["market_p50"]
            comp_df = comp_df.dropna(subset=["compa_ratio"])

            overall_cr = comp_df["compa_ratio"].median()

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Postings Analysed",  f"{len(comp_df):,}")
            col2.metric("Median Compa-Ratio", f"{overall_cr:.2f}",
                        help="1.00 = exactly at market median. >1.05 = above market.")
            col3.metric("% Above Market (>P50)", f"{(comp_df['compa_ratio']>1).mean()*100:.0f}%")
            col4.metric("% Below Market (<P50)", f"{(comp_df['compa_ratio']<1).mean()*100:.0f}%")

            st.divider()
            col1, col2 = st.columns(2)

            # Compa-ratio distribution
            with col1:
                st.markdown("#### Compa-Ratio Distribution")
                fig = px.histogram(
                    comp_df, x="compa_ratio", nbins=40,
                    color_discrete_sequence=["#667eea"],
                    labels={"compa_ratio": "Compa-Ratio"},
                    template="plotly_white",
                )
                fig.add_vline(x=1.0,  line_dash="dash", line_color="green",  annotation_text="Market P50")
                fig.add_vline(x=overall_cr, line_dash="dot", line_color="#d63031",
                              annotation_text=f"Co. Median {overall_cr:.2f}")
                fig.update_layout(height=360, margin=dict(l=0, r=0, t=20, b=0))
                st.plotly_chart(fig, use_container_width=True)

            # Compa-ratio by position level
            with col2:
                st.markdown("#### Avg Compa-Ratio by Position Level")
                cr_lvl = (
                    comp_df.groupby("positionLevels", observed=True)["compa_ratio"]
                    .mean().reset_index()
                )
                cr_lvl.columns = ["Level", "Avg Compa-Ratio"]
                cr_lvl["Color"] = cr_lvl["Avg Compa-Ratio"].apply(
                    lambda x: "#00b894" if x >= 1.0 else "#d63031"
                )
                fig = go.Figure(go.Bar(
                    x=cr_lvl["Level"], y=cr_lvl["Avg Compa-Ratio"],
                    marker_color=cr_lvl["Color"],
                    text=cr_lvl["Avg Compa-Ratio"].round(2), textposition="outside",
                ))
                fig.add_hline(y=1.0, line_dash="dash", line_color="gray")
                fig.update_layout(
                    template="plotly_white", height=360,
                    margin=dict(l=0, r=0, t=20, b=0),
                    yaxis_title="Avg Compa-Ratio",
                )
                st.plotly_chart(fig, use_container_width=True)

            # Company vs market by category
            st.markdown("#### Company Median vs Market Median by Category")
            cmp_cat = (
                comp_df.groupby("category", observed=True)["average_salary"]
                .median().reset_index()
                .rename(columns={"average_salary": "Company Median"})
            )
            mkt_cat = (
                df.groupby("category", observed=True)["average_salary"]
                .median().reset_index()
                .rename(columns={"average_salary": "Market Median"})
            )
            compare = cmp_cat.merge(mkt_cat, on="category").sort_values("Market Median", ascending=True)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Market Median", x=compare["Market Median"], y=compare["category"],
                orientation="h", marker_color="#b2bec3",
            ))
            fig.add_trace(go.Bar(
                name="Company Median", x=compare["Company Median"], y=compare["category"],
                orientation="h", marker_color="#6c5ce7",
            ))
            fig.update_layout(
                barmode="overlay", template="plotly_white",
                xaxis_title="Median Monthly Salary (SGD)",
                height=max(300, len(compare) * 30),
                margin=dict(l=0, r=0, t=20, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Improvement recommendations
            st.divider()
            st.markdown("#### 🔧 Improvement Recommendations")
            gaps = comp_df.groupby(["category", "positionLevels"], observed=True).agg(
                avg_cr=("compa_ratio", "mean"), n=("compa_ratio", "count")
            ).reset_index()
            gaps = gaps[gaps["n"] >= 3].sort_values("avg_cr")
            under = gaps[gaps["avg_cr"] < 0.9].head(5)
            over  = gaps[gaps["avg_cr"] > 1.15].head(5)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("🔴 **Underpaying vs Market (Compa-Ratio < 0.90)**")
                if len(under):
                    st.dataframe(
                        under.rename(columns={"category": "Category", "positionLevels": "Level",
                                              "avg_cr": "Avg Compa-Ratio", "n": "# Postings"})
                        .assign(**{"Avg Compa-Ratio": lambda x: x["Avg Compa-Ratio"].round(2)}),
                        hide_index=True, use_container_width=True,
                    )
                else:
                    st.success("No significantly underpaying segments found.")
            with c2:
                st.markdown("🟡 **Overpaying vs Market (Compa-Ratio > 1.15)**")
                if len(over):
                    st.dataframe(
                        over.rename(columns={"category": "Category", "positionLevels": "Level",
                                             "avg_cr": "Avg Compa-Ratio", "n": "# Postings"})
                        .assign(**{"Avg Compa-Ratio": lambda x: x["Avg Compa-Ratio"].round(2)}),
                        hide_index=True, use_container_width=True,
                    )
                else:
                    st.success("No significantly overpaying segments found.")


# ══════════════════════════════════════════════
# TAB 6 – Salary Recommender
# ══════════════════════════════════════════════
with tabs[5]:
    st.markdown("## 🎯 Salary Recommender")
    st.caption("Enter a role, level, and experience to get a data-driven salary offer range.")

    r1, r2, r3 = st.columns(3)
    with r1:
        rec_cat   = st.selectbox("Job Category",    sorted(df["category"].dropna().unique()), key="rec_cat")
    with r2:
        rec_level = st.selectbox("Position Level",  all_levels, key="rec_level")
    with r3:
        rec_exp   = st.number_input("Years of Experience", min_value=0, max_value=20, value=3, key="rec_exp")

    if st.button("🔍 Get Salary Range", type="primary"):
        # Filter matching data
        sub = df[
            (df["category"] == rec_cat) &
            (df["positionLevels"] == rec_level) &
            (df["minimumYearsExperience"] <= rec_exp)
        ]["average_salary"]

        if len(sub) < 10:
            # Widen the filter
            sub = df[
                (df["category"] == rec_cat) &
                (df["positionLevels"] == rec_level)
            ]["average_salary"]

        if len(sub) < 5:
            st.warning("Insufficient data for this combination. Try a broader category or level.")
        else:
            p10 = sub.quantile(0.10)
            p25 = sub.quantile(0.25)
            p50 = sub.quantile(0.50)
            p75 = sub.quantile(0.75)
            p90 = sub.quantile(0.90)

            st.divider()
            st.markdown(f"### Results for: **{rec_cat} · {rec_level} · {rec_exp} yrs exp**")
            st.caption(f"Based on {len(sub):,} matching postings")

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("P10 (Low End)",      f"${p10:,.0f}")
            c2.metric("P25 (Entry Offer)",  f"${p25:,.0f}")
            c3.metric("P50 (Market Rate)",  f"${p50:,.0f}", delta="benchmark")
            c4.metric("P75 (Competitive)",  f"${p75:,.0f}")
            c5.metric("P90 (Top of Market)",f"${p90:,.0f}")

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=p50,
                title={"text": "Market Median (SGD/mth)"},
                gauge={
                    "axis": {"range": [p10, p90]},
                    "bar": {"color": "#667eea"},
                    "steps": [
                        {"range": [p10, p25], "color": "#ff7675"},
                        {"range": [p25, p75], "color": "#fdcb6e"},
                        {"range": [p75, p90], "color": "#00b894"},
                    ],
                    "threshold": {
                        "line": {"color": "#2d3436", "width": 4},
                        "thickness": 0.75, "value": p50,
                    },
                },
            ))
            fig.update_layout(height=320, margin=dict(l=40, r=40, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

            # Histogram with percentile markers
            fig2 = px.histogram(
                sub, nbins=40, template="plotly_white",
                labels={"value": "Monthly Salary (SGD)"},
                color_discrete_sequence=["#a29bfe"],
            )
            for val, label, col in [
                (p25, "P25", "#fdcb6e"), (p50, "P50", "#6c5ce7"),
                (p75, "P75", "#fd79a8"), (p90, "P90", "#d63031"),
            ]:
                fig2.add_vline(x=val, line_dash="dash", line_color=col,
                               annotation_text=f"{label} ${val:,.0f}",
                               annotation_position="top right")
            fig2.update_layout(
                height=300, margin=dict(l=0, r=0, t=40, b=0),
                xaxis_title="Monthly Salary (SGD)", yaxis_title="# Postings",
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Decision guidance
            st.markdown("#### 📋 Offer Strategy Guide")
            col1, col2, col3 = st.columns(3)
            col1.info(f"**Cost-Conscious Offer**\n\n${p25:,.0f} – ${p50:,.0f}\n\nBelow market median. Use for candidates with less experience or where non-cash benefits are strong.")
            col2.success(f"**Competitive Offer**\n\n${p50:,.0f} – ${p75:,.0f}\n\nAt or above median. Recommended for most hires to ensure acceptance and retention.")
            col3.warning(f"**Aggressive Offer**\n\n${p75:,.0f} – ${p90:,.0f}\n\nTop-quartile. Use for critical roles, scarce skills, or counter-offer situations.")
