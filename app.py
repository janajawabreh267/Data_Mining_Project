"""
RetailMind Analytics
Interactive Data Mining Dashboard for Business Insights + Recommendation System

Responsible Students:
- Abd Alkareem Shafie
- Jana jwabreh
- Bushra hurani

Shared Tasks:
- Streamlit dashboard integration
- Connect preprocessing, clustering, association rules, recommender, anomaly detection,
  time series, customer profile, and business reporting modules.
- Build an interactive, user-friendly, service-oriented data mining application.

Project Idea:
Data Mining Dashboard for Business Insights with Smart Product Recommendations.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from modules.preprocessing import (
    load_data,
    preprocess_data,
    get_summary_stats,
    build_preprocessing_report_text,
)

from modules.clustering import (
    compute_customer_features,
    scale_customer_features,
    find_optimal_k,
    compare_clustering_methods,
    apply_clustering_method,
    apply_pca,
    apply_svd,
    dimensionality_comparison_table,
    explain_best_clustering_method,
    explain_clustering_metrics,
    build_segmentation_insights,
    build_segment_action_plan,
    build_customer_group_summary,
    plot_elbow,
    plot_pca_clusters,
    plot_svd_clusters,
    plot_cluster_profiles,
    plot_segment_distribution,
    plot_clustering_comparison,
)

from modules.association_rules import (
    build_basket_matrix,
    generate_association_rules,
    get_rules_summary_marketing,
    plot_rules_marketing_strength,
    plot_top_rules_marketing,
    interpret_rules_marketing,
    create_marketing_report,
    explain_metrics_simple,
)

from modules.recommender import (
    build_user_item_matrix,
    compute_item_similarity,
    get_product_stats,
    recommend_for_customer,
    recommend_for_basket,
    get_popular_products,
    get_recommender_quality_report,
    build_recommender_explanation,
    build_recommender_technical_explanation,
    build_recommendation_business_insight,
    build_recommendation_action_summary,
    plot_recommendations,
    plot_popular_products,
    plot_recommender_quality,
)

from modules.anomaly_detection import (
    build_transaction_features,
    detect_anomalies,
    plot_anomaly_scatter,
    plot_anomaly_score_distribution,
    get_top_anomalies,
)

from modules.time_series import (
    daily_revenue,
    monthly_revenue,
    sales_heatmap_data,
    forecast_sales,
    plot_daily_trend,
    plot_forecast,
    plot_heatmap,
)

from modules.reporting import build_insights_text, make_pdf_report


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="RetailMind Analytics",
    page_icon="🛒",
    layout="wide",
)


# ============================================================
# UI / UX Styling
# ============================================================

# The theme toggle is placed at the top of the sidebar.
# Light Mode: white + orange
# Dark Mode: black/charcoal + orange
with st.sidebar:
    theme_mode = st.toggle("🌙 Dark Mode", value=False)

if theme_mode:
    app_bg = "#0B0F14"
    sidebar_bg = "#111827"
    panel_bg = "#111827"
    panel_bg_2 = "#1F2937"
    header_bg = "linear-gradient(135deg,#111827,#1F2937,#431407)"
    card_bg = "linear-gradient(135deg,#111827,#1F2937)"
    input_bg = "#151C27"
    text_main = "#F9FAFB"
    text_muted = "#D1D5DB"
    accent = "#F97316"
    accent_2 = "#FB923C"
    accent_soft = "rgba(249,115,22,.14)"
    border = "#7C2D12"
    border_soft = "rgba(249,115,22,.35)"
    info_bg = "rgba(249,115,22,.10)"
    info_border = "rgba(249,115,22,.35)"
    success_text = "#BBF7D0"
    warning_text = "#FDBA74"
    chart_template = "plotly_dark"
    chart_bg = "#0B0F14"
    plot_bg = "#0B0F14"
    grid_color = "rgba(255,255,255,.12)"
    chart_font = "#F9FAFB"
    bc_1_bg       = "rgba(96,165,250,.12)"
    bc_1_border   = "#60A5FA"
    bc_1_title    = "#BAE6FD"
    bc_2_bg       = "rgba(167,139,250,.12)"
    bc_2_border   = "#A78BFA"
    bc_2_title    = "#C4B5FD"
    bc_3_bg       = "rgba(52,211,153,.12)"
    bc_3_border   = "#34D399"
    bc_3_title    = "#A7F3D0"
    bc_4_bg       = "rgba(251,191,36,.12)"
    bc_4_border   = "#FBBF24"
    bc_4_title    = "#FDE68A"
    controls_bg   = "rgba(167,139,250,.07)"
    controls_border= "rgba(167,139,250,.25)"
else:
    app_bg = "#FFFFFF"
    sidebar_bg = "#FFF7ED"
    panel_bg = "#FFFFFF"
    panel_bg_2 = "#FFF7ED"
    header_bg = "linear-gradient(135deg,#FFFFFF,#FFF7ED,#FFEDD5)"
    card_bg = "linear-gradient(135deg,#FFFFFF,#FFF7ED)"
    input_bg = "#FFFFFF"
    text_main = "#111827"
    text_muted = "#4B5563"
    accent = "#EA580C"
    accent_2 = "#F97316"
    accent_soft = "rgba(249,115,22,.10)"
    border = "#FDBA74"
    border_soft = "rgba(249,115,22,.30)"
    info_bg = "rgba(249,115,22,.08)"
    info_border = "rgba(249,115,22,.30)"
    success_text = "#166534"
    warning_text = "#9A3412"
    chart_template = "plotly_white"
    chart_bg = "#FFFFFF"
    plot_bg = "#FFFFFF"
    grid_color = "rgba(17,24,39,.12)"
    chart_font = "#111827"
    bc_1_bg       = "rgba(14,165,233,.08)"
    bc_1_border   = "#0EA5E9"
    bc_1_title    = "#0C4A6E"
    bc_2_bg       = "rgba(139,92,246,.08)"
    bc_2_border   = "#8B5CF6"
    bc_2_title    = "#4C1D95"
    bc_3_bg       = "rgba(16,185,129,.08)"
    bc_3_border   = "#10B981"
    bc_3_title    = "#064E3B"
    bc_4_bg       = "rgba(245,158,11,.08)"
    bc_4_border   = "#F59E0B"
    bc_4_title    = "#78350F"
    controls_bg   = "rgba(14,165,233,.05)"
    controls_border= "rgba(14,165,233,.25)"
    

st.markdown(
    f"""
<style>
/* Main page */
.stApp {{
    background-color: {app_bg};
    color: {text_main};
}}

html, body, [class*="css"] {{
    color: {text_main};
}}

/* Sidebar */
[data-testid="stSidebar"] {{
    background-color: {sidebar_bg};
    border-right: 1px solid {border_soft};
}}

[data-testid="stSidebar"] * {{
    color: {text_main};
}}

[data-testid="stSidebar"] hr {{
    border-color: {border_soft};
}}

/* Header */
.main-header {{
    background: {header_bg};
    padding: 2rem;
    border-radius: 18px;
    margin-bottom: 1rem;
    border: 1px solid {border};
    text-align: center;
    box-shadow: 0 8px 25px rgba(249,115,22,.12);
}}

.main-header h1 {{
    font-size: 2.6rem;
    margin: 0;
    background: linear-gradient(90deg,{accent},#FB923C,#F59E0B);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

.main-header p {{
    color: {text_muted};
    margin: .5rem 0 0 0;
}}

/* Cards */
.metric-card {{
    background: {card_bg};
    border: 1px solid {border};
    border-radius: 14px;
    padding: 1rem;
    text-align: center;
    min-height: 105px;
    box-shadow: 0 6px 18px rgba(249,115,22,.10);
}}

.metric-value {{
    font-size: 1.55rem;
    font-weight: 800;
    color: {accent};
}}

.metric-label {{
    font-size: .75rem;
    color: {text_muted};
    text-transform: uppercase;
    letter-spacing: .08em;
}}

.section-header {{
    font-size: 1.25rem;
    color: {text_main};
    border-left: 4px solid {accent};
    padding-left: 1rem;
    margin: 1.3rem 0 .8rem 0;
    font-weight: 800;
}}

.info-box {{
    background: {info_bg};
    border: 1px solid {info_border};
    border-radius: 12px;
    padding: 1rem;
    color: {text_muted};
    margin: .8rem 0;
}}

.success-box {{
    background: rgba(34,197,94,.10);
    border: 1px solid rgba(34,197,94,.35);
    border-radius: 12px;
    padding: 1rem;
    color: {success_text};
    margin: .8rem 0;
}}

.warning-box {{
    background: rgba(249,115,22,.10);
    border: 1px solid rgba(249,115,22,.35);
    border-radius: 12px;
    padding: 1rem;
    color: {warning_text};
    margin: .8rem 0;
}}

.small-note {{
    color: {text_muted};
    font-size: 0.88rem;
}}

/* Buttons */
.stButton > button,
.stDownloadButton > button {{
    border: 1px solid {accent} !important;
    color: {accent} !important;
    background: transparent !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    transition: all .2s ease-in-out;
}}

.stButton > button:hover,
.stDownloadButton > button:hover {{
    background-color: {accent} !important;
    color: white !important;
    border-color: {accent} !important;
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(249,115,22,.25);
}}

/* Inputs / select boxes / text areas */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] textarea,
textarea,
input {{
    background-color: {input_bg} !important;
    color: {text_main} !important;
    border-color: {border_soft} !important;
    border-radius: 10px !important;
}}

div[data-baseweb="select"] svg {{
    fill: {accent} !important;
}}

/* Dropdown menu */
ul[role="listbox"],
div[data-baseweb="popover"] {{
    background-color: {panel_bg} !important;
    color: {text_main} !important;
    border: 1px solid {border_soft} !important;
}}

li[role="option"] {{
    background-color: {panel_bg} !important;
    color: {text_main} !important;
}}

li[role="option"]:hover {{
    background-color: {accent_soft} !important;
}}

/* Sliders */
.stSlider [data-baseweb="slider"] div {{
    color: {accent} !important;
}}

.stSlider [role="slider"] {{
    background-color: {accent} !important;
    border-color: {accent} !important;
}}

/* Tabs */
button[data-baseweb="tab"] {{
    color: {text_muted} !important;
    border-radius: 10px 10px 0 0;
}}

button[data-baseweb="tab"][aria-selected="true"] {{
    color: {accent} !important;
    border-bottom: 3px solid {accent} !important;
    background-color: {accent_soft} !important;
}}

/* Radio and checkboxes */
.stRadio label,
.stCheckbox label,
.stToggle label {{
    color: {text_main} !important;
}}

/* File uploader */
[data-testid="stFileUploader"] section {{
    background-color: {panel_bg_2} !important;
    border: 1px dashed {accent} !important;
    border-radius: 14px !important;
}}

[data-testid="stFileUploader"] button {{
    border-color: {accent} !important;
    color: {accent} !important;
}}

/* Dataframes and tables */
[data-testid="stDataFrame"] {{
    border: 1px solid {border_soft};
    border-radius: 12px;
    overflow: hidden;
}}

/* Alerts */
[data-testid="stAlert"] {{
    border-radius: 12px;
    border: 1px solid {border_soft};
}}

/* Expander */
.streamlit-expanderHeader {{
    color: {text_main} !important;
    background-color: {panel_bg_2} !important;
    border-radius: 10px !important;
}}

/* Links and separators */
a {{
    color: {accent_2};
}}

hr {{
    border-color: {border_soft};
}}

/* Plotly toolbar blends better */
.modebar {{
    background: transparent !important;
}}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# Plotly Theme Helper
# ============================================================

def apply_plot_theme(fig):
    """Apply the active dashboard theme to Plotly charts created in app.py or modules."""
    fig.update_layout(
        template=chart_template,
        paper_bgcolor=chart_bg,
        plot_bgcolor=plot_bg,
        font=dict(color=chart_font),
        title_font=dict(color=chart_font),
        legend=dict(font=dict(color=chart_font)),
        margin=dict(l=20, r=20, t=55, b=20),
        colorway=[accent, "#FB923C", "#F59E0B", "#22C55E", "#3B82F6", "#A855F7"],
    )
    fig.update_xaxes(gridcolor=grid_color, zerolinecolor=grid_color, linecolor=grid_color)
    fig.update_yaxes(gridcolor=grid_color, zerolinecolor=grid_color, linecolor=grid_color)
    return fig


def themed_plotly_chart(fig, **kwargs):
    """Render Plotly chart with the selected UI theme."""
    st.plotly_chart(apply_plot_theme(fig), **kwargs)

# ============================================================
# Session State
# ============================================================

DEFAULT_SESSION_KEYS = {
    "df": None,
    "preprocessing_report": None,
    "customer_segments": None,
    "used_cluster_features": None,
    "clustering_scores": None,
    "best_clustering_method": None,
    "rules": None,
    "anomalies": None,
    "similarity_df": None,
    "user_item_matrix": None,
    "product_stats": None,
    "loaded_file": None,
}

for key, value in DEFAULT_SESSION_KEYS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# Helper UI Functions
# ============================================================

def metric_card(column, label: str, value: str):
    """Render a custom metric card."""
    column.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-value'>{value}</div>
            <div class='metric-label'>{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_info(text: str):
    """Render an info box."""
    st.markdown(f"<div class='info-box'>{text}</div>", unsafe_allow_html=True)


def show_success(text: str):
    """Render a success box."""
    st.markdown(f"<div class='success-box'>{text}</div>", unsafe_allow_html=True)


def add_recommendation_strength_columns(recs: pd.DataFrame) -> pd.DataFrame:
    """Add business-friendly strength labels if recommender scores are available."""
    if recs is None or recs.empty:
        return recs

    display_df = recs.copy()
    score_candidates = [
        "RecommendationScore", "Score", "Similarity", "similarity",
        "AvgSimilarity", "MatchScore", "match_score",
    ]
    score_col = next((col for col in score_candidates if col in display_df.columns), None)

    if score_col is not None:
        numeric_scores = pd.to_numeric(display_df[score_col], errors="coerce")
        if numeric_scores.notna().any():
            high_cut = numeric_scores.quantile(0.67)
            low_cut = numeric_scores.quantile(0.33)

            def strength_label(value):
                if pd.isna(value):
                    return "Recommended"
                if value >= high_cut:
                    return "High Match"
                if value >= low_cut:
                    return "Medium Match"
                return "Discovery Option"

            display_df["Match Strength"] = numeric_scores.apply(strength_label)
        else:
            display_df["Match Strength"] = "Recommended"
    else:
        display_df["Match Strength"] = "Recommended"

    display_df["Recommended Business Action"] = display_df["Match Strength"].map(
        {
            "High Match": "Promote first in email campaigns or checkout suggestions",
            "Medium Match": "Use as a secondary recommendation or bundle idea",
            "Discovery Option": "Test in campaigns and monitor response",
            "Recommended": "Use as a product recommendation opportunity",
        }
    ).fillna("Use as a product recommendation opportunity")

    return display_df


def business_recommendation_table(recs: pd.DataFrame) -> pd.DataFrame:
    """Keep recommendation results easy for a business user to read."""
    display_df = add_recommendation_strength_columns(recs)
    if display_df is None or display_df.empty:
        return display_df

    preferred_columns = [
        "Rank", "Description", "Product", "StockCode", "Match Strength",
        "Recommended Business Action", "Revenue", "TotalSold", "UniqueCustomers",
        "Orders", "Score", "Similarity", "RecommendationScore",
    ]
    existing = [col for col in preferred_columns if col in display_df.columns]
    remaining = [col for col in display_df.columns if col not in existing]
    return display_df[existing + remaining]


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown("## 🛒 RetailMind")
    st.caption("Data Mining Dashboard for Business Insights")

    uploaded_file = st.file_uploader(
        "Upload E-Commerce / Retail Dataset",
        type=["csv", "xlsx", "xls", "zip"],
    )

    st.markdown("---")

    st.markdown("### Flexible Dataset Support")
    st.markdown(
        """
        The preprocessing pipeline maps different retail datasets into a standard schema.

        **Core required meaning:**  
        Order ID, Product, Quantity, Date, Price
        """
    )

   
    st.markdown("---")

    st.caption("Dataset source note: Originally from UCI, accessed via Kaggle.")


# ============================================================
# Load + Preprocess Data
# ============================================================

if uploaded_file:
    if st.session_state.df is None or st.session_state.loaded_file != uploaded_file.name:
        with st.spinner("Loading and preprocessing dataset..."):
            raw_df = load_data(uploaded_file)
            cleaned_df, preprocessing_report = preprocess_data(raw_df)

            st.session_state.df = cleaned_df
            st.session_state.preprocessing_report = preprocessing_report
            st.session_state.loaded_file = uploaded_file.name

            # Reset dependent computations when a new file is uploaded.
            st.session_state.customer_segments = None
            st.session_state.used_cluster_features = None
            st.session_state.clustering_scores = None
            st.session_state.best_clustering_method = None
            st.session_state.rules = None
            st.session_state.anomalies = None
            st.session_state.similarity_df = None
            st.session_state.user_item_matrix = None
            st.session_state.product_stats = None


# ============================================================
# Header
# ============================================================

st.markdown(
    """
<div class="main-header">
    <h1>🛒 RetailMind Analytics</h1>
    <p>Interactive Data Mining Dashboard for Business Insights + Smart Product Recommendations</p>
</div>
""",
    unsafe_allow_html=True,
)

show_info(
    """
    <b>Project idea:</b> This system transforms e-commerce transaction data into useful
    business insights using integrated data mining techniques. It supports customer
    segmentation, product pattern discovery, sales trend analysis, anomaly detection,
    and smart recommendations.
    """
)

if st.session_state.df is None:
    show_info(
        """
        👈 Upload the <b>E-Commerce Data / Online Retail dataset</b> or another retail dataset
        to start. The system will standardize columns, clean the data, and generate insights.
        """
    )
    st.stop()


df = st.session_state.df
preprocessing_report = st.session_state.preprocessing_report
stats = get_summary_stats(df)


# ============================================================
# Tabs
# ============================================================

tabs = st.tabs(
    [
        "📊 Overview",
        "🧼 Data Quality",
        "👥 Customer Groups",
        "🛍️ What Sells Together?",
        "📈 Sales Trends",
        "🚨 Anomalies",
        "🎯 Recommendations",
        "👤 Customer Profile",
        "📄 Business Report",
    ]
)


# ============================================================
# Tab 1: Overview
# ============================================================

with tabs[0]:
    st.markdown(
        '<div class="section-header">Business Overview</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    metric_card(c1, "Records", f"{stats['total_records']:,}")
    metric_card(c2, "Customers", f"{stats['total_customers']:,}")
    metric_card(c3, "Products", f"{stats['total_products']:,}")
    metric_card(c4, "Orders", f"{stats['total_orders']:,}")
    metric_card(c5, "Revenue", f"£{stats['total_revenue']:,.0f}")

    st.markdown(
        '<div class="section-header">Business Insights</div>',
        unsafe_allow_html=True,
    )

    top_country = (
        df.groupby("Country")["TotalPrice"]
        .sum()
        .sort_values(ascending=False)
        .index[0]
    )

    top_product_name = (
        df.groupby("Description")["TotalPrice"]
        .sum()
        .sort_values(ascending=False)
        .index[0]
    )

    avg_order_value = stats["total_revenue"] / max(stats["total_orders"], 1)

    peak_hour = (
        df.groupby("Hour")["InvoiceNo"]
        .nunique()
        .sort_values(ascending=False)
        .index[0]
        if "Hour" in df.columns
        else "N/A"
    )

    show_success(
        f"""
        <b>Key business interpretation:</b><br>
        • Highest revenue country: <b>{top_country}</b>.<br>
        • Highest revenue product: <b>{top_product_name}</b>.<br>
        • Average order value: <b>£{avg_order_value:,.2f}</b>.<br>
        • Peak purchasing hour: <b>{peak_hour}:00</b>.<br>
        These insights help management prioritize countries, products, and campaign timing.
        """
    )

    st.markdown(
        '<div class="section-header">Global Revenue Map</div>',
        unsafe_allow_html=True,
    )

    country_rev = (
        df.groupby("Country")["TotalPrice"]
        .sum()
        .reset_index()
        .rename(columns={"TotalPrice": "Revenue"})
    )

    fig_map = px.choropleth(
        country_rev,
        locations="Country",
        locationmode="country names",
        color="Revenue",
        hover_name="Country",
        color_continuous_scale="Oranges",
        template=chart_template,
        title="Revenue Distribution by Country",
    )

    themed_plotly_chart(fig_map, width="stretch")

    left, right = st.columns(2)

    with left:
        top_products = (
            df.groupby("Description")
            .agg(
                Quantity=("Quantity", "sum"),
                Revenue=("TotalPrice", "sum"),
            )
            .sort_values("Revenue", ascending=False)
            .head(15)
            .reset_index()
        )

        fig_top_products = px.bar(
            top_products.sort_values("Revenue"),
            x="Revenue",
            y="Description",
            orientation="h",
            template=chart_template,
            title="Top Products by Revenue",
        )

        themed_plotly_chart(fig_top_products, width="stretch")

    with right:
        monthly = monthly_revenue(df)

        fig_monthly = px.line(
            monthly,
            x="Month",
            y="Revenue",
            markers=True,
            template=chart_template,
            title="Monthly Revenue Overview",
        )

        themed_plotly_chart(fig_monthly, width="stretch")

    st.markdown(
        '<div class="section-header">Cleaned Data Preview</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(df.head(100), width="stretch", height=300)


# ============================================================
# Tab 2: Data Quality / Preprocessing Report
# ============================================================

with tabs[1]:
    st.markdown(
        '<div class="section-header">Preprocessing Quality Report</div>',
        unsafe_allow_html=True,
    )

    show_info(
        """
        This section shows how the system cleaned and standardized the uploaded dataset.
        It is important because the dashboard is designed to work with multiple retail datasets,
        not only one fixed file.
        """
    )

    if preprocessing_report:
        c1, c2, c3, c4 = st.columns(4)

        metric_card(c1, "Original Rows", f"{preprocessing_report.get('original_rows', 0):,}")
        metric_card(c2, "Final Rows", f"{preprocessing_report.get('final_rows', 0):,}")
        metric_card(c3, "Dropped Rows", f"{preprocessing_report.get('total_dropped', 0):,}")
        metric_card(c4, "Retention Rate", f"{preprocessing_report.get('retention_rate', 0)}%")

        st.markdown(
            '<div class="section-header">Detected Column Mapping</div>',
            unsafe_allow_html=True,
        )

        mapping = preprocessing_report.get("detected_mapping", {})

        if mapping:
            mapping_df = pd.DataFrame(
                [{"Standard Column": k, "Detected Original Column": v} for k, v in mapping.items()]
            )
            st.dataframe(mapping_df, width="stretch")
        else:
            st.info("No column mapping was needed or detected.")

        st.markdown(
            '<div class="section-header">Cleaning Actions</div>',
            unsafe_allow_html=True,
        )

        cleaning_rows = [
            ("Dropped duplicates", preprocessing_report.get("dropped_duplicates", 0)),
            ("Dropped missing descriptions", preprocessing_report.get("dropped_missing_description", 0)),
            ("Dropped missing CustomerID", preprocessing_report.get("dropped_missing_customerid", 0)),
            ("Dropped cancelled / invalid invoices", preprocessing_report.get("dropped_cancelled_or_invalid_invoices", 0)),
            ("Dropped non-product stockcodes", preprocessing_report.get("dropped_non_product_stockcodes", 0)),
            ("Dropped invalid quantity / price", preprocessing_report.get("dropped_invalid_quantity_or_price", 0)),
            ("Dropped invalid dates", preprocessing_report.get("dropped_invalid_dates", 0)),
            ("Dropped rare products", preprocessing_report.get("dropped_rare_products", 0)),
        ]

        cleaning_df = pd.DataFrame(cleaning_rows, columns=["Cleaning Step", "Rows Affected"])
        st.dataframe(cleaning_df, width="stretch")

        st.markdown(
            '<div class="section-header">Missing Fields Handled Automatically</div>',
            unsafe_allow_html=True,
        )
        show_info(
            "If the uploaded dataset is missing optional fields, the system can generate safe replacements so the dashboard still works."
        )

        fallback_df = pd.DataFrame(
            [
                ("Customer ID", "Generated automatically" if preprocessing_report.get("generated_customerid", False) else "Already available"),
                ("Product Code", "Generated automatically" if preprocessing_report.get("generated_stockcode", False) else "Already available"),
                ("Country", "Generated automatically" if preprocessing_report.get("generated_country", False) else "Already available"),
            ],
            columns=["Field", "Status"],
        )

        st.dataframe(fallback_df, width="stretch")

        with st.expander("Full preprocessing report text"):
            st.text(build_preprocessing_report_text(preprocessing_report))

    else:
        st.warning("No preprocessing report is available.")


# ============================================================
# Tab 3: Segmentation
# ============================================================

with tabs[2]:
    st.markdown(
        '<div class="section-header">Customer Groups & Business Actions</div>',
        unsafe_allow_html=True,
    )

    show_info(
        """
        This section groups customers by buying behavior so the business can decide
        <b>who to reward, who to reactivate, and who needs attention</b>.
        The system can choose the best grouping strategy automatically.
        """
    )

    c1, c2, c3 = st.columns(3)
    c1.markdown(
        """
        <div class='metric-card'>
            <div class='metric-value'>🏆 VIP Customers</div>
            <div class='metric-label'>Identify loyal high-value customers for premium offers</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c2.markdown(
        """
        <div class='metric-card'>
            <div class='metric-value'>📣 Growth Customers</div>
            <div class='metric-label'>Find customers who can buy more with targeted campaigns</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c3.markdown(
        """
        <div class='metric-card'>
            <div class='metric-value'>⚠️ At-Risk Customers</div>
            <div class='metric-label'>Detect customers who may need reactivation offers</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-header">Simple Business Controls</div>',
        unsafe_allow_html=True,
    )

    show_info(
        """
        Recommended use: keep the strategy on <b>Automatic</b>. Advanced settings are available
        for the instructor or analyst, but the business user can simply click
        <b>Analyze Customer Groups</b>.
        """
    )

    customer_group_strategy = st.selectbox(
        "Customer Analysis Mode",
        [
            "Automatic (Recommended for Business Use)",
            "Balanced Customer Groups",
            "Special Behavior Detection",
            "Compare All Methods (Academic Analysis)",
        ],
        help=(
            "Choose how the system should analyze customers. Business users should keep Automatic. "
            "Advanced users can compare the technical methods from the details section."
        ),
    )

    show_customer_map = st.selectbox(
        "Customer Insights View",
        [
            "Show simple customer map",
            "Advanced analytical comparison (PCA & SVD)",
            "Hide technical maps",
        ],
        help="Choose how much visual analysis to show for customer groups.",
    )

    # Defaults keep the dashboard easy for business users.
    n_clusters = 4
    dbscan_eps_mode = "Auto"
    dbscan_eps = 0.8
    dbscan_min_samples = 8

    with st.expander("Advanced Analytics Details"):
        st.caption(
            "These settings are useful for the instructor or analyst. "
            "A normal business user can leave them unchanged."
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            n_clusters = st.slider(
                "Number of customer groups",
                2,
                8,
                4,
                help="How many customer groups should be created when using fixed grouping methods.",
            )

        with c2:
            dbscan_eps_mode = st.selectbox(
                "Unusual Customer Detection Mode",
                ["Auto", "Manual"],
                help="Auto lets the system estimate sensitivity for unusual customer detection. Manual lets analysts control it.",
            )

        with c3:
            dbscan_eps = st.slider(
                "Sensitivity to unusual customers",
                0.2,
                2.5,
                0.8,
                0.1,
                help="Lower values detect stricter unusual patterns; higher values create broader groups.",
            )

        with c4:
            dbscan_min_samples = st.slider(
                "Minimum group size",
                3,
                25,
                8,
                help="Minimum nearby customers needed to treat a pattern as a group instead of noise.",
            )

        st.markdown(
            """
            **Technical methods used behind the scenes:**  
            - KMeans and Hierarchical clustering create fixed customer groups.  
            - DBSCAN can detect unusual or noisy customer patterns.  
            - PCA and SVD help visualize customer groups in a 2D map.  
            - Silhouette, Davies-Bouldin, and Calinski-Harabasz scores help compare grouping quality.
            """
        )

    if customer_group_strategy == "Automatic (Recommended for Business Use)":
        selected_method_option = "Auto-select best by Silhouette"
    elif customer_group_strategy == "Balanced Customer Groups":
        selected_method_option = "KMeans"
    elif customer_group_strategy == "Special Behavior Detection":
        selected_method_option = "DBSCAN"
    else:
        selected_method_option = "Auto-select best by Silhouette"

    if show_customer_map == "Show simple customer map":
        dim_view = "PCA Only"
    elif show_customer_map == "Advanced analytical comparison (PCA & SVD)":
        dim_view = "PCA & SVD Comparison"
    else:
        dim_view = "Hide"

    run_seg = st.button("Analyze Customer Groups")

    if run_seg or st.session_state.customer_segments is not None:
        with st.spinner("Analyzing customer behavior and building customer groups..."):
            customer_features = compute_customer_features(df)
            customer_scaled, scaler, used_features = scale_customer_features(customer_features)

            k_range, inertias, silhouettes = find_optimal_k(customer_scaled)

            eps_value = None if dbscan_eps_mode == "Auto" else dbscan_eps

            clustering_scores, method_labels, best_method = compare_clustering_methods(
                customer_scaled,
                n_clusters=n_clusters,
                dbscan_eps=eps_value,
                dbscan_min_samples=dbscan_min_samples,
            )

            chosen_method = (
                best_method
                if selected_method_option == "Auto-select best by Silhouette"
                else selected_method_option
            )

            customer_segments = apply_clustering_method(
                customer_features,
                method_labels[chosen_method],
                chosen_method,
            )

            pca_df, explained_var = apply_pca(customer_scaled, customer_segments)
            svd_df, svd_explained = apply_svd(customer_scaled, customer_segments)
            comparison_df = dimensionality_comparison_table(explained_var, svd_explained)

            st.session_state.customer_segments = customer_segments
            st.session_state.used_cluster_features = used_features
            st.session_state.clustering_scores = clustering_scores
            st.session_state.best_clustering_method = best_method

        show_success(
            """
            <b>Customer groups are ready.</b><br>
            The system selected the most meaningful customer grouping approach for this dataset.<br>
            Use the segment names and suggested actions below to plan marketing campaigns,
            loyalty rewards, and reactivation offers.
            """
        )

        st.markdown(
            '<div class="section-header">Recommended Business Actions</div>',
            unsafe_allow_html=True,
        )

        a1, a2, a3 = st.columns(3)
        a1.markdown(
            """
            <div class='metric-card'>
                <div class='metric-value'>🏆 Reward</div>
                <div class='metric-label'>Give VIP customers loyalty rewards, premium bundles, and exclusive offers</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        a2.markdown(
            """
            <div class='metric-card'>
                <div class='metric-value'>📣 Grow</div>
                <div class='metric-label'>Target growth customers with promotions and product recommendations</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        a3.markdown(
            """
            <div class='metric-card'>
                <div class='metric-value'>🔁 Reactivate</div>
                <div class='metric-label'>Send win-back discounts to at-risk or inactive customers</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-header">Customer Segment Distribution</div>',
            unsafe_allow_html=True,
        )

        themed_plotly_chart(plot_segment_distribution(customer_segments), width="stretch")

        st.markdown(
            '<div class="section-header">What should the business do?</div>',
            unsafe_allow_html=True,
        )

        st.text(build_segmentation_insights(customer_segments))

        st.markdown(
            '<div class="section-header">Why each customer group matters</div>',
            unsafe_allow_html=True,
        )
        show_info(
            """
            This table converts the customer groups into clear marketing decisions.
            It helps a business user understand <b>why the group matters</b> and <b>what action to take</b>.
            """
        )
        segment_action_plan = build_segment_action_plan(customer_segments)
        st.dataframe(segment_action_plan, width="stretch", height=330)

        st.markdown(
            '<div class="section-header">Average Customer Behavior by Segment</div>',
            unsafe_allow_html=True,
        )

        show_info(
            """
            This chart explains how each customer group behaves on average.
            Use it to understand which groups spend more, buy more often, or have been inactive longer.
            """
        )
        themed_plotly_chart(plot_cluster_profiles(customer_segments), width="stretch")

        if dim_view in ["PCA & SVD Comparison", "PCA Only"]:
            st.markdown(
                '<div class="section-header">Customer Map</div>',
                unsafe_allow_html=True,
            )
            show_info(
                """
                This map gives a visual overview of customer groups.
                Customers close together have similar buying behavior.
                """
            )
            themed_plotly_chart(plot_pca_clusters(pca_df, explained_var), width="stretch")

        if dim_view == "PCA & SVD Comparison":
            themed_plotly_chart(plot_svd_clusters(svd_df, svd_explained), width="stretch")

        st.markdown(
            '<div class="section-header">Customer List for Campaign Planning</div>',
            unsafe_allow_html=True,
        )

        campaign_df = customer_segments.sort_values("TotalSpend", ascending=False).copy()
        campaign_df = campaign_df.rename(
            columns={
                "TotalSpend": "Total Spend",
                "AvgOrderValue": "Average Order Value",
                "CustomerActivityDays": "Activity Days",
                "ProductDiversity": "Product Variety",
            }
        )
        business_columns = [
            col for col in [
                "CustomerID",
                "Segment",
                "Total Spend",
                "Average Order Value",
                "OrderCount",
                "DaysSinceLastPurchase",
                "Product Variety",
                "Activity Days",
            ]
            if col in campaign_df.columns
        ]
        st.dataframe(
            campaign_df[business_columns],
            width="stretch",
            height=330,
        )

        with st.expander("Advanced Analytics Details"):
            st.markdown("### Technical method selected")
            st.write(f"Best method by evaluation metrics: {best_method}")
            st.write(f"Applied technical method: {chosen_method}")

            st.markdown("### Features used by the customer grouping engine")
            st.write(st.session_state.used_cluster_features)

            st.markdown("### Technical quality report")
            st.markdown(
                f"<div class='info-box'>{explain_clustering_metrics()}</div>",
                unsafe_allow_html=True,
            )

            themed_plotly_chart(plot_clustering_comparison(clustering_scores), width="stretch")
            st.dataframe(clustering_scores, width="stretch")

            themed_plotly_chart(plot_elbow(k_range, inertias, silhouettes), width="stretch")

            if dim_view == "PCA & SVD Comparison":
                st.markdown("### PCA vs SVD explained variance comparison")
                st.dataframe(comparison_df, width="stretch")

    else:
        st.info("Click Analyze Customer Groups to generate business-friendly customer segments.")


# ============================================================
# Tab 4: Basket Rules (MARKETING-FRIENDLY VERSION)
# ============================================================

with tabs[3]:
    st.markdown(
        '<div class="section-header">🛍️ Product Relationships — Smart Bundling for Sales</div>',
        unsafe_allow_html=True,
    )

    show_info(
        "<b>💡 What is this for?</b><br>"
        "This tool finds which products customers like to buy together. "
        "Use these insights to:<br>"
        "✅ <b>Bundle products</b> for bigger purchases<br>"
        "✅ <b>Cross-sell</b> on product pages<br>"
        "✅ <b>Plan email campaigns</b> with personalized recommendations<br>"
        "✅ <b>Improve store layout</b> by placing items near each other"
    )

    # ── Parameter Cards (Marketing Focused) ──────────────────
    card1, card2, card3 = st.columns(3)

    with card1:
        st.markdown(
            f"""<div style="background:{bc_1_bg};border-left:4px solid {bc_1_border};
            border-radius:10px;padding:1rem;margin-bottom:1rem;">
            <b style="color:{bc_1_title};">👥 Market Reach</b><br>
            <span style="font-size:0.9rem;color:{text_muted};">
            How many customers buy BOTH items?<br>
            <br>
            <b>3%</b> = 3 in 100 purchases<br>
            <b>Raise it</b> → Find common patterns<br>
            <b>Lower it</b> → Find niche opportunities
            </span></div>""",
            unsafe_allow_html=True,
        )

    with card2:
        st.markdown(
            f"""<div style="background:{bc_2_bg};border-left:4px solid {bc_2_border};
            border-radius:10px;padding:1rem;margin-bottom:1rem;">
            <b style="color:{bc_2_title};">🎯 Predictability</b><br>
            <span style="font-size:0.9rem;color:{text_muted};">
            If customer buys A, how often do they buy B?<br>
            <br>
            <b>30%</b> = 30% chance<br>
            <b>Raise it</b> → More reliable patterns<br>
            <b>Lower it</b> → Find hidden opportunities
            </span></div>""",
            unsafe_allow_html=True,
        )

    with card3:
        st.markdown(
            f"""<div style="background:{bc_3_bg};border-left:4px solid {bc_3_border};
            border-radius:10px;padding:1rem;margin-bottom:1rem;">
            <b style="color:{bc_3_title};">⚡ Auto-Calculated Metrics</b><br>
            <span style="font-size:0.9rem;color:{text_muted};">
            <b>Strength (Lift):</b> How much more likely?<br>
            <b>Market Reach:</b> How common is the pattern?<br>
            <b>Predictability:</b> How reliable is it?
            </span></div>""",
            unsafe_allow_html=True,
        )

    # ── Controls ──────────────────────────────────────────────
    countries = ["All"]
    if "United Kingdom" in df["Country"].dropna().unique():
        countries.append("United Kingdom")
    countries += sorted([c for c in df["Country"].dropna().unique().tolist() if c not in countries])

    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns(4)

    with ctrl1:
        country = st.selectbox(
            "🌍 Which country?", countries,
            index=1 if "United Kingdom" in countries else 0,
            help="Analyze products bought together in this country.",
        )
    with ctrl2:
        min_support = st.slider(
            "👥 Market Reach %", min_value=1, max_value=20, value=5, step=1,
            help="How many customers (%) buy BOTH items? Higher = more common patterns.",
        )
        min_support = min_support / 100
    with ctrl3:
        min_confidence = st.slider(
            "🎯 Predictability %", min_value=10, max_value=90, value=30, step=5,
            help="When A is bought, how often (%) is B also bought? Higher = more reliable.",
        )
        min_confidence = min_confidence / 100
    with ctrl4:
        st.write("")
        st.write("")
        run_analysis = st.button("🔍 Find Patterns", use_container_width=True, key="basket_analysis_button")

    # ── Reset rules when country changes ──────────────────────
    if "last_basket_country" not in st.session_state:
        st.session_state.last_basket_country = country

    if st.session_state.last_basket_country != country:
        st.session_state.rules = None
        st.session_state.last_basket_country = country

    # ── Run Analysis ──────────────────────────────────────────
    if run_analysis:
        with st.spinner("🔍 Analyzing which products are bought together..."):
            basket = build_basket_matrix(df, country)
            if basket.empty:
                st.session_state.rules = pd.DataFrame()
            else:
                rules = generate_association_rules(basket, min_support, min_confidence)
                st.session_state.rules = rules

    rules = st.session_state.rules

    if rules is not None and not rules.empty:
        # ── Success Message ────────────────────────────────────
        summary = get_rules_summary_marketing(rules)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "🔥 Hot Deals Found",
                summary['hot_deals'],
                "Ready for bundling"
            )
        with col2:
            st.metric(
                "💎 Strong Opportunities",
                summary['premium_opportunities'],
                "Cross-sell potential"
            )
        with col3:
            st.metric(
                "✅ Good Opportunities",
                summary['good_opportunities'],
                "Worth exploring"
            )
        with col4:
            st.metric(
                "📊 Total Patterns",
                summary['total_rules'],
                "Discovered"
            )

        # ── Top Rule Highlight ──────────────────────────────────
        show_success(
            f"""
            <b>🏆 Your Best Opportunity:</b><br>
            <br>
            <b>When customers buy:</b> {summary['strongest_rule_antecedent']}<br>
            <b>They also buy:</b> {summary['strongest_rule_consequent']}<br>
            <br>
            <b>{summary['strongest_rule_strength']}</b><br>
            {summary['strongest_rule_action']}<br>
            <br>
            <b>The Numbers:</b> Customers are <b>{summary['max_lift']:.1f}× more likely</b>
            to buy the second item when they buy the first.
            """
        )

        # ── Marketing-Focused Visualizations ────────────────────
        st.markdown('<div class="section-header">📊 Visual Analysis for Decisions</div>', unsafe_allow_html=True)

        fig1 = plot_rules_marketing_strength(rules)
        fig2 = plot_top_rules_marketing(rules)

        if fig1: themed_plotly_chart(fig1, use_container_width=True)
        if fig2: themed_plotly_chart(fig2, use_container_width=True)

        # ── Marketing Insights ──────────────────────────────────
        st.markdown('<div class="section-header">💡 What Should You Do?</div>', unsafe_allow_html=True)

        marketing_rules = interpret_rules_marketing(rules, top_n=8)

        for idx, rule in enumerate(marketing_rules, 1):
            col1, col2, col3 = st.columns([1.5, 2, 2])

            with col1:
                st.markdown(
                    f"""<div style="background:rgba(249,115,22,.10);border-left:4px solid {accent};
                    border-radius:8px;padding:0.8rem;height:100%;display:flex;align-items:center;justify-content:center;">
                    <b style="font-size:1.1rem;text-align:center;">{rule['strength']}</b>
                    </div>""",
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(f"""
                    **IF** customer buys **{rule['rule'].split('→')[0].strip()}**

                    **THEN** they also buy **{rule['rule'].split('→')[1].strip()}**
                """)

            with col3:
                st.markdown(f"""
                    **Relationship Strength:** {rule['lift']}

                    **Market Reach:** {rule['support']}

                    **Predictability:** {rule['confidence']}
                """)

            st.markdown(f"➜ {rule['action']}")
            st.divider()

        # ── Detailed Rules Table ────────────────────────────────
        st.markdown('<div class="section-header">📋 Detailed Rules for Your Team</div>', unsafe_allow_html=True)

        display_rules = rules[["antecedents", "consequents", "lift", "support", "confidence"]].copy()
        display_rules.columns = [
            "If Customer Buys",
            "They Also Buy",
            "Strength (Lift)",
            "Market Reach (%)",
            "Predictability (%)"
        ]

        display_rules["Market Reach (%)"] = (display_rules["Market Reach (%)"] * 100).round(1).astype(str) + "%"
        display_rules["Predictability (%)"] = (display_rules["Predictability (%)"] * 100).round(1).astype(str) + "%"

        st.dataframe(display_rules.head(50), use_container_width=True, height=400)

        # ── Download Options ────────────────────────────────────
        st.markdown('<div class="section-header">⬇️ Download for Your Team</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            csv_data = display_rules.to_csv(index=False)
            st.download_button(
                "📥 Download Rules as Excel",
                data=csv_data,
                file_name=f"product_bundles_{country}.csv",
                mime="text/csv",
            )

        with col2:
            report_text = create_marketing_report(rules)
            st.download_button(
                "📄 Download Marketing Report",
                data=report_text.encode('utf-8'),
                file_name=f"marketing_report_{country}.txt",
                mime="text/plain",
            )

        with col3:
            st.info("📧 Share the Excel file with your marketing team and use in email campaigns!")

        # ── Technical Details for Doctor ────────────────────────
        with st.expander("📊 Technical Details (For Dr. & Advanced Users)"):
            st.markdown(explain_metrics_simple())

            st.markdown("### What Each Metric Means:")
            st.markdown(f"""
            - **Support (Market Reach):** {rules['support'].mean()*100:.1f}% average - how common are these combinations
            - **Confidence (Predictability):** {rules['confidence'].mean()*100:.1f}% average - reliability of the pattern
            - **Lift (Strength):** {rules['lift'].mean():.2f}x average - how much stronger than random chance
            """)

            st.markdown("### Statistical Distribution:")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Highest Lift", f"{rules['lift'].max():.2f}x")
                st.metric("Lowest Lift", f"{rules['lift'].min():.2f}x")

            with col2:
                st.metric("Highest Confidence", f"{rules['confidence'].max()*100:.1f}%")
                st.metric("Lowest Confidence", f"{rules['confidence'].min()*100:.1f}%")

            st.markdown("### All Rules (Full Technical Data):")
            st.dataframe(rules, use_container_width=True, height=500)

    elif rules is not None and rules.empty:
        show_info(
            "❌ <b>No patterns found yet</b><br>"
            "Try <b>raising the Market Reach %</b> (find more common items) "
            "or <b>lowering the Predictability %</b> (find hidden opportunities)."
        )
    else:
        show_info(
            "👈 Set your preferences above and click <b>'Find Patterns'</b> to discover which products are bought together."
        )
# ============================================================
# Tab 5: Time Series
# ============================================================

with tabs[4]:
    st.markdown(
        '<div class="section-header">Sales Trends / Time Series Analysis</div>',
        unsafe_allow_html=True,
    )

    show_info(
        """
        Time series analysis helps the business understand revenue trends,
        seasonal patterns, peak buying times, and expected future sales.
        """
    )

    daily = daily_revenue(df)

    themed_plotly_chart(plot_daily_trend(daily), width="stretch")

    a, b = st.columns(2)

    with a:
        heat = sales_heatmap_data(df)
        themed_plotly_chart(plot_heatmap(heat), width="stretch")

    with b:
        periods = st.slider("Forecast horizon (days)", 7, 90, 30)
        fc = forecast_sales(daily, periods)
        themed_plotly_chart(plot_forecast(daily.tail(120), fc), width="stretch")

    show_info(
        """
        The forecast uses a lightweight seasonal-naive trend model to keep the application
        fast and usable on student laptops while still providing a practical business forecast.
        """
    )


# ============================================================
# Tab 6: Anomaly Detection
# ============================================================

with tabs[5]:
    st.markdown(
        '<div class="section-header">Transaction Anomaly Detection</div>',
        unsafe_allow_html=True,
    )

    show_info(
        """
        Anomaly detection identifies unusual invoices, such as very high-value orders,
        unusual quantities, or abnormal transaction patterns.
        """
    )

    contamination = st.slider("Expected anomaly rate", 0.01, 0.15, 0.05, 0.01)

    if st.button("Detect Unusual Invoices"):
        with st.spinner("Running Isolation Forest..."):
            features = build_transaction_features(df)
            st.session_state.anomalies = detect_anomalies(features, contamination)

    features_with_anomalies = st.session_state.anomalies

    if features_with_anomalies is not None:
        n = int(features_with_anomalies["IsAnomaly"].sum())

        st.success(
            f"Detected {n:,} unusual invoices from {len(features_with_anomalies):,} invoices."
        )

        a, b = st.columns(2)

        with a:
            themed_plotly_chart(plot_anomaly_scatter(features_with_anomalies), width="stretch")

        with b:
            themed_plotly_chart(plot_anomaly_score_distribution(features_with_anomalies), width="stretch")

        st.dataframe(
            get_top_anomalies(features_with_anomalies),
            width="stretch",
            height=360,
        )

    else:
        st.info("Run anomaly detection to flag unusual high-value or unusual-quantity invoices.")


# ============================================================
# Tab 7: Recommendations
# ============================================================

with tabs[6]:
    st.markdown(
        '<div class="section-header">Smart Product Recommendations</div>',
        unsafe_allow_html=True,
    )

    show_info(
        """
        This section helps the business decide <b>what products to recommend, what products to bundle,</b>
        and how to increase sales using previous purchase behavior. The customer does not need to understand
        recommendation algorithms; the dashboard turns purchase history into practical sales actions.
        """
    )

    r1, r2, r3 = st.columns(3)
    r1.markdown(
        """
        <div class='metric-card'>
            <div class='metric-value'>🛒 What to recommend?</div>
            <div class='metric-label'>Suggest products the customer is likely to buy next</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    r2.markdown(
        """
        <div class='metric-card'>
            <div class='metric-value'>📦 What to bundle?</div>
            <div class='metric-label'>Find products that fit well with the current basket</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    r3.markdown(
        """
        <div class='metric-card'>
            <div class='metric-value'>📈 How to grow sales?</div>
            <div class='metric-label'>Support cross-selling and increase average order value</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.similarity_df is None:
        with st.spinner("Preparing product recommendations from purchase history..."):
            matrix = build_user_item_matrix(df)
            product_stats = get_product_stats(df)
            similarity_df = compute_item_similarity(matrix)

            st.session_state.user_item_matrix = matrix
            st.session_state.product_stats = product_stats
            st.session_state.similarity_df = similarity_df

    matrix = st.session_state.user_item_matrix
    similarity_df = st.session_state.similarity_df
    product_stats = st.session_state.product_stats

    rec_mode = st.radio(
        "Choose recommendation service",
        [
            "Customer-Based Recommendations",
            "Basket Add-on Recommendations",
        ],
        horizontal=True,
        help="Choose whether to recommend products for one customer or suggest add-ons for a current basket.",
    )

    if rec_mode == "Customer-Based Recommendations":
        show_info(
            """
            Use this when you know the customer ID. The system reviews what this customer bought before
            and suggests products they are likely to buy next.
            """
        )

        st.markdown(
            '<div class="section-header">How the business can use this</div>',
            unsafe_allow_html=True,
        )
        b1, b2, b3 = st.columns(3)
        b1.markdown(
            """
            <div class='metric-card'>
                <div class='metric-value'>📧 Email Campaign</div>
                <div class='metric-label'>Send personalized product suggestions to the customer</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        b2.markdown(
            """
            <div class='metric-card'>
                <div class='metric-value'>🛒 Checkout Offer</div>
                <div class='metric-label'>Show relevant products before checkout</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        b3.markdown(
            """
            <div class='metric-card'>
                <div class='metric-value'>🎯 Targeted Sales</div>
                <div class='metric-label'>Focus sales effort on products with stronger match potential</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns([2, 1])

        with c1:
            customer_id = st.text_input("Customer ID", placeholder="Example: 17850")

        with c2:
            top_n = st.slider("Number of recommendations", 5, 20, 10, key="customer_top_n")

        if st.button("Get Product Recommendations") and customer_id:
            recs = recommend_for_customer(
                customer_id.strip(),
                matrix,
                similarity_df,
                product_stats,
                top_n,
            )

            if "Message" in recs.columns:
                st.warning(recs["Message"].iloc[0])
                show_info(
                    """
                    This customer does not have enough purchase history in the dataset.
                    The system will show popular products instead as a safe fallback.
                    """
                )

                popular = get_popular_products(df, top_n)
                st.dataframe(business_recommendation_table(popular), width="stretch")
                themed_plotly_chart(plot_popular_products(popular), width="stretch")

            else:
                recs_display = business_recommendation_table(recs)
                show_success(build_recommendation_business_insight(recs))
                show_info(build_recommendation_action_summary(recs))

                st.markdown(
                    '<div class="section-header">Recommended Business Actions</div>',
                    unsafe_allow_html=True,
                )
                show_info(
                    """
                    Start with products marked as <b>High Match</b>. Use them in email campaigns,
                    product pages, and checkout suggestions. Medium matches can be tested as secondary offers.
                    """
                )

                themed_plotly_chart(plot_recommendations(recs), width="stretch")

                st.markdown(
                    '<div class="section-header">Recommendation List for Sales Team</div>',
                    unsafe_allow_html=True,
                )
                st.dataframe(recs_display, width="stretch")

    else:
        show_info(
            """
            Use this when a customer already has products in the basket.
            The system suggests add-on products that can be promoted before checkout to increase basket value.
            """
        )

        st.markdown(
            '<div class="section-header">Best use cases</div>',
            unsafe_allow_html=True,
        )
        u1, u2, u3 = st.columns(3)
        u1.markdown(
            """
            <div class='metric-card'>
                <div class='metric-value'>📦 Bundle Offer</div>
                <div class='metric-label'>Suggest products that fit naturally with the current basket</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        u2.markdown(
            """
            <div class='metric-card'>
                <div class='metric-value'>⬆️ Upsell Moment</div>
                <div class='metric-label'>Recommend add-ons while the customer is ready to buy</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        u3.markdown(
            """
            <div class='metric-card'>
                <div class='metric-value'>🛍️ Store Layout</div>
                <div class='metric-label'>Place related products closer in the online or physical store</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        product_options = sorted(similarity_df.columns.tolist())

        selected_products = st.multiselect(
            "Products currently in the basket",
            product_options,
        )

        top_n = st.slider("Number of add-on recommendations", 5, 20, 10, key="basket_top_n")

        if st.button("Suggest Add-on Products"):
            basket_recs = recommend_for_basket(
                selected_products,
                similarity_df,
                product_stats,
                top_n,
            )

            if "Message" in basket_recs.columns:
                st.warning(basket_recs["Message"].iloc[0])
            else:
                basket_recs_display = business_recommendation_table(basket_recs)
                show_success(build_recommendation_business_insight(basket_recs))
                show_info(build_recommendation_action_summary(basket_recs))

                st.markdown(
                    '<div class="section-header">Recommended Business Actions</div>',
                    unsafe_allow_html=True,
                )
                show_info(
                    """
                    Use the strongest add-on products as checkout suggestions, bundle offers,
                    or product-page recommendations. This helps increase average order value.
                    """
                )

                themed_plotly_chart(plot_recommendations(basket_recs), width="stretch")

                st.markdown(
                    '<div class="section-header">Suggested Add-on Products</div>',
                    unsafe_allow_html=True,
                )
                st.dataframe(basket_recs_display, width="stretch")

    st.markdown(
        '<div class="section-header">Popular Products Backup Plan</div>',
        unsafe_allow_html=True,
    )

    show_info(
        """
        If a customer is new or has no history, the system can still recommend popular products.
        This makes the recommendation service useful even in cold-start cases.
        """
    )

    popular = get_popular_products(df, 15)

    themed_plotly_chart(plot_popular_products(popular), width="stretch")
    st.dataframe(business_recommendation_table(popular), width="stretch")

    with st.expander("Advanced Analytics Details"):
        st.markdown("### Recommendation Engine Details")
        st.write(
            "The recommendation engine analyzes previous purchase behavior to suggest products "
            "customers are likely to buy together. Advanced similarity analysis is used internally "
            "to improve recommendation quality and avoid recommending products the customer already bought."
        )
        with st.expander("Show full technical method"):
            st.write(build_recommender_technical_explanation())

        st.markdown("### Recommendation Coverage Indicators")
        quality_report = get_recommender_quality_report(matrix, similarity_df)

        q1, q2, q3, q4, q5 = st.columns(5)

        metric_card(q1, "Customers", f"{quality_report['customers_in_matrix']:,}")
        metric_card(q2, "Products", f"{quality_report['products_in_matrix']:,}")
        metric_card(q3, "Recommendation Coverage", f"{quality_report['matrix_density']}")
        metric_card(q4, "Avg Products / Customer", f"{quality_report['avg_products_per_customer']}")
        metric_card(q5, "Match Quality", f"{quality_report['avg_similarity']}")

        themed_plotly_chart(plot_recommender_quality(quality_report), width="stretch")


# ============================================================
# Tab 8: Customer Profile
# ============================================================

with tabs[7]:
    st.markdown(
        '<div class="section-header">Customer Profile 360°</div>',
        unsafe_allow_html=True,
    )

    show_info(
        """
        This page connects multiple parts of the project around one customer:
        spending behavior, purchase timeline, segmentation result, and recommendation potential.
        """
    )

    customer_options = sorted(df["CustomerID"].dropna().astype(str).unique().tolist())

    selected_customer = st.selectbox(
        "Select CustomerID",
        customer_options[:5000],
    )

    cdf = df[df["CustomerID"].astype(str) == str(selected_customer)]

    total_spend = cdf["TotalPrice"].sum()
    orders = cdf["InvoiceNo"].nunique()
    products = cdf["StockCode"].nunique()

    a, b, c = st.columns(3)

    metric_card(a, "Customer Revenue", f"£{total_spend:,.2f}")
    metric_card(b, "Orders", f"{orders:,}")
    metric_card(c, "Unique Products", f"{products:,}")

    if st.session_state.customer_segments is not None:
        row = st.session_state.customer_segments[
            st.session_state.customer_segments["CustomerID"].astype(str) == str(selected_customer)
        ]

        if not row.empty:
            st.info(
                f"Segment: {row.iloc[0]['Segment']} | "
                f"Days Since Last Purchase: {row.iloc[0]['DaysSinceLastPurchase']} days | "
                f"Orders: {row.iloc[0]['OrderCount']} | "
                f"Total Spend: £{row.iloc[0]['TotalSpend']:,.2f}"
            )
        else:
            st.info("Run segmentation first to show this customer's segment.")
    else:
        st.info("Run segmentation first to show this customer's segment.")

    timeline = (
        cdf.groupby("InvoiceDate")["TotalPrice"]
        .sum()
        .reset_index()
    )

    themed_plotly_chart(
        px.line(
            timeline,
            x="InvoiceDate",
            y="TotalPrice",
            markers=True,
            template=chart_template,
            title="Customer Purchase Timeline",
        ),
        width="stretch",
    )

    st.dataframe(
        cdf.sort_values("InvoiceDate", ascending=False).head(50),
        width="stretch",
        height=280,
    )


# ============================================================
# Tab 9: Business Report
# ============================================================

with tabs[8]:
    st.markdown(
        '<div class="section-header">Automated Business Report</div>',
        unsafe_allow_html=True,
    )

    show_info(
        """
        The report summarizes key business insights generated by the system.
        It can be downloaded as PDF or TXT and included in the final project submission.
        """
    )

    # The reporting module was originally written for older segment column names.
    # Our current customer grouping module uses clearer business-friendly names,
    # so we create a safe compatibility copy only for report generation.
    report_segments = st.session_state.customer_segments

    if report_segments is not None:
        report_segments = report_segments.copy()

        compatibility_columns = {
            "TotalSpend": "Monetary",
            "OrderCount": "Frequency",
            "DaysSinceLastPurchase": "Recency",
        }

        for new_col, old_col in compatibility_columns.items():
            if new_col in report_segments.columns and old_col not in report_segments.columns:
                report_segments[old_col] = report_segments[new_col]

    report_text = build_insights_text(
        df,
        report_segments,
        st.session_state.rules,
    )

    if preprocessing_report:
        report_text += "\n\n"
        report_text += build_preprocessing_report_text(preprocessing_report)

    if st.session_state.customer_segments is not None:
        report_text += "\n\n"
        report_text += build_segmentation_insights(st.session_state.customer_segments)

    st.text_area("Generated Report", report_text, height=420)

    pdf_bytes = make_pdf_report(report_text)

    st.download_button(
        "⬇️ Export Report as PDF",
        data=pdf_bytes,
        file_name="RetailMind_Business_Report.pdf",
        mime="application/pdf",
    )

    st.download_button(
        "⬇️ Export Report as TXT",
        data=report_text.encode("utf-8"),
        file_name="RetailMind_Business_Report.txt",
        mime="text/plain",
    )