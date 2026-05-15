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
    compute_rfm,
    scale_rfm,
    find_optimal_k,
    compare_clustering_methods,
    apply_clustering_method,
    apply_pca,
    apply_svd,
    dimensionality_comparison_table,
    explain_best_clustering_method,
    explain_clustering_metrics,
    build_segmentation_insights,
    plot_elbow,
    plot_pca_clusters,
    plot_svd_clusters,
    plot_cluster_profiles,
    plot_segment_distribution,
    plot_clustering_comparison,
)

from modules.association_rules import (
    build_basket_matrix,
    run_association_rules,
    plot_rules_scatter,
    plot_top_rules_bar,
    interpret_rules,
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
    build_recommendation_business_insight,
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
    theme_mode = st.toggle("🌙 Dark Mode", value=True)

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
    "rfm": None,
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

    st.markdown("### Techniques used")
    st.markdown(
        """
        ✅ Flexible preprocessing  
        ✅ Clustering comparison  
        ✅ PCA & SVD  
        ✅ FP-Growth / Apriori  
        ✅ Time Series + Forecasting  
        ✅ Anomaly Detection  
        ✅ Recommendation System  
        ✅ Customer Profile  
        ✅ PDF / TXT Report  
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
            st.session_state.rfm = None
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
        "👥 Segmentation",
        "🛍️ Basket Rules",
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
            '<div class="section-header">Fallback Columns</div>',
            unsafe_allow_html=True,
        )

        fallback_df = pd.DataFrame(
            [
                ("Generated CustomerID", preprocessing_report.get("generated_customerid", False)),
                ("Generated StockCode", preprocessing_report.get("generated_stockcode", False)),
                ("Generated Country", preprocessing_report.get("generated_country", False)),
            ],
            columns=["Fallback", "Applied"],
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
        '<div class="section-header">RFM Customer Segmentation + Clustering Comparison + PCA / SVD</div>',
        unsafe_allow_html=True,
    )

    show_info(
        """
        Customers are represented using enhanced RFM features:
        Recency, Frequency, Monetary, AvgOrderValue, ProductDiversity, and CustomerActivityDays.
        The system compares KMeans, Hierarchical Clustering, and DBSCAN, then selects the best
        method using clustering metrics.
        """
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        n_clusters = st.slider("KMeans / Hierarchical clusters", 2, 8, 4)

    with c2:
        dbscan_eps_mode = st.selectbox("DBSCAN eps mode", ["Auto", "Manual"])

    with c3:
        dbscan_eps = st.slider("Manual DBSCAN eps", 0.2, 2.5, 0.8, 0.1)

    with c4:
        dbscan_min_samples = st.slider("DBSCAN min samples", 3, 25, 8)

    dim_view = st.selectbox(
        "Dimensionality View",
        ["PCA & SVD Comparison", "PCA Only", "SVD Only"],
    )

    selected_method_option = st.selectbox(
        "Clustering method to apply after comparison",
        [
            "Auto-select best by Silhouette",
            "KMeans",
            "Hierarchical (Agglomerative)",
            "DBSCAN",
        ],
    )

    run_seg = st.button("Run and Compare Clustering Methods")

    if run_seg or st.session_state.rfm is not None:
        with st.spinner("Computing customer features, comparing clustering methods, and projecting with PCA/SVD..."):
            rfm = compute_rfm(df)
            rfm_scaled, scaler, used_features = scale_rfm(rfm)

            k_range, inertias, silhouettes = find_optimal_k(rfm_scaled)

            eps_value = None if dbscan_eps_mode == "Auto" else dbscan_eps

            clustering_scores, method_labels, best_method = compare_clustering_methods(
                rfm_scaled,
                n_clusters=n_clusters,
                dbscan_eps=eps_value,
                dbscan_min_samples=dbscan_min_samples,
            )

            chosen_method = (
                best_method
                if selected_method_option == "Auto-select best by Silhouette"
                else selected_method_option
            )

            rfm_clustered = apply_clustering_method(
                rfm,
                method_labels[chosen_method],
                chosen_method,
            )

            pca_df, explained_var = apply_pca(rfm_scaled, rfm_clustered)
            svd_df, svd_explained = apply_svd(rfm_scaled, rfm_clustered)
            comparison_df = dimensionality_comparison_table(explained_var, svd_explained)

            st.session_state.rfm = rfm_clustered
            st.session_state.used_cluster_features = used_features
            st.session_state.clustering_scores = clustering_scores
            st.session_state.best_clustering_method = best_method

        st.success(explain_best_clustering_method(best_method, clustering_scores))
        st.info(f"Applied method in the dashboard: {chosen_method}")

        st.markdown(
            '<div class="section-header">Features Used for Clustering</div>',
            unsafe_allow_html=True,
        )

        st.write(st.session_state.used_cluster_features)

        st.markdown(
            '<div class="section-header">Clustering Quality Report</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"<div class='info-box'>{explain_clustering_metrics()}</div>",
            unsafe_allow_html=True,
        )

        themed_plotly_chart(plot_clustering_comparison(clustering_scores), width="stretch")
        st.dataframe(clustering_scores, width="stretch")

        a, b = st.columns(2)

        with a:
            themed_plotly_chart(plot_segment_distribution(rfm_clustered), width="stretch")

        with b:
            themed_plotly_chart(plot_elbow(k_range, inertias, silhouettes), width="stretch")

        if dim_view in ["PCA & SVD Comparison", "PCA Only"]:
            themed_plotly_chart(plot_pca_clusters(pca_df, explained_var), width="stretch")

        if dim_view in ["PCA & SVD Comparison", "SVD Only"]:
            themed_plotly_chart(plot_svd_clusters(svd_df, svd_explained), width="stretch")

        if dim_view == "PCA & SVD Comparison":
            st.markdown(
                '<div class="section-header">PCA vs SVD Explained Variance Comparison</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(comparison_df, width="stretch")

            show_info(
                """
                PCA and SVD are both dimensionality reduction methods.
                PCA is commonly used to visualize customer segments by preserving variance.
                SVD is shown as an alternative projection for comparison.
                """
            )

        themed_plotly_chart(plot_cluster_profiles(rfm_clustered), width="stretch")

        st.markdown(
            '<div class="section-header">Business Interpretation of Segments</div>',
            unsafe_allow_html=True,
        )

        st.text(build_segmentation_insights(rfm_clustered))

        st.dataframe(
            rfm_clustered.sort_values("Monetary", ascending=False),
            width="stretch",
            height=330,
        )

    else:
        st.info("Click Run and Compare Clustering Methods to evaluate KMeans, Hierarchical, and DBSCAN.")


# ============================================================
# Tab 4: Association Rules
# ============================================================

with tabs[3]:
    st.markdown(
        '<div class="section-header">Market Basket Analysis — Apriori / FP-Growth</div>',
        unsafe_allow_html=True,
    )

    show_info(
        """
        Association rules discover products that are frequently purchased together.
        This supports cross-selling, bundling, and store layout decisions.
        """
    )

    countries = ["All"]

    if "United Kingdom" in df["Country"].dropna().unique():
        countries.append("United Kingdom")

    countries += sorted(
        [
            c for c in df["Country"].dropna().unique().tolist()
            if c not in countries
        ]
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        country = st.selectbox(
            "Country",
            countries,
            index=1 if "United Kingdom" in countries else 0,
        )

    with c2:
        algorithm = st.selectbox("Algorithm", ["FP-Growth", "Apriori"])

    with c3:
        min_support = st.slider("Min Support", 0.005, 0.20, 0.03, 0.005)

    with c4:
        min_confidence = st.slider("Min Confidence", 0.10, 1.0, 0.30, 0.05)

    with c5:
        min_lift = st.slider("Min Lift", 1.0, 10.0, 1.2, 0.1)

    if st.button("Mine Basket Rules"):
        with st.spinner("Mining product relationships..."):
            basket = build_basket_matrix(df, country)
            rules = run_association_rules(
                basket,
                algorithm,
                min_support,
                min_confidence,
                min_lift,
            )
            st.session_state.rules = rules

    rules = st.session_state.rules

    if rules is not None and not rules.empty:
        st.success(f"Found {len(rules):,} rules using {algorithm}.")

        themed_plotly_chart(plot_rules_scatter(rules), width="stretch")
        themed_plotly_chart(plot_top_rules_bar(rules), width="stretch")

        st.dataframe(
            rules[["antecedents", "consequents", "support", "confidence", "lift"]],
            width="stretch",
            height=360,
        )

        st.markdown(
            '<div class="section-header">Business Interpretation of Strong Rules</div>',
            unsafe_allow_html=True,
        )

        for insight in interpret_rules(rules, top_n=5):
            st.markdown(f"- {insight}")

    else:
        st.info("Run rule mining to discover cross-selling opportunities.")


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
        '<div class="section-header">Smart Product Recommendation System</div>',
        unsafe_allow_html=True,
    )

    show_info(build_recommender_explanation())

    if st.session_state.similarity_df is None:
        with st.spinner("Building item-based recommendation engine..."):
            matrix = build_user_item_matrix(df)
            product_stats = get_product_stats(df)
            similarity_df = compute_item_similarity(matrix)

            st.session_state.user_item_matrix = matrix
            st.session_state.product_stats = product_stats
            st.session_state.similarity_df = similarity_df

    matrix = st.session_state.user_item_matrix
    similarity_df = st.session_state.similarity_df
    product_stats = st.session_state.product_stats

    st.markdown(
        '<div class="section-header">Recommender Quality Report</div>',
        unsafe_allow_html=True,
    )

    quality_report = get_recommender_quality_report(matrix, similarity_df)

    q1, q2, q3, q4, q5 = st.columns(5)

    metric_card(q1, "Customers", f"{quality_report['customers_in_matrix']:,}")
    metric_card(q2, "Products", f"{quality_report['products_in_matrix']:,}")
    metric_card(q3, "Density", f"{quality_report['matrix_density']}")
    metric_card(q4, "Avg Products / Customer", f"{quality_report['avg_products_per_customer']}")
    metric_card(q5, "Avg Similarity", f"{quality_report['avg_similarity']}")

    themed_plotly_chart(plot_recommender_quality(quality_report), width="stretch")

    rec_mode = st.radio(
        "Recommendation Mode",
        ["Customer-based recommendation", "Basket-based recommendation"],
        horizontal=True,
    )

    if rec_mode == "Customer-based recommendation":
        c1, c2 = st.columns([2, 1])

        with c1:
            customer_id = st.text_input("Customer ID", placeholder="Example: 17850")

        with c2:
            top_n = st.slider("Top N", 5, 20, 10, key="customer_top_n")

        if st.button("Get Customer Recommendations") and customer_id:
            recs = recommend_for_customer(
                customer_id.strip(),
                matrix,
                similarity_df,
                product_stats,
                top_n,
            )

            if "Message" in recs.columns:
                st.warning(recs["Message"].iloc[0])

                popular = get_popular_products(df, top_n)
                st.dataframe(popular, width="stretch")
                themed_plotly_chart(plot_popular_products(popular), width="stretch")

            else:
                themed_plotly_chart(plot_recommendations(recs), width="stretch")
                st.dataframe(recs, width="stretch")
                show_success(build_recommendation_business_insight(recs))

    else:
        product_options = sorted(similarity_df.columns.tolist())

        selected_products = st.multiselect(
            "Select products currently in the basket",
            product_options,
        )

        top_n = st.slider("Top N", 5, 20, 10, key="basket_top_n")

        if st.button("Get Basket Recommendations"):
            basket_recs = recommend_for_basket(
                selected_products,
                similarity_df,
                product_stats,
                top_n,
            )

            if "Message" in basket_recs.columns:
                st.warning(basket_recs["Message"].iloc[0])
            else:
                themed_plotly_chart(plot_recommendations(basket_recs), width="stretch")
                st.dataframe(basket_recs, width="stretch")
                show_success(build_recommendation_business_insight(basket_recs))

    st.markdown(
        '<div class="section-header">Popular Products Fallback</div>',
        unsafe_allow_html=True,
    )

    popular = get_popular_products(df, 15)

    themed_plotly_chart(plot_popular_products(popular), width="stretch")
    st.dataframe(popular, width="stretch")


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

    if st.session_state.rfm is not None:
        row = st.session_state.rfm[
            st.session_state.rfm["CustomerID"].astype(str) == str(selected_customer)
        ]

        if not row.empty:
            st.info(
                f"Segment: {row.iloc[0]['Segment']} | "
                f"Recency: {row.iloc[0]['Recency']} days | "
                f"Frequency: {row.iloc[0]['Frequency']} | "
                f"Monetary: £{row.iloc[0]['Monetary']:,.2f}"
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

    report_text = build_insights_text(
        df,
        st.session_state.rfm,
        st.session_state.rules,
    )

    if preprocessing_report:
        report_text += "\n\n"
        report_text += build_preprocessing_report_text(preprocessing_report)

    if st.session_state.rfm is not None:
        report_text += "\n\n"
        report_text += build_segmentation_insights(st.session_state.rfm)

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