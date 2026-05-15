"""
Module 2: Flexible Customer Segmentation, Clustering Comparison,
and Dimensionality Reduction (PCA + SVD)

Responsible Student: Abd Alkareem Shafie
ID : 12241203

Purpose:
- Build customer-level features from retail/e-commerce transactions.
- Support multiple datasets after preprocessing standardizes columns.
- Apply RFM-based customer segmentation.
- Add business-oriented customer features.
- Scale and log-transform skewed features.
- Compare multiple clustering algorithms.
- Select the best clustering method using evaluation metrics.
- Visualize customer segments using PCA and SVD.

Mandatory requirements covered:
- Clustering
- Dimensionality Reduction

Techniques included:
- KMeans
- Agglomerative / Hierarchical Clustering
- DBSCAN
- PCA
- Truncated SVD
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.neighbors import NearestNeighbors


# ============================================================
# 1. Configuration
# ============================================================

RFM_BASE_FEATURES = ["Recency", "Frequency", "Monetary"]

OPTIONAL_CUSTOMER_FEATURES = [
    "AvgOrderValue",
    "ProductDiversity",
    "CustomerActivityDays",
]

CANDIDATE_CLUSTER_FEATURES = RFM_BASE_FEATURES + OPTIONAL_CUSTOMER_FEATURES


# ============================================================
# 2. Safe Column Checking
# ============================================================

def _require_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    """
    Check that required columns exist before running clustering.

    The preprocessing module should standardize column names. If these
    columns are missing, clustering cannot be performed correctly.
    """
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns for clustering: {missing}. "
            "Please check preprocessing column mapping."
        )


def _get_product_column(df: pd.DataFrame) -> str | None:
    """
    Choose the best available product identifier.

    Priority:
    1. StockCode
    2. Description
    3. None

    This keeps clustering flexible across different retail datasets.
    """
    if "StockCode" in df.columns:
        return "StockCode"

    if "Description" in df.columns:
        return "Description"

    return None


# ============================================================
# 3. Customer Feature Engineering: Enhanced RFM
# ============================================================

@st.cache_data(show_spinner=False)
def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute enhanced RFM customer features.

    RFM:
    - Recency: Days since last purchase
    - Frequency: Number of unique invoices/orders
    - Monetary: Total spending

    Extra business features:
    - AvgOrderValue: Average spending per order
    - ProductDiversity: Number of distinct products bought
    - CustomerActivityDays: Days between first and last purchase

    These features make clustering more meaningful for business decisions.
    """
    _require_columns(
        df,
        ["CustomerID", "InvoiceDate", "InvoiceNo", "TotalPrice"],
    )

    product_col = _get_product_column(df)

    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    agg_dict = {
        "Recency": ("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        "Frequency": ("InvoiceNo", "nunique"),
        "Monetary": ("TotalPrice", "sum"),
        "FirstPurchase": ("InvoiceDate", "min"),
        "LastPurchase": ("InvoiceDate", "max"),
    }

    if product_col is not None:
        agg_dict["ProductDiversity"] = (product_col, "nunique")

    rfm = df.groupby("CustomerID").agg(**agg_dict).reset_index()

    # If no product column exists, create a neutral fallback.
    if "ProductDiversity" not in rfm.columns:
        rfm["ProductDiversity"] = 1

    # Average order value gives better business interpretation than total spending alone.
    rfm["AvgOrderValue"] = (
        rfm["Monetary"] / rfm["Frequency"].replace(0, np.nan)
    ).fillna(0)

    # Customer lifetime / activity period.
    rfm["CustomerActivityDays"] = (
        rfm["LastPurchase"] - rfm["FirstPurchase"]
    ).dt.days.clip(lower=0)

    numeric_cols = [
        "Recency",
        "Frequency",
        "Monetary",
        "ProductDiversity",
        "AvgOrderValue",
        "CustomerActivityDays",
    ]

    for col in numeric_cols:
        rfm[col] = pd.to_numeric(rfm[col], errors="coerce").fillna(0)

    rfm["Monetary"] = rfm["Monetary"].round(2)
    rfm["AvgOrderValue"] = rfm["AvgOrderValue"].round(2)

    return rfm.drop(columns=["FirstPurchase", "LastPurchase"])


# ============================================================
# 4. Flexible Feature Selection + Scaling
# ============================================================

def get_available_cluster_features(rfm: pd.DataFrame) -> list[str]:
    """
    Select only available numeric features for clustering.

    This allows the module to work with datasets where some optional
    features may not exist.
    """
    available = []

    for col in CANDIDATE_CLUSTER_FEATURES:
        if col in rfm.columns and pd.api.types.is_numeric_dtype(rfm[col]):
            available.append(col)

    if len(available) < 3:
        raise ValueError(
            "Not enough numeric customer features for clustering. "
            f"Available features: {available}"
        )

    return available


def scale_rfm(rfm: pd.DataFrame):
    """
    Prepare customer features for clustering.

    Steps:
    1. Select available customer features dynamically.
    2. Apply log1p to skewed positive features.
    3. Apply StandardScaler because clustering is distance-based.

    Returns:
    - scaled feature matrix
    - fitted scaler
    - list of used features
    """
    used_features = get_available_cluster_features(rfm)

    features = rfm[used_features].copy()

    # Retail behavior is usually skewed:
    # many small customers, few very large customers.
    log_features = [
        "Frequency",
        "Monetary",
        "AvgOrderValue",
        "ProductDiversity",
        "CustomerActivityDays",
    ]

    for col in log_features:
        if col in features.columns:
            features[col] = np.log1p(features[col].clip(lower=0))

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    return scaled, scaler, used_features


# ============================================================
# 5. KMeans: Find Good K
# ============================================================

def find_optimal_k(rfm_scaled, k_range=range(2, 9)):
    """
    Test several k values for KMeans.

    Metrics:
    - Inertia: used for elbow method.
    - Silhouette: used to evaluate cluster separation.
    """
    inertias = []
    silhouettes = []

    n_samples = len(rfm_scaled)

    for k in k_range:
        if k >= n_samples:
            continue

        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(rfm_scaled)

        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(rfm_scaled, labels))

    valid_k = list(range(2, 2 + len(inertias)))

    return valid_k, inertias, silhouettes


def plot_elbow(k_range, inertias, silhouettes):
    """
    Plot KMeans elbow curve.

    The elbow helps decide a reasonable number of clusters.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=list(k_range),
            y=inertias,
            mode="lines+markers",
            name="Inertia",
        )
    )

    fig.update_layout(
        title="Elbow Method — Choosing Number of Clusters",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Inertia",
        template="plotly_dark",
        height=400,
    )

    return fig


# ============================================================
# 6. DBSCAN Adaptive Helper
# ============================================================

def estimate_dbscan_eps(X, min_samples: int = 8) -> float:
    """
    Estimate a reasonable DBSCAN eps value using nearest-neighbor distances.

    This makes DBSCAN more flexible across datasets instead of using a fixed eps.
    """
    if len(X) <= min_samples:
        return 0.8

    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors.fit(X)

    distances, _ = neighbors.kneighbors(X)
    kth_distances = np.sort(distances[:, -1])

    # Use 90th percentile as a simple stable heuristic.
    eps = float(np.percentile(kth_distances, 90))

    if eps <= 0 or np.isnan(eps):
        eps = 0.8

    return round(eps, 3)


# ============================================================
# 7. Safe Clustering Metrics
# ============================================================

def _safe_cluster_metrics(X, labels):
    """
    Compute clustering metrics safely.

    DBSCAN may mark points as noise with label -1.
    Metrics are computed only on valid non-noise clusters.
    """
    labels = np.asarray(labels)

    mask = labels != -1
    eval_X = X[mask]
    eval_labels = labels[mask]

    n_clusters = len(set(eval_labels))
    n_noise = int(np.sum(labels == -1))

    if n_clusters < 2 or len(eval_X) < 3:
        return {
            "silhouette": np.nan,
            "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
        }

    return {
        "silhouette": float(silhouette_score(eval_X, eval_labels)),
        "davies_bouldin": float(davies_bouldin_score(eval_X, eval_labels)),
        "calinski_harabasz": float(calinski_harabasz_score(eval_X, eval_labels)),
        "n_clusters": n_clusters,
        "n_noise": n_noise,
    }


# ============================================================
# 8. Clustering Model Comparison
# ============================================================

@st.cache_data(show_spinner=False)
def compare_clustering_methods(
    rfm_scaled,
    n_clusters: int = 4,
    dbscan_eps: float | None = None,
    dbscan_min_samples: int = 8,
):
    """
    Compare multiple clustering algorithms.

    Algorithms:
    - KMeans
    - Hierarchical / Agglomerative Clustering
    - DBSCAN

    Selection rule:
    - Primary: highest Silhouette Score
    - Secondary: lower Davies-Bouldin Index
    - Invalid methods are ignored
    """
    if dbscan_eps is None:
        dbscan_eps = estimate_dbscan_eps(rfm_scaled, dbscan_min_samples)

    methods = {}

    # KMeans works well for compact and business-interpretable segments.
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    methods["KMeans"] = km.fit_predict(rfm_scaled)

    # Agglomerative clustering provides hierarchical grouping.
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    methods["Hierarchical (Agglomerative)"] = agg.fit_predict(rfm_scaled)

    # DBSCAN can detect noise/outlier customers.
    db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    methods["DBSCAN"] = db.fit_predict(rfm_scaled)

    rows = []

    for method_name, labels in methods.items():
        metrics = _safe_cluster_metrics(rfm_scaled, labels)

        rows.append(
            {
                "Method": method_name,
                "Silhouette Score": metrics["silhouette"],
                "Davies-Bouldin Index": metrics["davies_bouldin"],
                "Calinski-Harabasz Score": metrics["calinski_harabasz"],
                "Clusters Found": metrics["n_clusters"],
                "Noise Points": metrics["n_noise"],
                "Status": (
                    "Valid"
                    if not np.isnan(metrics["silhouette"])
                    else "Not suitable for this dataset/settings"
                ),
            }
        )

    comparison = pd.DataFrame(rows)

    valid = comparison.dropna(subset=["Silhouette Score"]).copy()

    if valid.empty:
        best_method = "KMeans"
    else:
        valid = valid.sort_values(
            by=["Silhouette Score", "Davies-Bouldin Index"],
            ascending=[False, True],
        )
        best_method = valid.iloc[0]["Method"]

    return comparison, methods, best_method


def explain_best_clustering_method(best_method: str, comparison_df: pd.DataFrame) -> str:
    """
    Generate a business-friendly explanation of the selected clustering method.
    """
    row = comparison_df[comparison_df["Method"] == best_method].iloc[0]

    score = row["Silhouette Score"]
    clusters = int(row["Clusters Found"])

    if best_method == "KMeans":
        reason = (
            "KMeans produced clear and stable customer groups, which makes it "
            "suitable for marketing segmentation and business interpretation."
        )
    elif best_method.startswith("Hierarchical"):
        reason = (
            "Hierarchical clustering produced meaningful grouped customer behavior "
            "and is useful when we want to understand nested customer relationships."
        )
    else:
        reason = (
            "DBSCAN was selected because it discovered density-based customer groups "
            "and separated noisy or unusual customers."
        )

    return (
        f"Best method: {best_method}. "
        f"It achieved the highest valid Silhouette Score ({score:.3f}) "
        f"with {clusters} valid clusters. {reason}"
    )


def explain_clustering_metrics() -> str:
    """
    Explain clustering evaluation metrics in simple language.
    """
    return (
        "Silhouette Score measures how well-separated and compact the clusters are "
        "(higher is better). Davies-Bouldin measures similarity between clusters "
        "(lower is better). Calinski-Harabasz measures separation between clusters "
        "relative to compactness (higher is better)."
    )


def plot_clustering_comparison(comparison_df: pd.DataFrame):
    """
    Plot comparison of clustering methods using Silhouette Score.
    """
    fig = px.bar(
        comparison_df,
        x="Method",
        y="Silhouette Score",
        color="Method",
        text="Silhouette Score",
        title="Clustering Method Comparison — Silhouette Score",
        template="plotly_dark",
        height=420,
        color_discrete_sequence=px.colors.qualitative.Bold,
    )

    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(showlegend=False)

    return fig


# ============================================================
# 9. Attach Labels + Business Segment Names
# ============================================================

@st.cache_data(show_spinner=False)
def apply_clustering_method(
    rfm: pd.DataFrame,
    labels,
    method_name: str,
):
    """
    Attach cluster labels to RFM data and create business-friendly segment names.

    The segment naming uses full RFM logic:
    - Lower Recency is better
    - Higher Frequency is better
    - Higher Monetary value is better
    """
    rfm = rfm.copy()

    rfm["Cluster"] = pd.Series(labels).astype(str).values
    rfm["ClusteringMethod"] = method_name

    labels_map = {"-1": "Noise / Outlier Customers"}

    non_noise = rfm[rfm["Cluster"] != "-1"].copy()

    if not non_noise.empty:
        cluster_summary = non_noise.groupby("Cluster")[
            ["Recency", "Frequency", "Monetary"]
        ].mean()

        ranked = cluster_summary.copy()

        # Lower recency is better, so ascending=True gives recent customers higher rank.
        ranked["RecencyScore"] = ranked["Recency"].rank(ascending=False)

        # Higher frequency and monetary are better.
        ranked["FrequencyScore"] = ranked["Frequency"].rank(ascending=True)
        ranked["MonetaryScore"] = ranked["Monetary"].rank(ascending=True)

        ranked["RFMScore"] = ranked[
            ["RecencyScore", "FrequencyScore", "MonetaryScore"]
        ].mean(axis=1)

        ranked = ranked.sort_values("RFMScore", ascending=False)

        segment_names = [
            "Champions",
            "Loyal High Value",
            "Potential Loyalists",
            "Needs Attention",
            "At Risk",
            "Low Activity",
            "Dormant",
            "Other Segment",
        ]

        for i, cluster_id in enumerate(ranked.index):
            labels_map[cluster_id] = (
                segment_names[i]
                if i < len(segment_names)
                else f"Segment {i + 1}"
            )

    rfm["Segment"] = rfm["Cluster"].map(labels_map).fillna("Other Segment")

    return rfm


@st.cache_data(show_spinner=False)
def apply_kmeans(rfm: pd.DataFrame, rfm_scaled, n_clusters: int = 4):
    """
    Backward-compatible KMeans function.

    Some parts of the app may still call apply_kmeans directly.
    """
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(rfm_scaled)

    return apply_clustering_method(rfm, labels, "KMeans"), km


# ============================================================
# 10. PCA Dimensionality Reduction
# ============================================================

@st.cache_data(show_spinner=False)
def apply_pca(rfm_scaled, rfm_with_clusters: pd.DataFrame):
    """
    Reduce enhanced customer features to 2D using PCA for visualization.

    PCA helps us visually inspect whether customer segments are separated.
    """
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(rfm_scaled)

    keep_cols = [
        "CustomerID",
        "Segment",
        "Recency",
        "Frequency",
        "Monetary",
        "AvgOrderValue",
        "ProductDiversity",
        "CustomerActivityDays",
    ]

    keep_cols = [col for col in keep_cols if col in rfm_with_clusters.columns]

    pca_df = rfm_with_clusters[keep_cols].copy()
    pca_df["PC1"] = components[:, 0]
    pca_df["PC2"] = components[:, 1]

    explained = pca.explained_variance_ratio_

    return pca_df, explained


# ============================================================
# 11. SVD Dimensionality Reduction
# ============================================================

@st.cache_data(show_spinner=False)
def apply_svd(rfm_scaled, rfm_with_clusters: pd.DataFrame):
    """
    Reduce enhanced customer features to 2D using Truncated SVD.

    SVD is used as an alternative dimensionality reduction method
    and compared with PCA.
    """
    svd = TruncatedSVD(n_components=2, random_state=42)
    components = svd.fit_transform(rfm_scaled)

    keep_cols = [
        "CustomerID",
        "Segment",
        "Recency",
        "Frequency",
        "Monetary",
        "AvgOrderValue",
        "ProductDiversity",
        "CustomerActivityDays",
    ]

    keep_cols = [col for col in keep_cols if col in rfm_with_clusters.columns]

    svd_df = rfm_with_clusters[keep_cols].copy()
    svd_df["SVD1"] = components[:, 0]
    svd_df["SVD2"] = components[:, 1]

    explained = svd.explained_variance_ratio_

    return svd_df, explained


def dimensionality_comparison_table(pca_explained, svd_explained):
    """
    Compare PCA and SVD explained variance.
    """
    return pd.DataFrame(
        {
            "Method": ["PCA", "SVD"],
            "Component 1 Explained Variance": [
                float(pca_explained[0]),
                float(svd_explained[0]),
            ],
            "Component 2 Explained Variance": [
                float(pca_explained[1]),
                float(svd_explained[1]),
            ],
            "Total Explained Variance": [
                float(sum(pca_explained)),
                float(sum(svd_explained)),
            ],
        }
    )


# ============================================================
# 12. Visualizations
# ============================================================

def plot_pca_clusters(pca_df: pd.DataFrame, explained_variance):
    """
    Scatter plot of PCA components colored by customer segment.
    """
    hover_cols = [
        col for col in [
            "CustomerID",
            "Recency",
            "Frequency",
            "Monetary",
            "AvgOrderValue",
            "ProductDiversity",
        ]
        if col in pca_df.columns
    ]

    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="Segment",
        hover_data=hover_cols,
        title=(
            "Customer Segments — PCA Projection "
            f"(Explained Variance: {sum(explained_variance) * 100:.1f}%)"
        ),
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Bold,
        height=500,
    )

    fig.update_traces(marker=dict(size=5, opacity=0.7))

    return fig


def plot_svd_clusters(svd_df: pd.DataFrame, explained_variance):
    """
    Scatter plot of SVD components colored by customer segment.
    """
    hover_cols = [
        col for col in [
            "CustomerID",
            "Recency",
            "Frequency",
            "Monetary",
            "AvgOrderValue",
            "ProductDiversity",
        ]
        if col in svd_df.columns
    ]

    fig = px.scatter(
        svd_df,
        x="SVD1",
        y="SVD2",
        color="Segment",
        hover_data=hover_cols,
        title=(
            "Customer Segments — SVD Projection "
            f"(Explained Variance: {sum(explained_variance) * 100:.1f}%)"
        ),
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Bold,
        height=500,
    )

    fig.update_traces(marker=dict(size=5, opacity=0.7))

    return fig


def plot_cluster_profiles(rfm_with_clusters: pd.DataFrame):
    """
    Bar chart showing average customer behavior metrics per segment.
    """
    profile_features = [
        "Recency",
        "Frequency",
        "Monetary",
        "AvgOrderValue",
        "ProductDiversity",
        "CustomerActivityDays",
    ]

    profile_features = [
        col for col in profile_features if col in rfm_with_clusters.columns
    ]

    summary = (
        rfm_with_clusters
        .groupby("Segment")[profile_features]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        summary.melt(
            id_vars="Segment",
            var_name="Metric",
            value_name="Average",
        ),
        x="Segment",
        y="Average",
        color="Metric",
        barmode="group",
        title="Average Customer Metrics per Segment",
        template="plotly_dark",
        height=450,
    )

    return fig


def plot_segment_distribution(rfm_with_clusters: pd.DataFrame):
    """
    Pie chart of customer counts per segment.
    """
    counts = rfm_with_clusters["Segment"].value_counts().reset_index()
    counts.columns = ["Segment", "Count"]

    fig = px.pie(
        counts,
        values="Count",
        names="Segment",
        title="Customer Segment Distribution",
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Bold,
        height=400,
    )

    return fig


# ============================================================
# 13. Business Interpretation
# ============================================================

def build_segmentation_insights(rfm_with_clusters: pd.DataFrame) -> str:
    """
    Generate short business insights from customer segments.

    This supports the Business Report and discussion.
    """
    if rfm_with_clusters.empty:
        return "No customer segments were generated."

    lines = [
        "Customer Segmentation Insights",
        "------------------------------",
    ]

    segment_summary = (
        rfm_with_clusters
        .groupby("Segment")
        .agg(
            Customers=("CustomerID", "nunique"),
            AvgRecency=("Recency", "mean"),
            AvgFrequency=("Frequency", "mean"),
            AvgMonetary=("Monetary", "mean"),
            TotalRevenue=("Monetary", "sum"),
        )
        .sort_values("TotalRevenue", ascending=False)
    )

    top_segment = segment_summary.index[0]

    lines.append(
        f"- Highest revenue segment: {top_segment} "
        f"with total customer monetary value of "
        f"{segment_summary.iloc[0]['TotalRevenue']:.2f}."
    )

    if "Champions" in segment_summary.index:
        lines.append(
            "- Champions are the most valuable customers and should be targeted "
            "with loyalty rewards and premium offers."
        )

    if "At Risk" in segment_summary.index:
        lines.append(
            "- At Risk customers should be targeted with retention campaigns "
            "because their recency is weaker than stronger segments."
        )

    lines.append(
        "- Segmentation helps the business design targeted marketing strategies "
        "instead of treating all customers the same."
    )

    return "\n".join(lines)