"""
Module 4: Smart Product Recommender System
Responsible Student: Abd Alkareem Shafie
ID 12241203

# hint:
# This recommender is unsupervised and does not require product ratings.
# It uses purchase behavior to infer product relationships.

Purpose:
- Build a flexible recommendation module for retail/e-commerce datasets.
- Support item-based collaborative filtering using customer-product behavior.
- Reduce heavy-buyer bias using binary interaction normalization.
- Provide recommendations for existing customers.
- Provide basket-based recommendations from selected products.
- Provide popular-product fallback for cold-start customers.
- Add business-friendly explanations for recommendation results.

Requirement covered:
- Recommender Systems

Why this module is important:
A recommender system helps the business increase cross-selling opportunities
by suggesting products that are commonly purchased by similar customers or
related to items already purchased.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. Configuration
# ============================================================

DEFAULT_TOP_PRODUCTS = 300
DEFAULT_TOP_N_RECOMMENDATIONS = 10

# Candidate columns used by the recommender.
# The preprocessing module should already standardize these columns.
REQUIRED_COLUMNS = [
    "CustomerID",
    "Description",
    "Quantity",
    "TotalPrice",
]


# ============================================================
# 2. Validation Helpers
# ============================================================

def _require_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    """
    Validate that all required columns exist.

    This makes the recommender safer when testing multiple datasets.
    """
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns for recommender system: {missing}. "
            "Please check preprocessing output."
        )


def _clean_product_name(product: str) -> str:
    """
    Normalize product names for matching.

    This helps when users type or select products with extra spaces.
    """
    return str(product).strip().upper()


# ============================================================
# 3. Product Popularity Statistics
# ============================================================

@st.cache_data(show_spinner=False)
def get_product_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute product-level statistics used in recommendation explanations.

    Outputs:
    - Total sold quantity
    - Total revenue
    - Number of unique customers
    - Number of unique invoices
    """
    _require_columns(df, REQUIRED_COLUMNS)

    stats = (
        df.groupby("Description")
        .agg(
            TotalSold=("Quantity", "sum"),
            Revenue=("TotalPrice", "sum"),
            UniqueCustomers=("CustomerID", "nunique"),
            Orders=("InvoiceNo", "nunique") if "InvoiceNo" in df.columns else ("Description", "count"),
        )
        .reset_index()
    )

    stats["Revenue"] = stats["Revenue"].round(2)

    return stats.sort_values(
        ["UniqueCustomers", "TotalSold", "Revenue"],
        ascending=False,
    )


# ============================================================
# 4. Build Customer-Product Matrix
# ============================================================

@st.cache_data(show_spinner=False)
def build_user_item_matrix(
    df: pd.DataFrame,
    top_products: int = DEFAULT_TOP_PRODUCTS,
    binary: bool = True,
) -> pd.DataFrame:
    """
    Build a Customer x Product interaction matrix.

    Parameters:
    - top_products:
        Limits the matrix to the most popular products to keep computation efficient.
    - binary:
        If True, values become 1/0 based on whether the customer bought the product.
        This reduces dominance of heavy buyers and usually improves recommendation quality.

    Why binary normalization?
    In retail data, a few customers may buy very large quantities. If we use raw quantity,
    those heavy buyers dominate similarity scores. Binary interaction focuses on purchase
    relationships instead of purchase volume.
    """
    _require_columns(df, ["CustomerID", "Description", "Quantity"])

    df = df.copy()
    df["Description"] = df["Description"].astype(str).str.strip().str.upper()

    # Select top products by number of unique customers first, then quantity.
    product_rank = (
        df.groupby("Description")
        .agg(
            UniqueCustomers=("CustomerID", "nunique"),
            TotalQuantity=("Quantity", "sum"),
        )
        .sort_values(["UniqueCustomers", "TotalQuantity"], ascending=False)
        .head(top_products)
        .index
    )

    df_filtered = df[df["Description"].isin(product_rank)]

    matrix = df_filtered.pivot_table(
        index="CustomerID",
        columns="Description",
        values="Quantity",
        aggfunc="sum",
        fill_value=0,
    )

    if binary:
        matrix = matrix.gt(0).astype(int)

    return matrix


# ============================================================
# 5. Item-Based Collaborative Filtering
# ============================================================

@st.cache_data(show_spinner=False)
def compute_item_similarity(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute item-item cosine similarity.

    Each product is represented by a vector of customers who purchased it.
    Products bought by similar customers receive higher similarity scores.

    Extra safety:
    - If there are fewer than 2 products, recommendations are not meaningful,
      so an empty DataFrame is returned instead of crashing.
    - The diagonal is set to zero using a writable NumPy copy to avoid
      "underlying array is read-only" errors with Streamlit caching.
    """
    if matrix.empty or matrix.shape[1] < 2:
        return pd.DataFrame()

    item_matrix = matrix.T

    similarity = cosine_similarity(item_matrix)

    similarity_df = pd.DataFrame(
        similarity,
        index=item_matrix.index,
        columns=item_matrix.index,
    )

    # Remove self-similarity safely.
    similarity_df = similarity_df.copy()
    arr = similarity_df.to_numpy(copy=True)
    np.fill_diagonal(arr, 0)
    similarity_df.iloc[:, :] = arr

    return similarity_df


# ============================================================
# 6. Customer-Based Recommendations
# ============================================================


def recommend_for_customer(
    customer_id: str,
    matrix: pd.DataFrame,
    similarity_df: pd.DataFrame,
    product_stats: pd.DataFrame | None = None,
    top_n: int = DEFAULT_TOP_N_RECOMMENDATIONS,
) -> pd.DataFrame:
    """
    Recommend products for an existing customer.

    Method:
    1. Get products already purchased by the customer.
    2. Sum similarity scores from purchased products to all other products.
    3. Exclude already purchased products.
    4. Return top N recommended products.

    This is item-based collaborative filtering.
    """
    customer_id = str(customer_id).strip()

    if matrix.empty or similarity_df.empty:
        return pd.DataFrame(
            {"Message": ["Recommendation matrix is empty. Please check dataset size."]}
        )

    if customer_id not in matrix.index.astype(str):
        return pd.DataFrame(
            {"Message": [f"Customer {customer_id} not found. Showing popular products is recommended for cold-start users."]}
        )

    # Ensure index comparison works even if CustomerID type differs.
    matrix_index_as_str = matrix.index.astype(str)
    actual_customer_index = matrix.index[matrix_index_as_str == customer_id][0]

    customer_purchases = matrix.loc[actual_customer_index]
    purchased = customer_purchases[customer_purchases > 0].index.tolist()

    if not purchased:
        return pd.DataFrame(
            {"Message": ["No purchase history found for this customer."]}
        )

    scores = pd.Series(0.0, index=matrix.columns)

    for item in purchased:
        if item in similarity_df.columns:
            scores = scores.add(similarity_df[item], fill_value=0)

    # Remove already purchased products.
    scores = scores.drop(labels=[p for p in purchased if p in scores.index], errors="ignore")

    scores = scores.sort_values(ascending=False).head(top_n)

    recommendations = pd.DataFrame(
        {
            "Rank": range(1, len(scores) + 1),
            "Product": scores.index,
            "RecommendationScore": scores.values.round(4),
            "Reason": [
                "Similar to products previously purchased by this customer"
                for _ in range(len(scores))
            ],
        }
    )

    if product_stats is not None and not product_stats.empty:
        recommendations = recommendations.merge(
            product_stats[
                ["Description", "TotalSold", "Revenue", "UniqueCustomers"]
            ],
            left_on="Product",
            right_on="Description",
            how="left",
        ).drop(columns=["Description"], errors="ignore")

    return recommendations


# ============================================================
# 7. Basket-Based Recommendations
# ============================================================

def recommend_for_basket(
    selected_products: list[str],
    similarity_df: pd.DataFrame,
    product_stats: pd.DataFrame | None = None,
    top_n: int = DEFAULT_TOP_N_RECOMMENDATIONS,
) -> pd.DataFrame:
    """
    Recommend products based on a selected basket.

    Example:
    If the basket contains Product A and Product B,
    the system recommends products most similar to those basket items.

    This is useful for:
    - Cross-selling
    - Bundling
    - "Customers who bought this also bought..." logic
    """
    if similarity_df.empty:
        return pd.DataFrame({"Message": ["Similarity matrix is empty."]})

    selected_products = [_clean_product_name(p) for p in selected_products if str(p).strip()]

    valid_products = [p for p in selected_products if p in similarity_df.columns]

    if not valid_products:
        return pd.DataFrame(
            {"Message": ["No selected basket products were found in the similarity matrix."]}
        )

    scores = pd.Series(0.0, index=similarity_df.columns)

    for item in valid_products:
        scores = scores.add(similarity_df[item], fill_value=0)

    # Exclude products already in basket.
    scores = scores.drop(labels=valid_products, errors="ignore")

    scores = scores.sort_values(ascending=False).head(top_n)

    recommendations = pd.DataFrame(
        {
            "Rank": range(1, len(scores) + 1),
            "Product": scores.index,
            "RecommendationScore": scores.values.round(4),
            "Reason": [
                "Frequently related to products in the selected basket"
                for _ in range(len(scores))
            ],
        }
    )

    if product_stats is not None and not product_stats.empty:
        recommendations = recommendations.merge(
            product_stats[
                ["Description", "TotalSold", "Revenue", "UniqueCustomers"]
            ],
            left_on="Product",
            right_on="Description",
            how="left",
        ).drop(columns=["Description"], errors="ignore")

    return recommendations


# ============================================================
# 8. Popular Products Fallback
# ============================================================

def get_popular_products(df: pd.DataFrame, top_n: int = DEFAULT_TOP_N_RECOMMENDATIONS) -> pd.DataFrame:
    """
    Return popular products as a fallback for cold-start cases.

    Cold-start means:
    - New customer
    - Customer not found in the dataset
    - No purchase history available

    Ranking uses both:
    - number of unique customers
    - sold quantity
    - revenue
    """
    _require_columns(df, REQUIRED_COLUMNS)

    popular = (
        df.groupby("Description")
        .agg(
            TotalSold=("Quantity", "sum"),
            Revenue=("TotalPrice", "sum"),
            UniqueCustomers=("CustomerID", "nunique"),
            Orders=("InvoiceNo", "nunique") if "InvoiceNo" in df.columns else ("Description", "count"),
        )
        .sort_values(["UniqueCustomers", "TotalSold", "Revenue"], ascending=False)
        .head(top_n)
        .reset_index()
    )

    popular["Revenue"] = popular["Revenue"].round(2)
    popular.insert(0, "Rank", range(1, len(popular) + 1))

    return popular


# ============================================================
# 9. Recommendation Evaluation / Quality Indicators
# ============================================================

def get_recommender_quality_report(
    matrix: pd.DataFrame,
    similarity_df: pd.DataFrame,
) -> dict:
    """
    Generate simple quality indicators for the recommender.

    This is not supervised evaluation, but it helps explain system coverage.
    """
    if matrix.empty:
        return {
            "customers_in_matrix": 0,
            "products_in_matrix": 0,
            "matrix_density": 0,
            "avg_products_per_customer": 0,
            "avg_similarity": 0,
        }

    total_cells = matrix.shape[0] * matrix.shape[1]
    non_zero = int((matrix > 0).sum().sum())

    density = non_zero / total_cells if total_cells else 0

    if similarity_df.empty:
        avg_similarity = 0
    else:
        values = similarity_df.values
        avg_similarity = float(values[values > 0].mean()) if np.any(values > 0) else 0

    return {
        "customers_in_matrix": int(matrix.shape[0]),
        "products_in_matrix": int(matrix.shape[1]),
        "matrix_density": round(float(density), 4),
        "avg_products_per_customer": round(float((matrix > 0).sum(axis=1).mean()), 2),
        "avg_similarity": round(avg_similarity, 4),
    }


def build_recommender_explanation() -> str:
    """
    Return a simple explanation of how recommendations are generated.

    Useful in the dashboard and during discussion.
    """
    return (
        "The recommender uses item-based collaborative filtering. "
        "First, it builds a customer-product matrix using purchase history. "
        "Then it converts purchases to binary interactions to reduce heavy-buyer bias. "
        "After that, it computes cosine similarity between products. "
        "Products similar to what a customer already bought are recommended, while already purchased products are excluded."
    )


def build_recommendation_business_insight(recommendations: pd.DataFrame) -> str:
    """
    Generate a short business interpretation for recommendation results.
    """
    if recommendations.empty:
        return "No recommendations were generated."

    if "Message" in recommendations.columns:
        return str(recommendations["Message"].iloc[0])

    top_product = recommendations.iloc[0]["Product"]

    return (
        f"The top recommended product is '{top_product}'. "
        "These recommendations can support cross-selling and help increase average order value."
    )


# ============================================================
# 10. Visualizations
# ============================================================

def plot_recommendations(recommendations: pd.DataFrame):
    """
    Bar chart for recommended products and their recommendation scores.
    """
    if recommendations.empty or "Message" in recommendations.columns:
        return None

    fig = px.bar(
        recommendations.sort_values("RecommendationScore", ascending=True),
        x="RecommendationScore",
        y="Product",
        orientation="h",
        title="Recommended Products",
        template="plotly_dark",
        color="RecommendationScore",
        color_continuous_scale="Teal",
        height=450,
    )

    return fig


def plot_popular_products(popular: pd.DataFrame):
    """
    Bar chart of popular products used as fallback recommendations.
    """
    if popular.empty:
        return None

    fig = px.bar(
        popular.sort_values("TotalSold", ascending=True),
        x="TotalSold",
        y="Description",
        orientation="h",
        title="Popular Products Fallback",
        template="plotly_dark",
        color="Revenue",
        color_continuous_scale="Sunset",
        height=450,
    )

    return fig


def plot_recommender_quality(report: dict):
    """
    Visualize basic recommender quality indicators.
    """
    metrics = pd.DataFrame(
        {
            "Metric": [
                "Customers",
                "Products",
                "Matrix Density",
                "Avg Products / Customer",
                "Avg Similarity",
            ],
            "Value": [
                report.get("customers_in_matrix", 0),
                report.get("products_in_matrix", 0),
                report.get("matrix_density", 0),
                report.get("avg_products_per_customer", 0),
                report.get("avg_similarity", 0),
            ],
        }
    )

    fig = px.bar(
        metrics,
        x="Metric",
        y="Value",
        text="Value",
        title="Recommender Quality Indicators",
        template="plotly_dark",
        height=400,
        color="Metric",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False)

    return fig