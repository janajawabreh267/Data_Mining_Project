"""
Module 3: Market Basket Analysis — Association Rule Mining (Apriori)
Responsible Student: Jana jawabreh

Technique: Apriori Algorithm
- Finds frequent itemsets using minimum support threshold
- Generates association rules using minimum confidence threshold
- Calculates lift to identify real (non-coincidental) relationships
"""

import pandas as pd

from mlxtend.frequent_patterns import apriori, association_rules
import plotly.express as px

import streamlit as st


@st.cache_data(show_spinner=False)
def build_basket_matrix(
    df: pd.DataFrame,
    country: str = 'United Kingdom',
    max_products: int = 80,
) -> pd.DataFrame:
    """
    Create a boolean invoice-product basket matrix.

    Preprocessing steps applied:
    - Filter by country if specified
    - Remove cancelled invoices (InvoiceNo starting with 'C')
    - Remove negative/zero quantity rows (returns)
    - Remove null descriptions
    - Filter out invoices with too many items (outliers)
    - Limit to top N products by total quantity sold for performance
    """
    work = df.copy()

    if country and country != 'All':
        work = work[work['Country'] == country]

    # Remove cancelled invoices
    work = work[~work['InvoiceNo'].astype(str).str.startswith('C')]

    # Remove returns (negative/zero quantity)
    work = work[work['Quantity'] > 0]

    # Remove null descriptions
    work = work.dropna(subset=['Description'])

    # Filter out invoices with too many items (outliers that distort rules)
    invoice_sizes = work.groupby('InvoiceNo').size()
    valid_invoices = invoice_sizes[invoice_sizes <= 30].index
    work = work[work['InvoiceNo'].isin(valid_invoices)]

    if work.empty:
        return pd.DataFrame()

    # Limit to top N products by total quantity sold
    top_products = (
        work.groupby('Description')['Quantity']
        .sum()
        .nlargest(max_products)
        .index
    )
    work = work[work['Description'].isin(top_products)]

    basket = (
        work.groupby(['InvoiceNo', 'Description'])['Quantity']
        .sum()
        .unstack(fill_value=0)
    )

    # Convert to bool — much lighter than int64
    return basket.gt(0).astype(bool)


@st.cache_data(show_spinner=False)
def generate_association_rules(
    basket: pd.DataFrame,
    min_support: float = 0.03,
    min_confidence: float = 0.3,
) -> pd.DataFrame:
    """
    Run the full association rule mining pipeline:
    1. Find frequent itemsets (items with support >= min_support)
    2. Generate candidate rules from frequent itemsets
    3. Calculate lift for each rule

    Lift interpretation:
    - Lift > 1: Items are positively correlated
    - Lift = 1: Items are independent
    - Lift < 1: Items are negatively correlated

    Returns a DataFrame of rules sorted by lift (descending) then confidence.
    """
    if basket.empty:
        return pd.DataFrame()

    # Step 1: Find frequent itemsets — low_memory=True for performance
    frequent_itemsets = apriori(
        basket,
        min_support=min_support,
        use_colnames=True,
        low_memory=True,
    )

    if frequent_itemsets.empty:
        return pd.DataFrame()

    # Step 2: Generate rules using confidence as primary metric
    rules = association_rules(
        frequent_itemsets,
        metric='confidence',
        min_threshold=min_confidence,
    )

    if rules.empty:
        return pd.DataFrame()

    # Keep only positive relationships (lift > 1)
    rules = rules[rules['lift'] > 1].copy()

    if rules.empty:
        return pd.DataFrame()

    # Convert frozensets to readable strings
    rules['antecedents'] = rules['antecedents'].apply(
        lambda x: ', '.join(sorted(list(x)))
    )
    rules['consequents'] = rules['consequents'].apply(
        lambda x: ', '.join(sorted(list(x)))
    )

    # Round for display
    for col in ['support', 'confidence', 'lift']:
        if col in rules.columns:
            rules[col] = rules[col].round(4)

    return rules.sort_values(
        ['lift', 'confidence'], ascending=False
    ).reset_index(drop=True)


def get_rules_summary(rules: pd.DataFrame) -> dict:
    """
    Compute summary statistics for discovered rules.
    'Strongest rule' is determined by highest lift value.
    """
    if rules.empty:
        return {}

    top_rule = rules.iloc[0]
    return {
        'total_rules': len(rules),
        'avg_lift': round(rules['lift'].mean(), 2),
        'avg_confidence': round(rules['confidence'].mean(), 2),
        'max_lift': round(rules['lift'].max(), 2),
        'strongest_rule_antecedent': top_rule['antecedents'],
        'strongest_rule_consequent': top_rule['consequents'],
    }


def interpret_rules(rules: pd.DataFrame, top_n: int = 5) -> list:
    """
    Generate business-friendly interpretations for the strongest rules.

    Lift tiers:
    - Lift > 10: extremely strong
    - Lift > 5:  strong bundle opportunity
    - Lift > 2:  moderate cross-sell opportunity
    - Lift >= 1: weak but real relationship
    - Lift < 1:  negative correlation
    """
    if rules is None or rules.empty:
        return ["No strong rules found. Try lowering the Minimum Support or Confidence thresholds."]

    insights = []
    for _, row in rules.head(top_n).iterrows():
        lift = row['lift']

        if lift > 10:
            strength = "🔥 Extremely strong relationship — very high cross-sell potential."
        elif lift > 5:
            strength = "💡 Strong bundle opportunity — consider packaging these together."
        elif lift > 2:
            strength = "📦 Moderate cross-sell opportunity — worth promoting together."
        elif lift >= 1:
            strength = "📌 Weak but real relationship — monitor for bundling potential."
        else:
            strength = "❌ Negative correlation — these items are bought less together than expected."

        insights.append(
            f"Customers who buy **{row['antecedents']}** also buy "
            f"**{row['consequents']}** "
            f"(Support={row['support']}, Confidence={row['confidence']}, "
            f"Lift={lift}). {strength}"
        )
    return insights


def plot_rules_scatter(
    rules: pd.DataFrame,
    min_support: float = 0.0,
    min_confidence: float = 0.0,
):
    """
    Scatter plot: Support (x) vs Confidence (y), bubble size = Lift.
    """
    if rules.empty:
        return None

    fig = px.scatter(
        rules.head(80),
        x='support', y='confidence',
        size='lift', color='lift',
        hover_data=['antecedents', 'consequents', 'lift'],
        title='Association Rules — Support vs Confidence vs Lift',
        template='plotly_white',
        color_continuous_scale='RdYlGn',
        height=500,
        labels={
            'support': 'Support (How Common)',
            'confidence': 'Confidence (Reliability)',
            'lift': 'Lift (Relationship Strength)',
        },
    )

    if min_support > 0:
        fig.add_vline(
            x=min_support,
            line_dash='dash',
            line_color='rgba(229, 57, 53, 0.7)',
            annotation_text=f'Min Support = {min_support}',
            annotation_position='top right',
        )
    if min_confidence > 0:
        fig.add_hline(
            y=min_confidence,
            line_dash='dash',
            line_color='rgba(56, 142, 60, 0.7)',
            annotation_text=f'Min Confidence = {min_confidence}',
            annotation_position='bottom right',
        )

    fig.update_layout(
        font=dict(family="Segoe UI, sans-serif"),
        paper_bgcolor='rgba(245,245,245,1)',
        plot_bgcolor='rgba(255,255,255,1)',
    )

    return fig


def plot_top_rules_bar(rules: pd.DataFrame, top_n: int = 15):
    """Horizontal bar chart of top rules ranked by Lift."""
    if rules.empty:
        return None

    top = rules.head(top_n).copy()
    top['rule'] = top['antecedents'] + ' → ' + top['consequents']

    fig = px.bar(
        top, x='lift', y='rule', orientation='h',
        color='confidence',
        title=f'Top {top_n} Product Relationships by Lift',
        template='plotly_white',
        color_continuous_scale='Blues',
        height=520,
        labels={
            'lift': 'Lift (Relationship Strength)',
            'confidence': 'Confidence (Reliability)',
        },
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    fig.update_layout(
        font=dict(family="Segoe UI, sans-serif"),
        paper_bgcolor='rgba(245,245,245,1)',
        plot_bgcolor='rgba(255,255,255,1)',
    )
    return fig