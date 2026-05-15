"""
Module 5: Transaction Anomaly Detection — Isolation Forest
Responsible Student: Bushra Hurani

Requirement covered: Anomaly Detection
"""

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler



# 1. Safe Column Validation


def _require_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns for anomaly detection: {missing}. "
            "Please make sure preprocessing has been applied correctly."
        )



# 2. Feature Engineering for Anomaly Detection

def build_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build invoice-level features for suspicious transaction detection.
    """

    _require_columns(
        df,
        [
            "InvoiceNo",
            "CustomerID",
            "StockCode",
            "Quantity",
            "UnitPrice",
            "TotalPrice",
            "InvoiceDate",
        ],
    )

    data = df.copy()
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], errors="coerce")
    data = data.dropna(subset=["InvoiceDate"])

    features = (
        data.groupby("InvoiceNo")
        .agg(
            TotalAmount=("TotalPrice", "sum"),
            UniqueProducts=("StockCode", "nunique"),
            TotalQuantity=("Quantity", "sum"),
            AvgUnitPrice=("UnitPrice", "mean"),
            CustomerID=("CustomerID", "first"),
            InvoiceDate=("InvoiceDate", "first"),
            Country=("Country", "first") if "Country" in data.columns else ("CustomerID", "first"),
        )
        .reset_index()
    )

    features["Hour"] = features["InvoiceDate"].dt.hour
    features["IsWeekend"] = features["InvoiceDate"].dt.dayofweek.isin([5, 6]).astype(int)

    customer_avg = features.groupby("CustomerID")["TotalAmount"].transform("mean")
    features["CustomerAvgAmount"] = customer_avg
    features["AmountDeviation"] = features["TotalAmount"] - customer_avg
    features["AmountRatioToCustomerAvg"] = (
        features["TotalAmount"] / customer_avg.replace(0, np.nan)
    )
    features["AmountRatioToCustomerAvg"] = (
        features["AmountRatioToCustomerAvg"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    numeric_cols = [
        "TotalAmount",
        "UniqueProducts",
        "TotalQuantity",
        "AvgUnitPrice",
        "CustomerAvgAmount",
        "AmountDeviation",
        "AmountRatioToCustomerAvg",
    ]

    for col in numeric_cols:
        features[col] = pd.to_numeric(features[col], errors="coerce").fillna(0)

    features["TotalAmount"] = features["TotalAmount"].round(2)
    features["AvgUnitPrice"] = features["AvgUnitPrice"].round(2)
    features["CustomerAvgAmount"] = features["CustomerAvgAmount"].round(2)
    features["AmountDeviation"] = features["AmountDeviation"].round(2)
    features["AmountRatioToCustomerAvg"] = features["AmountRatioToCustomerAvg"].round(2)

    return features


# 3. Isolation Forest Model

def detect_anomalies(
    features: pd.DataFrame,
    contamination: float = 0.05,
) -> pd.DataFrame:
    """
    Detect unusual invoices using Isolation Forest.

    contamination:
    expected proportion of anomalies in the dataset.
    """

    _require_columns(
        features,
        [
            "TotalAmount",
            "UniqueProducts",
            "TotalQuantity",
            "AvgUnitPrice",
            "Hour",
            "IsWeekend",
            "AmountDeviation",
            "AmountRatioToCustomerAvg",
        ],
    )

    features = features.copy()

    feature_cols = [
        "TotalAmount",
        "UniqueProducts",
        "TotalQuantity",
        "AvgUnitPrice",
        "Hour",
        "IsWeekend",
        "AmountDeviation",
        "AmountRatioToCustomerAvg",
    ]

    X = features[feature_cols].copy()

    skewed_cols = [
        "TotalAmount",
        "UniqueProducts",
        "TotalQuantity",
        "AvgUnitPrice",
        "AmountDeviation",
        "AmountRatioToCustomerAvg",
    ]

    for col in skewed_cols:
        X[col] = np.sign(X[col]) * np.log1p(np.abs(X[col]))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
    )

    labels = model.fit_predict(X_scaled)
    raw_scores = model.decision_function(X_scaled)

    features["AnomalyLabel"] = labels
    features["IsAnomaly"] = features["AnomalyLabel"] == -1
    features["AnomalyScoreRaw"] = raw_scores.round(4)

    return features


# 4. Summary + Business Insights

def get_anomaly_summary(features: pd.DataFrame) -> dict:
    if features.empty:
        return {
            "total_invoices": 0,
            "anomaly_invoices": 0,
            "anomaly_rate": 0,
            "highest_anomaly_amount": 0,
            "highest_anomaly_quantity": 0,
            "highest_anomaly_ratio": 0,
        }

    anomalies = features[features["IsAnomaly"]]

    return {
        "total_invoices": len(features),
        "anomaly_invoices": int(features["IsAnomaly"].sum()),
        "anomaly_rate": round(
            int(features["IsAnomaly"].sum()) / max(len(features), 1) * 100,
            2,
        ),
        "highest_anomaly_amount": round(float(anomalies["TotalAmount"].max()), 2)
        if not anomalies.empty else 0,
        "highest_anomaly_quantity": round(float(anomalies["TotalQuantity"].max()), 2)
        if not anomalies.empty else 0,
        "highest_anomaly_ratio": round(float(anomalies["AmountRatioToCustomerAvg"].max()), 2)
        if not anomalies.empty else 0,
    }


def explain_anomaly_detection() -> str:
    return (
        "Isolation Forest is an unsupervised anomaly detection algorithm. "
        "It identifies unusual invoices by isolating records that behave differently "
        "from normal purchase patterns. This is suitable for retail data because "
        "we usually do not have labeled fraud or anomaly examples."
    )


def build_anomaly_insights(features: pd.DataFrame) -> str:
    summary = get_anomaly_summary(features)

    return (
        "Anomaly Detection Insights\n"
        "--------------------------\n"
        f"- Total invoices analyzed: {summary['total_invoices']:,}\n"
        f"- Anomalous invoices detected: {summary['anomaly_invoices']:,} "
        f"({summary['anomaly_rate']}%)\n"
        f"- Highest anomalous invoice amount: {summary['highest_anomaly_amount']:,.2f}\n"
        f"- Highest anomalous purchased quantity: {summary['highest_anomaly_quantity']:,.2f}\n"
        f"- Highest ratio compared to customer average: {summary['highest_anomaly_ratio']:,.2f}\n\n"
        "Business Meaning:\n"
        "- Anomalies may indicate unusual bulk purchases, VIP/high-value orders, "
        "data-entry errors, or suspicious transactions.\n"
        "- These results should be reviewed by a business user before making final decisions."
    )


# 5. Tables

def get_top_anomalies(
    features: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    cols = [
        "InvoiceNo",
        "CustomerID",
        "Country",
        "TotalAmount",
        "UniqueProducts",
        "TotalQuantity",
        "AvgUnitPrice",
        "Hour",
        "AmountDeviation",
        "AmountRatioToCustomerAvg",
        "InvoiceDate",
        "AnomalyScoreRaw",
    ]

    available_cols = [col for col in cols if col in features.columns]

    return (
        features[features["IsAnomaly"]]
        .sort_values("AnomalyScoreRaw", ascending=True)
        .head(top_n)[available_cols]
        .reset_index(drop=True)
    )


# 6. Visualizations

def plot_anomaly_scatter(features: pd.DataFrame):
    features = features.copy()
    features["Type"] = features["IsAnomaly"].map(
        {True: "Unusual order", False: "Regular Order"}
    )

    fig = px.scatter(
        features,
        x="TotalQuantity",
        y="TotalAmount",
        color="Type",
        hover_data=[
            "InvoiceNo",
            "CustomerID",
            "Country",
            "UniqueProducts",
            "Hour",
            "AmountRatioToCustomerAvg",
        ],
        title="Unusual Order Detection — Smart Pattern Check",
        template="plotly_dark",
        opacity=0.65,
        height=480,
    )

    return fig


def plot_anomaly_score_distribution(features: pd.DataFrame):
    fig = px.histogram(
        features,
        x="AnomalyScoreRaw",
        color="IsAnomaly",
        nbins=50,
        title="Unusual Order Score Distribution",
        template="plotly_dark",
        labels={
            "AnomalyScoreRaw": "Unusual order Score",
            "IsAnomaly": "Is Unusual order",
        },
        height=380,
    )

    return fig


def plot_anomaly_by_hour(features: pd.DataFrame):
    hourly = (
        features.groupby("Hour")
        .agg(
            TotalInvoices=("InvoiceNo", "count"),
            Anomalies=("IsAnomaly", "sum"),
        )
        .reset_index()
    )

    hourly["AnomalyRate"] = (
        hourly["Anomalies"] / hourly["TotalInvoices"].replace(0, np.nan) * 100
    ).fillna(0)

    fig = px.bar(
        hourly,
        x="Hour",
        y="AnomalyRate",
        title="Anomaly Rate by Hour",
        template="plotly_dark",
        labels={"AnomalyRate": "Anomaly Rate (%)"},
        height=380,
    )

    return fig