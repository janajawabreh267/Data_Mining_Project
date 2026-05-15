"""
Module 1: Flexible Data Loading, Cleaning & Feature Engineering
Responsible Student: Abd Alkareem Shafie
ID: 12241203

Purpose:
- Load CSV/XLSX/ZIP retail and e-commerce datasets.
- Support multiple datasets from the same business domain.
- Standardize different column names into one common schema.
- Handle special datasets such as Olist multi-file ZIP and customer-summary datasets.
- Clean noisy records using business-aware rules.
- Engineer features for dashboard, clustering, recommender, association rules,
  time series, and anomaly detection.

Supported final schema:
- InvoiceNo
- StockCode
- Description
- Quantity
- InvoiceDate
- UnitPrice
- TotalPrice
- CustomerID
- Country
"""

import zipfile
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# 1. Configuration
# ============================================================

MIN_PRODUCT_FREQ = 5
OUTLIER_LOWER_Q = 0.01
OUTLIER_UPPER_Q = 0.99

NON_PRODUCT_STOCKCODES = {
    "POST", "DOT", "M", "D", "S", "C2", "BANK CHARGES",
    "AMAZONFEE", "CRUK", "B", "PADS"
}

COLUMN_ALIASES = {
    "InvoiceNo": [
        "InvoiceNo", "Invoice", "InvoiceID", "Invoice No", "Invoice_No",
        "OrderID", "Order ID", "Order_ID", "TransactionID", "Transaction ID",
        "Transaction ID", "BillNo", "Bill No", "order_id"
    ],
    "StockCode": [
        "StockCode", "Stock Code", "ProductID", "Product ID", "Product_ID",
        "ProductCode", "Product Code", "SKU", "ItemID", "Item ID",
        "ItemCode", "Item Code", "product_id", "Product ID"
    ],
    "Description": [
        "Description", "ProductName", "Product Name", "Product_Name",
        "ItemName", "Item Name", "Item_Name", "Product", "Item",
        "ProductDescription", "Product Description", "Product Category",
        "Category", "Sub-Category", "product_category_name",
        "product_category_name_english", "Membership Type"
    ],
    "Quantity": [
        "Quantity", "Qty", "QTY", "Units", "UnitsSold", "Units Sold",
        "QuantityOrdered", "Quantity Ordered", "OrderQuantity",
        "Order Quantity", "Items Purchased"
    ],
    "InvoiceDate": [
        "InvoiceDate", "Invoice Date", "Invoice_Date", "OrderDate",
        "Order Date", "Order_Date", "TransactionDate", "Transaction Date",
        "PurchaseDate", "Purchase Date", "Date", "order_purchase_timestamp"
    ],
    "UnitPrice": [
        "UnitPrice", "Unit Price", "Unit_Price", "Price", "ItemPrice",
        "Item Price", "SalesPrice", "Sales Price", "UnitCost", "Unit Cost",
        "Price per Unit", "price"
    ],
    "TotalPrice": [
        "TotalPrice", "Total Price", "Total_Price", "Total Amount",
        "TotalAmount", "Sales", "Revenue", "Amount", "Total Spend"
    ],
    "CustomerID": [
        "CustomerID", "Customer ID", "Customer_ID", "UserID", "User ID",
        "User_ID", "ClientID", "Client ID", "Client_ID", "Customer",
        "customer_id", "customer_unique_id"
    ],
    "Country": [
        "Country", "Region", "Location", "Market", "CustomerCountry",
        "Customer Country", "ShipCountry", "Ship Country", "City",
        "customer_city", "customer_state"
    ],
}

CORE_REQUIRED_COLUMNS = [
    "Description",
    "Quantity",
    "InvoiceDate",
]


# ============================================================
# 2. Data Loading
# ============================================================

@st.cache_data(show_spinner=False)
def load_data(uploaded_file) -> pd.DataFrame:
    """
    Load CSV/XLSX/ZIP datasets.

    ZIP support:
    - Olist multi-file dataset is automatically merged.
    - ZIP containing one CSV/XLSX is loaded directly.
    """
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file, encoding="ISO-8859-1")

    if file_name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file, engine="openpyxl", sheet_name=0)

    if file_name.endswith(".zip"):
        uploaded_file.seek(0)
        return load_zip_dataset(uploaded_file)

    raise ValueError("Unsupported file type. Please upload CSV, Excel, or ZIP.")


def load_zip_dataset(uploaded_file) -> pd.DataFrame:
    """
    Load a ZIP dataset.

    If Olist files are detected, merge them into transaction-level records.
    Otherwise, read the first CSV/XLSX file found inside the ZIP.
    """
    with zipfile.ZipFile(uploaded_file) as z:
        names = z.namelist()
        lower_names = [n.lower() for n in names]

        is_olist = any("olist_orders_dataset" in n for n in lower_names) and \
                   any("olist_order_items_dataset" in n for n in lower_names)

        if is_olist:
            return _load_olist_from_zip(z)

        for name in names:
            if name.lower().endswith(".csv"):
                with z.open(name) as f:
                    return pd.read_csv(f, encoding="ISO-8859-1")
            if name.lower().endswith((".xlsx", ".xls")):
                with z.open(name) as f:
                    return pd.read_excel(BytesIO(f.read()), engine="openpyxl", sheet_name=0)

    raise ValueError("No readable CSV/XLSX file found inside ZIP.")


def _read_zip_csv(z: zipfile.ZipFile, contains_name: str) -> pd.DataFrame:
    """Read a CSV from ZIP by partial filename."""
    for name in z.namelist():
        if contains_name.lower() in name.lower():
            with z.open(name) as f:
                return pd.read_csv(f, encoding="ISO-8859-1")
    return pd.DataFrame()


def _load_olist_from_zip(z: zipfile.ZipFile) -> pd.DataFrame:
    """
    Convert Olist multi-file dataset into the standard transaction-level format.

    Files used:
    - olist_orders_dataset.csv
    - olist_order_items_dataset.csv
    - olist_customers_dataset.csv
    - olist_products_dataset.csv
    - product_category_name_translation.csv
    """
    orders = _read_zip_csv(z, "olist_orders_dataset")
    items = _read_zip_csv(z, "olist_order_items_dataset")
    customers = _read_zip_csv(z, "olist_customers_dataset")
    products = _read_zip_csv(z, "olist_products_dataset")
    translation = _read_zip_csv(z, "product_category_name_translation")

    if orders.empty or items.empty:
        raise ValueError("Olist ZIP is missing required order/order_items files.")

    df = items.merge(
        orders[["order_id", "customer_id", "order_purchase_timestamp"]],
        on="order_id",
        how="left",
    )

    if not customers.empty:
        customer_cols = ["customer_id"]
        if "customer_unique_id" in customers.columns:
            customer_cols.append("customer_unique_id")
        if "customer_state" in customers.columns:
            customer_cols.append("customer_state")
        if "customer_city" in customers.columns:
            customer_cols.append("customer_city")

        df = df.merge(customers[customer_cols], on="customer_id", how="left")

    if not products.empty:
        product_cols = ["product_id"]
        if "product_category_name" in products.columns:
            product_cols.append("product_category_name")

        df = df.merge(products[product_cols], on="product_id", how="left")

    if (
        not translation.empty
        and "product_category_name" in df.columns
        and "product_category_name" in translation.columns
    ):
        try:
            df = df.merge(
                translation,
                on="product_category_name",
                how="left",
            )
        except Exception:
            # Some Olist ZIP versions contain incompatible translation files.
            # The dashboard can still work using product_id or product_category_name.
            pass

    description_col = "product_id"
    if "product_category_name_english" in df.columns:
        description_col = "product_category_name_english"
    elif "product_category_name" in df.columns:
        description_col = "product_category_name"

    # Convert Olist relational tables into the same transaction schema used by the dashboard.
    # This makes Olist compatible with clustering, recommender, association rules,
    # time-series analysis, and anomaly detection without changing the rest of the app.
    standard = pd.DataFrame()
    standard["InvoiceNo"] = df["order_id"]
    standard["StockCode"] = df["product_id"]
    standard["Description"] = df[description_col].fillna(df["product_id"])
    standard["Quantity"] = 1
    standard["InvoiceDate"] = df["order_purchase_timestamp"]
    standard["UnitPrice"] = df["price"]
    standard["CustomerID"] = (
        df["customer_unique_id"] if "customer_unique_id" in df.columns else df["customer_id"]
    )
    standard["Country"] = (
        "Brazil-" + df["customer_state"].astype(str)
        if "customer_state" in df.columns else "Brazil"
    )

    return standard


# ============================================================
# 3. Flexible Column Standardization
# ============================================================

def _normalize_col_name(name: str) -> str:
    """Normalize column names for flexible matching."""
    return (
        str(name)
        .strip()
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )


def standardize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Map dataset-specific column names into standard schema.

    Returns:
    - dataframe with standardized columns
    - mapping report
    """
    df = df.copy()

    normalized_existing = {
        _normalize_col_name(col): col for col in df.columns
    }

    rename_dict = {}
    detected_mapping = {}

    for standard_col, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            normalized_alias = _normalize_col_name(alias)
            if normalized_alias in normalized_existing:
                original_col = normalized_existing[normalized_alias]
                if standard_col not in df.columns:
                    rename_dict[original_col] = standard_col
                    detected_mapping[standard_col] = original_col
                break

    df.rename(columns=rename_dict, inplace=True)

    return df, {
        "detected_mapping": detected_mapping,
        "standardized_columns": list(df.columns),
    }


def _is_customer_summary_dataset(df: pd.DataFrame) -> bool:
    """
    Detect customer-summary datasets.

    Example:
    E-commerce Customer Behavior dataset has:
    Customer ID, Total Spend, Items Purchased, Days Since Last Purchase,
    but no invoice/order-level transaction rows.
    """
    cols = {_normalize_col_name(c) for c in df.columns}

    has_customer = "customerid" in cols or "customer" in cols
    has_total_spend = "totalspend" in cols
    has_items = "itemspurchased" in cols
    lacks_order_date = not any(c in cols for c in ["invoicedate", "orderdate", "date"])

    return has_customer and has_total_spend and has_items and lacks_order_date


def _convert_customer_summary_to_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert customer-summary dataset into pseudo transaction-level rows.

    This allows the dashboard to run on customer behavior datasets.
    Note:
    - Association rules and recommender will be less meaningful because there are
      no real baskets.
    - Clustering and business overview still work.
    """
    df, _ = standardize_columns(df.copy())

    # Some public e-commerce datasets are already aggregated per customer.
    # Instead of rejecting them, we create safe pseudo-transactions so the
    # dashboard remains flexible. The app/report should still mention that
    # basket/recommender outputs are less meaningful in summary mode.
    result = pd.DataFrame()

    result["CustomerID"] = df["CustomerID"].astype(str)
    result["InvoiceNo"] = "SUMMARY_" + result["CustomerID"]
    result["Description"] = (
        df["Description"].astype(str)
        if "Description" in df.columns else "CUSTOMER SUMMARY"
    )
    result["StockCode"] = result["Description"]
    result["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(1)

    total = pd.to_numeric(df["TotalPrice"], errors="coerce").fillna(0)
    qty = result["Quantity"].replace(0, np.nan)
    result["UnitPrice"] = (total / qty).fillna(total)

    if "Country" in df.columns:
        result["Country"] = df["Country"].astype(str)
    else:
        result["Country"] = "Unknown"

    if "Days Since Last Purchase" in df.columns:
        days = pd.to_numeric(df["Days Since Last Purchase"], errors="coerce").fillna(0)
    elif "DaysSinceLastPurchase" in df.columns:
        days = pd.to_numeric(df["DaysSinceLastPurchase"], errors="coerce").fillna(0)
    else:
        days = 0

    today = pd.Timestamp.today().normalize()
    result["InvoiceDate"] = today - pd.to_timedelta(days, unit="D")

    return result


def validate_required_columns(df: pd.DataFrame) -> None:
    """
    Validate minimum required columns.

    Price requirement is flexible:
    - Either UnitPrice exists
    - Or TotalPrice exists and UnitPrice can be derived from Quantity
    """
    missing = [col for col in CORE_REQUIRED_COLUMNS if col not in df.columns]

    has_price = "UnitPrice" in df.columns or "TotalPrice" in df.columns

    if missing or not has_price:
        if not has_price:
            missing.append("UnitPrice or TotalPrice")

        raise ValueError(
            "Missing required columns after preprocessing column mapping: "
            f"{missing}. Please upload a retail/e-commerce dataset containing "
            "order ID, product, quantity, date, and price/sales amount."
        )


# ============================================================
# 4. Main Preprocessing Pipeline
# ============================================================

def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Full flexible cleaning and feature engineering pipeline.

    Returns:
    - cleaned dataframe
    - report dictionary explaining preprocessing actions
    """
    df = df.copy()
    report = {}

    report["original_rows"] = len(df)
    report["original_columns"] = list(df.columns)
    report["summary_dataset_mode"] = False

    if _is_customer_summary_dataset(df):
        df = _convert_customer_summary_to_transactions(df)
        report["summary_dataset_mode"] = True

    # Step 1: Standardize columns.
    df, mapping_report = standardize_columns(df)
    report.update(mapping_report)

    # Step 2: Validate columns.
    validate_required_columns(df)

    # Step 3: Fallback optional columns.
    if "InvoiceNo" not in df.columns:
        df["InvoiceNo"] = ["ORDER_" + str(i) for i in range(len(df))]
        report["generated_invoiceno"] = True
    else:
        report["generated_invoiceno"] = False

    if "CustomerID" not in df.columns:
        df["CustomerID"] = ["CUSTOMER_" + str(i) for i in range(len(df))]
        report["generated_customerid"] = True
    else:
        report["generated_customerid"] = False

    if "StockCode" not in df.columns:
        df["StockCode"] = df["Description"].astype(str)
        report["generated_stockcode"] = True
    else:
        report["generated_stockcode"] = False

    if "Country" not in df.columns:
        df["Country"] = "Unknown"
        report["generated_country"] = True
    else:
        report["generated_country"] = False

    # Step 4: Drop duplicates.
    before = len(df)
    df.drop_duplicates(inplace=True)
    report["dropped_duplicates"] = before - len(df)

    # Step 5: Clean Description.
    before = len(df)
    df = df[df["Description"].notna()]
    df["Description"] = df["Description"].astype(str).str.strip().str.upper()
    df = df[(df["Description"] != "") & (df["Description"] != "NAN")]
    report["dropped_missing_description"] = before - len(df)

    # Step 6: Clean CustomerID.
    before = len(df)
    df = df[df["CustomerID"].notna()]
    df["CustomerID"] = df["CustomerID"].astype(str).str.strip()
    df["CustomerID"] = df["CustomerID"].str.replace(r"\.0$", "", regex=True)
    df = df[(df["CustomerID"] != "") & (df["CustomerID"].str.lower() != "nan")]
    report["dropped_missing_customerid"] = before - len(df)

    # Step 7: Clean InvoiceNo and remove cancellations.
    before = len(df)
    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
    df = df[~df["InvoiceNo"].str.upper().str.startswith("C")]
    df = df[df["InvoiceNo"].str.len() > 0]
    report["dropped_cancelled_or_invalid_invoices"] = before - len(df)

    # Step 8: Clean StockCode.
    before = len(df)
    df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()
    df = df[~df["StockCode"].isin(NON_PRODUCT_STOCKCODES)]
    report["dropped_non_product_stockcodes"] = before - len(df)

    # Step 9: Numeric conversion.
    before = len(df)
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")

    if "UnitPrice" in df.columns:
        df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")

    if "TotalPrice" in df.columns:
        df["TotalPrice"] = pd.to_numeric(df["TotalPrice"], errors="coerce")

    df.dropna(subset=["Quantity"], inplace=True)
    report["dropped_non_numeric_quantity"] = before - len(df)

    # Step 10: Derive UnitPrice or TotalPrice when needed.
    if "UnitPrice" not in df.columns and "TotalPrice" in df.columns:
        df["UnitPrice"] = df["TotalPrice"] / df["Quantity"].replace(0, np.nan)
        report["derived_unitprice_from_totalprice"] = True
    else:
        report["derived_unitprice_from_totalprice"] = False

    if "TotalPrice" not in df.columns and "UnitPrice" in df.columns:
        df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
        report["derived_totalprice_from_unitprice"] = True
    else:
        report["derived_totalprice_from_unitprice"] = False

    before = len(df)
    df.dropna(subset=["UnitPrice", "TotalPrice"], inplace=True)
    report["dropped_non_numeric_price"] = before - len(df)

    # Step 11: Remove invalid values.
    before = len(df)
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0) & (df["TotalPrice"] > 0)]
    report["dropped_invalid_quantity_or_price"] = before - len(df)

    # Step 12: Parse dates.
    before = len(df)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df.dropna(subset=["InvoiceDate"], inplace=True)
    report["dropped_invalid_dates"] = before - len(df)

    # Step 13: Round financial values.
    df["UnitPrice"] = df["UnitPrice"].round(2)
    df["TotalPrice"] = df["TotalPrice"].round(2)

    # Step 14: Outlier clipping.
    # We clip instead of dropping to reduce the extreme influence of very large
    # transactions while preserving useful records for business analysis.
    report["outlier_clipping_applied"] = False
    report["outlier_method"] = "Quantile clipping on TotalPrice using 1% and 99% caps"

    if len(df) > 100:
        q_low = df["TotalPrice"].quantile(OUTLIER_LOWER_Q)
        q_high = df["TotalPrice"].quantile(OUTLIER_UPPER_Q)

        df["TotalPrice"] = df["TotalPrice"].clip(lower=q_low, upper=q_high)

        report["outlier_clipping_applied"] = True
        report["totalprice_clip_lower"] = round(float(q_low), 2)
        report["totalprice_clip_upper"] = round(float(q_high), 2)
    else:
        report["totalprice_clip_lower"] = None
        report["totalprice_clip_upper"] = None

    # Step 15: Rare product filtering.
    # The threshold is adaptive: small datasets keep rare items, while larger
    # datasets remove extremely rare products to improve basket/recommender stability.
    before = len(df)
    product_counts = df["Description"].value_counts()

    if len(df) < 1000:
        min_freq = 1
    elif len(df) < 10000:
        min_freq = min(3, MIN_PRODUCT_FREQ)
    else:
        min_freq = MIN_PRODUCT_FREQ

    valid_products = product_counts[product_counts >= min_freq].index
    df = df[df["Description"].isin(valid_products)]

    report["rare_product_min_frequency"] = min_freq
    report["dropped_rare_products"] = before - len(df)

    # Step 16: Clean Country.
    df["Country"] = (
        df["Country"]
        .astype(str)
        .str.strip()
        .replace({"": "Unknown", "nan": "Unknown", "NaN": "Unknown"})
    )

    # Step 17: Time features.
    df["Month"] = df["InvoiceDate"].dt.month
    df["MonthName"] = df["InvoiceDate"].dt.month_name()
    df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()
    df["DayOfWeekNum"] = df["InvoiceDate"].dt.dayofweek
    df["Hour"] = df["InvoiceDate"].dt.hour
    df["IsWeekend"] = df["InvoiceDate"].dt.dayofweek.isin([5, 6]).astype(int)
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)

    # Step 18: Business preprocessing insights.
    # These summary values help the dashboard/report explain the cleaned dataset
    # in business terms, not only technical cleaning numbers.
    if len(df):
        country_revenue = df.groupby("Country")["TotalPrice"].sum().sort_values(ascending=False)
        product_revenue = df.groupby("Description")["TotalPrice"].sum().sort_values(ascending=False)

        report["top_revenue_country"] = str(country_revenue.index[0]) if not country_revenue.empty else "Unknown"
        report["top_revenue_country_revenue"] = round(float(country_revenue.iloc[0]), 2) if not country_revenue.empty else 0
        report["top_revenue_product"] = str(product_revenue.index[0]) if not product_revenue.empty else "Unknown"
        report["top_revenue_product_revenue"] = round(float(product_revenue.iloc[0]), 2) if not product_revenue.empty else 0
        report["average_order_value"] = round(
            float(df["TotalPrice"].sum() / max(df["InvoiceNo"].nunique(), 1)),
            2,
        )
    else:
        report["top_revenue_country"] = "Unknown"
        report["top_revenue_country_revenue"] = 0
        report["top_revenue_product"] = "Unknown"
        report["top_revenue_product_revenue"] = 0
        report["average_order_value"] = 0

    # Step 19: Final metadata.
    report["final_rows"] = len(df)
    report["final_columns"] = list(df.columns)
    report["total_dropped"] = report["original_rows"] - report["final_rows"]
    report["retention_rate"] = (
        round(report["final_rows"] / report["original_rows"] * 100, 1)
        if report["original_rows"] else 0
    )
    report["date_min"] = df["InvoiceDate"].min() if len(df) else None
    report["date_max"] = df["InvoiceDate"].max() if len(df) else None
    report["unique_customers"] = df["CustomerID"].nunique() if len(df) else 0
    report["unique_products"] = df["StockCode"].nunique() if len(df) else 0
    report["unique_orders"] = df["InvoiceNo"].nunique() if len(df) else 0

    df.reset_index(drop=True, inplace=True)
    return df, report


# ============================================================
# 5. Dashboard Summary Statistics
# ============================================================

def get_summary_stats(df: pd.DataFrame) -> dict:
    """Return high-level business statistics for dashboard KPIs."""
    if df.empty:
        return {
            "total_records": 0,
            "total_customers": 0,
            "total_products": 0,
            "total_revenue": 0,
            "total_orders": 0,
            "date_range": (None, None),
            "top_countries": pd.Series(dtype=float),
        }

    return {
        "total_records": len(df),
        "total_customers": df["CustomerID"].nunique(),
        "total_products": df["StockCode"].nunique(),
        "total_revenue": df["TotalPrice"].sum(),
        "total_orders": df["InvoiceNo"].nunique(),
        "date_range": (df["InvoiceDate"].min(), df["InvoiceDate"].max()),
        "top_countries": (
            df.groupby("Country")["TotalPrice"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        ),
    }


# ============================================================
# 6. Preprocessing Report Text
# ============================================================

def build_preprocessing_report_text(report: dict) -> str:
    """Convert preprocessing report dictionary into readable text."""
    lines = [
        "Preprocessing Summary",
        "---------------------",
        f"Original rows: {report.get('original_rows', 0):,}",
        f"Final rows: {report.get('final_rows', 0):,}",
        f"Total dropped: {report.get('total_dropped', 0):,}",
        f"Retention rate: {report.get('retention_rate', 0)}%",
        "",
        "Dataset Mode",
        "------------",
        f"Customer-summary mode: {report.get('summary_dataset_mode', False)}",
        "",
        "Column Mapping",
        "--------------",
    ]

    detected = report.get("detected_mapping", {})
    if detected:
        for standard_col, original_col in detected.items():
            lines.append(f"- {original_col} → {standard_col}")
    else:
        lines.append("- No column mapping was needed or detected.")

    lines.extend([
        "",
        "Business Preprocessing Insights",
        "-------------------------------",
        f"- Top revenue country after cleaning: {report.get('top_revenue_country', 'Unknown')} "
        f"({report.get('top_revenue_country_revenue', 0):,.2f})",
        f"- Top revenue product after cleaning: {report.get('top_revenue_product', 'Unknown')} "
        f"({report.get('top_revenue_product_revenue', 0):,.2f})",
        f"- Average order value after cleaning: {report.get('average_order_value', 0):,.2f}",
        "",
        "Cleaning Actions",
        "----------------",
        f"- Dropped duplicates: {report.get('dropped_duplicates', 0):,}",
        f"- Dropped missing descriptions: {report.get('dropped_missing_description', 0):,}",
        f"- Dropped missing CustomerID: {report.get('dropped_missing_customerid', 0):,}",
        f"- Dropped cancelled/invalid invoices: {report.get('dropped_cancelled_or_invalid_invoices', 0):,}",
        f"- Dropped non-product stockcodes: {report.get('dropped_non_product_stockcodes', 0):,}",
        f"- Dropped invalid quantity/price: {report.get('dropped_invalid_quantity_or_price', 0):,}",
        f"- Dropped invalid dates: {report.get('dropped_invalid_dates', 0):,}",
        f"- Dropped rare products: {report.get('dropped_rare_products', 0):,}",
        "",
        "Feature Engineering",
        "-------------------",
        "- TotalPrice = Quantity × UnitPrice or loaded directly from Sales/Amount column",
        "- Month, MonthName, DayOfWeek, DayOfWeekNum, Hour, IsWeekend, YearMonth",
    ])

    if report.get("generated_invoiceno", False):
        lines.append("- InvoiceNo was missing and generated automatically.")

    if report.get("derived_unitprice_from_totalprice", False):
        lines.append("- UnitPrice was derived from TotalPrice / Quantity.")

    if report.get("derived_totalprice_from_unitprice", False):
        lines.append("- TotalPrice was derived from Quantity × UnitPrice.")

    if report.get("outlier_clipping_applied", False):
        lines.extend([
            "",
            "Outlier Handling",
            "----------------",
            f"- Method: {report.get('outlier_method', 'Quantile clipping')}",
            "- TotalPrice was clipped using 1% and 99% quantiles.",
            f"- Lower cap: {report.get('totalprice_clip_lower')}",
            f"- Upper cap: {report.get('totalprice_clip_upper')}",
        ])

    if report.get("generated_customerid", False):
        lines.append("- CustomerID was missing and generated automatically.")

    if report.get("generated_stockcode", False):
        lines.append("- StockCode was missing and generated from Description.")

    if report.get("generated_country", False):
        lines.append("- Country was missing and filled as Unknown.")

    return "\n".join(lines)