"""
Module 6: Sales Trends, Heatmap, and Forecasting
Responsible Student: Bushra Hurani

Requirement covered: Time Series Analysis
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _require_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for time series: {missing}")


def daily_revenue(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df, ["InvoiceDate", "TotalPrice"])

    data = df.copy()
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], errors="coerce")
    data = data.dropna(subset=["InvoiceDate"])

    daily = (
        data.set_index("InvoiceDate")
        .resample("D")["TotalPrice"]
        .sum()
        .reset_index()
    )

    daily.columns = ["Date", "Revenue"]
    daily["Revenue_MA7"] = daily["Revenue"].rolling(7, min_periods=1).mean()
    daily["Revenue_MA30"] = daily["Revenue"].rolling(30, min_periods=1).mean()

    return daily


def monthly_revenue(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df, ["InvoiceDate", "TotalPrice"])

    data = df.copy()
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], errors="coerce")
    data = data.dropna(subset=["InvoiceDate"])

    monthly = (
        data.set_index("InvoiceDate")
        .resample("MS")["TotalPrice"]
        .sum()
        .reset_index()
    )

    monthly.columns = ["Month", "Revenue"]
    monthly["MonthName"] = monthly["Month"].dt.strftime("%Y-%m")

    return monthly


def sales_heatmap_data(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df, ["InvoiceDate", "InvoiceNo"])

    data = df.copy()
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], errors="coerce")
    data = data.dropna(subset=["InvoiceDate"])

    data["DayOfWeek"] = data["InvoiceDate"].dt.day_name()
    data["Hour"] = data["InvoiceDate"].dt.hour

    order = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"
    ]

    heat = data.pivot_table(
        index="DayOfWeek",
        columns="Hour",
        values="InvoiceNo",
        aggfunc="nunique",
        fill_value=0,
    )

    return heat.reindex(order).fillna(0)


def get_time_series_summary(df: pd.DataFrame) -> dict:
    daily = daily_revenue(df)
    monthly = monthly_revenue(df)

    if daily.empty:
        return {
            "total_revenue": 0,
            "average_daily_revenue": 0,
            "best_day": None,
            "best_day_revenue": 0,
            "best_month": None,
            "best_month_revenue": 0,
            "growth_rate": 0,
        }

    total_revenue = daily["Revenue"].sum()
    average_daily = daily["Revenue"].mean()

    best_day_row = daily.loc[daily["Revenue"].idxmax()]
    best_month_row = monthly.loc[monthly["Revenue"].idxmax()] if not monthly.empty else None

    if len(monthly) >= 2 and monthly["Revenue"].iloc[0] != 0:
        growth_rate = (
            (monthly["Revenue"].iloc[-1] - monthly["Revenue"].iloc[0])
            / monthly["Revenue"].iloc[0]
            * 100
        )
    else:
        growth_rate = 0

    return {
        "total_revenue": round(float(total_revenue), 2),
        "average_daily_revenue": round(float(average_daily), 2),
        "best_day": best_day_row["Date"],
        "best_day_revenue": round(float(best_day_row["Revenue"]), 2),
        "best_month": best_month_row["MonthName"] if best_month_row is not None else None,
        "best_month_revenue": round(float(best_month_row["Revenue"]), 2)
        if best_month_row is not None else 0,
        "growth_rate": round(float(growth_rate), 2),
    }


def _seasonal_naive_forecast(history: pd.DataFrame, periods: int) -> pd.DataFrame:
    last_date = history["Date"].max()
    future_dates = pd.date_range(
        last_date + pd.Timedelta(days=1),
        periods=periods,
        freq="D",
    )

    last_7 = history["Revenue"].tail(7).to_numpy()

    if len(last_7) == 0:
        base = np.zeros(periods)
    else:
        base = np.resize(last_7, periods)

    recent = history["Revenue"].tail(30).reset_index(drop=True)
    trend = 0.0

    if len(recent) > 2:
        x = np.arange(len(recent))
        trend = np.polyfit(x, recent.values, 1)[0]

    steps = np.arange(1, periods + 1)
    forecast = np.maximum(base + trend * steps, 0)

    residual_std = max(float(history["Revenue"].tail(60).std() or 0), 1.0)

    out = pd.DataFrame(
        {
            "Date": future_dates,
            "Forecast": forecast,
        }
    )

    out["Lower"] = np.maximum(out["Forecast"] - 1.96 * residual_std, 0)
    out["Upper"] = out["Forecast"] + 1.96 * residual_std
    out["Model"] = "Seasonal naive fallback"

    return out


def forecast_sales(daily: pd.DataFrame, periods: int = 30) -> pd.DataFrame:
    """
    Forecast daily revenue.

    Uses ARIMA(1,1,1) when statsmodels is available and enough data exists.
    Otherwise, falls back to a 7-day seasonal naive model with recent trend.
    """

    history = daily.copy().sort_values("Date")

    if history.empty:
        return pd.DataFrame()

    y = history.set_index("Date")["Revenue"].asfreq("D").fillna(0)

    if len(y) < 35:
        return _seasonal_naive_forecast(history, periods)

    try:
        from statsmodels.tsa.arima.model import ARIMA

        model = ARIMA(y, order=(1, 1, 1))
        fitted = model.fit()
        pred = fitted.get_forecast(steps=periods)

        mean = pred.predicted_mean
        ci = pred.conf_int(alpha=0.05)

        out = pd.DataFrame(
            {
                "Date": mean.index,
                "Forecast": np.maximum(mean.to_numpy(), 0),
                "Lower": np.maximum(ci.iloc[:, 0].to_numpy(), 0),
                "Upper": np.maximum(ci.iloc[:, 1].to_numpy(), 0),
            }
        )

        out["Model"] = "ARIMA(1,1,1)"
        return out

    except Exception:
        return _seasonal_naive_forecast(history, periods)


def build_time_series_insights(df: pd.DataFrame, forecast: pd.DataFrame | None = None) -> str:
    summary = get_time_series_summary(df)

    lines = [
        "Time Series Analysis Insights",
        "-----------------------------",
        f"- Total revenue over the selected period: {summary['total_revenue']:,.2f}.",
        f"- Average daily revenue: {summary['average_daily_revenue']:,.2f}.",
        f"- Best sales day: {summary['best_day']} with revenue {summary['best_day_revenue']:,.2f}.",
        f"- Best sales month: {summary['best_month']} with revenue {summary['best_month_revenue']:,.2f}.",
        f"- Revenue growth from first to last month: {summary['growth_rate']}%.",
    ]

    if forecast is not None and not forecast.empty:
        model_name = forecast["Model"].iloc[0] if "Model" in forecast.columns else "Forecast model"
        avg_forecast = forecast["Forecast"].mean()
        lines.extend(
            [
                "",
                "Forecasting Summary",
                "-------------------",
                f"- Forecasting model used: {model_name}.",
                f"- Average forecasted daily revenue: {avg_forecast:,.2f}.",
                "- Forecasts are estimates and should be used for planning, not as guaranteed values.",
            ]
        )

    return "\n".join(lines)


def plot_daily_trend(daily: pd.DataFrame):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=daily["Date"],
            y=daily["Revenue"],
            mode="lines",
            name="Daily Revenue",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=daily["Date"],
            y=daily["Revenue_MA7"],
            mode="lines",
            name="7-day Moving Average",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=daily["Date"],
            y=daily["Revenue_MA30"],
            mode="lines",
            name="30-day Moving Average",
        )
    )

    fig.update_layout(
        title="Daily Revenue Trend with Moving Averages",
        template="plotly_dark",
        height=480,
        xaxis_title="Date",
        yaxis_title="Revenue",
    )

    return fig


def plot_monthly_revenue(monthly: pd.DataFrame):
    fig = px.bar(
        monthly,
        x="MonthName",
        y="Revenue",
        title="Monthly Revenue Trend",
        template="plotly_dark",
        height=420,
    )

    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Revenue",
    )

    return fig


def plot_forecast(daily: pd.DataFrame, forecast: pd.DataFrame):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=daily["Date"],
            y=daily["Revenue"],
            mode="lines",
            name="Past Sales"        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast["Date"],
            y=forecast["Forecast"],
            mode="lines",
            name="Predicted Sales"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast["Date"],
            y=forecast["Upper"],
            mode="lines",
            name="Upper Expected Range",
            line=dict(width=0),
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast["Date"],
            y=forecast["Lower"],
            mode="lines",
           name="Lower Expected Range",
            fill="tonexty",
            line=dict(width=0),
            showlegend=True,
        )
    )

    model_name = (
        forecast["Model"].iloc[0]
        if "Model" in forecast.columns and not forecast.empty
        else "Forecast"
    )

    fig.update_layout(
       title="Future Sales Prediction",       
         template="plotly_dark",
        height=480,
        xaxis_title="Date",
        yaxis_title="Revenue",
    )

    return fig


def plot_heatmap(heat: pd.DataFrame):
    fig = px.imshow(
        heat,
        aspect="auto",
        color_continuous_scale="Viridis",
        template="plotly_dark",
        title="Purchase Activity Heatmap — Day of Week × Hour",
    )

    fig.update_layout(
        height=470,
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week",
    )

    return fig