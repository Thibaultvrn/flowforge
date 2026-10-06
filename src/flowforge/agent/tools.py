"""Deterministic metrics over the operational scored orders.

The operational dataset is the chronological test period scored by the
trained Logistic Regression baseline. The language model may call the
LangChain tools in this module. It never reads the parquet file and never
computes the metrics itself.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
from langchain.tools import tool

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_OPERATIONAL_PATH = (
    _PROJECT_ROOT / "data" / "07_model_output" / "operational_scored_orders.parquet"
)

ORDER_ID = "Order Id"
ACTUAL_LATE = "Late_delivery_risk"
PREDICTED_PROBABILITY = "predicted_late_probability"
NET_SALES = "net_sales"
GROSS_SALES = "gross_sales"
DELAY_DAYS = "delay_days"
EXPOSURE = "expected_revenue_exposure"
ORDER_REGION = "Order Region"
SHIPPING_MODE = "Shipping Mode"
CUSTOMER_SEGMENT = "Customer Segment"

HIGH_RISK_THRESHOLD = 0.7

_REQUIRED_COLUMNS = [
    ORDER_ID,
    ACTUAL_LATE,
    PREDICTED_PROBABILITY,
    GROSS_SALES,
    NET_SALES,
    DELAY_DAYS,
    EXPOSURE,
    ORDER_REGION,
    SHIPPING_MODE,
    CUSTOMER_SEGMENT,
]
_TEXT_COLUMNS = (ORDER_REGION, SHIPPING_MODE, CUSTOMER_SEGMENT)

_EXPOSURE_DEFINITION = (
    "expected_revenue_exposure = predicted_late_probability * net_sales. "
    "Prioritization proxy only. Not a financial loss, cost, or margin impact."
)


def _limit(n: int) -> int:
    try:
        count = int(n)
    except (TypeError, ValueError) as exc:
        raise ValueError("n must be a positive integer.") from exc
    if count < 1:
        raise ValueError("n must be a positive integer.")
    return count


def _rate(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _money(value: float) -> float:
    return round(float(value), 2)


def _dump(payload: dict) -> str:
    return json.dumps(payload, indent=2)


@lru_cache(maxsize=1)
def _load_scored_orders() -> pd.DataFrame:
    if not _OPERATIONAL_PATH.is_file():
        raise FileNotFoundError(
            f"Required dataset not found: {_OPERATIONAL_PATH}. Run `kedro run` first."
        )
    scored = pd.read_parquet(_OPERATIONAL_PATH)

    missing = [c for c in _REQUIRED_COLUMNS if c not in scored.columns]
    if missing:
        raise ValueError(f"operational_scored_orders is missing columns: {missing}")
    if not scored[ORDER_ID].is_unique:
        raise ValueError("operational_scored_orders contains duplicate Order Id values.")

    for column in _TEXT_COLUMNS:
        scored[column] = scored[column].str.strip()
    return scored


def get_scored_orders() -> pd.DataFrame:
    """Return the operational scored orders (chronological test period).

    The actual outcome is ``Late_delivery_risk``. ``delay_days`` is real
    shipping days minus scheduled shipping days.
    """
    return _load_scored_orders().copy()


def get_network_summary() -> dict:
    """Summarize the operational scored network."""
    scored = get_scored_orders()
    return {
        "operational_orders": int(len(scored)),
        "average_predicted_late_probability": _rate(scored[PREDICTED_PROBABILITY].mean()),
        "actual_late_delivery_rate": _rate(scored[ACTUAL_LATE].mean()),
        "total_gross_sales": _money(scored[GROSS_SALES].sum()),
        "total_net_sales": _money(scored[NET_SALES].sum()),
        "total_expected_revenue_exposure": _money(scored[EXPOSURE].sum()),
        "average_delay_days": _rate(scored[DELAY_DAYS].mean(), digits=4),
        "high_risk_orders": int(scored[PREDICTED_PROBABILITY].ge(HIGH_RISK_THRESHOLD).sum()),
        "high_risk_threshold": HIGH_RISK_THRESHOLD,
        "exposure_definition": _EXPOSURE_DEFINITION,
    }


def _aggregate(scored: pd.DataFrame, group_column: str) -> pd.DataFrame:
    frame = scored.assign(
        high_risk_orders=scored[PREDICTED_PROBABILITY].ge(HIGH_RISK_THRESHOLD)
    )
    return frame.groupby(group_column, as_index=False).agg(
        orders=(ORDER_ID, "size"),
        average_predicted_late_probability=(PREDICTED_PROBABILITY, "mean"),
        actual_late_rate=(ACTUAL_LATE, "mean"),
        average_delay_days=(DELAY_DAYS, "mean"),
        total_net_sales=(NET_SALES, "sum"),
        total_expected_revenue_exposure=(EXPOSURE, "sum"),
        high_risk_orders=("high_risk_orders", "sum"),
    )


def _metric_records(
    table: pd.DataFrame, label_column: str, *, include_delay: bool
) -> list[dict]:
    records: list[dict] = []
    for row in table.to_dict(orient="records"):
        record = {
            label_column: row[label_column],
            "orders": int(row["orders"]),
            "average_predicted_late_probability": _rate(
                row["average_predicted_late_probability"]
            ),
            "actual_late_rate": _rate(row["actual_late_rate"]),
        }
        if include_delay:
            record["average_delay_days"] = _rate(row["average_delay_days"], digits=4)
        record["total_net_sales"] = _money(row["total_net_sales"])
        record["total_expected_revenue_exposure"] = _money(
            row["total_expected_revenue_exposure"]
        )
        record["high_risk_orders"] = int(row["high_risk_orders"])
        records.append(record)
    return records


def get_risk_hotspots(n: int = 10) -> dict:
    """Return the Order Regions with the highest total expected revenue exposure."""
    count = _limit(n)
    table = _aggregate(get_scored_orders(), ORDER_REGION)
    table = table.sort_values(
        by=[
            "total_expected_revenue_exposure",
            "average_predicted_late_probability",
            "high_risk_orders",
            ORDER_REGION,
        ],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).head(count)
    regions = _metric_records(table, ORDER_REGION, include_delay=False)
    return {
        "returned": len(regions),
        "ranked_by": "total_expected_revenue_exposure",
        "high_risk_threshold": HIGH_RISK_THRESHOLD,
        "exposure_definition": _EXPOSURE_DEFINITION,
        "regions": regions,
    }


def get_priority_orders(n: int = 20) -> dict:
    """Return orders ranked by the precomputed ``expected_revenue_exposure``."""
    count = _limit(n)
    ranked = get_scored_orders().sort_values(
        by=[EXPOSURE, ORDER_ID],
        ascending=[False, True],
        kind="mergesort",
    ).head(count)

    orders: list[dict] = []
    for row in ranked.to_dict(orient="records"):
        orders.append(
            {
                ORDER_ID: int(row[ORDER_ID]),
                PREDICTED_PROBABILITY: _rate(row[PREDICTED_PROBABILITY]),
                NET_SALES: _money(row[NET_SALES]),
                EXPOSURE: _money(row[EXPOSURE]),
                ORDER_REGION: row[ORDER_REGION],
                SHIPPING_MODE: row[SHIPPING_MODE],
                CUSTOMER_SEGMENT: row[CUSTOMER_SEGMENT],
            }
        )
    return {
        "returned": len(orders),
        "ranked_by": EXPOSURE,
        "definition": _EXPOSURE_DEFINITION,
        "orders": orders,
    }


def compare_shipping_modes() -> dict:
    """Compare operational scored orders across Shipping Mode."""
    table = _aggregate(get_scored_orders(), SHIPPING_MODE)
    table = table.sort_values(
        by=["average_predicted_late_probability", SHIPPING_MODE],
        ascending=[False, True],
        kind="mergesort",
    )
    modes = _metric_records(table, SHIPPING_MODE, include_delay=True)
    return {
        "returned": len(modes),
        "ranked_by": "average_predicted_late_probability",
        "high_risk_threshold": HIGH_RISK_THRESHOLD,
        "exposure_definition": _EXPOSURE_DEFINITION,
        "shipping_modes": modes,
    }


@tool("network_summary")
def network_summary() -> str:
    """Return precomputed network metrics for the operational scored orders.

    The operational orders are the latest chronological period, scored by the
    trained model. Use this for overall context before a recommendation. The
    payload includes operational order count, average predicted late
    probability, actual late-delivery rate, total gross sales, total net sales,
    total expected_revenue_exposure, average delay in days (real shipping days
    minus scheduled days), and the count of orders with predicted probability
    >= 0.7. Quote these figures. Do not recompute them.
    """
    return _dump(get_network_summary())


@tool("risk_hotspots")
def risk_hotspots(n: int = 10) -> str:
    """Return Order Regions ranked by total expected_revenue_exposure.

    Use this to find geographic late-delivery hotspots. Each region includes
    order count, average predicted late probability, actual late rate, total
    net sales, total expected_revenue_exposure, and the count of orders with
    predicted probability >= 0.7. expected_revenue_exposure is a prioritization
    proxy, not financial loss. All metrics are precomputed. Do not recompute them.

    Args:
        n: How many regions to return, highest total exposure first.
    """
    return _dump(get_risk_hotspots(n))


@tool("priority_orders")
def priority_orders(n: int = 20) -> str:
    """Return operational orders ranked by expected_revenue_exposure.

    expected_revenue_exposure is predicted late probability times net sales.
    Use it only as a prioritization proxy for which orders to review first.
    It is not financial loss, cost, or margin impact. Each order includes
    Order Id, predicted probability, net sales, expected_revenue_exposure,
    Order Region, Shipping Mode, and Customer Segment. Figures are precomputed.

    Args:
        n: How many orders to return, highest exposure first.
    """
    return _dump(get_priority_orders(n))


@tool("compare_shipping_modes")
def compare_shipping_modes_tool() -> str:
    """Compare Shipping Mode values on the operational scored orders.

    Use this when the question is about shipping-mode performance. Each mode
    includes order count, average predicted late probability, actual late rate,
    average delay in days, total net sales, total expected_revenue_exposure,
    and the count of orders with predicted probability >= 0.7. All metrics are
    precomputed. Do not recompute them.
    """
    return _dump(compare_shipping_modes())


DECISION_TOOLS = [
    network_summary,
    risk_hotspots,
    priority_orders,
    compare_shipping_modes_tool,
]
