"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 1.7.0
"""

import pandas as pd

ID_COLUMN = "Order Id"
TARGET_COLUMN = "Late_delivery_risk"
ORDER_DATE_COLUMN = "order date (DateOrders)"

# Unknown at order creation time, or pure identifiers.
EXCLUDED_COLUMNS = [
    "Late_delivery_risk",
    "Delivery Status",
    "shipping date (DateOrders)",
    "Days for shipping (real)",
    "delay_days",
    "Order Id",
    "Customer Id",
]

NUMERICAL_FEATURES = [
    "Days for shipment (scheduled)",
    "gross_sales",
    "net_sales",
    "total_quantity",
    "total_discount",
    "total_profit",
    "discount_rate",
    "profit_margin",
    "average_unit_value",
    "number_order_lines",
    "number_products",
    "number_categories",
]

CATEGORICAL_FEATURES = [
    "Shipping Mode",
    "Order Region",
    "Order Country",
    "Order State",
    "Order City",
    "Market",
    "Customer Segment",
]

DATE_FEATURES = ["order_year", "order_month", "order_day_of_week", "order_hour"]

FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + DATE_FEATURES

EXPECTED_ORDER_COUNT = 65_752


def build_order_features(df: pd.DataFrame) -> pd.DataFrame:
    order_date = pd.to_datetime(df[ORDER_DATE_COLUMN], format="%m/%d/%Y %H:%M")

    features = df[[ID_COLUMN, TARGET_COLUMN, *NUMERICAL_FEATURES, *CATEGORICAL_FEATURES]].copy()
    features["order_year"] = order_date.dt.year
    features["order_month"] = order_date.dt.month
    features["order_day_of_week"] = order_date.dt.dayofweek
    features["order_hour"] = order_date.dt.hour

    leaked = set(FEATURE_COLUMNS) & set(EXCLUDED_COLUMNS + [ORDER_DATE_COLUMN])
    assert len(features) == EXPECTED_ORDER_COUNT, f"Expected {EXPECTED_ORDER_COUNT} rows, got {len(features)}"
    assert features[ID_COLUMN].is_unique, "Order Id is not unique"
    assert set(features[TARGET_COLUMN].unique()) <= {0, 1}, "Late_delivery_risk not binary"
    assert not leaked, f"Leakage columns among features: {leaked}"
    assert set(features.columns) == {ID_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS}

    print("Feature columns:")
    for col in FEATURE_COLUMNS:
        print(f"  {col}")
    print("Shape:", features.shape)
    nulls = features.isna().sum()
    print("Null counts:")
    print(nulls[nulls > 0])

    return features
