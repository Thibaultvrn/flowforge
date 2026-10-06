"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 1.7.0
"""

import pandas as pd


def inspect_orders(df: pd.DataFrame) -> pd.DataFrame:
    print("Shape:", df.shape)
    print("Unique Order Id:", df["Order Id"].nunique())
    print("Late_delivery_risk distribution:")
    print(df["Late_delivery_risk"].value_counts(normalize=True))
    print("Rows per Order Id:")
    print(df.groupby("Order Id").size().describe())
    return df


ORDER_LEVEL_COLUMNS = [
    "Late_delivery_risk",
    "Delivery Status",
    "Shipping Mode",
    "order date (DateOrders)",
    "shipping date (DateOrders)",
    "Days for shipment (scheduled)",
    "Days for shipping (real)",
    "Order Region",
    "Order Country",
    "Order State",
    "Market",
    "Customer Id",
    "Customer Segment",
]


def audit_order_consistency(df: pd.DataFrame) -> pd.DataFrame:
    n_orders = df["Order Id"].nunique()
    nunique = df.groupby("Order Id")[ORDER_LEVEL_COLUMNS].nunique(dropna=False)
    inconsistent = (nunique > 1).sum()

    report = pd.DataFrame(
        {
            "column": inconsistent.index,
            "inconsistent_orders": inconsistent.values,
            "inconsistent_percentage": (inconsistent.values / n_orders * 100).round(4),
        }
    )

    print("Order-level consistency:")
    print(report.to_string(index=False))

    nulls = df.isna().sum()
    print("Null values per column:")
    print(nulls[nulls > 0])
    print("Duplicated rows:", df.duplicated().sum())
    print("Unique Order Id:", n_orders)

    return report


ITEM_LEVEL_COLUMNS = [
    "Sales",
    "Sales per customer",
    "Order Item Total",
    "Order Item Quantity",
    "Order Item Product Price",
    "Product Price",
    "Order Item Discount",
    "Order Item Discount Rate",
    "Benefit per order",
    "Order Profit Per Order",
    "Order Item Profit Ratio",
    "Product Card Id",
    "Product Name",
    "Category Id",
    "Category Name",
]


def audit_item_level_columns(df: pd.DataFrame) -> pd.DataFrame:
    nunique = df.groupby("Order Id")[ITEM_LEVEL_COLUMNS].nunique(dropna=False)
    varies = (nunique > 1).sum()

    report = pd.DataFrame(
        {
            "column": ITEM_LEVEL_COLUMNS,
            "orders_with_multiple_values": varies.values,
            "percentage_with_multiple_values": (varies.values / len(nunique) * 100).round(4),
            "mean_unique_values_per_order": nunique.mean().round(4).values,
            "max_unique_values_per_order": nunique.max().values,
        }
    )

    print("Item-level column variation across orders:")
    print(report.to_string(index=False))

    sizes = df.groupby("Order Id").size()
    example_ids = sizes[sizes > 1].index[:5]
    with pd.option_context("display.max_columns", None, "display.width", 250):
        for order_id in example_ids:
            print(f"\nOrder Id {order_id}:")
            print(df.loc[df["Order Id"] == order_id, ITEM_LEVEL_COLUMNS].to_string(index=False))

    return report


ORDER_FIRST_COLUMNS = [
    "Late_delivery_risk",
    "Delivery Status",
    "Shipping Mode",
    "order date (DateOrders)",
    "shipping date (DateOrders)",
    "Days for shipment (scheduled)",
    "Days for shipping (real)",
    "Order Region",
    "Order Country",
    "Order State",
    "Order City",
    "Market",
    "Customer Id",
    "Customer Segment",
]

EXPECTED_ORDER_COUNT = 65_752


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.where(denominator != 0)).fillna(0.0)


# Visualization only; must never enter the predictive feature set.
COORDINATE_COLUMNS = ["Latitude", "Longitude"]


def _validate_constant_coordinates(df: pd.DataFrame) -> None:
    distinct = df.groupby("Order Id")[COORDINATE_COLUMNS].nunique(dropna=False)
    inconsistent = distinct[(distinct > 1).any(axis=1)]
    if not inconsistent.empty:
        raise ValueError(
            f"{len(inconsistent)} Order Ids have multiple distinct coordinates, "
            f"e.g. {inconsistent.index[:5].tolist()}"
        )


def aggregate_orders(df: pd.DataFrame) -> pd.DataFrame:
    _validate_constant_coordinates(df)

    orders = (
        df.groupby("Order Id", sort=True)
        .agg(
            **{col: (col, "first") for col in ORDER_FIRST_COLUMNS + COORDINATE_COLUMNS},
            gross_sales=("Sales", "sum"),
            net_sales=("Order Item Total", "sum"),
            total_quantity=("Order Item Quantity", "sum"),
            total_discount=("Order Item Discount", "sum"),
            total_profit=("Order Profit Per Order", "sum"),
            number_order_lines=("Sales", "size"),
            number_products=("Product Card Id", "nunique"),
            number_categories=("Category Id", "nunique"),
        )
        .reset_index()
    )

    orders["delay_days"] = (
        orders["Days for shipping (real)"] - orders["Days for shipment (scheduled)"]
    )
    orders["discount_rate"] = _safe_divide(orders["total_discount"], orders["gross_sales"])
    orders["profit_margin"] = _safe_divide(orders["total_profit"], orders["net_sales"])
    orders["average_unit_value"] = _safe_divide(orders["gross_sales"], orders["total_quantity"])

    assert len(orders) == EXPECTED_ORDER_COUNT, f"Expected {EXPECTED_ORDER_COUNT} rows, got {len(orders)}"
    assert orders["Order Id"].is_unique, "Order Id is not unique"
    assert orders["Order Id"].notna().all(), "Order Id contains nulls"
    assert set(orders["Late_delivery_risk"].unique()) <= {0, 1}, "Late_delivery_risk not binary"
    assert (orders["total_quantity"] > 0).all(), "Non-positive total_quantity found"

    print("Shape:", orders.shape)
    print("Unique orders:", orders["Order Id"].nunique())
    print("Late_delivery_risk distribution:")
    print(orders["Late_delivery_risk"].value_counts(normalize=True))
    with pd.option_context("display.max_columns", None, "display.width", 250):
        print(
            orders[
                [
                    "gross_sales",
                    "net_sales",
                    "total_quantity",
                    "total_profit",
                    "delay_days",
                    "number_products",
                    "number_categories",
                ]
            ].describe()
        )

    return orders
