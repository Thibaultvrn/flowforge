"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 1.7.0
"""

from kedro.pipeline import Node, Pipeline

from .nodes import (
    aggregate_orders,
    audit_item_level_columns,
    audit_order_consistency,
    inspect_orders,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=inspect_orders,
                inputs="raw_orders",
                outputs="inspected_orders",
                name="inspect_orders_node",
            ),
            Node(
                func=audit_order_consistency,
                inputs="inspected_orders",
                outputs="order_consistency_report",
                name="audit_order_consistency_node",
            ),
            Node(
                func=audit_item_level_columns,
                inputs="inspected_orders",
                outputs="item_level_audit_report",
                name="audit_item_level_columns_node",
            ),
            Node(
                func=aggregate_orders,
                inputs="inspected_orders",
                outputs="orders_primary",
                name="aggregate_orders_node",
            ),
        ]
    )
