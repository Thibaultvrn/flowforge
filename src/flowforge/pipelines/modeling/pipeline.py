"""
This is a boilerplate pipeline 'modeling'
generated using Kedro 1.7.0
"""

from kedro.pipeline import Node, Pipeline

from .nodes import (
    build_operational_scored_orders,
    create_temporal_split,
    score_test_orders,
    train_logistic_baseline,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=create_temporal_split,
                inputs=["order_features", "orders_primary"],
                outputs=["train_orders", "validation_orders", "test_orders"],
                name="create_temporal_split_node",
            ),
            Node(
                func=train_logistic_baseline,
                inputs=["train_orders", "validation_orders"],
                outputs=[
                    "logistic_model",
                    "validation_logistic_predictions",
                    "logistic_validation_metrics",
                ],
                name="train_logistic_baseline_node",
            ),
            Node(
                func=score_test_orders,
                inputs=["logistic_model", "test_orders"],
                outputs=["test_logistic_predictions", "test_logistic_metrics"],
                name="score_test_orders_node",
            ),
            Node(
                func=build_operational_scored_orders,
                inputs=["test_logistic_predictions", "orders_primary"],
                outputs="operational_scored_orders",
                name="build_operational_scored_orders_node",
            ),
        ]
    )
