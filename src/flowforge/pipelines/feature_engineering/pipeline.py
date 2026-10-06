"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 1.7.0
"""

from kedro.pipeline import Node, Pipeline

from .nodes import build_order_features


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=build_order_features,
                inputs="orders_primary",
                outputs="order_features",
                name="build_order_features_node",
            )
        ]
    )
