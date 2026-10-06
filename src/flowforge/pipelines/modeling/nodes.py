"""
This is a boilerplate pipeline 'modeling'
generated using Kedro 1.7.0
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ID_COLUMN = "Order Id"
TARGET_COLUMN = "Late_delivery_risk"
ORDER_DATE_COLUMN = "order date (DateOrders)"

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15


def _describe_split(name: str, split: pd.DataFrame, dates: pd.Series, total: int) -> None:
    print(f"\n{name}:")
    print(f"  rows: {len(split)} ({len(split) / total:.2%})")
    print(f"  date range: {dates.min()} -> {dates.max()}")
    print("  Late_delivery_risk distribution:")
    print(split[TARGET_COLUMN].value_counts(normalize=True).sort_index().to_string())


def create_temporal_split(
    order_features: pd.DataFrame, orders_primary: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_columns = list(order_features.columns)

    dates = orders_primary[[ID_COLUMN, ORDER_DATE_COLUMN]].assign(
        _order_datetime=lambda d: pd.to_datetime(d[ORDER_DATE_COLUMN], format="%m/%d/%Y %H:%M")
    )[[ID_COLUMN, "_order_datetime"]]

    merged = order_features.merge(dates, on=ID_COLUMN, how="left", validate="one_to_one")
    assert merged["_order_datetime"].notna().all(), "Missing order dates after merge"

    merged = merged.sort_values(["_order_datetime", ID_COLUMN], kind="stable").reset_index(drop=True)

    total = len(merged)
    train_end = int(total * TRAIN_FRACTION)
    validation_end = int(total * (TRAIN_FRACTION + VALIDATION_FRACTION))

    splits = {
        "Train": merged.iloc[:train_end],
        "Validation": merged.iloc[train_end:validation_end],
        "Test": merged.iloc[validation_end:],
    }

    for name, split in splits.items():
        _describe_split(name, split, split["_order_datetime"], total)

    train, validation, test = (s[output_columns].reset_index(drop=True) for s in splits.values())
    return train, validation, test


CLASSIFICATION_THRESHOLD = 0.5


def _split_feature_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    candidates = [c for c in df.columns if c not in (ID_COLUMN, TARGET_COLUMN)]
    numerical = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in candidates if c not in numerical]
    return numerical, categorical


def train_logistic_baseline(
    train_orders: pd.DataFrame, validation_orders: pd.DataFrame
) -> tuple[Pipeline, pd.DataFrame, dict]:
    numerical, categorical = _split_feature_types(train_orders)
    features = numerical + categorical
    print("Numerical features:", numerical)
    print("Categorical features:", categorical)

    model = Pipeline(
        [
            (
                "preprocessor",
                ColumnTransformer(
                    [
                        ("num", Pipeline([("scaler", StandardScaler())]), numerical),
                        ("cat", Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical),
                    ]
                ),
            ),
            ("classifier", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ]
    )
    model.fit(train_orders[features], train_orders[TARGET_COLUMN])

    y_true = validation_orders[TARGET_COLUMN]
    y_proba = model.predict_proba(validation_orders[features])[:, 1]
    y_pred = (y_proba >= CLASSIFICATION_THRESHOLD).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": CLASSIFICATION_THRESHOLD,
        "validation_prevalence": float(y_true.mean()),
        "mean_predicted_probability": float(y_proba.mean()),
        "min_predicted_probability": float(y_proba.min()),
        "median_predicted_probability": float(pd.Series(y_proba).median()),
        "max_predicted_probability": float(y_proba.max()),
    }

    print("\nLogistic Regression validation metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    predictions = pd.DataFrame(
        {
            ID_COLUMN: validation_orders[ID_COLUMN].to_numpy(),
            TARGET_COLUMN: y_true.to_numpy(),
            "predicted_late_probability": y_proba,
            "predicted_class": y_pred,
        }
    )

    return model, predictions, metrics


def score_test_orders(model: Pipeline, test_orders: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    y_true = test_orders[TARGET_COLUMN]
    y_proba = model.predict_proba(test_orders.drop(columns=[ID_COLUMN, TARGET_COLUMN]))[:, 1]
    y_pred = (y_proba >= CLASSIFICATION_THRESHOLD).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": CLASSIFICATION_THRESHOLD,
        "test_prevalence": float(y_true.mean()),
        "mean_predicted_probability": float(y_proba.mean()),
    }

    print("\nLogistic Regression test metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    predictions = pd.DataFrame(
        {
            ID_COLUMN: test_orders[ID_COLUMN].to_numpy(),
            TARGET_COLUMN: y_true.to_numpy(),
            "predicted_late_probability": y_proba,
            "predicted_class": y_pred,
        }
    )
    return predictions, metrics


OPERATIONAL_BUSINESS_COLUMNS = [
    "gross_sales",
    "net_sales",
    "total_profit",
    "delay_days",
    "Shipping Mode",
    "Order Region",
    "Order Country",
    "Order City",
    "Market",
    "Customer Segment",
    "Latitude",
    "Longitude",
]


def build_operational_scored_orders(
    test_predictions: pd.DataFrame, orders_primary: pd.DataFrame
) -> pd.DataFrame:
    business = orders_primary[[ID_COLUMN, ORDER_DATE_COLUMN, *OPERATIONAL_BUSINESS_COLUMNS]]
    scored = test_predictions.merge(business, on=ID_COLUMN, how="left", validate="one_to_one")
    assert scored["net_sales"].notna().all(), "Scored orders missing from orders_primary"

    # Prioritization proxy only, not an estimate of expected financial loss.
    scored["expected_revenue_exposure"] = scored["predicted_late_probability"] * scored["net_sales"]

    order_dates = pd.to_datetime(scored.pop(ORDER_DATE_COLUMN), format="%m/%d/%Y %H:%M")

    print("\nOperational scored orders:", len(scored))
    print(f"Period: {order_dates.min()} -> {order_dates.max()}")
    print(f"Average predicted probability: {scored['predicted_late_probability'].mean():.4f}")
    print(f"Total expected_revenue_exposure: {scored['expected_revenue_exposure'].sum():,.2f}")
    print("Top 5 orders by expected_revenue_exposure:")
    with pd.option_context("display.max_columns", None, "display.width", 250):
        print(
            scored.nlargest(5, "expected_revenue_exposure")[
                [
                    ID_COLUMN,
                    "predicted_late_probability",
                    "net_sales",
                    "expected_revenue_exposure",
                    "Shipping Mode",
                    "Order Region",
                    "Market",
                ]
            ].to_string(index=False)
        )

    return scored
