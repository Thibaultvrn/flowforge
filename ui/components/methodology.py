"""Methodology page: how FlowForge predicts, scores, and briefs."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from components.architecture import (
    agent_flow,
    architecture_diagram,
    leakage_panel,
    pipeline_summary,
    stack_panel,
    timeline,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_METRICS_PATH = PROJECT_ROOT / "data" / "08_reporting" / "logistic_test_metrics.json"

AGGREGATED_FIELDS = [
    "gross_sales — sum of Sales",
    "net_sales — sum of Order Item Total",
    "total_quantity — sum of Order Item Quantity",
    "total_discount — sum of Order Item Discount",
    "total_profit — sum of Order Profit Per Order",
    "number_order_lines — row count per Order Id",
    "number_products — unique Product Card Id",
    "number_categories — unique Category Id",
]


def _load_test_metrics() -> dict[str, float]:
    return json.loads(TEST_METRICS_PATH.read_text(encoding="utf-8"))


def _section(title: str, kicker: str = "") -> None:
    kicker_html = f'<div class="meth-kicker meth-section-kicker">{kicker}</div>' if kicker else ""
    st.markdown(f'{kicker_html}<h2 class="meth-h">{title}</h2>', unsafe_allow_html=True)


def _metric_card(label: str, value: str, note: str, *, lead: bool = False) -> str:
    cls = "meth-metric lead" if lead else "meth-metric"
    return (
        f'<div class="{cls}"><div class="meth-metric-label">{label}</div>'
        f'<div class="meth-metric-value">{value}</div>'
        f'<div class="meth-metric-note">{note}</div></div>'
    )


def render() -> None:
    st.markdown(
        '<div class="ff-header"><div><div class="ff-title">Methodology</div>'
        '<div class="ff-subtitle">How FlowForge turns order-creation signals into an operational brief</div></div>'
        '<div class="ff-scope">Static project documentation<br/>No live model calls on this page</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="meth-lead">FlowForge is a supply-chain decision-intelligence prototype. '
        "It predicts late-delivery risk when an order is created, converts those probabilities "
        "into a revenue-weighted ranking, and asks a language model only to explain the ranking.</p>",
        unsafe_allow_html=True,
    )

    _business_problem()
    _data()
    _leakage()
    _temporal()
    _mathematics()
    _performance()
    _scoring()
    _architecture()
    _agent()
    _stack()
    _kedro()
    _limitations()
    _summary()


def _business_problem() -> None:
    _section("The business problem", "01")
    st.markdown(
        "Late delivery is expensive to discover after the fact. FlowForge estimates the probability "
        "that an order will be late **at order creation**, then uses that probability to decide "
        "where attention should go."
    )
    cols = st.columns(5, gap="small")
    steps = [
        ("Predict", "P(late) from creation-time features"),
        ("Quantify exposure", "Probability × net sales"),
        ("Identify hotspots", "Regions ranked by exposure"),
        ("Prioritize", "Orders to review first"),
        ("Brief", "A short operational narrative"),
    ]
    for col, (title, body) in zip(cols, steps):
        col.markdown(
            f'<div class="meth-card compact"><div class="meth-card-title">{title}</div>'
            f'<p>{body}</p></div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        "**Prediction** produces a probability. **Decision support** ranks work and writes the brief. "
        "The language model never computes the probability, the ranking, or the exposure."
    )


def _data() -> None:
    _section("Data and unit of analysis", "02")
    left, right = st.columns([3, 2], gap="medium")
    with left:
        st.markdown(
            "The DataCo extract is **item-level**: one row per product line on an order. "
            "Late delivery, however, is an **order-level** outcome. Several lines can share one "
            "`Order Id`, so the pipeline aggregates to **exactly one row per order** before modeling."
        )
        st.markdown(
            f'<div class="meth-facts">'
            f'<div><b>180,519</b> order-item rows</div>'
            f'<div><b>65,752</b> unique orders</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    with right:
        items = "".join(f"<li>{field}</li>" for field in AGGREGATED_FIELDS)
        st.markdown(
            f'<div class="meth-card"><div class="meth-kicker">Aggregated per Order Id</div>'
            f"<ul>{items}</ul></div>",
            unsafe_allow_html=True,
        )


def _leakage() -> None:
    _section("Leakage prevention", "03")
    st.markdown(
        "The prediction is made **when the order is created**, before actual shipment or delivery "
        "is known. Anything that is only observed after shipping is withheld from the model. "
        "That is a prediction-time availability rule, not a causal identification strategy."
    )
    st.markdown(leakage_panel(), unsafe_allow_html=True)
    st.caption(
        "Identifiers (`Order Id`, `Customer Id`) are also excluded as features. "
        "`Order Id` is retained only as a join key. Latitude and Longitude are kept for the map, not for the model."
    )


def _temporal() -> None:
    _section("Temporal validation", "04")
    st.markdown(
        "Orders are sorted by order datetime and split **chronologically**: first 70% train, "
        "next 15% validation, final 15% test. There is no shuffle. Future orders cannot leak "
        "into a model fitted on the past."
    )
    st.markdown(timeline(), unsafe_allow_html=True)
    st.caption(
        "Boundaries are the first and last order timestamps in each slice of the 65,752-order table. "
        "The operational snapshot scored for the Control Tower is the test period."
    )


def _mathematics() -> None:
    _section("Mathematics", "05")
    st.markdown(
        "The baseline is **binary Logistic Regression**. For order *i*, the target is "
        r"$Y_i \in \{0,1\}$, late versus not late, modelled as"
    )
    st.latex(r"Y_i \mid X_i \sim \mathrm{Bernoulli}(p_i)")
    st.markdown("The linear predictor is mapped to a probability with the logit link:")
    st.latex(r"\log\frac{p_i}{1-p_i} = x_i^{\top}\beta \qquad p_i = \frac{1}{1+e^{-x_i^{\top}\beta}}")
    st.markdown(
        "Intuitively: the model produces a real-valued score $x_i^{\\top}\\beta$; the sigmoid "
        "compresses that score into $[0,1]$; that number is the **predicted late-delivery probability**."
    )
    st.markdown("Fitting maximises the Bernoulli log-likelihood")
    st.latex(
        r"\ell(\beta)=\sum_i \bigl[ y_i\log p_i + (1-y_i)\log(1-p_i) \bigr]"
    )
    st.markdown(
        "scikit-learn's `LogisticRegression` with `solver=\"lbfgs\"` applies **L2 regularization** "
        "by default. Equivalently, it maximises a penalised likelihood, or minimises penalised log-loss:"
    )
    st.latex(r"\max_{\beta}\ \ell(\beta) - \lambda\|\beta\|_2^{2}")
    st.markdown(
        r"The penalty keeps coefficients from exploding and stabilises estimation in a wide one-hot "
        r"feature space. $C$ is the **inverse** regularization strength; FlowForge leaves the "
        r"library default $C=1$. This is a **predictive** model, not an inferential one: "
        r"coefficients are not used for hypothesis tests."
    )
    with st.expander("Preprocessing"):
        st.markdown(
            """
Numerical columns go through `StandardScaler`. Categorical columns go through
`OneHotEncoder(handle_unknown="ignore")`. Both sit in a `ColumnTransformer` in front
of the classifier. Date parts (`order_year`, `order_month`, `order_day_of_week`,
`order_hour`) are treated as numeric. There is no class weighting and no probability calibration.
"""
        )


def _performance() -> None:
    _section("Model performance", "06")
    metrics = _load_test_metrics()
    st.markdown(
        "The numbers below are the **final chronological test** evaluation of the current baseline. "
        "Validation was used only while fitting; the test set was scored once."
    )
    c1, c2, c3, c4 = st.columns(4, gap="small")
    c1.markdown(
        _metric_card("ROC-AUC", f"{metrics['roc_auc']:.4f}", "Ranking / discrimination", lead=True),
        unsafe_allow_html=True,
    )
    c2.markdown(
        _metric_card("Brier score", f"{metrics['brier_score']:.4f}", "Mean squared probability error", lead=True),
        unsafe_allow_html=True,
    )
    c3.markdown(
        _metric_card("Accuracy", f"{metrics['accuracy']:.4f}", "Share correct at threshold 0.5"),
        unsafe_allow_html=True,
    )
    c4.markdown(
        _metric_card("F1", f"{metrics['f1_score']:.4f}", "Balance of precision and recall"),
        unsafe_allow_html=True,
    )
    c5, c6, c7 = st.columns(3, gap="small")
    c5.markdown(
        _metric_card("Precision", f"{metrics['precision']:.4f}", "Of predicted late, share actually late"),
        unsafe_allow_html=True,
    )
    c6.markdown(
        _metric_card("Recall", f"{metrics['recall']:.4f}", "Of actually late, share predicted late"),
        unsafe_allow_html=True,
    )
    c7.markdown(
        _metric_card("Threshold", "0.50", "Used only for class metrics"),
        unsafe_allow_html=True,
    )
    st.markdown(
        "**ROC-AUC** (~0.75) is the probability that a late order is ranked above an on-time order. "
        "FlowForge cares about this because exposure ranking depends on relative risk. "
        "**Brier score** is the mean squared error of the predicted probabilities; lower is better. "
        "The model is **not** claimed to be calibrated — no calibration step was run. "
        "Accuracy, precision, recall and F1 use a 0.5 cutoff and are secondary to the probability scores."
    )


def _scoring() -> None:
    _section("Business scoring", "07")
    st.markdown(
        "Each scored test order carries a prioritization weight equal to predicted late probability "
        "times net sales:"
    )
    st.latex(r"E_i = p_i \times R_i")
    st.markdown(
        r"where $p_i$ is `predicted_late_probability` and $R_i$ is `net_sales`. "
        "**This is not expected financial loss.** It is a **prioritization proxy**: revenue that would "
        "be affected if the late-delivery event occurred, weighted by the model's probability. "
        "Two orders with the same $p_i$ are not equal operationally if one carries much more revenue."
    )


def _architecture() -> None:
    _section("System architecture", "08")
    st.markdown(
        "Computation flows downward. Each band is a different trust boundary. "
        "The language model sits **above** the numbers, not inside them."
    )
    st.markdown(architecture_diagram(), unsafe_allow_html=True)
    st.markdown(
        "**Design principle:** the LLM does not calculate business metrics. Python tools compute "
        "probabilities, counts, sales, exposure and rankings. The language model selects approved "
        "tools, reads their JSON, and writes the executive brief."
    )


def _agent() -> None:
    _section("Agent design", "09")
    st.markdown(
        "The Control Tower exposes **three fixed actions** — Analyze Hotspots, Prioritize Interventions, "
        "Compare Shipping Modes. There is no chat, no free-text prompt, and no path to arbitrary Python. "
        "The agent may only call `network_summary`, `risk_hotspots`, `priority_orders`, and "
        "`compare_shipping_modes`."
    )
    st.markdown(agent_flow(), unsafe_allow_html=True)
    st.markdown(
        "**Trust boundary:** the LLM is not the source of numerical truth. Deterministic tools are. "
        "A public session is limited to three successful briefs."
    )


def _stack() -> None:
    _section("Technology stack", "10")
    st.markdown("Only libraries and services that this repository actually uses:")
    st.markdown(stack_panel(), unsafe_allow_html=True)


def _kedro() -> None:
    _section("Reproducibility / Kedro", "11")
    cols = st.columns(4, gap="small")
    defs = [
        ("Node", "One named computation with typed inputs and outputs."),
        ("Pipeline", "A DAG of nodes. Default run is data processing → features → modeling."),
        ("Data Catalog", "Logical names mapped to files (CSV, Parquet, pickle, JSON)."),
        ("Configuration", "Paths and load/save args live in YAML, not in Python."),
    ]
    for col, (title, body) in zip(cols, defs):
        col.markdown(
            f'<div class="meth-card compact"><div class="meth-card-title">{title}</div><p>{body}</p></div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        "That split is what makes a rerun (`kedro run`) reconstruct the same artifacts: "
        "modularity, traceability, and separation of parameters from computation."
    )


def _limitations() -> None:
    _section("Current limitations", "12")
    st.markdown(
        """
- Historical **DataCo** orders, not a live warehouse feed.
- Map coordinates are **recorded store locations**, not origin–destination shipment routes.
- `expected_revenue_exposure` is a **prioritization proxy**, not expected loss.
- The classifier is an intentional **Logistic Regression baseline**, not a tuned production model.
- No causal claim is made about shipping-mode or region effects.
- No inventory, capacity, or routing optimization is executed.
- Public demo actions are the three fixed analyses, with a per-session call limit.
"""
    )


def _summary() -> None:
    _section("End-to-end", "13")
    st.markdown(pipeline_summary(), unsafe_allow_html=True)
    st.markdown(
        '<p class="meth-close">FlowForge separates prediction, deterministic business logic and language '
        "generation so that the LLM explains decisions without becoming the source of quantitative truth.</p>",
        unsafe_allow_html=True,
    )
