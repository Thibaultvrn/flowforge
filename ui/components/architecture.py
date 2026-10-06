"""Static HTML diagrams for the Methodology page."""

from __future__ import annotations

from html import escape


def leakage_panel() -> str:
    available = [
        "Scheduled shipping days",
        "Shipping mode, market, geography",
        "Customer segment",
        "Order composition and sales totals",
        "Calendar features from order date",
    ]
    withheld = [
        "Delivery Status",
        "Days for shipping (real)",
        "shipping date (DateOrders)",
        "delay_days",
        "Late_delivery_risk (target, not a feature)",
    ]
    left = "".join(f"<li>{escape(item)}</li>" for item in available)
    right = "".join(f"<li>{escape(item)}</li>" for item in withheld)
    return f"""
<div class="meth-compare">
  <div class="meth-compare-col meth-available">
    <div class="meth-kicker">Available at prediction time</div>
    <div class="meth-card-title">Order-creation inputs</div>
    <ul>{left}</ul>
  </div>
  <div class="meth-compare-col meth-withheld">
    <div class="meth-kicker">Outcome / post-shipment</div>
    <div class="meth-card-title">Withheld from the model</div>
    <ul>{right}</ul>
  </div>
</div>
"""


def timeline() -> str:
    periods = [
        ("Train 70%", "2015-01-01", "2017-03-16", "70"),
        ("Validation 15%", "2017-03-16", "2017-09-05", "15"),
        ("Test 15%", "2017-09-05", "2018-01-31", "15"),
    ]
    cells = "".join(
        f'<div class="meth-tl-seg meth-tl-{width}">'
        f'<div class="meth-tl-label">{escape(label)}</div>'
        f'<div class="meth-tl-dates">{escape(start)} → {escape(end)}</div>'
        f"</div>"
        for label, start, end, width in periods
    )
    return f'<div class="meth-timeline" aria-label="Chronological train, validation and test periods">{cells}</div>'


def architecture_diagram() -> str:
    layers = [
        (
            "Data / ML layer",
            [
                "Raw DataCo CSV",
                "Kedro data processing",
                "Order-level dataset",
                "Feature engineering",
                "Temporal split",
                "Logistic Regression",
                "Operational risk scores",
            ],
        ),
        (
            "Deterministic decision layer",
            [
                "network_summary",
                "risk_hotspots",
                "priority_orders",
                "compare_shipping_modes",
            ],
        ),
        (
            "Language model layer",
            [
                "LangChain agent",
                "Fixed action intent",
                "OpenAI (brief only)",
            ],
        ),
        (
            "Presentation layer",
            [
                "Operational Brief",
                "Streamlit Control Tower",
            ],
        ),
    ]
    bands = []
    for title, nodes in layers:
        pills = "".join(f'<span class="meth-pill">{escape(n)}</span>' for n in nodes)
        bands.append(
            f'<div class="meth-layer"><div class="meth-layer-title">{escape(title)}</div>'
            f'<div class="meth-pills">{pills}</div></div>'
        )
    return f'<div class="meth-arch">{"".join(bands)}</div>'


def agent_flow() -> str:
    steps = [
        ("1", "User action", "One of three dashboard buttons"),
        ("2", "Fixed agent intent", "No free-text, no arbitrary tools"),
        ("3", "Approved Python tool", "Counts, rates, exposure, rankings"),
        ("4", "Structured tool output", "JSON the model must quote"),
        ("5", "LLM explanation", "Headline, summary, recommendations"),
    ]
    cells = "".join(
        f'<div class="meth-flow-step"><div class="meth-flow-n">{n}</div>'
        f'<div class="meth-flow-title">{escape(title)}</div>'
        f'<div class="meth-flow-body">{escape(body)}</div></div>'
        for n, title, body in steps
    )
    return f'<div class="meth-flow">{cells}</div>'


def pipeline_summary() -> str:
    steps = ["Order", "Features", "P(late)", "Revenue exposure", "Deterministic analytics", "LLM decision brief"]
    parts: list[str] = []
    for i, step in enumerate(steps):
        if i:
            parts.append('<span class="meth-pipe-arrow" aria-hidden="true">→</span>')
        parts.append(f'<div class="meth-pipe-step">{escape(step)}</div>')
    return f'<div class="meth-pipe">{"".join(parts)}</div>'


def stack_panel() -> str:
    groups = [
        ("Data pipeline", "Kedro · Parquet · pandas"),
        ("ML", "scikit-learn"),
        ("Agent", "LangChain · OpenAI"),
        ("Interface", "Streamlit · deck.gl · custom SVG Sankey"),
        ("Engineering", "Git · GitHub · python-dotenv"),
    ]
    cells = "".join(
        f'<div class="meth-stack"><div class="meth-kicker">{escape(role)}</div>'
        f'<div class="meth-stack-tools">{escape(tools)}</div></div>'
        for role, tools in groups
    )
    return f'<div class="meth-stack-grid">{cells}</div>'
