import html
import sys
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# The flowforge package lives in src/ and is not necessarily pip-installed.
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATA_PATH = PROJECT_ROOT / "data" / "03_primary" / "orders.parquet"

AI_ACTIONS = {
    "Analyze Hotspots": "analyze_hotspots",
    "Prioritize Interventions": "prioritize_interventions",
    "Compare Shipping Modes": "compare_shipping_modes",
}
AI_RESULT_KEY = "ai_result"
AI_LAST_ACTION_KEY = "ai_last_action"
AI_CALL_COUNT_KEY = "ai_call_count"
AI_CALL_LIMIT = 3
AI_LIMIT_MESSAGE = "AI demo limit reached for this session."

TARGET = "Late_delivery_risk"
FILTER_COLUMNS = ["Market", "Shipping Mode", "Customer Segment"]
GRID_PRECISION = 1

HEIGHT_METRICS = {
    "Order Volume": "orders",
    "Gross Sales": "gross_sales",
    "Late Delivery Rate": "late_rate",
}
MAX_ELEVATION = 400_000

CSS = """
<style>
:root {
    --bg: #0b0d10;
    --panel: #12151a;
    --border: #232830;
    --text: #d6dae0;
    --muted: #7d8590;
    --accent: #d97a2b;
}
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background: var(--bg);
    color: var(--text);
}
[data-testid="stSidebar"] {
    background: var(--panel);
    border-right: 1px solid var(--border);
}
.block-container { padding-top: 1.2rem; padding-bottom: 1rem; max-width: 100%; }
html, body, p, div, span, label { font-size: 13px; }
h1, h2, h3, h4 { color: var(--text); letter-spacing: 0.04em; }

.ff-header {
    display: flex; align-items: baseline; gap: 14px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px; margin-bottom: 12px;
}
.ff-title { font-size: 20px; font-weight: 700; letter-spacing: 0.32em; color: var(--text); }
.ff-subtitle { font-size: 11px; letter-spacing: 0.18em; text-transform: uppercase; color: var(--muted); }

.ff-kpi {
    background: var(--panel); border: 1px solid var(--border);
    border-left: 2px solid var(--accent);
    padding: 10px 12px;
}
.ff-kpi-label { font-size: 10px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted); }
.ff-kpi-value { font-size: 22px; font-weight: 600; color: var(--text); font-variant-numeric: tabular-nums; }

.ff-section {
    font-size: 10px; letter-spacing: 0.18em; text-transform: uppercase;
    color: var(--muted); border-bottom: 1px solid var(--border);
    padding: 4px 0; margin: 14px 0 8px 0;
}
.ff-status {
    font-size: 11px; color: var(--muted); border: 1px dashed var(--border);
    padding: 8px; margin-top: 8px; font-family: ui-monospace, monospace;
}

.ff-brief {
    background: var(--panel); border: 1px solid var(--border);
    border-left: 2px solid var(--accent);
    padding: 14px 16px; margin-bottom: 10px;
}
.ff-brief-tag {
    font-size: 10px; letter-spacing: 0.16em; text-transform: uppercase;
    color: var(--accent); margin-bottom: 6px;
}
.ff-brief-headline { font-size: 19px; font-weight: 600; color: var(--text); line-height: 1.35; margin-bottom: 8px; }
.ff-brief-summary { font-size: 13px; color: var(--text); line-height: 1.55; margin-bottom: 12px; max-width: 110ch; }
.ff-brief-label {
    font-size: 10px; letter-spacing: 0.16em; text-transform: uppercase;
    color: var(--muted); margin: 10px 0 6px 0;
}
.ff-brief ol { margin: 0; padding-left: 20px; }
.ff-brief li { font-size: 12.5px; color: var(--text); line-height: 1.5; margin-bottom: 3px; }
.ff-metric-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 6px; }
.ff-metric {
    background: var(--bg); border: 1px solid var(--border);
    padding: 7px 9px; font-size: 11.5px; color: var(--text);
    font-variant-numeric: tabular-nums; line-height: 1.4;
}
.ff-error {
    background: var(--panel); border: 1px solid #5a2a2a; border-left: 2px solid #b0453a;
    padding: 10px 12px; font-size: 12px; color: var(--text); margin-bottom: 10px;
}

[data-testid="stDataFrame"] { border: 1px solid var(--border); }
.stButton > button {
    width: 100%; background: var(--panel); color: var(--muted);
    border: 1px solid var(--border); border-radius: 2px;
    font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase;
}
div[data-baseweb="select"] > div { background: var(--bg); border-color: var(--border); }
[data-testid="stMultiSelectTagsContainer"] span[data-tag] {
    background: #1c2128; border: 1px solid var(--border);
    border-radius: 2px; color: var(--text); font-size: 11px;
}
</style>
"""


@st.cache_data
def load_orders() -> pd.DataFrame:
    return pd.read_parquet(DATA_PATH)


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown('<div class="ff-section">Filters</div>', unsafe_allow_html=True)
    mask = pd.Series(True, index=df.index)
    for col in FILTER_COLUMNS:
        options = sorted(df[col].dropna().unique())
        selected = st.sidebar.multiselect(col, options, default=options)
        mask &= df[col].isin(selected)
    return df[mask]


def kpi(label: str, value: str) -> str:
    return f'<div class="ff-kpi"><div class="ff-kpi-label">{label}</div><div class="ff-kpi-value">{value}</div></div>'


def render_kpis(df: pd.DataFrame) -> None:
    values = [
        ("Total Orders", f"{len(df):,}"),
        ("Late Delivery Rate", f"{df[TARGET].mean():.1%}" if len(df) else "-"),
        ("Gross Sales", f"${df['gross_sales'].sum() / 1e6:,.2f}M"),
        ("Average Delay", f"{df['delay_days'].mean():+.2f} d" if len(df) else "-"),
    ]
    for col, (label, value) in zip(st.columns(4), values):
        col.markdown(kpi(label, value), unsafe_allow_html=True)


def aggregate_geo(df: pd.DataFrame) -> pd.DataFrame:
    geo = (
        df.assign(
            lat=df["Latitude"].round(GRID_PRECISION),
            lon=df["Longitude"].round(GRID_PRECISION),
        )
        .groupby(["lat", "lon"])
        .agg(
            orders=(TARGET, "size"),
            gross_sales=("gross_sales", "sum"),
            late_rate=(TARGET, "mean"),
        )
        .reset_index()
    )
    return geo


def render_map(df: pd.DataFrame, metric_label: str) -> None:
    geo = aggregate_geo(df)
    if geo.empty:
        st.info("No orders match the current filters.")
        return

    metric = HEIGHT_METRICS[metric_label]
    peak = geo[metric].max() or 1
    geo["elevation"] = geo[metric] / peak * MAX_ELEVATION
    geo["color_r"] = (60 + geo["late_rate"] * 157).astype(int)
    geo["color_g"] = (110 + geo["late_rate"] * 12).astype(int)
    geo["color_b"] = (140 - geo["late_rate"] * 97).astype(int)
    geo["late_rate_pct"] = (geo["late_rate"] * 100).round(1)
    geo["gross_sales_fmt"] = geo["gross_sales"].map("{:,.0f}".format)

    layer = pdk.Layer(
        "ColumnLayer",
        data=geo,
        get_position="[lon, lat]",
        get_elevation="elevation",
        get_fill_color="[color_r, color_g, color_b, 220]",
        radius=30_000,
        elevation_scale=1,
        extruded=True,
        pickable=True,
        auto_highlight=True,
    )
    view = pdk.ViewState(
        latitude=float((geo["lat"] * geo["orders"]).sum() / geo["orders"].sum()),
        longitude=float((geo["lon"] * geo["orders"]).sum() / geo["orders"].sum()),
        zoom=2.4,
        pitch=50,
        bearing=-15,
    )
    tooltip = {
        "html": (
            "<b>{lat}, {lon}</b><br/>Orders: {orders}<br/>"
            "Gross sales: ${gross_sales_fmt}<br/>Late rate: {late_rate_pct}%"
        ),
        "style": {"backgroundColor": "#12151a", "color": "#d6dae0", "fontSize": "11px"},
    }
    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=view,
            map_provider="carto",
            map_style="dark",
            tooltip=tooltip,
        ),
        height=520,
    )


def render_region_table(df: pd.DataFrame) -> None:
    regions = (
        df.groupby("Order Region")
        .agg(orders=(TARGET, "size"), late_rate=(TARGET, "mean"), avg_delay=("delay_days", "mean"))
        .sort_values("late_rate", ascending=False)
        .head(10)
        .reset_index()
    )
    st.dataframe(
        regions,
        hide_index=True,
        width="stretch",
        column_config={
            "Order Region": "Region",
            "orders": st.column_config.NumberColumn("Orders", format="%d"),
            "late_rate": st.column_config.ProgressColumn("Late Rate", format="percent", min_value=0, max_value=1),
            "avg_delay": st.column_config.NumberColumn("Avg Delay (d)", format="%.2f"),
        },
    )


def render_shipping_table(df: pd.DataFrame) -> None:
    modes = (
        df.groupby("Shipping Mode")
        .agg(
            orders=(TARGET, "size"),
            late_rate=(TARGET, "mean"),
            scheduled=("Days for shipment (scheduled)", "mean"),
            actual=("Days for shipping (real)", "mean"),
            avg_delay=("delay_days", "mean"),
            gross_sales=("gross_sales", "sum"),
        )
        .sort_values("late_rate", ascending=False)
        .reset_index()
    )
    st.dataframe(
        modes,
        hide_index=True,
        width="stretch",
        column_config={
            "Shipping Mode": "Mode",
            "orders": st.column_config.NumberColumn("Orders", format="%d"),
            "late_rate": st.column_config.ProgressColumn("Late Rate", format="percent", min_value=0, max_value=1),
            "scheduled": st.column_config.NumberColumn("Sched. (d)", format="%.2f"),
            "actual": st.column_config.NumberColumn("Actual (d)", format="%.2f"),
            "avg_delay": st.column_config.NumberColumn("Avg Delay (d)", format="%.2f"),
            "gross_sales": st.column_config.NumberColumn("Gross Sales", format="$%.0f"),
        },
    )


def run_ai_action(label: str, action: str) -> dict:
    try:
        from flowforge.agent import run_decision_action

        return {"label": label, "brief": run_decision_action(action)}
    except RuntimeError as exc:
        if "OPENAI_API_KEY" in str(exc):
            return {"label": label, "error": "OpenAI API key is not configured."}
        return {"label": label, "error": "The AI decision analysis did not complete. Please retry."}
    except ImportError:
        return {"label": label, "error": "The AI decision layer is unavailable. Check the installed dependencies."}
    except Exception as exc:
        return {
            "label": label,
            "error": f"The AI decision analysis failed ({type(exc).__name__}). Please retry.",
        }


def init_ai_state() -> None:
    st.session_state.setdefault(AI_RESULT_KEY, None)
    st.session_state.setdefault(AI_LAST_ACTION_KEY, None)
    st.session_state.setdefault(AI_CALL_COUNT_KEY, 0)


def handle_ai_click(label: str, action: str) -> None:
    # Drop the previous brief first so it can never outlive a newer click.
    st.session_state[AI_RESULT_KEY] = None
    st.session_state[AI_LAST_ACTION_KEY] = label

    if st.session_state[AI_CALL_COUNT_KEY] >= AI_CALL_LIMIT:
        st.session_state[AI_RESULT_KEY] = {"label": label, "error": AI_LIMIT_MESSAGE}
        return

    with st.spinner("Running AI decision analysis..."):
        result = run_ai_action(label, action)
    if "brief" in result:
        st.session_state[AI_CALL_COUNT_KEY] += 1
    st.session_state[AI_RESULT_KEY] = result


def render_ai_operations() -> None:
    st.markdown('<div class="ff-section">AI Operations</div>', unsafe_allow_html=True)
    clicked = [
        (label, action)
        for label, action in AI_ACTIONS.items()
        if st.button(label, key=f"ai_{action}")
    ]
    if clicked:
        handle_ai_click(*clicked[0])

    last_action = st.session_state[AI_LAST_ACTION_KEY]
    result = st.session_state[AI_RESULT_KEY]
    if last_action is None:
        status = "Decision layer ready. Scope: operational test period."
    elif result is not None and result.get("error") == AI_LIMIT_MESSAGE:
        status = f"Last run blocked: {last_action}"
    elif result is not None and "error" in result:
        status = f"Last run failed: {last_action}"
    else:
        status = f"Last run: {last_action}"
    usage = f"AI demo calls used: {st.session_state[AI_CALL_COUNT_KEY]} / {AI_CALL_LIMIT}"
    st.markdown(
        f'<div class="ff-status">{html.escape(status)}<br/>{html.escape(usage)}</div>',
        unsafe_allow_html=True,
    )


def render_ai_brief() -> None:
    result = st.session_state[AI_RESULT_KEY]
    if result is None:
        return

    if "error" in result:
        st.markdown(f'<div class="ff-error">{html.escape(result["error"])}</div>', unsafe_allow_html=True)
        return

    brief = result["brief"]
    recommendations = "".join(f"<li>{html.escape(r)}</li>" for r in brief.get("recommendations", []))
    metrics = "".join(
        f'<div class="ff-metric">{html.escape(m)}</div>' for m in brief.get("supporting_metrics", [])
    )
    st.markdown(
        f"""
<div class="ff-brief">
  <div class="ff-brief-tag">AI Decision Brief / {html.escape(result["label"])}</div>
  <div class="ff-brief-headline">{html.escape(brief.get("headline", ""))}</div>
  <div class="ff-brief-summary">{html.escape(brief.get("summary", ""))}</div>
  <div class="ff-brief-label">Recommendations</div>
  <ol>{recommendations}</ol>
  <div class="ff-brief-label">Supporting Metrics</div>
  <div class="ff-metric-grid">{metrics}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="FlowForge Control Tower", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    init_ai_state()
    st.markdown(
        '<div class="ff-header"><span class="ff-title">FLOWFORGE</span>'
        '<span class="ff-subtitle">Supply Chain Decision Intelligence</span></div>',
        unsafe_allow_html=True,
    )

    df = apply_filters(load_orders())
    render_kpis(df)

    main_col, side_col = st.columns([4, 1], gap="medium")
    # Side column first so a button click updates session state before the brief renders.
    with side_col:
        render_ai_operations()

    with main_col:
        render_ai_brief()
        st.markdown('<div class="ff-section">Geographic Order Distribution</div>', unsafe_allow_html=True)
        metric_label = st.radio(
            "Column height", list(HEIGHT_METRICS), horizontal=True, label_visibility="collapsed"
        )
        render_map(df, metric_label)

        left, right = st.columns(2, gap="medium")
        with left:
            st.markdown('<div class="ff-section">Top 10 Regions by Late Delivery Rate</div>', unsafe_allow_html=True)
            render_region_table(df)
        with right:
            st.markdown('<div class="ff-section">Shipping Mode Performance</div>', unsafe_allow_html=True)
            render_shipping_table(df)


main()
