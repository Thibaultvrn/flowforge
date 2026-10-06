import html
import math
import sys
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# The flowforge package lives in src/ and is not necessarily pip-installed.
for path in (PROJECT_ROOT / "src", Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from components.geo_scale import build_scale, hex_to_rgb  # noqa: E402
from components.sankey import sankey_svg  # noqa: E402

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

MAP_METRICS = {
    "Order Volume": "orders",
    "Gross Sales": "gross_sales",
    "Late Delivery Rate": "late_rate",
}
MARKER_RADIUS_PX = 4.5
# (radius multiple of the marker, alpha 0-255); outer ring diameter stays at 1.5x the marker.
RIPPLE_RINGS = [(1.25, 95), (1.5, 45)]
MAP_HEIGHT = 470
MAP_WIDTH_ESTIMATE_PX = 760

ACCENT = "#c2622d"
OUTCOME_COLUMN = "Delivery outcome"
OUTCOME_LABELS = {1: "Late", 0: "Not late"}
SANKEY_WIDTH = 1100
SANKEY_HEIGHT = 380

CSS = """
<style>
:root {
    --bg: #f5f6f8;
    --card: #ffffff;
    --border: #e4e7eb;
    --border-strong: #d5dae0;
    --text: #1d2433;
    --muted: #6b7280;
    --accent: #c2622d;
    --accent-soft: #f6ebe3;
}
.block-container { padding-top: 3.2rem; padding-bottom: 3rem; max-width: 1480px; }
.stButton { margin-bottom: -6px; }
[data-testid="stSidebar"] { border-right: 1px solid var(--border); }
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .ff-card-title) {
    background: var(--card); box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
}
[data-testid="stSidebar"] label p { font-size: 12.5px; color: var(--muted); }

.ff-header { display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 8px 16px; margin-bottom: 1.4rem; }
.ff-title { font-size: 24px; font-weight: 650; color: var(--text); letter-spacing: -0.01em; line-height: 1.2; }
.ff-subtitle { font-size: 13.5px; color: var(--muted); margin-top: 2px; }
.ff-scope { font-size: 12px; color: var(--muted); text-align: right; line-height: 1.5; }

.ff-kpi {
    background: var(--card); border: 1px solid var(--border); border-radius: 8px;
    padding: 14px 16px 13px 16px;
}
.ff-kpi-label { font-size: 12px; color: var(--muted); font-weight: 500; }
.ff-kpi-value {
    font-size: 24px; font-weight: 600; color: var(--text); margin-top: 4px;
    font-variant-numeric: tabular-nums; letter-spacing: -0.01em;
}
.ff-kpi-note { font-size: 11.5px; color: var(--muted); margin-top: 2px; }

.ff-card-title { font-size: 15px; font-weight: 600; color: var(--text); line-height: 1.3; }
.ff-card-sub { font-size: 12.5px; color: var(--muted); margin: 2px 0 10px 0; line-height: 1.45; }
.ff-footnote { font-size: 11.5px; color: var(--muted); margin-top: 8px; line-height: 1.5; }

.ff-legend { display: flex; flex-wrap: wrap; gap: 14px; align-items: center; font-size: 11.5px; color: var(--muted); margin-top: 8px; }
.ff-swatch { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; vertical-align: -1px; }
.ff-legend-lead { color: var(--text); font-weight: 500; }
.ff-legend-note { font-style: italic; }

.ff-status { font-size: 11.5px; color: var(--muted); margin: 6px 0 2px 0; }
[data-testid="stMain"] hr { margin: 0.5rem 0 0.6rem 0; }
.ff-status b { color: var(--text); font-weight: 600; }

.ff-brief-eyebrow {
    font-size: 11px; font-weight: 600; color: var(--accent); letter-spacing: 0.06em;
    text-transform: uppercase; margin: 6px 0 6px 0;
}
.ff-brief-headline { font-size: 17px; font-weight: 600; color: var(--text); line-height: 1.4; margin-bottom: 8px; }
.ff-brief-summary { font-size: 13.5px; color: #374151; line-height: 1.6; margin-bottom: 14px; }
.ff-brief-label { font-size: 12px; font-weight: 600; color: var(--text); margin: 12px 0 6px 0; }
.ff-brief ol { margin: 0; padding-left: 18px; }
.ff-brief li { font-size: 13px; color: #374151; line-height: 1.55; margin-bottom: 4px; padding-left: 2px; }
.ff-brief li::marker { color: var(--accent); font-weight: 600; }
.ff-figures { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
.ff-figure {
    background: var(--bg); border-radius: 6px; padding: 7px 10px;
    font-size: 12px; color: #374151; line-height: 1.4; font-variant-numeric: tabular-nums;
}
.ff-empty { font-size: 13px; color: var(--muted); line-height: 1.6; padding: 6px 0 2px 0; }
.ff-empty ul { margin: 6px 0 0 0; padding-left: 18px; }
.ff-notice {
    border: 1px solid var(--border-strong); border-left: 3px solid var(--accent);
    background: var(--accent-soft); border-radius: 6px;
    padding: 10px 12px; font-size: 13px; color: var(--text); margin-top: 8px;
}

.stButton > button {
    width: 100%; border-radius: 6px; font-size: 13px; font-weight: 500;
    padding: 0.3rem 0.75rem; min-height: 2.2rem; justify-content: flex-start;
}
[data-testid="stMultiSelectTagsContainer"] span[data-tag] {
    background: var(--accent-soft); color: var(--text); border-radius: 4px; font-size: 12px;
}

.meth-kicker { font-size: 11px; font-weight: 600; color: var(--accent); letter-spacing: 0.08em; text-transform: uppercase; }
.meth-section-kicker { display: block; margin-top: 1.75rem; }
.meth-h { font-size: 22px; font-weight: 650; color: var(--text); letter-spacing: -0.015em; margin: 0 0 0.55rem 0; }
.meth-lead { font-size: 15px; color: #374151; line-height: 1.65; max-width: 78ch; margin: 0 0 1.6rem 0; }
.meth-close { font-size: 15px; color: var(--text); line-height: 1.65; max-width: 78ch; margin: 0.8rem 0 0 0; }
.meth-card {
    background: var(--card); border: 1px solid var(--border); border-radius: 8px;
    padding: 12px 14px; height: 100%;
}
.meth-card.compact p { font-size: 12.5px; color: #4b5563; line-height: 1.45; margin: 4px 0 0 0; }
.meth-card-title { font-size: 14px; font-weight: 600; color: var(--text); }
.meth-card ul { margin: 8px 0 0 0; padding-left: 18px; }
.meth-card li { font-size: 12.5px; color: #374151; line-height: 1.45; margin-bottom: 3px; }
.meth-facts { display: flex; gap: 24px; margin: 12px 0 4px 0; font-size: 14px; color: var(--text); }
.meth-facts b { font-size: 20px; font-weight: 650; display: block; font-variant-numeric: tabular-nums; }
.meth-compare { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 8px 0 4px 0; }
.meth-compare-col { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; }
.meth-available { border-left: 3px solid #64748b; }
.meth-withheld { border-left: 3px solid var(--accent); }
.meth-compare ul { margin: 8px 0 0 0; padding-left: 18px; }
.meth-compare li { font-size: 13px; color: #374151; line-height: 1.45; margin-bottom: 3px; }
.meth-timeline { display: grid; grid-template-columns: 7fr 1.5fr 1.5fr; gap: 6px; margin: 10px 0 6px 0; }
.meth-tl-seg { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; }
.meth-tl-70 { border-top: 3px solid #94a3b8; }
.meth-tl-15 { border-top: 3px solid var(--accent); }
.meth-tl-label { font-size: 12.5px; font-weight: 600; color: var(--text); }
.meth-tl-dates { font-size: 12px; color: var(--muted); margin-top: 2px; font-variant-numeric: tabular-nums; }
.meth-metric { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; height: 100%; }
.meth-metric.lead { border-left: 3px solid var(--accent); }
.meth-metric-label { font-size: 12px; color: var(--muted); font-weight: 500; }
.meth-metric-value { font-size: 24px; font-weight: 650; color: var(--text); margin-top: 4px; font-variant-numeric: tabular-nums; letter-spacing: -0.02em; }
.meth-metric-note { font-size: 12px; color: var(--muted); margin-top: 2px; line-height: 1.4; }
.meth-arch { display: flex; flex-direction: column; gap: 8px; margin: 8px 0; }
.meth-layer { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; }
.meth-layer-title { font-size: 11px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); margin-bottom: 8px; }
.meth-pills { display: flex; flex-wrap: wrap; gap: 6px; }
.meth-pill { background: var(--bg); border: 1px solid var(--border); border-radius: 999px; padding: 4px 10px; font-size: 12.5px; color: var(--text); }
.meth-flow { display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; margin: 8px 0; }
.meth-flow-step { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; }
.meth-flow-n { font-size: 11px; font-weight: 650; color: var(--accent); }
.meth-flow-title { font-size: 13.5px; font-weight: 600; color: var(--text); margin: 2px 0 4px 0; }
.meth-flow-body { font-size: 12.5px; color: #4b5563; line-height: 1.4; }
.meth-stack-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; margin: 8px 0; }
.meth-stack { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 12px; }
.meth-stack-tools { font-size: 13.5px; color: var(--text); margin-top: 6px; line-height: 1.4; }
.meth-pipe { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin: 8px 0 4px 0; }
.meth-pipe-step { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 8px 12px; font-size: 13px; font-weight: 600; color: var(--text); }
.meth-pipe-arrow { color: var(--muted); font-size: 14px; }
@media (max-width: 1100px) {
    .meth-compare, .meth-flow, .meth-stack-grid, .meth-timeline { grid-template-columns: 1fr; }
}
</style>
"""


@st.cache_data
def load_orders() -> pd.DataFrame:
    return pd.read_parquet(DATA_PATH)


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown('<div class="ff-card-title">Filters</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="ff-card-sub">Applies to the dashboard metrics and charts.</div>', unsafe_allow_html=True)
    mask = pd.Series(True, index=df.index)
    for col in FILTER_COLUMNS:
        options = sorted(df[col].dropna().unique())
        selected = st.sidebar.multiselect(col, options, default=options)
        mask &= df[col].isin(selected)
    return df[mask]


def card_header(title: str, subtitle: str = "") -> None:
    sub = f'<div class="ff-card-sub">{subtitle}</div>' if subtitle else '<div style="height:8px"></div>'
    st.markdown(f'<div class="ff-card-title">{title}</div>{sub}', unsafe_allow_html=True)


def kpi(label: str, value: str, note: str) -> str:
    return (
        f'<div class="ff-kpi"><div class="ff-kpi-label">{label}</div>'
        f'<div class="ff-kpi-value">{value}</div><div class="ff-kpi-note">{note}</div></div>'
    )


def render_kpis(df: pd.DataFrame, total_orders: int) -> None:
    values = [
        ("Total Orders", f"{len(df):,}", f"of {total_orders:,} in dataset"),
        ("Late Delivery Rate", f"{df[TARGET].mean():.1%}" if len(df) else "-", "Actual late deliveries"),
        ("Gross Sales", f"${df['gross_sales'].sum() / 1e6:,.2f}M", "Sum of order lines"),
        ("Average Delay", f"{df['delay_days'].mean():+.2f} d" if len(df) else "-", "Real minus scheduled days"),
    ]
    for col, (label, value, note) in zip(st.columns(4, gap="small"), values):
        col.markdown(kpi(label, value, note), unsafe_allow_html=True)


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


def _fit_view(geo: pd.DataFrame) -> pdk.ViewState:
    lat_min, lat_max = geo["lat"].min(), geo["lat"].max()
    lon_min, lon_max = geo["lon"].min(), geo["lon"].max()
    lon_span = max(lon_max - lon_min, 1.0) * 1.15
    zoom = math.log2(MAP_WIDTH_ESTIMATE_PX * 360 / (lon_span * 512))
    return pdk.ViewState(
        latitude=float((lat_min + lat_max) / 2),
        longitude=float((lon_min + lon_max) / 2),
        zoom=float(min(max(zoom, 0.6), 6)),
        pitch=0,
        bearing=0,
    )


def render_map(df: pd.DataFrame, metric_label: str) -> None:
    geo = aggregate_geo(df)
    if geo.empty:
        st.info("No orders match the current filters.")
        return

    scale = build_scale(metric_label, geo[MAP_METRICS[metric_label]])
    legend = scale.legend()
    geo["bin"] = scale.bins(geo[MAP_METRICS[metric_label]])
    geo["color"] = geo["bin"].map(lambda b: hex_to_rgb(legend[b][0]))
    geo["late_rate_pct"] = (geo["late_rate"] * 100).round(1)
    geo["gross_sales_fmt"] = geo["gross_sales"].map("{:,.0f}".format)
    geo["orders_fmt"] = geo["orders"].map("{:,}".format)
    # Higher values are drawn last so they stay visible where markers overlap.
    geo = geo.sort_values("bin", kind="stable")

    ripples = [
        pdk.Layer(
            "ScatterplotLayer",
            data=geo,
            get_position="[lon, lat]",
            get_radius=MARKER_RADIUS_PX * multiple,
            radius_units="pixels",
            get_line_color=f"[color[0], color[1], color[2], {alpha}]",
            line_width_units="pixels",
            get_line_width=1,
            stroked=True,
            filled=False,
            pickable=False,
        )
        for multiple, alpha in reversed(RIPPLE_RINGS)
    ]
    markers = pdk.Layer(
        "ScatterplotLayer",
        data=geo,
        get_position="[lon, lat]",
        get_radius=MARKER_RADIUS_PX,
        radius_units="pixels",
        get_fill_color="[color[0], color[1], color[2], 235]",
        get_line_color=[255, 255, 255, 220],
        line_width_units="pixels",
        get_line_width=0.75,
        stroked=True,
        filled=True,
        pickable=True,
        auto_highlight=True,
    )
    tooltip = {
        "html": (
            "<div style='font-weight:600;margin-bottom:2px'>{lat}\u00b0, {lon}\u00b0</div>"
            "Orders: {orders_fmt}<br/>Gross sales: ${gross_sales_fmt}<br/>Late rate: {late_rate_pct}%"
        ),
        "style": {
            "backgroundColor": "#ffffff",
            "color": "#1d2433",
            "fontSize": "12px",
            "border": "1px solid #e4e7eb",
            "borderRadius": "6px",
            "boxShadow": "0 2px 8px rgba(16,24,40,0.08)",
        },
    }
    st.pydeck_chart(
        pdk.Deck(
            layers=[*ripples, markers],
            initial_view_state=_fit_view(geo),
            views=[pdk.View("MapView", controller={"dragRotate": False, "touchRotate": False})],
            map_provider="carto",
            map_style="light",
            tooltip=tooltip,
        ),
        height=MAP_HEIGHT,
    )

    swatches = "".join(
        f'<span><span class="ff-swatch" style="background:{color}"></span>{html.escape(label)}</span>'
        for color, label in legend
    )
    st.markdown(
        f'<div class="ff-legend"><span class="ff-legend-lead">Marker size is constant. '
        f"Color represents {html.escape(metric_label.lower())}.</span>{swatches}"
        f'<span class="ff-legend-note">{html.escape(scale.description)}</span></div>',
        unsafe_allow_html=True,
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


def render_flow_view(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No orders match the current filters.")
        return
    flows = df.assign(**{OUTCOME_COLUMN: df[TARGET].map(OUTCOME_LABELS)})
    svg = sankey_svg(
        flows,
        ["Market", "Shipping Mode", OUTCOME_COLUMN],
        rate_column=TARGET,
        node_colors={"Late": ACCENT, "Not late": "#94a3b8"},
        link_colors={"Late": ACCENT, "Not late": "#94a3b8"},
        width=SANKEY_WIDTH,
        height=SANKEY_HEIGHT,
    )
    # st.html sanitizes inline SVG away; an iframe keeps it and its hover titles.
    # The markup is generated here from dataset values with every label escaped.
    st.iframe(
        '<body style="margin:0;overflow:hidden;'
        "font-family:'Source Sans 3','Source Sans Pro','Segoe UI',system-ui,sans-serif;\">"
        f"{svg}</body>",
        height="content",
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


def render_decision_controls() -> None:
    clicked = [
        (label, action)
        for label, action in AI_ACTIONS.items()
        if st.button(label, key=f"ai_{action}", width="stretch")
    ]
    if clicked:
        handle_ai_click(*clicked[0])

    last_action = st.session_state[AI_LAST_ACTION_KEY]
    result = st.session_state[AI_RESULT_KEY]
    if last_action is None:
        status = "Ready"
    elif result is not None and result.get("error") == AI_LIMIT_MESSAGE:
        status = f"Last run blocked: {last_action}"
    elif result is not None and "error" in result:
        status = f"Last run failed: {last_action}"
    else:
        status = f"Last run: {last_action}"
    usage = f"AI demo calls used: {st.session_state[AI_CALL_COUNT_KEY]} / {AI_CALL_LIMIT}"
    st.markdown(
        f'<div class="ff-status"><b>{html.escape(status)}</b> &nbsp;\u00b7&nbsp; {html.escape(usage)}</div>',
        unsafe_allow_html=True,
    )


def render_decision_brief() -> None:
    result = st.session_state[AI_RESULT_KEY]
    if result is None:
        st.markdown(
            '<div class="ff-empty">Select an analysis to generate an operational brief for the '
            "latest scored period.<ul>"
            "<li><b>Analyze Hotspots</b>: regions ranked by revenue exposure</li>"
            "<li><b>Prioritize Interventions</b>: orders to review first</li>"
            "<li><b>Compare Shipping Modes</b>: where risk concentrates by mode</li>"
            "</ul></div>",
            unsafe_allow_html=True,
        )
        return

    if "error" in result:
        st.markdown(f'<div class="ff-notice">{html.escape(result["error"])}</div>', unsafe_allow_html=True)
        return

    brief = result["brief"]
    recommendations = "".join(f"<li>{html.escape(r)}</li>" for r in brief.get("recommendations", []))
    figures = "".join(
        f'<div class="ff-figure">{html.escape(m)}</div>' for m in brief.get("supporting_metrics", [])
    )
    st.markdown(
        f"""
<div class="ff-brief">
  <div class="ff-brief-eyebrow">{html.escape(result["label"])}</div>
  <div class="ff-brief-headline">{html.escape(brief.get("headline", ""))}</div>
  <div class="ff-brief-summary">{html.escape(brief.get("summary", ""))}</div>
  <div class="ff-brief-label">Recommended actions</div>
  <ol>{recommendations}</ol>
  <div class="ff-brief-label">Supporting figures</div>
  <div class="ff-figures">{figures}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_dashboard() -> None:
    orders = load_orders()
    df = apply_filters(orders)

    st.markdown(
        '<div class="ff-header"><div><div class="ff-title">FlowForge</div>'
        '<div class="ff-subtitle">Supply Chain Decision Intelligence</div></div>'
        '<div class="ff-scope">DataCo supply chain orders<br/>Order-level view, one row per order</div></div>',
        unsafe_allow_html=True,
    )
    render_kpis(df, len(orders))
    st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)

    map_col, brief_col = st.columns([3, 2], gap="medium")
    with brief_col, st.container(border=True):
        card_header(
            "Operational Brief",
            "Decision assistant for the latest scored period. All figures come from "
            "deterministic analytics; the language model only summarizes them.",
        )
        render_decision_controls()
        st.divider()
        render_decision_brief()

    with map_col, st.container(border=True):
        card_header("Geographic Distribution", "Orders aggregated to a 0.1\u00b0 grid of recorded store coordinates.")
        metric_label = st.segmented_control(
            "Map metric", list(MAP_METRICS), default="Order Volume", label_visibility="collapsed"
        ) or "Order Volume"
        render_map(df, metric_label)

    left, right = st.columns(2, gap="medium")
    with left, st.container(border=True):
        card_header("Top 10 Regions by Late Delivery Rate", "Ranked by actual late rate within the current filters.")
        render_region_table(df)
    with right, st.container(border=True):
        card_header("Shipping Mode Performance", "Scheduled versus actual shipping days by mode.")
        render_shipping_table(df)

    with st.container(border=True):
        card_header(
            "Flow View",
            "How orders move from market to shipping mode to delivery outcome. "
            "Band width is proportional to order count; hover a band for counts and late rate.",
        )
        render_flow_view(df)
        st.markdown(
            '<div class="ff-footnote">Aggregated category flows, not physical shipment routes. '
            "Delivery outcome is the actual Late_delivery_risk flag.</div>",
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(page_title="FlowForge Control Tower", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    init_ai_state()
    page = st.navigation(
        [
            st.Page(render_dashboard, title="Dashboard", icon=":material/space_dashboard:", default=True),
            st.Page("methodology_page.py", title="Methodology", icon=":material/menu_book:", url_path="methodology"),
        ]
    )
    page.run()


main()
