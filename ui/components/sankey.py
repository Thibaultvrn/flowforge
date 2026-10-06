"""Dependency-free SVG Sankey diagram for categorical order flows."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape

import pandas as pd


@dataclass
class _Node:
    stage: int
    label: str
    value: int
    x: float = 0.0
    y: float = 0.0
    h: float = 0.0
    out_offset: float = 0.0
    in_offset: float = 0.0


def _ribbon(x0: float, y0: float, x1: float, y1: float, h: float) -> str:
    xm = (x0 + x1) / 2
    return (
        f"M{x0:.1f},{y0:.1f} C{xm:.1f},{y0:.1f} {xm:.1f},{y1:.1f} {x1:.1f},{y1:.1f} "
        f"L{x1:.1f},{y1 + h:.1f} C{xm:.1f},{y1 + h:.1f} {xm:.1f},{y0 + h:.1f} {x0:.1f},{y0 + h:.1f} Z"
    )


def sankey_svg(
    df: pd.DataFrame,
    stages: list[str],
    *,
    rate_column: str | None = None,
    node_colors: dict[str, str] | None = None,
    link_colors: dict[str, str] | None = None,
    width: int = 1100,
    height: int = 380,
    label_space: int = 170,
    node_width: int = 10,
    node_gap: int = 12,
) -> str:
    """Render order counts flowing through consecutive categorical stages.

    Links into a node listed in ``link_colors`` take that color; others use a
    neutral tone. ``rate_column`` (a 0/1 column) adds its mean to link tooltips.
    """
    node_colors = node_colors or {}
    link_colors = link_colors or {}
    total = len(df)
    if total == 0:
        return ""

    nodes: dict[tuple[int, str], _Node] = {}
    order: list[list[_Node]] = []
    for i, column in enumerate(stages):
        counts = df[column].value_counts()
        stage_nodes = [_Node(i, str(label), int(v)) for label, v in counts.items()]
        order.append(stage_nodes)
        nodes.update({(i, n.label): n for n in stage_nodes})

    usable = height - 20
    scale = min((usable - node_gap * (len(s) - 1)) / total for s in order)
    inner_width = width - 2 * label_space - node_width
    for i, stage_nodes in enumerate(order):
        x = label_space + inner_width * i / (len(stages) - 1)
        stage_height = sum(max(n.value * scale, 1.0) for n in stage_nodes) + node_gap * (len(stage_nodes) - 1)
        y = (height - stage_height) / 2
        for n in stage_nodes:
            n.x, n.y, n.h = x, y, max(n.value * scale, 1.0)
            y += n.h + node_gap

    links_svg: list[str] = []
    for i in range(len(stages) - 1):
        src_col, dst_col = stages[i], stages[i + 1]
        agg = {"orders": (src_col, "size")}
        if rate_column:
            agg["rate"] = (rate_column, "mean")
        flows = df.groupby([src_col, dst_col], observed=True).agg(**agg).reset_index()
        rank = {n.label: k for k, n in enumerate(order[i + 1])}
        src_rank = {n.label: k for k, n in enumerate(order[i])}
        flows["_dst_rank"] = flows[dst_col].astype(str).map(rank)
        flows["_src_rank"] = flows[src_col].astype(str).map(src_rank)

        out_sorted = flows.sort_values(["_src_rank", "_dst_rank"])
        positions: dict[tuple[str, str], float] = {}
        for row in out_sorted.itertuples(index=False):
            src = nodes[(i, str(row[0]))]
            positions[(str(row[0]), str(row[1]))] = src.y + src.out_offset
            src.out_offset += row.orders * scale

        in_sorted = flows.sort_values(["_dst_rank", "_src_rank"])
        for row in in_sorted.itertuples(index=False):
            src_label, dst_label = str(row[0]), str(row[1])
            src, dst = nodes[(i, src_label)], nodes[(i + 1, dst_label)]
            h = row.orders * scale
            y0 = positions[(src_label, dst_label)]
            y1 = dst.y + dst.in_offset
            dst.in_offset += h
            color = link_colors.get(dst_label, "#cbd5e1")
            tip = f"{src_label} \u2192 {dst_label}: {row.orders:,} orders ({row.orders / total:.1%} of total)"
            if rate_column:
                tip += f", late rate {row.rate:.1%}"
            links_svg.append(
                f'<path d="{_ribbon(src.x + node_width, y0, dst.x, y1, h)}" '
                f'fill="{color}" fill-opacity="0.45"><title>{escape(tip)}</title></path>'
            )

    def label_anchor(stage: int, x: float) -> tuple[float, str]:
        return (x - 8, "end") if stage == 0 else (x + node_width + 8, "start")

    nodes_svg: list[str] = []
    last = len(stages) - 1
    for stage_nodes in order:
        for n in stage_nodes:
            color = node_colors.get(n.label, "#475569")
            nodes_svg.append(
                f'<rect x="{n.x:.1f}" y="{n.y:.1f}" width="{node_width}" height="{n.h:.1f}" '
                f'fill="{color}" rx="1"><title>{escape(n.label)}: {n.value:,} orders</title></rect>'
            )
            tx, anchor = label_anchor(n.stage, n.x)
            ty = n.y + n.h / 2
            halo = ' class="halo"' if n.stage not in (0, last) else ""
            nodes_svg.append(
                f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="{anchor}" dominant-baseline="middle"{halo}>'
                f'<tspan class="lbl">{escape(n.label)}</tspan>'
                f'<tspan class="val" dx="6">{n.value / total:.0%}</tspan></text>'
            )

    headers: list[str] = []
    for i, column in enumerate(stages):
        hx, anchor = label_anchor(i, order[i][0].x)
        headers.append(f'<text x="{hx:.1f}" y="10" text-anchor="{anchor}" class="hdr">{escape(column)}</text>')

    return f"""
<svg viewBox="0 0 {width} {height + 14}" width="100%" style="display:block" preserveAspectRatio="xMidYMid meet"
     xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Order flow diagram">
  <style>
    text {{ font-family: inherit; font-size: 12px; fill: #1d2433; }}
    .val {{ fill: #6b7280; font-size: 11px; }}
    .hdr {{ fill: #6b7280; font-size: 10.5px; letter-spacing: 0.08em; text-transform: uppercase; }}
    .halo {{ paint-order: stroke; stroke: #ffffff; stroke-width: 3px; stroke-linejoin: round; }}
    path:hover {{ fill-opacity: 0.75; }}
  </style>
  {"".join(headers)}
  <g transform="translate(0,14)">{"".join(links_svg)}{"".join(nodes_svg)}</g>
</svg>
"""
