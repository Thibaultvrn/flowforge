"""Binned sequential color scales for the geographic map.

Values are only mapped to colors for display; the underlying metrics are
never modified.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

# Quiet slate for low values through a restrained burnt orange for high values.
PALETTE = ["#b9c3cd", "#e8c8a4", "#dfa06b", "#c86d33", "#8f3d12"]
N_BINS = len(PALETTE)

LATE_RATE_DOMAIN = (0.30, 0.80)


@dataclass(frozen=True)
class ColorScale:
    metric_label: str
    description: str
    to_unit: Callable[[pd.Series], pd.Series]
    from_unit: Callable[[float], float]
    fmt: Callable[[float], str]

    def bins(self, values: pd.Series) -> pd.Series:
        t = self.to_unit(values).clip(0, 1)
        return np.minimum((t * N_BINS).astype(int), N_BINS - 1)

    def legend(self) -> list[tuple[str, str]]:
        edges = [self.from_unit(k / N_BINS) for k in range(1, N_BINS)]
        labels = [f"< {self.fmt(edges[0])}"]
        labels += [f"{self.fmt(lo)} \u2013 {self.fmt(hi)}" for lo, hi in zip(edges, edges[1:])]
        labels.append(f"\u2265 {self.fmt(edges[-1])}")
        return list(zip(PALETTE, labels))


def _compact(value: float) -> str:
    for threshold, suffix in ((1e6, "M"), (1e3, "k")):
        if value >= threshold:
            return f"{value / threshold:.1f}{suffix}".replace(".0", "")
    return f"{value:.0f}"


def _log_scale(metric_label: str, values: pd.Series, money: bool) -> ColorScale:
    lo = max(float(values.min()), 1e-9)
    hi = max(float(values.max()), lo * 10)
    span = math.log(hi / lo)
    prefix = "$" if money else ""
    return ColorScale(
        metric_label=metric_label,
        description=f"log scale, {prefix}{_compact(lo)} to {prefix}{_compact(hi)} per location",
        to_unit=lambda s: np.log(s.clip(lower=lo) / lo) / span,
        from_unit=lambda t: lo * math.exp(t * span),
        fmt=lambda v: f"{prefix}{_compact(v)}",
    )


def build_scale(metric_label: str, values: pd.Series) -> ColorScale:
    if metric_label == "Late Delivery Rate":
        lo, hi = LATE_RATE_DOMAIN
        return ColorScale(
            metric_label=metric_label,
            description=f"linear scale, clipped to {lo:.0%} \u2013 {hi:.0%}",
            to_unit=lambda s: (s - lo) / (hi - lo),
            from_unit=lambda t: lo + t * (hi - lo),
            fmt=lambda v: f"{v:.0%}",
        )
    return _log_scale(metric_label, values, money=metric_label == "Gross Sales")


def hex_to_rgb(color: str) -> list[int]:
    return [int(color[i : i + 2], 16) for i in (1, 3, 5)]
