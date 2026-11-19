# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional

import numpy as np
from translations import tr


try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - optional dependency
    go = None


PLOTLY_AVAILABLE = go is not None


def plot_breakthrough_interactive(
    time_min,
    c_out_mg_l,
    c_out_c0,
    breakthrough_ratio: float,
    breakthrough_time_min: Optional[float] = None,
    ebct_min: Optional[float] = None,
    title: str = "Breakthrough curve",
    lang: str = "uk",
):
    if go is None:
        return None

    time = np.asarray(time_min, dtype=float)
    concentration = np.asarray(c_out_mg_l, dtype=float)
    ratio = np.asarray(c_out_c0, dtype=float)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time,
            y=ratio,
            mode="lines",
            name="C/C0",
            customdata=np.column_stack([concentration]),
            hovertemplate=tr(lang, "ui_plot_hover"),
        )
    )
    fig.add_hline(
        y=float(breakthrough_ratio),
        line_dash="dash",
        line_color="red",
        annotation_text=tr(lang, "breakthrough_boundary_label").format(value=f"{breakthrough_ratio:.3f}"),
        annotation_position="bottom right",
    )
    if breakthrough_time_min is not None and np.isfinite(breakthrough_time_min):
        fig.add_vline(
            x=float(breakthrough_time_min),
            line_dash="dashdot",
            line_color="orange",
            annotation_text=tr(lang, "ui_plot_model_tb").format(relation="=", value=f"{breakthrough_time_min:.2f}"),
            annotation_position="top",
        )
    if ebct_min is not None and np.isfinite(ebct_min):
        fig.add_vline(
            x=float(ebct_min),
            line_dash="dot",
            line_color="gray",
            annotation_text=tr(lang, "ebct_label").format(value=f"{ebct_min:.2f}"),
            annotation_position="bottom",
        )

    fig.update_layout(
        title=title,
        xaxis_title=tr(lang, "time_min"),
        yaxis_title="C/C0",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
