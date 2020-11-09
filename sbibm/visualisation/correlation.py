from __future__ import annotations

from math import exp, log  # noqa
from typing import Any, Dict, List, Optional

import pandas as pd
import deneb as den
import altair as alt

import sbibm
from sbibm.utils.io import get_ndarray_from_csv
from deneb.utils import save
from sbibm.visualisation.colors import COLORS_RGB_STR
from sbibm.utils.torch import sample


def fig_correlation(
    df: pd.DataFrame,
    metrics: List[str] = ["C2ST", "MMD", "KSD", "MEDDIST"],
    config: str = "manuscript",
    title: Optional[str] = None,
    title_dx: int = 0,
    width: Optional[int] = None,
    height: Optional[int] = None,
    keywords: Dict[str, Any] = {},
    style: Dict[str, Any] = {},
):
    keywords["sparse"] = True
    keywords["limits"] = [0.0, 1.0]
    keywords["font_size"] = 14
    keywords["rotate_outwards"] = True

    if config == "manuscript":
        style["font_family"] = "Inter"
        keywords["width"] = 200 if width is None else width
        keywords["height"] = 200 if height is None else height

    if config == "streamlit":
        keywords["width"] = None if width is None else width
        keywords["height"] = None if height is None else height

    alt.themes.enable("default")

    den.set_style(
        extra={
            "config": {
                "axisX": {
                    "domain": False,
                    "domainWidth": 0,
                    "ticks": False,
                    "tickWidth": 0,
                    "grid": False,
                },
                "axisY": {
                    "domain": False,
                    "domainWidth": 0,
                    "ticks": False,
                    "tickWidth": 0,
                    "grid": False,
                },
            }
        },
        **style,
    )

    chart = den.correlation_matrix(df, metrics=metrics, **keywords)

    if title is not None:
        chart = chart.properties(title={"text": [title],}).configure_title(
            fontSize=12, offset=10, orient="top", anchor="middle", dx=title_dx
        )

    if config == "manuscript":
        chart = chart.configure_text(font="Inter")

    return chart
