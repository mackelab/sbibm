from typing import Any, Dict, List, Optional

import altair as alt
import deneb as den
import pandas as pd


def fig_metric(
    df: pd.DataFrame,
    metric: str,
    title: Optional[str] = None,
    title_dx: int = 0,
    width: Optional[int] = None,
    height: Optional[int] = None,
    labels: bool = True,
    keywords: Dict[str, Any] = {},
    style: Dict[str, Any] = {},
    default_color: str = "#000000",
    colors_dict: Dict[str, Any] = {},
    config: Optional[str] = None,
):
    """Plots metrics

    Args:
        df: Dataframe which at least has columns `algorithm`, `num_simulations` and
            a column titled accordingly to `metric`. 
        metric: Metric to plot, should be a column in `df`.
        title: Title for plot
        title_dx: x-direction offset for title
        labels: Whether to plot labels
        seed: Seed
        width: Width
        height: Height
        default_color: Default color of samples
        colors_dict: Dictionary of colors
        config: Optional string to load predefined config
        style: Optional dictionary for `den.set_style`
        keywords: Optional dictionary passed on to `den.lineplot`

    Returns:
        Chart

    Note:
        Due to an open issue on vega-lite, it is difficult to sort columns in
        non-alphabetical fashion (i.e., ordered by algorithm name). As a workaround,
        consider prepending an algorithm with a space to have it listed first, e.g.
        `df.loc[df["algorithm"] == "REJ-ABC", "algorithm"] = " REJ-ABC"`.
        See also: https://github.com/vega/vega-lite/issues/5366/
    """
    colors = {}
    for algorithm in df.algorithm.unique():
        algorithm_first = algorithm.split("_")[-1].strip()
        if algorithm_first not in colors_dict:
            colors[algorithm] = default_color
        else:
            colors[algorithm] = colors_dict[algorithm_first]

    keywords["column_labels"] = labels
    keywords["color"] = den.colorscale(colors, shorthand="algorithm:N")

    if config == "manuscript":
        keywords["width"] = 700 / len(df.algorithm.unique()) if width is None else width
        keywords["height"] = 60 if height is None else height
        style["font_size"] = 12
        style["font_family"] = "Inter"

    if config == "streamlit":
        keywords["width"] = None if width is None else width
        keywords["height"] = None if height is None else height
        style["font_size"] = 16
        # style["font_size_label"] = 16
        # style["font_size_title"] = 16

    keywords["limits"] = None
    keywords["log_y"] = False
    keywords["y_axis"] = alt.Axis(title=metric)

    if metric == "MMD":
        keywords["y_axis"] = alt.Axis(title="MMD²")

    if metric == "C2ST":
        keywords["limits"] = [0.5, 1.0]

    if metric == "RT":
        keywords["log_y"] = True
        keywords["limits"] = [0.001, 1000.0]
        keywords["y_axis"] = alt.Axis(
            values=[0.001, 0.01, 0.1, 0.0, 1.0, 10.0, 100.0, 1000.0]
        )

    alt.themes.enable("default")

    den.set_style(
        extra={
            "config": {
                "axisX": {
                    "grid": False,
                    "labelAngle": 270,
                    "domain": False,
                    "domainWidth": 0,
                    "ticks": True,
                    "tickWidth": 0,
                    "minExtent": 0,
                },
                "axisY": {
                    "domain": False,
                    "domainWidth": 0,
                    "ticks": True,
                    "tickWidth": 0,
                    "grid": True,
                    "titlePadding": 0,
                    "tickCount": 5.0,
                },
            }
        },
        **style,
    )

    chart = den.lineplot(
        df.sort_values("algorithm"),
        x="num_simulations:O",
        y=f"{metric}:Q",
        error_extent="ci",
        column="algorithm:N",
        independent_y=False,
        row_title="",
        column_title="Number of Simulations",
        title_orient="bottom",
        **keywords,
    )

    chart = chart.configure_point(size=50).configure_line(size=1.5)

    if title is not None:
        chart = chart.properties(title={"text": [title],}).configure_title(
            offset=10, orient="top", anchor="middle", dx=title_dx,
        )

    return chart
