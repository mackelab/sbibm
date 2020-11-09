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


def fig_metric(
    df: pd.DataFrame,
    metric: str,
    config: str = "manuscript",
    title: Optional[str] = None,
    title_dx: int = 0,
    width: Optional[int] = None,
    height: Optional[int] = None,
    labels: bool = True,
    keywords: Dict[str, Any] = {},
    style: Dict[str, Any] = {},
    default_color: str = "#000000",
):
    colors = COLORS_RGB_STR
    for algorithm in df.algorithm.unique():
        algorithm_first = algorithm.split("_")[-1].split("-")[0]
        if algorithm_first not in colors:
            colors[algorithm] = default_color
        else:
            colors[algorithm] = colors[algorithm_first]

    keywords["column_labels"] = labels
    keywords["color"] = den.colorscale(colors, shorthand="algorithm:N")

    if config == "manuscript":
        keywords["width"] = 700 / len(df.algorithm.unique()) if width is None else width
        keywords["height"] = 60 if height is None else height
        style["font_family"] = "Inter"

    if config == "streamlit":
        keywords["width"] = None if width is None else width
        keywords["height"] = None if height is None else height
        style["font_family"] = "Inter"
        style["font_size"] = 16
        style["font_size_label"] = 16
        style["font_size_title"] = 16

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
                    # "titleX": 10,
                    "tickCount": 5.0,
                },
            }
        },
        **style,
    )

    chart = den.lineplot(
        df,
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
            fontSize=12, offset=10, orient="top", anchor="middle", dx=title_dx
        )

    return chart


def fig_posterior(
    task_name: str,
    num_observation: int = 1,
    num_samples: int = 1000,
    prior: bool = False,
    reference: bool = True,
    true_parameter: bool = False,
    samples_path: Optional[str] = None,
    samples_name: Optional[str] = None,
    samples_color: Optional[str] = None,
    title: Optional[str] = None,
    title_dx: int = 0,
    legend: bool = True,
    seed: int = 101,
    config: str = "manuscript",
    width: Optional[int] = None,
    height: Optional[int] = None,
    **kwargs: Any,
):
    # Samples to plot
    task = sbibm.get_task(task_name)
    samples = []
    labels_samples = []
    colors = {}

    samples_prior = task.get_prior()(num_samples=num_samples)
    if prior:
        sample_name = "Prior"
        samples.append(samples_prior.numpy())
        labels_samples.append(sample_name)
        colors[sample_name].append("#ccc")

    if reference:
        sample_name = "Posterior"
        samples_reference = sample(
            task.get_reference_posterior_samples(
                num_observation=num_observation
            ).numpy(),
            num_samples,
            replace=False,
            seed=seed,
        )
        samples.append(samples_reference)
        labels_samples.append(sample_name)
        colors[sample_name] = "#888888"

    if true_parameter:
        sample_name = "True parameter"
        samples.append(
            task.get_true_parameters(num_observation=num_observation)
            .repeat(num_samples, 1)
            .numpy()
        )
        labels_samples.append(sample_name)
        colors[sample_name] = "#000"

    if samples_path is not None:
        samples_ = get_ndarray_from_csv(samples_path)
        samples_algorithm = sample(samples_, num_samples, replace=False, seed=seed)
        samples.append(samples_algorithm)
        if samples_name is None:
            sample_name = "Algorithm"
        else:
            sample_name = samples_name
        labels_samples.append(sample_name)
        if samples_color is not None:
            colors[sample_name] = samples_color
        else:
            if sample_name in COLORS_RGB_STR:
                colors[sample_name] = COLORS_RGB_STR[samples_name]
            else:
                colors[sample_name] = "#888"

    if len(samples) == 0:
        return None

    for s in samples:
        assert s.shape[0] == num_samples

    numbers_unicode = ["₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉", "₁₀"]
    labels_dim = [f"θ{numbers_unicode[i+1]}" for i in range(task.dim_parameters)]

    df = den.np2df(
        samples=[sample for sample in samples],
        field="sample",
        labels_samples=labels_samples,
        labels_dim=labels_dim,
    )

    style = {}
    keywords = {}

    keywords["color"] = den.colorscale(colors, shorthand="sample:N", legend=legend)

    limits_auto = "prior"
    _LIMITS_ = {
        "gaussian_linear": [-1.0, +1.0],
        "gaussian_linear_uniform": [-1.0, +1.0],
        "two_moons": [-1.0, +1.0],
        "slcp": [-3.0, +3.0],
    }
    if task_name in _LIMITS_:
        limits = _LIMITS_[task_name]
    else:
        if limits_auto == "prior":
            limits = [
                list(i)
                for i in zip(
                    samples_prior.min(dim=0)[0].tolist(),
                    samples_prior.max(dim=0)[0].tolist(),
                )
            ]
        else:
            limits = None
    keywords["limits"] = limits

    keywords["num_bins"] = 40

    if config == "manuscript":
        style["font_family"] = "Inter"
        keywords["width"] = 100 if width is None else width
        keywords["height"] = 100 if height is None else height

    if config == "streamlit":
        size = 500 / task.dim_parameters
        keywords["width"] = size if width is None else width
        keywords["height"] = size if height is None else height
        style["font_family"] = "Inter"
        style["font_size"] = 16
        style["font_size_label"] = 16
        style["font_size_title"] = 16

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

    chart = den.pairplot(
        df, field="sample", scatter_size=1.0, bar_opacity=0.4, **keywords,
    )

    if title is not None:
        chart = chart.properties(title={"text": [title],}).configure_title(
            fontSize=12, offset=10, orient="top", anchor="middle", dx=title_dx
        )

    return chart
