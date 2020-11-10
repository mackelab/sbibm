from typing import Any, Dict, List, Optional

import altair as alt
import deneb as den
import torch

import sbibm
from sbibm.utils.io import get_ndarray_from_csv
from sbibm.utils.torch import sample


def fig_posterior(
    task_name: str,
    num_observation: int = 1,
    num_samples: int = 1000,
    prior: bool = False,
    reference: bool = True,
    true_parameter: bool = False,
    samples_path: Optional[str] = None,
    samples_tensor: Optional[torch.Tensor] = None,
    samples_name: Optional[str] = None,
    samples_color: Optional[str] = None,
    title: Optional[str] = None,
    title_dx: int = 0,
    legend: bool = True,
    seed: int = 101,
    config: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    default_color: str = "#000000",
    colors_dict: Dict[str, Any] = {},
    **kwargs: Any,
):
    """Plots posteriors samples for given task

    Args:
        task_name: Name of the task to plot posteriors for
        num_observation: Observation number
        num_samples: Number of samples to use for plotting
        prior: Whether or not to plot prior samples
        reference: Whether or not to plot reference posterior samples
        samples_path: If specified, will load samples from disk from path
        samples_tensor: Instead of a path, samples can also be passed as torch.Tensor
        samples_name: Name for samples, defaults to "Algorithm"
        samples_color: Optional string for color of samples
        title: Title for plot
        title_dx: x-direction offset for title
        legend: Whether to plot a legend
        seed: Seed
        config: Optional string to load predefined config
        width: Width
        height: Height
        default_color: Default color of samples
        colors_dict: Dictionary of colors

    Returns:
        Chart
    """
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
        colors[sample_name] = "#888"

    if true_parameter:
        sample_name = "True parameter"
        samples.append(
            task.get_true_parameters(num_observation=num_observation)
            .repeat(num_samples, 1)
            .numpy()
        )
        labels_samples.append(sample_name)
        colors[sample_name] = "#000"

    if samples_tensor is not None or samples_path is not None:
        if samples_tensor is not None:
            samples_ = samples_tensor.numpy()
        else:
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
            if sample_name in colors_dict:
                colors[sample_name] = colors_dict[samples_name]
            else:
                colors[sample_name] = default_color

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
