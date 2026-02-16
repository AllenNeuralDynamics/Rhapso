"""
Detection QC plotting — diagnostic figures for IP detection parameter sweeps.

All plotting functions accept an output_dir and save PNGs directly.
The caller (R2R step or capsule) is responsible for setting output_dir
to the appropriate location (e.g. /results/ on Code Ocean).
"""
import logging
import os
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np

from Rhapso.evaluation.detection_qc.sweep_analyzer import SweepTrialResult

logger = logging.getLogger(__name__)

# Colorblind-safe palette (IBM Design)
COLORS_PASS = "#648FFF"
COLORS_FAIL = "#DC267F"
COLORS_NEUTRAL = "#785EF0"
COLORS_SELECTED = "#FE6100"


def plot_sweep_ip_counts(
    trials: List[SweepTrialResult],
    target_interest_points: int,
    output_dir: str,
    filename: str = "sweep_ip_counts.png",
) -> str:
    """Bar chart of mean IP count per trial, colored by pass/fail.

    Parameters
    ----------
    trials : list of SweepTrialResult
        All sweep trials.
    target_interest_points : int
        Target threshold (drawn as horizontal line).
    output_dir : str
        Directory to save the figure.
    filename : str
        Output filename.

    Returns
    -------
    str
        Full path to the saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(6, len(trials) * 0.8), 5))

    labels = []
    means = []
    colors = []
    for t in trials:
        labels.append(f"s{t.multiscale}\n\u03c3={t.sigma}")
        means.append(t.mean_ip_count)
        colors.append(COLORS_PASS if t.success else COLORS_FAIL)

    x = np.arange(len(trials))
    bars = ax.bar(x, means, color=colors, edgecolor="black", linewidth=0.5)

    ax.axhline(
        y=target_interest_points,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=f"Target ({target_interest_points})",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Mean IP Count per View")
    ax.set_xlabel("Trial (scale / sigma)")
    ax.set_title("Parameter Sweep: Mean IP Count by Trial")
    ax.legend(loc="upper right")

    # Annotate pass/fail
    for i, (bar, trial) in enumerate(zip(bars, trials)):
        rate_text = f"{trial.success_rate * 100:.0f}%"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + target_interest_points * 0.02,
            rate_text,
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
        )

    fig.tight_layout()
    output_path = os.path.join(output_dir, filename)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved sweep IP counts plot to {output_path}")
    return output_path


def plot_sweep_success_rates(
    trials: List[SweepTrialResult],
    output_dir: str,
    filename: str = "sweep_success_rates.png",
) -> str:
    """Bar chart of view success rates per trial.

    Parameters
    ----------
    trials : list of SweepTrialResult
        All sweep trials.
    output_dir : str
        Directory to save the figure.
    filename : str
        Output filename.

    Returns
    -------
    str
        Full path to the saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(6, len(trials) * 0.8), 4))

    labels = [f"s{t.multiscale}\n\u03c3={t.sigma}" for t in trials]
    rates = [t.success_rate * 100 for t in trials]
    colors = [COLORS_PASS if t.success else COLORS_FAIL for t in trials]

    x = np.arange(len(trials))
    ax.bar(x, rates, color=colors, edgecolor="black", linewidth=0.5)

    ax.axhline(y=50, color="black", linestyle="--", linewidth=1.0, label="50% threshold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Views Meeting Target (%)")
    ax.set_xlabel("Trial (scale / sigma)")
    ax.set_title("Parameter Sweep: View Success Rate")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right")

    fig.tight_layout()
    output_path = os.path.join(output_dir, filename)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved sweep success rates plot to {output_path}")
    return output_path


def plot_per_view_ip_distribution(
    trials: List[SweepTrialResult],
    target_interest_points: int,
    output_dir: str,
    filename: str = "per_view_ip_distribution.png",
) -> str:
    """Box plot of IP counts per view for each trial.

    Parameters
    ----------
    trials : list of SweepTrialResult
        All sweep trials (must have view_metrics populated).
    target_interest_points : int
        Target threshold line.
    output_dir : str
        Directory to save the figure.
    filename : str
        Output filename.

    Returns
    -------
    str
        Full path to the saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(6, len(trials) * 0.8), 5))

    data = []
    labels = []
    for t in trials:
        counts = [vm.ip_count for vm in t.view_metrics]
        data.append(counts if counts else [0])
        labels.append(f"s{t.multiscale}\n\u03c3={t.sigma}")

    bp = ax.boxplot(
        data,
        patch_artist=True,
        labels=labels,
        medianprops=dict(color="black", linewidth=1.5),
    )

    for i, (patch, trial) in enumerate(zip(bp["boxes"], trials)):
        patch.set_facecolor(COLORS_PASS if trial.success else COLORS_FAIL)
        patch.set_alpha(0.6)

    ax.axhline(
        y=target_interest_points,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=f"Target ({target_interest_points})",
    )

    ax.set_ylabel("IP Count per View")
    ax.set_xlabel("Trial (scale / sigma)")
    ax.set_title("Parameter Sweep: IP Count Distribution per View")
    ax.legend(loc="upper right")

    fig.tight_layout()
    output_path = os.path.join(output_dir, filename)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved per-view IP distribution plot to {output_path}")
    return output_path


def plot_sigma_vs_multiscale_heatmap(
    trials: List[SweepTrialResult],
    output_dir: str,
    filename: str = "sigma_multiscale_heatmap.png",
) -> str:
    """Heatmap of mean IP count across sigma x multiscale grid.

    Parameters
    ----------
    trials : list of SweepTrialResult
        All sweep trials.
    output_dir : str
        Directory to save the figure.
    filename : str
        Output filename.

    Returns
    -------
    str
        Full path to the saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build unique sorted axes
    multiscales = sorted(set(t.multiscale for t in trials))
    sigmas = sorted(set(t.sigma for t in trials), reverse=True)

    if len(multiscales) < 2 and len(sigmas) < 2:
        logger.info("Skipping heatmap — fewer than 2 unique values on both axes")
        return ""

    grid = np.full((len(sigmas), len(multiscales)), np.nan)
    success_grid = np.full((len(sigmas), len(multiscales)), False)

    ms_idx = {ms: i for i, ms in enumerate(multiscales)}
    sig_idx = {s: i for i, s in enumerate(sigmas)}

    for t in trials:
        r = sig_idx[t.sigma]
        c = ms_idx[t.multiscale]
        grid[r, c] = t.mean_ip_count
        success_grid[r, c] = t.success

    fig, ax = plt.subplots(figsize=(max(5, len(multiscales) * 1.2), max(4, len(sigmas) * 0.8)))

    im = ax.imshow(grid, cmap="YlOrRd", aspect="auto")
    cbar = fig.colorbar(im, ax=ax, label="Mean IP Count")

    ax.set_xticks(range(len(multiscales)))
    ax.set_xticklabels([f"s{ms}" for ms in multiscales])
    ax.set_yticks(range(len(sigmas)))
    ax.set_yticklabels([f"\u03c3={s}" for s in sigmas])
    ax.set_xlabel("Multiscale Level")
    ax.set_ylabel("Sigma")
    ax.set_title("Parameter Sweep: Mean IP Count Heatmap")

    # Annotate cells
    for r in range(len(sigmas)):
        for c in range(len(multiscales)):
            val = grid[r, c]
            if np.isnan(val):
                continue
            marker = "\u2713" if success_grid[r, c] else "\u2717"
            ax.text(
                c, r, f"{val:.0f}\n{marker}",
                ha="center", va="center", fontsize=8,
                color="white" if val > np.nanmax(grid) * 0.6 else "black",
            )

    fig.tight_layout()
    output_path = os.path.join(output_dir, filename)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved sigma vs multiscale heatmap to {output_path}")
    return output_path


def generate_all_plots(
    trials: List[SweepTrialResult],
    target_interest_points: int,
    output_dir: str,
) -> List[str]:
    """Generate all detection QC plots.

    Parameters
    ----------
    trials : list of SweepTrialResult
        All sweep trials.
    target_interest_points : int
        IP count target threshold.
    output_dir : str
        Directory to save all figures.

    Returns
    -------
    list of str
        Paths to all generated plot files.
    """
    if not trials:
        logger.warning("No trials to plot")
        return []

    paths = []

    path = plot_sweep_ip_counts(trials, target_interest_points, output_dir)
    if path:
        paths.append(path)

    path = plot_sweep_success_rates(trials, output_dir)
    if path:
        paths.append(path)

    # Only generate per-view plots if view metrics are populated
    has_view_metrics = any(t.view_metrics for t in trials)
    if has_view_metrics:
        path = plot_per_view_ip_distribution(trials, target_interest_points, output_dir)
        if path:
            paths.append(path)

    path = plot_sigma_vs_multiscale_heatmap(trials, output_dir)
    if path:
        paths.append(path)

    logger.info(f"Generated {len(paths)} QC plots in {output_dir}")
    return paths
