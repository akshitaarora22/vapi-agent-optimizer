"""
results/visualize.py

Generate plots and reports from optimization results.
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import Dict, List


def plot_optimization_curve(convergence_data: Dict, output_path: str = "results/optimization_curve.png") -> None:
    """Plot reward per iteration and running best."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f0f0f")
    ax.set_facecolor("#1a1a1a")

    iters = convergence_data["iterations"]
    rewards = convergence_data["rewards"]
    running_best = convergence_data["running_best"]

    ax.scatter(iters, rewards, color="#4ade80", s=60, zorder=5, label="Iteration reward", alpha=0.8)
    ax.plot(iters, running_best, color="#facc15", linewidth=2.5, label="Running best", zorder=4)
    ax.fill_between(iters, 0, running_best, alpha=0.08, color="#facc15")

    ax.set_xlabel("Optimization Iteration", color="#888", fontsize=11)
    ax.set_ylabel("Reward (0–1)", color="#888", fontsize=11)
    ax.set_title("GP-BO Optimization Curve — Dental Scheduler", color="#eee", fontsize=13, pad=15)
    ax.tick_params(colors="#666")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    ax.set_ylim(0, 1.05)
    ax.legend(facecolor="#222", labelcolor="#ccc", framealpha=0.8)
    ax.grid(True, color="#2a2a2a", linestyle="--", alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {output_path}")


def plot_before_after(
    baseline_scores: Dict[str, float],
    final_scores: Dict[str, float],
    output_path: str = "results/before_after.png",
) -> None:
    """Bar chart comparing baseline vs final agent on each dimension."""
    dimensions = ["booking_success", "turn_efficiency", "info_completeness", "naturalness", "reward"]
    labels = ["Booking\nSuccess", "Turn\nEfficiency", "Info\nComplete", "Naturalness", "REWARD"]

    x = np.arange(len(dimensions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#0f0f0f")
    ax.set_facecolor("#1a1a1a")

    b1 = ax.bar(x - width / 2, [baseline_scores.get(d, 0) for d in dimensions],
                width, label="Baseline", color="#ef4444", alpha=0.85, zorder=3)
    b2 = ax.bar(x + width / 2, [final_scores.get(d, 0) for d in dimensions],
                width, label="Optimized", color="#4ade80", alpha=0.85, zorder=3)

    # Value labels on bars
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 0.01,
            f"{h:.2f}", ha="center", va="bottom", fontsize=9, color="#ccc"
        )

    # Highlight the reward bars
    for bar in [b1[-1], b2[-1]]:
        bar.set_edgecolor("#fff")
        bar.set_linewidth(2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="#bbb", fontsize=10)
    ax.set_ylabel("Score (0–1)", color="#888")
    ax.set_title("Before vs After Optimization — Dental Scheduler Agent", color="#eee", fontsize=13, pad=15)
    ax.set_ylim(0, 1.15)
    ax.tick_params(colors="#666")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.legend(facecolor="#222", labelcolor="#ccc", framealpha=0.8)
    ax.grid(True, axis="y", color="#2a2a2a", linestyle="--", alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {output_path}")


def print_results_table(
    baseline_scores: Dict[str, float],
    final_scores: Dict[str, float],
    baseline_config: Dict[str, int],
    final_config: Dict[str, int],
) -> None:
    """Print a clean results table to stdout."""
    from rich.console import Console
    from rich.table import Table
    from rich import box
    from agent.config import PROMPT_AXES

    console = Console()

    # Scores table
    table = Table(title="📊 Optimization Results", box=box.ROUNDED, style="dim")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Baseline", justify="center")
    table.add_column("Optimized", justify="center")
    table.add_column("Δ Change", justify="center")

    dims = ["booking_success", "turn_efficiency", "info_completeness", "naturalness", "reward"]
    for d in dims:
        b = baseline_scores.get(d, 0)
        f = final_scores.get(d, 0)
        delta = f - b
        delta_str = f"[green]+{delta:.3f}[/green]" if delta > 0 else f"[red]{delta:.3f}[/red]"
        style = "bold" if d == "reward" else ""
        table.add_row(d, f"{b:.3f}", f"{f:.3f}", delta_str, style=style)

    console.print(table)

    # Config table
    config_table = Table(title="🔧 Best Configuration Found", box=box.ROUNDED, style="dim")
    config_table.add_column("Axis", style="cyan")
    config_table.add_column("Value Index", justify="center")
    config_table.add_column("Snippet", style="dim", max_width=60)

    for axis, idx in final_config.items():
        config_table.add_row(axis, str(idx), PROMPT_AXES[axis][idx][:80] + "...")

    console.print(config_table)


def _make_serializable(obj):
    """Recursively convert numpy types and other non-serializable objects to Python natives."""
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(x) for x in obj]
    if isinstance(obj, float) and (obj != obj):  # NaN
        return None
    return obj


def save_full_results(
    baseline_scores: Dict,
    final_scores: Dict,
    baseline_config: Dict,
    final_config: Dict,
    optimization_history: List,
    path: str = "results/full_results.json",
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = _make_serializable({
        "baseline": {"config": baseline_config, "scores": baseline_scores},
        "final": {"config": final_config, "scores": final_scores},
        "improvement": {
            k: final_scores.get(k, 0) - baseline_scores.get(k, 0)
            for k in baseline_scores
        },
        "optimization_history": optimization_history,
    })
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved full results: {path}")
