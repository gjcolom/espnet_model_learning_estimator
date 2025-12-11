"""Plotting functions for training visualization."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .models import EpochMetrics
from .analysis import compute_relative_improvements, compute_improvements


def plot_loss_curves(
    epochs: list[EpochMetrics],
    output_path: Optional[Path] = None,
    show: bool = True,
    metric: str = 'generator_loss'
) -> None:
    """Plot linear-scale loss curves.

    Args:
        epochs: List of epoch metrics
        output_path: Path to save figure (optional)
        show: Whether to display interactively
        metric: Base metric name (without train_/valid_ prefix)
    """
    epoch_nums = [e.epoch for e in epochs]
    train_losses = [getattr(e, f'train_{metric}') for e in epochs]
    valid_losses = [getattr(e, f'valid_{metric}') for e in epochs]

    plt.figure(figsize=(12, 6))
    plt.plot(epoch_nums, train_losses, label=f'Train {metric}', alpha=0.8)
    plt.plot(epoch_nums, valid_losses, label=f'Valid {metric}', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Progress - {metric} (Linear Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_loss_curves_log(
    epochs: list[EpochMetrics],
    output_path: Optional[Path] = None,
    show: bool = True,
    metric: str = 'generator_loss'
) -> None:
    """Plot log-scale loss curves.

    Args:
        epochs: List of epoch metrics
        output_path: Path to save figure (optional)
        show: Whether to display interactively
        metric: Base metric name (without train_/valid_ prefix)
    """
    epoch_nums = [e.epoch for e in epochs]
    train_losses = [getattr(e, f'train_{metric}') for e in epochs]
    valid_losses = [getattr(e, f'valid_{metric}') for e in epochs]

    plt.figure(figsize=(12, 6))
    plt.semilogy(epoch_nums, train_losses, label=f'Train {metric}', alpha=0.8)
    plt.semilogy(epoch_nums, valid_losses, label=f'Valid {metric}', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title(f'Training Progress - {metric} (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_improvement_curve(
    epochs: list[EpochMetrics],
    output_path: Optional[Path] = None,
    show: bool = True,
    metric: str = 'valid_generator_loss',
    threshold: Optional[float] = None,
    smoothing_window: int = 1,
    use_relative: bool = True
) -> None:
    """Plot improvement curve (epoch vs loss delta).

    Args:
        epochs: List of epoch metrics
        output_path: Path to save figure (optional)
        show: Whether to display interactively
        metric: Full metric name (e.g., 'valid_generator_loss')
        threshold: Threshold line to display (optional)
        smoothing_window: Rolling average window size
        use_relative: Use percentage or absolute improvement
    """
    if use_relative:
        improvements = compute_relative_improvements(epochs, metric)
        ylabel = 'Improvement (%)'
        title_suffix = '(Relative %)'
    else:
        improvements = compute_improvements(epochs, metric)
        ylabel = 'Improvement (absolute)'
        title_suffix = '(Absolute)'

    if not improvements:
        print("Not enough data to plot improvements")
        return

    epoch_nums = [e for e, _ in improvements]
    imp_values = [i for _, i in improvements]

    # Apply smoothing if requested
    if smoothing_window > 1:
        imp_values = np.convolve(
            imp_values,
            np.ones(smoothing_window) / smoothing_window,
            mode='valid'
        )
        epoch_nums = epoch_nums[smoothing_window-1:]

    plt.figure(figsize=(12, 6))
    plt.plot(epoch_nums, imp_values, label='Improvement', alpha=0.8)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    if threshold is not None:
        plt.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, label=f'Threshold ({threshold})')

    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(f'Epoch-to-Epoch Improvement {title_suffix}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_combined(
    epochs: list[EpochMetrics],
    output_path: Optional[Path] = None,
    show: bool = True,
    metric: str = 'generator_loss',
    threshold: Optional[float] = None,
    smoothing_window: int = 10
) -> None:
    """Create combined plot with all visualizations.

    Args:
        epochs: List of epoch metrics
        output_path: Path to save figure (optional)
        show: Whether to display interactively
        metric: Base metric name
        threshold: Threshold line for improvement graph
        smoothing_window: Smoothing window size
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epoch_nums = [e.epoch for e in epochs]
    train_losses = [getattr(e, f'train_{metric}') for e in epochs]
    valid_losses = [getattr(e, f'valid_{metric}') for e in epochs]

    # Linear scale
    axes[0, 0].plot(epoch_nums, train_losses, label='Train', alpha=0.8)
    axes[0, 0].plot(epoch_nums, valid_losses, label='Valid', alpha=0.8)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss (Linear Scale)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Log scale
    axes[0, 1].semilogy(epoch_nums, train_losses, label='Train', alpha=0.8)
    axes[0, 1].semilogy(epoch_nums, valid_losses, label='Valid', alpha=0.8)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss (log)')
    axes[0, 1].set_title('Loss (Log Scale)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Improvement curve (relative)
    improvements = compute_relative_improvements(epochs, f'valid_{metric}')
    if improvements:
        imp_epochs = [e for e, _ in improvements]
        imp_values = [i for _, i in improvements]

        # Smoothed version
        if len(imp_values) >= smoothing_window:
            smoothed = np.convolve(imp_values, np.ones(smoothing_window) / smoothing_window, mode='valid')
            smoothed_epochs = imp_epochs[smoothing_window-1:]
            axes[1, 0].plot(smoothed_epochs, smoothed, label=f'Smoothed (window={smoothing_window})', alpha=0.9, linewidth=2)

        axes[1, 0].plot(imp_epochs, imp_values, label='Raw', alpha=0.3)
        axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        if threshold:
            axes[1, 0].axhline(y=threshold, color='red', linestyle='--', alpha=0.7, label=f'Threshold ({threshold}%)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Improvement (%)')
        axes[1, 0].set_title('Epoch-to-Epoch Improvement (Relative)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # Validation loss trend with rolling average
    if valid_losses:
        window = min(20, len(valid_losses) // 5) if len(valid_losses) > 5 else 1
        if window > 1:
            rolling_avg = np.convolve(valid_losses, np.ones(window) / window, mode='valid')
            rolling_epochs = epoch_nums[window-1:]
            axes[1, 1].plot(rolling_epochs, rolling_avg, label=f'Rolling Avg (window={window})', linewidth=2)
        axes[1, 1].plot(epoch_nums, valid_losses, label='Valid Loss', alpha=0.3)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Validation Loss')
        axes[1, 1].set_title('Validation Loss Trend')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close()
