"""Training analysis functions."""

from typing import Optional
import numpy as np

from .models import EpochMetrics, PlateauInfo, AnalysisResult


def compute_improvements(
    epochs: list[EpochMetrics],
    metric: str = 'valid_generator_loss'
) -> list[tuple[int, float]]:
    """Compute epoch-to-epoch improvements for a metric.

    Args:
        epochs: List of epoch metrics
        metric: Attribute name to analyze

    Returns:
        List of (epoch, improvement) tuples.
        Positive improvement = loss decreased (good).
    """
    improvements = []

    for i in range(1, len(epochs)):
        prev_val = getattr(epochs[i-1], metric)
        curr_val = getattr(epochs[i], metric)

        if prev_val is not None and curr_val is not None:
            improvement = prev_val - curr_val
            improvements.append((epochs[i].epoch, improvement))

    return improvements


def compute_relative_improvements(
    epochs: list[EpochMetrics],
    metric: str = 'valid_generator_loss'
) -> list[tuple[int, float]]:
    """Compute epoch-to-epoch relative (percentage) improvements.

    Args:
        epochs: List of epoch metrics
        metric: Attribute name to analyze

    Returns:
        List of (epoch, relative_improvement) tuples as percentages.
    """
    improvements = []

    for i in range(1, len(epochs)):
        prev_val = getattr(epochs[i-1], metric)
        curr_val = getattr(epochs[i], metric)

        if prev_val is not None and curr_val is not None and prev_val != 0:
            relative_improvement = (prev_val - curr_val) / prev_val * 100
            improvements.append((epochs[i].epoch, relative_improvement))

    return improvements


def detect_plateaus(
    improvements: list[tuple[int, float]],
    threshold: float = 0.01,
    consecutive_epochs: int = 10,
    use_relative: bool = False
) -> list[PlateauInfo]:
    """Detect plateau regions where improvement is below threshold.

    Args:
        improvements: List of (epoch, improvement) tuples
        threshold: Improvement threshold (absolute or relative %)
        consecutive_epochs: Number of consecutive epochs below threshold to confirm plateau
        use_relative: If True, threshold is interpreted as percentage

    Returns:
        List of PlateauInfo objects
    """
    plateaus = []
    current_plateau_start = None
    consecutive_count = 0
    plateau_improvements = []

    for epoch, improvement in improvements:
        below_threshold = improvement < threshold

        if below_threshold:
            if current_plateau_start is None:
                current_plateau_start = epoch
                plateau_improvements = [improvement]
            else:
                plateau_improvements.append(improvement)
            consecutive_count += 1
        else:
            if consecutive_count >= consecutive_epochs:
                plateaus.append(PlateauInfo(
                    start_epoch=current_plateau_start,
                    end_epoch=epoch - 1,
                    avg_improvement=float(np.mean(plateau_improvements)),
                    epochs_in_plateau=consecutive_count
                ))
            current_plateau_start = None
            consecutive_count = 0
            plateau_improvements = []

    # Check if we ended in a plateau
    if consecutive_count >= consecutive_epochs:
        plateaus.append(PlateauInfo(
            start_epoch=current_plateau_start,
            end_epoch=improvements[-1][0],
            avg_improvement=float(np.mean(plateau_improvements)),
            epochs_in_plateau=consecutive_count
        ))

    return plateaus


def suggest_stop_epoch(
    epochs: list[EpochMetrics],
    improvements: list[tuple[int, float]],
    plateaus: list[PlateauInfo],
    min_epochs: int = 50
) -> Optional[int]:
    """Suggest an optimal stopping epoch based on plateau analysis.

    Args:
        epochs: List of epoch metrics
        improvements: List of (epoch, improvement) tuples
        plateaus: Detected plateau regions
        min_epochs: Minimum epochs before considering early stop

    Returns:
        Suggested stop epoch, or None if no recommendation
    """
    if not plateaus:
        return None

    # Find the first significant plateau after min_epochs
    for plateau in plateaus:
        if plateau.start_epoch >= min_epochs:
            return plateau.start_epoch

    return None


def analyze_training(
    epochs: list[EpochMetrics],
    metric: str = 'valid_generator_loss',
    threshold: float = 0.1,
    consecutive_epochs: int = 10,
    use_relative: bool = True
) -> AnalysisResult:
    """Perform complete training analysis.

    Args:
        epochs: List of epoch metrics
        metric: Metric to analyze
        threshold: Plateau detection threshold
        consecutive_epochs: Consecutive epochs below threshold for plateau
        use_relative: Use relative (%) or absolute improvements

    Returns:
        AnalysisResult with complete analysis
    """
    if use_relative:
        improvements = compute_relative_improvements(epochs, metric)
    else:
        improvements = compute_improvements(epochs, metric)

    plateaus = detect_plateaus(improvements, threshold, consecutive_epochs, use_relative)
    suggested_stop = suggest_stop_epoch(epochs, improvements, plateaus)

    # Calculate total improvement
    if epochs:
        first_val = getattr(epochs[0], metric)
        last_val = getattr(epochs[-1], metric)
        if first_val and last_val:
            total_improvement = (first_val - last_val) / first_val * 100
        else:
            total_improvement = 0.0
    else:
        total_improvement = 0.0

    # Calculate recent improvement rate (last 20 epochs)
    if len(improvements) >= 20:
        recent = improvements[-20:]
        recent_improvement_rate = float(np.mean([imp for _, imp in recent]))
    elif improvements:
        recent_improvement_rate = float(np.mean([imp for _, imp in improvements]))
    else:
        recent_improvement_rate = 0.0

    return AnalysisResult(
        epochs=epochs,
        plateaus=plateaus,
        suggested_stop_epoch=suggested_stop,
        total_improvement=total_improvement,
        recent_improvement_rate=recent_improvement_rate
    )
