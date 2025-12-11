"""ESPnet Training Log Analyzer package."""

from .models import EpochMetrics, PlateauInfo, AnalysisResult
from .parser import parse_log_file
from .analysis import (
    compute_improvements,
    compute_relative_improvements,
    detect_plateaus,
    suggest_stop_epoch,
    analyze_training,
)
from .plotting import (
    plot_loss_curves,
    plot_loss_curves_log,
    plot_improvement_curve,
    plot_combined,
)
from .report import print_summary

__version__ = "0.1.0"
__all__ = [
    "EpochMetrics",
    "PlateauInfo",
    "AnalysisResult",
    "parse_log_file",
    "compute_improvements",
    "compute_relative_improvements",
    "detect_plateaus",
    "suggest_stop_epoch",
    "analyze_training",
    "plot_loss_curves",
    "plot_loss_curves_log",
    "plot_improvement_curve",
    "plot_combined",
    "print_summary",
]
