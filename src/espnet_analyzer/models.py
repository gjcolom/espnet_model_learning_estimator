"""Data models for ESPnet training analysis."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EpochMetrics:
    """Metrics for a single epoch."""
    epoch: int
    # Training metrics
    train_generator_loss: Optional[float] = None
    train_mel_loss: Optional[float] = None
    train_kl_loss: Optional[float] = None
    train_dur_loss: Optional[float] = None
    train_adv_loss: Optional[float] = None
    train_feat_match_loss: Optional[float] = None
    train_discriminator_loss: Optional[float] = None
    # Validation metrics
    valid_generator_loss: Optional[float] = None
    valid_mel_loss: Optional[float] = None
    valid_kl_loss: Optional[float] = None
    valid_dur_loss: Optional[float] = None
    valid_adv_loss: Optional[float] = None
    valid_feat_match_loss: Optional[float] = None
    valid_discriminator_loss: Optional[float] = None
    # Timing
    train_time_seconds: Optional[float] = None


@dataclass
class PlateauInfo:
    """Information about detected plateau."""
    start_epoch: int
    end_epoch: int
    avg_improvement: float
    epochs_in_plateau: int


@dataclass
class AnalysisResult:
    """Complete analysis results."""
    epochs: list[EpochMetrics]
    plateaus: list[PlateauInfo] = field(default_factory=list)
    suggested_stop_epoch: Optional[int] = None
    total_improvement: float = 0.0
    recent_improvement_rate: float = 0.0
