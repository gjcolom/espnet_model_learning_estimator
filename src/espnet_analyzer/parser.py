"""Log file parsing for ESPnet training logs."""

import re
from pathlib import Path
from typing import Optional

from .models import EpochMetrics


def parse_time_string(time_str: str) -> float:
    """Parse time strings like '25 minutes and 17.14 seconds' to seconds."""
    total_seconds = 0.0

    # Match weeks
    weeks_match = re.search(r'(\d+)\s*weeks?', time_str)
    if weeks_match:
        total_seconds += int(weeks_match.group(1)) * 7 * 24 * 3600

    # Match days
    days_match = re.search(r'(\d+)\s*days?', time_str)
    if days_match:
        total_seconds += int(days_match.group(1)) * 24 * 3600

    # Match hours
    hours_match = re.search(r'(\d+)\s*hours?', time_str)
    if hours_match:
        total_seconds += int(hours_match.group(1)) * 3600

    # Match minutes
    minutes_match = re.search(r'(\d+)\s*minutes?', time_str)
    if minutes_match:
        total_seconds += int(minutes_match.group(1)) * 60

    # Match seconds
    seconds_match = re.search(r'([\d.]+)\s*seconds?', time_str)
    if seconds_match:
        total_seconds += float(seconds_match.group(1))

    return total_seconds


def extract_metric(text: str, metric_name: str) -> Optional[float]:
    """Extract a metric value from text."""
    pattern = rf'{metric_name}=([\d.e+-]+)'
    match = re.search(pattern, text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def parse_log_file(log_path: Path) -> list[EpochMetrics]:
    """Parse an ESPnet training log file and extract epoch metrics.

    Args:
        log_path: Path to the training log file

    Returns:
        List of EpochMetrics, sorted by epoch number
    """
    epochs = []

    # Pattern for epoch results line
    epoch_pattern = re.compile(r'(\d+)epoch results:')

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Look for epoch results lines
            epoch_match = epoch_pattern.search(line)
            if not epoch_match:
                continue

            epoch_num = int(epoch_match.group(1))

            # Split into train and valid sections
            train_section = ""
            valid_section = ""

            if '[train]' in line:
                train_start = line.index('[train]')
                train_section = line[train_start:]
                if '[valid]' in train_section:
                    valid_start = train_section.index('[valid]')
                    valid_section = train_section[valid_start:]
                    train_section = train_section[:valid_start]

            metrics = EpochMetrics(epoch=epoch_num)

            # Extract training metrics
            metrics.train_generator_loss = extract_metric(train_section, 'generator_loss')
            metrics.train_mel_loss = extract_metric(train_section, 'generator_mel_loss')
            metrics.train_kl_loss = extract_metric(train_section, 'generator_kl_loss')
            metrics.train_dur_loss = extract_metric(train_section, 'generator_dur_loss')
            metrics.train_adv_loss = extract_metric(train_section, 'generator_adv_loss')
            metrics.train_feat_match_loss = extract_metric(train_section, 'generator_feat_match_loss')
            metrics.train_discriminator_loss = extract_metric(train_section, 'discriminator_loss')

            # Extract validation metrics
            metrics.valid_generator_loss = extract_metric(valid_section, 'generator_loss')
            metrics.valid_mel_loss = extract_metric(valid_section, 'generator_mel_loss')
            metrics.valid_kl_loss = extract_metric(valid_section, 'generator_kl_loss')
            metrics.valid_dur_loss = extract_metric(valid_section, 'generator_dur_loss')
            metrics.valid_adv_loss = extract_metric(valid_section, 'generator_adv_loss')
            metrics.valid_feat_match_loss = extract_metric(valid_section, 'generator_feat_match_loss')
            metrics.valid_discriminator_loss = extract_metric(valid_section, 'discriminator_loss')

            # Extract timing
            time_match = re.search(r'time=([\d\w\s.]+?)(?:,|$)', train_section)
            if time_match:
                metrics.train_time_seconds = parse_time_string(time_match.group(1))

            epochs.append(metrics)

    return sorted(epochs, key=lambda x: x.epoch)
