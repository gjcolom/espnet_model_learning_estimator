#!/usr/bin/env python3
"""
ESPnet Training Log Parser & Analyzer - CLI Entry Point

Parses ESPnet training logs to extract epoch-wise metrics,
generate visualizations, and detect training plateaus.
"""

import sys
import argparse
from pathlib import Path

# Add src to path for package imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from espnet_analyzer import (
    parse_log_file,
    analyze_training,
    print_summary,
    plot_loss_curves,
    plot_loss_curves_log,
    plot_improvement_curve,
    plot_combined,
)


def main():
    parser = argparse.ArgumentParser(
        description='ESPnet Training Log Parser & Analyzer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s train.log
  %(prog)s train.log --output-dir ./graphs --threshold 0.05
  %(prog)s train.log --metric mel_loss --no-show
        """
    )

    parser.add_argument('log_file', type=Path, help='Path to ESPnet training log file')
    parser.add_argument('--output-dir', '-o', type=Path, default=None,
                        help='Directory to save output graphs')
    parser.add_argument('--metric', '-m', type=str, default='generator_loss',
                        choices=['generator_loss', 'mel_loss', 'discriminator_loss'],
                        help='Primary metric to analyze (default: generator_loss)')
    parser.add_argument('--threshold', '-t', type=float, default=0.1,
                        help='Improvement threshold for plateau detection (default: 0.1%%)')
    parser.add_argument('--consecutive', '-c', type=int, default=10,
                        help='Consecutive epochs below threshold to confirm plateau (default: 10)')
    parser.add_argument('--smoothing', '-s', type=int, default=10,
                        help='Smoothing window for improvement graph (default: 10)')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display graphs interactively')
    parser.add_argument('--absolute', action='store_true',
                        help='Use absolute improvement instead of relative percentage')

    args = parser.parse_args()

    if not args.log_file.exists():
        print(f"Error: Log file not found: {args.log_file}")
        return 1

    # Create output directory if specified
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    # Parse log file
    print(f"Parsing log file: {args.log_file}")
    epochs = parse_log_file(args.log_file)

    if not epochs:
        print("Error: No epoch data found in log file.")
        return 1

    print(f"Found {len(epochs)} epochs")

    # Map metric name
    metric_map = {
        'generator_loss': 'generator_loss',
        'mel_loss': 'mel_loss',
        'discriminator_loss': 'discriminator_loss'
    }
    metric = metric_map.get(args.metric, 'generator_loss')
    valid_metric = f'valid_{metric}'

    # Analyze training
    result = analyze_training(
        epochs,
        metric=valid_metric,
        threshold=args.threshold,
        consecutive_epochs=args.consecutive,
        use_relative=not args.absolute
    )

    # Print summary
    print_summary(result, valid_metric)

    # Generate plots
    show = not args.no_show

    if args.output_dir:
        plot_loss_curves(epochs, args.output_dir / 'loss_linear.png', show=False, metric=metric)
        plot_loss_curves_log(epochs, args.output_dir / 'loss_log.png', show=False, metric=metric)
        plot_improvement_curve(
            epochs, args.output_dir / 'improvement.png', show=False,
            metric=valid_metric, threshold=args.threshold,
            smoothing_window=args.smoothing, use_relative=not args.absolute
        )
        plot_combined(
            epochs, args.output_dir / 'combined.png', show=False,
            metric=metric, threshold=args.threshold, smoothing_window=args.smoothing
        )
        print(f"Graphs saved to: {args.output_dir}")

    if show:
        plot_combined(
            epochs, show=True,
            metric=metric, threshold=args.threshold, smoothing_window=args.smoothing
        )

    return 0


if __name__ == '__main__':
    exit(main())
