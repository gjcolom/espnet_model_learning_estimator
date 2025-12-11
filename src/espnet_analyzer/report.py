"""Report generation for training analysis."""

from .models import AnalysisResult


def print_summary(result: AnalysisResult, metric: str = 'valid_generator_loss') -> None:
    """Print analysis summary to console.

    Args:
        result: Analysis result to summarize
        metric: Metric name being analyzed
    """
    print("\n" + "=" * 60)
    print("ESPnet Training Analysis Summary")
    print("=" * 60)

    if not result.epochs:
        print("No epochs found in log file.")
        return

    first = result.epochs[0]
    last = result.epochs[-1]

    print(f"\nEpochs analyzed: {first.epoch} to {last.epoch} ({len(result.epochs)} total)")

    first_loss = getattr(first, metric)
    last_loss = getattr(last, metric)

    if first_loss and last_loss:
        print(f"\n{metric}:")
        print(f"  First epoch: {first_loss:.4f}")
        print(f"  Last epoch:  {last_loss:.4f}")
        print(f"  Total improvement: {result.total_improvement:.2f}%")

    print(f"\nRecent improvement rate (last 20 epochs): {result.recent_improvement_rate:.4f}% per epoch")

    if result.plateaus:
        print(f"\nDetected {len(result.plateaus)} plateau region(s):")
        for i, p in enumerate(result.plateaus, 1):
            print(f"  {i}. Epochs {p.start_epoch}-{p.end_epoch} "
                  f"({p.epochs_in_plateau} epochs, avg improvement: {p.avg_improvement:.4f}%)")
    else:
        print("\nNo significant plateaus detected.")

    if result.suggested_stop_epoch:
        print(f"\n>>> Suggested stop epoch: {result.suggested_stop_epoch}")
    else:
        print("\n>>> No early stopping recommended yet.")

    print("=" * 60 + "\n")
