# ESPnet Training Log Analyzer

A tool to parse ESPnet training logs, visualize training progress, and detect diminishing returns to help decide when to stop training.

## Features

- **Log Parsing**: Extracts epoch-wise metrics from ESPnet training logs (works with ESPnet v1 and v2)
- **Visualization**: Generates loss curves (linear and log scale), improvement graphs, and trend analysis
- **Plateau Detection**: Identifies when training improvements fall below a threshold
- **Early Stop Recommendation**: Suggests optimal stopping points based on diminishing returns

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/espnet-training-analyzer.git
cd espnet-training-analyzer

# Run the setup script to create a virtual environment
./setup_env.sh
```

## Usage

```bash
# Basic usage - analyze a log file and show graphs
./analyze.sh /path/to/train.log

# Save graphs to a directory without displaying
./analyze.sh /path/to/train.log --output-dir ./graphs --no-show

# Customize analysis parameters
./analyze.sh /path/to/train.log \
    --metric mel_loss \
    --threshold 0.05 \
    --consecutive 20 \
    --smoothing 15
```

### Command Line Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output-dir` | `-o` | None | Directory to save output graphs |
| `--metric` | `-m` | `generator_loss` | Metric to analyze (`generator_loss`, `mel_loss`, `discriminator_loss`) |
| `--threshold` | `-t` | `0.1` | Improvement threshold (%) for plateau detection |
| `--consecutive` | `-c` | `10` | Consecutive epochs below threshold to confirm plateau |
| `--smoothing` | `-s` | `10` | Smoothing window for improvement graph |
| `--no-show` | | False | Don't display graphs interactively |
| `--absolute` | | False | Use absolute improvement instead of percentage |

## Output

The tool generates four graphs:

1. **Loss (Linear Scale)**: Training and validation loss over epochs
2. **Loss (Log Scale)**: Same data on logarithmic scale - reveals small improvements that flatten on linear scale
3. **Epoch-to-Epoch Improvement**: Shows the rate of improvement per epoch - helps identify when you've hit diminishing returns
4. **Validation Loss Trend**: Rolling average of validation loss to smooth out noise

### Example Output

```
============================================================
ESPnet Training Analysis Summary
============================================================

Epochs analyzed: 1 to 415 (415 total)

valid_generator_loss:
  First epoch: 47.1140
  Last epoch:  35.9330
  Total improvement: 23.73%

Recent improvement rate (last 20 epochs): -0.0828% per epoch

No significant plateaus detected.

>>> No early stopping recommended yet.
============================================================
```

## Supported Log Formats

The parser looks for epoch summary lines in the format:
```
[timestamp] (trainer:XXX) INFO: Nepoch results: [train] ... [valid] ...
```

This is the standard ESPnet output format for both GAN-based models (VITS, etc.) and non-GAN models.

## Requirements

- Python 3.8+
- matplotlib
- numpy

## License

MIT License
