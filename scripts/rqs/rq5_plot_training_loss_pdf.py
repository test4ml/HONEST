#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ5: Plot Training Loss as PDF

Generate publication-ready PDF plots from LLaMA-Factory training logs.
Designed for single-column academic papers.

Usage:
    python scripts/rqs/rq5_plot_training_loss_pdf.py --log-dir data/examples/rq5_results/lora_adapter
"""

import argparse
import json
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import matplotlib.font_manager as fm

# Get project root and add custom SimSun font
script_dir = Path(__file__).parent.parent.parent
font_path = script_dir / 'fonts' / 'SimSun.ttf'
if font_path.exists():
    fm.fontManager.addfont(str(font_path))
    font_prop = fm.FontProperties(fname=str(font_path))
    font_name = font_prop.get_name()
    print(f"Using font: {font_name} from {font_path}")
else:
    font_name = 'DejaVu Sans'
    print(f"Warning: Font not found at {font_path}, using DejaVu Sans")

# Configure matplotlib for Chinese and single-column paper
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.family'] = font_name
matplotlib.rcParams['font.size'] = 9
matplotlib.rcParams['axes.linewidth'] = 0.8
matplotlib.rcParams['grid.linewidth'] = 0.5
matplotlib.rcParams['lines.linewidth'] = 1.2
matplotlib.rcParams['legend.frameon'] = True
matplotlib.rcParams['legend.fancybox'] = False
matplotlib.rcParams['legend.shadow'] = False
matplotlib.rcParams['legend.edgecolor'] = 'gray'
matplotlib.rcParams['legend.framealpha'] = 0.9
matplotlib.rcParams['axes.unicode_minus'] = False  # Fix minus sign display


def parse_trainer_log(log_file: Path):
    """Parse trainer_log.jsonl and extract training metrics."""
    steps = []
    losses = []
    learning_rates = []
    eval_steps = []
    eval_losses = []

    with open(log_file, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                # Only parse training loss entries (skip eval-only entries)
                if 'loss' in data:
                    steps.append(data.get('current_steps', 0))
                    losses.append(data['loss'])
                    learning_rates.append(data.get('lr', 0))
                # Parse eval loss from log file as well
                if 'eval_loss' in data:
                    eval_steps.append(data.get('current_steps', 0))
                    eval_losses.append(data['eval_loss'])

    return steps, losses, learning_rates, eval_steps, eval_losses


def parse_eval_results(output_dir: Path):
    """Parse evaluation results from checkpoints."""
    # Try to get eval loss from trainer_state.json
    state_file = output_dir / 'trainer_state.json'
    eval_steps = []
    eval_losses = []

    if state_file.exists():
        with open(state_file, 'r') as f:
            state = json.load(f)
            for log in state.get('log_history', []):
                if 'eval_loss' in log:
                    eval_steps.append(log.get('step', 0))
                    eval_losses.append(log['eval_loss'])

    return eval_steps, eval_losses


def plot_training_loss_pdf(log_dir: Path, output_dir: Path = None):
    """Generate training loss plots in PDF format for single-column papers."""
    if output_dir is None:
        output_dir = log_dir

    log_file = log_dir / 'trainer_log.jsonl'
    if not log_file.exists():
        print(f"Error: trainer_log.jsonl not found in {log_dir}")
        return

    # Parse data
    steps, losses, learning_rates, eval_steps_from_log, eval_losses_from_log = parse_trainer_log(log_file)
    eval_steps_from_state, eval_losses_from_state = parse_eval_results(log_dir)

    # Merge eval data from both sources (prefer trainer_state.json if available)
    if eval_steps_from_state and eval_losses_from_state:
        eval_steps = eval_steps_from_state
        eval_losses = eval_losses_from_state
    else:
        eval_steps = eval_steps_from_log
        eval_losses = eval_losses_from_log

    if not steps:
        print("Error: No training data found")
        return

    # Single-column paper: width ~3.5 inches (8.5 cm)
    fig_width = 7  # Two plots side by side
    fig_height = 2.5

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
    fig.patch.set_facecolor('white')

    # Plot 1: Training Loss
    ax1.plot(steps, losses, 'b-', linewidth=1.2, label='Training Loss')
    ax1.set_xlabel('Training Steps', fontsize=9)
    ax1.set_ylabel('Loss', fontsize=9)
    ax1.set_title('(a) Training Loss', fontsize=10, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.set_xlim(left=0)
    ax1.legend(frameon=True, fancybox=False, shadow=False, edgecolor='gray',
               fontsize=8, loc='upper right')

    # Plot 2: Training + Eval Loss
    ax2.plot(steps, losses, 'b-', linewidth=1.2, label='Training Loss', alpha=0.7)
    if eval_steps and eval_losses:
        ax2.plot(eval_steps, eval_losses, 'r-', linewidth=1.5, marker='o', markersize=3,
                label='Validation Loss')
    ax2.set_xlabel('Training Steps', fontsize=9)
    ax2.set_ylabel('Loss', fontsize=9)
    ax2.set_title('(b) Training & Validation Loss', fontsize=10, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.set_xlim(left=0)
    ax2.legend(frameon=True, fancybox=False, shadow=False, edgecolor='gray',
               fontsize=8, loc='upper right')

    plt.tight_layout()

    # Save as PDF
    output_file = output_dir / 'training_loss.pdf'
    plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close(fig)

    # Create separate plot for eval loss only (single-column width)
    if eval_steps and eval_losses:
        fig2, ax = plt.subplots(1, 1, figsize=(3.5, 2.5))
        fig2.patch.set_facecolor('white')

        ax.plot(eval_steps, eval_losses, 'r-', linewidth=1.5, marker='o', markersize=3,
                label='Validation Loss')
        ax.set_xlabel('Training Steps', fontsize=9)
        ax.set_ylabel('Loss', fontsize=9)
        ax.set_title('Validation Loss', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_xlim(left=0)
        ax.legend(frameon=True, fancybox=False, shadow=False, edgecolor='gray',
                  fontsize=8, loc='upper right')

        plt.tight_layout()

        output_file2 = output_dir / 'training_eval_loss.pdf'
        plt.savefig(output_file2, format='pdf', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file2}")
        plt.close(fig2)

    print("\nTraining metrics:")
    print(f"  Total steps: {max(steps) if steps else 0}")
    print(f"  Final training loss: {losses[-1] if losses else 0:.6f}")
    if eval_losses:
        print(f"  Final validation loss: {eval_losses[-1]:.6f}")
    print(f"  Min training loss: {min(losses) if losses else 0:.6f}")
    if eval_losses:
        print(f"  Min validation loss: {min(eval_losses):.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate PDF plots from LLaMA-Factory training logs"
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        required=True,
        help='Path to training output directory (containing trainer_log.jsonl)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for PDF files (default: same as log-dir)'
    )

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir) if args.output_dir else log_dir

    if not log_dir.exists():
        print(f"Error: Log directory not found: {log_dir}")
        return

    plot_training_loss_pdf(log_dir, output_dir)


if __name__ == '__main__':
    main()
