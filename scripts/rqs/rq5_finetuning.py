#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQ5 LLaMA-Factory Fine-tuning Runner

Run LLaMA-Factory LoRA fine-tuning on the training data.

This script:
1. Prepares the LLaMA-Factory configuration
2. Runs the fine-tuning process
3. Saves the fine-tuned model adapter

Prerequisites:
- LLaMA-Factory installed and configured
- Base model (Qwen2.5-7B-Instruct) available

Usage:
    # Run fine-tuning with default parameters
    python scripts/rqs/rq5_finetuning.py --train-file data/examples/rq5_results/train_sharedgpt.jsonl

    # Run with custom parameters
    python scripts/rqs/rq5_finetuning.py --train-file train_sharedgpt.jsonl --output-dir ./rq5_lora_adapter --num-epochs 3
"""

import os
import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, Any
import shutil


def create_llamafactory_config(
    train_file: str,
    val_file: str,
    output_dir: str,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    num_epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation: int = 4,
    learning_rate: float = 5e-5,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    cutoff_len: int = 4096,
    gradient_checkpointing: bool = True,
    save_steps: int = 100,
) -> Dict[str, Any]:
    """
    Create LLaMA-Factory configuration for LoRA fine-tuning.

    Returns the configuration dictionary that can be saved to a YAML file.
    """
    config = {
        "model_name_or_path": model_name,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_target": "all",
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,

        "dataset": "rq5_dataset",
        "dataset_dir": str(Path(train_file).parent),
        "template": "qwen",
        "cutoff_len": cutoff_len,

        "output_dir": output_dir,
        "logging_steps": 10,
        "save_steps": save_steps,
        "plot_loss": True,
        "overwrite_output_dir": True,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation,
        "learning_rate": learning_rate,
        "num_train_epochs": num_epochs,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "bf16": True,
        "flash_attn": "auto",
        "gradient_checkpointing": gradient_checkpointing,
        "ddp_timeout": 180000000,

        "val_size": 0.1,
        "per_device_eval_batch_size": batch_size,
        "eval_strategy": "steps",
        "eval_steps": save_steps,

        "preprocessing_num_workers": 8,

        # System prompt configuration
        "system_prompt": "You are a careful and logical assistant. When answering questions: 1) Think step by step through the reasoning. 2) Pay attention to all the information given in the question. 3) Ensure your conclusion follows logically from the premises. 4) Be consistent in your answers.",
    }

    return config


def create_dataset_info(train_file: str, val_file: str, dataset_dir: str) -> Dict[str, Any]:
    """Create dataset info for LLaMA-Factory"""
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    dataset_info = {
        "rq5_dataset": {
            "file_name": str(Path(train_file).name),
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations"
            }
        }
    }

    # Save dataset info
    info_file = dataset_dir / "dataset_info.json"
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)

    print(f"Dataset info saved to {info_file}")
    return dataset_info


def save_config(config: Dict[str, Any], output_file: str):
    """Save configuration to YAML file"""
    import yaml

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print(f"Configuration saved to {output_file}")


def run_llamafactory_train(
    config: Dict[str, Any],
    llamafactory_path: str = None,
    gpus: str = None
) -> bool:
    """
    Run LLaMA-Factory training.

    Args:
        config: Configuration dictionary for training
        llamafactory_path: Path to LLaMA-Factory directory

    Returns:
        True if training succeeded, False otherwise
    """
    # Try to find LLaMA-Factory
    if llamafactory_path is None:
        possible_paths = [
            "$LLAMAFACTORY_PATH",
            Path.home() / "LLaMA-Factory",
            "/opt/LLaMA-Factory",
            "./LLaMA-Factory",
        ]
        for path in possible_paths:
            if Path(path).exists():
                llamafactory_path = str(path)
                break

    if llamafactory_path is None or not Path(llamafactory_path).exists():
        print(f"Error: LLaMA-Factory not found at {llamafactory_path}")
        print("Please specify the path with --llamafactory-path")
        return False

    print(f"Using LLaMA-Factory at: {llamafactory_path}")

    # Build command with all arguments
    cmd = [
        "llamafactory-cli",
        "train",
    ]

    # Convert config to command-line arguments
    arg_mapping = {
        "model_name_or_path": "--model_name",
        "stage": "--stage",
        "do_train": "--do_train",
        "finetuning_type": "--finetuning_type",
        "lora_target": "--lora_target",
        "lora_r": "--lora_r",
        "lora_alpha": "--lora_alpha",
        "lora_dropout": "--lora_dropout",
        "dataset": "--dataset",
        "dataset_dir": "--dataset_dir",
        "template": "--template",
        "cutoff_len": "--cutoff_len",
        "output_dir": "--output_dir",
        "logging_steps": "--logging_steps",
        "save_steps": "--save_steps",
        "plot_loss": "--plot_loss",
        "overwrite_output_dir": "--overwrite_output_dir",
        "per_device_train_batch_size": "--per_device_train_batch_size",
        "gradient_accumulation_steps": "--gradient_accumulation_steps",
        "learning_rate": "--learning_rate",
        "num_train_epochs": "--num_train_epochs",
        "lr_scheduler_type": "--lr_scheduler_type",
        "warmup_ratio": "--warmup_ratio",
        "bf16": "--bf16",
        "flash_attn": "--flash_attn",
        "gradient_checkpointing": "--gradient_checkpointing",
        "ddp_timeout": "--ddp_timeout",
        "val_size": "--val_size",
        "per_device_eval_batch_size": "--per_device_eval_batch_size",
        "eval_strategy": "--eval_strategy",
        "eval_steps": "--eval_steps",
        "preprocessing_num_workers": "--preprocessing_num_workers",
    }

    for key, value in config.items():
        if value is None or value is False:
            continue
        # Handle lora_target specially - needs comma-separated string
        if key == "lora_target" and isinstance(value, list):
            cmd.extend(["--lora_target", ",".join(value)])
        elif key == "lora_target":
            cmd.extend(["--lora_target", str(value)])
        else:
            arg_name = arg_mapping.get(key)
            if arg_name:
                if value is True:
                    cmd.append(arg_name)
                else:
                    cmd.extend([arg_name, str(value)])

    print(f"Running command: {' '.join(cmd)}")

    # Set up environment with GPU selection
    env = os.environ.copy()
    if gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpus
        print(f"Using GPUs: {gpus}")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True,
            env=env
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"Error running training: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="RQ5: Run LLaMA-Factory fine-tuning"
    )
    parser.add_argument(
        "--train-file", "-t",
        type=str,
        required=True,
        help="Path to training data in SharedGPT format"
    )
    parser.add_argument(
        "--val-file", "-v",
        type=str,
        default=None,
        help="Path to validation data in SharedGPT format (optional)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="data/examples/rq5_results/lora_adapter",
        help="Output directory for LoRA adapter"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model name"
    )
    parser.add_argument(
        "--num-epochs", "-e",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=4,
        help="Batch size per device"
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning-rate", "-lr",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--cutoff-len",
        type=int,
        default=4096,
        help="Maximum sequence length (default: 4096, lower to save memory)"
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing (enabled by default to save memory)"
    )
    parser.add_argument(
        "--no-flash-attention",
        action="store_true",
        help="Disable flash attention (enabled by default for faster training)"
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="Save checkpoint every N steps (default: 100)"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="GPU IDs to use (e.g., '2,3' for GPUs 2 and 3)"
    )
    parser.add_argument(
        "--llamafactory-path",
        type=str,
        default=None,
        help="Path to LLaMA-Factory directory"
    )
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="Only generate config file, don't run training"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("RQ5 LLaMA-Factory Fine-tuning")
    print("=" * 70)
    print(f"Train file: {args.train_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Base model: {args.model_name}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")

    # Validate input file
    if not Path(args.train_file).exists():
        print(f"Error: Training file not found: {args.train_file}")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataset info
    print("\nPreparing dataset configuration...")
    create_dataset_info(
        args.train_file,
        args.val_file,
        str(Path(args.train_file).parent)
    )

    # Create configuration
    print("\nCreating fine-tuning configuration...")
    config = create_llamafactory_config(
        train_file=args.train_file,
        val_file=args.val_file,
        output_dir=str(output_dir),
        model_name=args.model_name,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        cutoff_len=args.cutoff_len,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        save_steps=args.save_steps,
    )

    # Override flash_attn based on command-line argument
    if args.no_flash_attention:
        config["flash_attn"] = False

    # Save configuration
    config_file = output_dir / "config.yaml"
    save_config(config, str(config_file))

    # Estimate training
    print("\nTraining configuration:")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation}")
    print(f"  Max sequence length: {args.cutoff_len}")
    print(f"  Save checkpoint every: {args.save_steps} steps")
    print(f"  Gradient checkpointing: {not args.no_gradient_checkpointing}")
    print(f"  Flash attention: {not args.no_flash_attention}")
    if args.gpus:
        print(f"  Using GPUs: {args.gpus}")
    print(f"  LoRA rank: {args.lora_r}")
    print(f"  LoRA alpha: {args.lora_alpha}")

    if args.config_only:
        print("\nConfig file generated. Exiting (--config-only flag set).")
        return

    # Run training
    print("\n" + "=" * 70)
    print("Starting fine-tuning...")
    print("=" * 70)

    success = run_llamafactory_train(
        config,
        args.llamafactory_path,
        args.gpus
    )

    if success:
        print("\n" + "=" * 70)
        print("Fine-tuning completed successfully!")
        print("=" * 70)
        print(f"LoRA adapter saved to: {output_dir}")
    else:
        print("\n" + "=" * 70)
        print("Fine-tuning failed!")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    exit(main() or 0)
