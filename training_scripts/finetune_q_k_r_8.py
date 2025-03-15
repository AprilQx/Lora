import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import numpy as np
import argparse
import os
import json
from pathlib import Path
import logging
import math
import sys
from datetime import datetime
import wandb

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.preprocessor import text_to_numeric, numeric_to_text, load_data
from src.models.qwen import load_qwen

from utils.lora_flop_tracker import LoRAFLOPTracker
from src.models.lora import train_lora

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create results directory
RESULTS_DIR = Path(project_root) / "results"  # Use the project_root variable
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR = RESULTS_DIR / "finetune_figures"
FIGURES_DIR.mkdir(exist_ok=True)




def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train a LoRA model for time series forecasting")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha scaling")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--max_steps", type=int, default=10000, help="Maximum training steps")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--alpha", type=float, default=10.0, help="Scaling parameter for LLMTIME")
    parser.add_argument("--precision", type=int, default=3, help="Decimal precision for LLMTIME")
    parser.add_argument("--output_dir", type=str, default="results/models", help="Output directory")
    parser.add_argument("--train_file", type=str, default="../data/processed/train_texts.txt", help="Path to training data")
    parser.add_argument("--val_file", type=str, default="../data/processed/val_texts.txt", help="Path to validation data")
    args = parser.parse_args()
    
    # Initialize FLOP tracker
    
    torch.mps.empty_cache()  # For MPS (Apple Silicon)
    flop_tracker = LoRAFLOPTracker(
        experiment_name=f"lora_r{args.lora_r}_a{args.lora_alpha}_lr{args.learning_rate:.0e}",
        hidden_size=896,  # Qwen2.5-0.5B-Instruct
        num_attention_heads=14,
        num_hidden_layers=24,
        intermediate_size=4864,
        head_dim=64,
        vocab_size=151936,
        max_budget=1e17,
        lora_r=args.lora_r,
        lora_target_modules=["q_proj", "v_proj"]
    )
    
    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_qwen()
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    
    # Load training and validation data
    logger.info(f"Loading training data from {args.train_file}")
    with open(args.train_file, 'r') as f:
        train_texts = f.readlines()
    
    logger.info(f"Loading validation data from {args.val_file}")
    with open(args.val_file, 'r') as f:
        val_texts = f.readlines()
    
    logger.info(f"Loaded {len(train_texts)} training sequences and {len(val_texts)} validation sequences")
    
    
    # Train with LoRA
    logger.info("Starting LoRA training...")
    trained_model, history = train_lora(
        model=model,
        tokenizer=tokenizer,
        train_texts=train_texts,
        val_texts=val_texts,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        max_length=args.max_length,
        output_dir=args.output_dir,
        flop_tracker=flop_tracker
    )
    
    # Generate FLOP report
    flop_report = flop_tracker.generate_report(
        os.path.join(args.output_dir, f"flop_report_r{args.lora_r}_a{args.lora_alpha}_lr{args.learning_rate:.0e}.json")
    )
    
    logger.info(f"FLOP Usage: {flop_report['total_flops']:.2e} ({flop_report['budget_used_percent']:.2f}% of budget)")
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()