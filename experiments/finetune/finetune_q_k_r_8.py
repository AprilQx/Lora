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
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.preprocessor import text_to_numeric, numeric_to_text, load_data
from src.models.qwen import load_qwen

from utils.lora_flop_tracker import LoRAFLOPTracker
from src.models.finetune_lora import train_lora

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
    # Example usage - will need to be modified for actual use
    parser = argparse.ArgumentParser(description="Train a LoRA model for time series forecasting")
    
    # Basic parameters
    parser.add_argument("--train_file", type=str, default="../data/processed3/train_texts.txt", required=True,help="Path to training data file")
    parser.add_argument("--val_file", type=str, default="../data/processed3/train_texts.txt", required=True, help="Path to validation data file")
    parser.add_argument("--output_dir", type=str, default="results/models", help="Output directory for models")
    
    # LoRA parameters
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.5, help="LoRA dropout")
    
    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--max_steps", type=int, default=2000, help="Maximum training steps")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate every n steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save model every n steps")
    
    # FLOP tracking
    parser.add_argument("--max_flops", type=float, default=1e17, help="Maximum allowed FLOPS (scientific notation)")
    
    # Wandb parameters
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for tracking")
    parser.add_argument("--wandb_project", type=str, default="lora-finetuning", help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Weights & Biases entity name")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")

    
    args = parser.parse_args()

    # Set global random seed
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    logger.info(f"Setting global random seed to: {args.random_seed}")
    
    # Create FLOP tracker
    flop_tracker = LoRAFLOPTracker(max_budget=args.max_flops)
    
    # Load model and tokenizer
    model, tokenizer = load_qwen()
    
    # Load training data
    with open(args.train_file, 'r') as f:
        train_texts = [line.strip() for line in f]
    
    # Train model
    trained_model, history = train_lora(
        model=model,
        tokenizer=tokenizer,
        train_texts=train_texts,
        val_data_path=args.val_file,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        max_length=args.max_length,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        output_dir=args.output_dir,
        flop_tracker=flop_tracker,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        random_seed=args.random_seed,
        precision=3
    )
if __name__ == "__main__":
    main()