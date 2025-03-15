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

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.preprocessor import load_and_preprocess, text_to_numeric, numeric_to_text, load_data
from src.models.qwen import load_qwen
from src.evaluation import evaluate_forecasting
from utils.flop_tracker import FLOPTracker

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
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR = RESULTS_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# LoRA implementation
class LoRALinear(nn.Module):
    """
    LoRA implementation for Linear layers:
    y = W*x + b + (A*B)*x * (alpha/r)
    """
    def __init__(self, original_linear: nn.Linear, r: int, alpha: int = None, dropout: float = 0.0):
        super().__init__()
        assert isinstance(original_linear, nn.Linear)
        
        # Store original layer and freeze it
        self.original_linear = original_linear
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False
        
        # Get dimensions
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        
        # LoRA parameters
        self.r = r
        self.alpha = alpha if alpha is not None else r
        
        # Get device from original layer
        device = original_linear.weight.device
        
        # Define LoRA matrices
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features, device=device))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, device=device))
        
        # Optional dropout layer
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize A matrix (B is initialized to zeros)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
    
    def forward(self, x):
        # Original output
        base_output = self.original_linear(x)
        
        # LoRA path with dropout
        lora_output = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
        
        # Combine with scaling factor
        return base_output + lora_output * (self.alpha / self.r)


def apply_lora_to_model(model, r=8, alpha=16, dropout=0.0, target_modules=None):
    """
    Apply LoRA to specific modules in the model.
    
    Args:
        model: The model to add LoRA to
        r: LoRA rank
        alpha: LoRA alpha (scaling)
        dropout: Dropout rate for LoRA layers
        target_modules: List of module types to apply LoRA to (if None, apply to Q and V projections only)
    
    Returns:
        Modified model
    """
    # If no target modules specified, default to Q and V projections
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]
    
    # Track parameters to train
    trainable_params = []
    
    # Apply LoRA to attention modules
    for layer in model.model.layers:
        for name, module in layer.self_attn.named_modules():
            # Check if this is a target module
            if name in target_modules and isinstance(module, nn.Linear):
                logger.info(f"Applying LoRA (r={r}, alpha={alpha}) to {name}")
                
                # Replace with LoRA module
                if name == "q_proj":
                    layer.self_attn.q_proj = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
                    trainable_params.extend([p for p in layer.self_attn.q_proj.parameters() if p.requires_grad])
                elif name == "v_proj":
                    layer.self_attn.v_proj = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
                    trainable_params.extend([p for p in layer.self_attn.v_proj.parameters() if p.requires_grad])
    
    # Add bias parameter if it exists
    if model.lm_head.bias is not None and model.lm_head.bias.requires_grad:
        trainable_params.append(model.lm_head.bias)
    
    return model, trainable_params


def process_sequences(texts, tokenizer, max_length=512, stride=256, add_eos=False):
    """
    Process text sequences into tokenized chunks for training.
    
    Args:
        texts: List of text sequences
        tokenizer: Tokenizer to use
        max_length: Maximum length of each chunk
        stride: Stride between consecutive chunks
        add_eos: Whether to add EOS token
        
    Returns:
        List of tokenized sequences
    """
    all_input_ids = []
    all_labels = []
    
    for text in texts:
        # Apply tokenization scheme
        encoding = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        seq_ids = encoding.input_ids[0]
        
        # If sequence is too long, create chunks with sliding window
        if len(seq_ids) > max_length:
            # Create sliding windows
            for i in range(0, len(seq_ids) - max_length + 1, stride):
                chunk = seq_ids[i:i + max_length]
                all_input_ids.append(chunk)
                all_labels.append(chunk.clone())  # For autoregressive loss
        else:
            # Pad sequence if it's shorter than max_length
            if len(seq_ids) < max_length:
                padding = torch.full((max_length - len(seq_ids),), tokenizer.pad_token_id)
                input_ids = torch.cat([seq_ids, padding])
                
                # # Create attention mask
                # attention_mask = torch.cat([
                #     torch.ones(len(seq_ids), dtype=torch.long),
                #     torch.zeros(max_length - len(seq_ids), dtype=torch.long)
                # ])
                
                # For shorter sequences, only use the actual sequence as labels
                labels = torch.cat([seq_ids, torch.full((max_length - len(seq_ids),), -100)])
            else:
                input_ids = seq_ids
                #attention_mask = torch.ones(max_length, dtype=torch.long)
                labels = seq_ids.clone()
            
            all_input_ids.append(input_ids)
            all_labels.append(labels)
    
    return torch.stack(all_input_ids), torch.stack(all_labels)


def train_lora(
    model, 
    tokenizer, 
    train_texts, 
    val_texts, 
    lora_r=8, 
    lora_alpha=16, 
    lora_dropout=0.0,
    learning_rate=1e-4,
    batch_size=4,
    max_steps=10000,
    max_length=512,
    eval_steps=500,
    save_steps=1000,
    device=None,
    output_dir="results/models",
    flop_tracker=None
):
    """
    Train a model with LoRA.
    
    Args:
        model: The base model
        tokenizer: The tokenizer
        train_texts: Training text samples
        val_texts: Validation text samples
        lora_r: LoRA rank
        lora_alpha: LoRA alpha (scaling)
        lora_dropout: Dropout rate for LoRA layers
        learning_rate: Learning rate
        batch_size: Batch size
        max_steps: Maximum training steps
        max_length: Maximum sequence length
        eval_steps: Evaluate every n steps
        save_steps: Save model every n steps
        device: Device to use (if None, use CUDA if available)
        output_dir: Directory to save models
        flop_tracker: FLOPTracker instance for monitoring FLOP usage
        
    Returns:
        Trained model and training history
    """
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Apply LoRA to model
    model, trainable_params = apply_lora_to_model(
        model, 
        r=lora_r, 
        alpha=lora_alpha, 
        dropout=lora_dropout
    )
    
    # Move model to device
    model = model.to(device)
    
    # Prepare training data
    logger.info("Tokenizing training data...")
    train_input_ids, train_labels = process_sequences(
        train_texts, 
        tokenizer, 
        max_length=max_length, 
        stride=max_length // 2
    )
    
    logger.info("Tokenizing validation data...")
    val_input_ids, val_labels = process_sequences(
        val_texts, 
        tokenizer, 
        max_length=max_length, 
        stride=max_length
    )
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(train_input_ids, train_labels)
    val_dataset = TensorDataset(val_input_ids, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    
    # Setup scheduler
    steps_per_epoch = len(train_loader)
    num_warmup_steps = min(100, max_steps // 10)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=learning_rate,
        total_steps=max_steps,
        pct_start=num_warmup_steps / max_steps,
        div_factor=25.0,
        final_div_factor=100.0,
        anneal_strategy='linear'
    )
    
    # Training loop
    model.train()
    step = 0
    best_val_loss = float("inf")
    history = {
        "train_loss": [],
        "val_loss": [],
        "steps": []
    }
    
    logger.info(f"Starting training with batch_size={batch_size}, lr={learning_rate}, lora_r={lora_r}, lora_alpha={lora_alpha}")
    
    while step < max_steps:
        # Training epoch
        progress_bar = tqdm(train_loader, desc=f"Step {step}")
        epoch_loss = 0
        
        for batch_idx, (input_ids, labels) in enumerate(progress_bar):
            # Track FLOPS if tracker provided
            if flop_tracker is not None:
                flop_tracker.log_training_step(
                    seq_len=max_length,
                    batch_size=batch_size, 
                    description=f"Training step {step}"
                )
            
            # Move batch to device
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            
            # Update parameters
            optimizer.step()
            scheduler.step()
            
            # Log progress
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])
            
            # Periodic evaluation
            if step > 0 and step % eval_steps == 0:
                val_loss = evaluate(model, val_loader, device, flop_tracker)
                logger.info(f"Step {step}: Validation loss = {val_loss:.4f}")
                
                # Save history
                history["train_loss"].append(epoch_loss / (batch_idx + 1))
                history["val_loss"].append(val_loss)
                history["steps"].append(step)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_lora_model(model, os.path.join(output_dir, f"best_lora_r{lora_r}_a{lora_alpha}_lr{learning_rate:.0e}"))
                
                # Reset for next evaluation
                model.train()
            
            # Periodic saving
            if step > 0 and step % save_steps == 0:
                save_lora_model(model, os.path.join(output_dir, f"step_{step}_lora_r{lora_r}_a{lora_alpha}_lr{learning_rate:.0e}"))
            
            # Increment step
            step += 1
            if step >= max_steps:
                break
    
    # Final evaluation
    final_val_loss = evaluate(model, val_loader, device, flop_tracker)
    logger.info(f"Final validation loss: {final_val_loss:.4f}")
    
    # Save final model
    save_lora_model(model, os.path.join(output_dir, f"final_lora_r{lora_r}_a{lora_alpha}_lr{learning_rate:.0e}"))
    
    # Save training history
    history_path = os.path.join(output_dir, f"history_r{lora_r}_a{lora_alpha}_lr{learning_rate:.0e}.json")
    with open(history_path, 'w') as f:
        json.dump(history, f)
    
    return model, history


def evaluate(model, val_loader, device, flop_tracker=None):
    """
    Evaluate the model on validation data.
    
    Args:
        model: The model to evaluate
        val_loader: DataLoader for validation data
        device: Device to use
        flop_tracker: Optional FLOPTracker for monitoring FLOP usage
        
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for input_ids, labels in tqdm(val_loader, desc="Evaluating"):
            # Track FLOPS if tracker provided
            if flop_tracker is not None:
                flop_tracker.log_training_step(
                    seq_len=input_ids.shape[1],
                    batch_size=input_ids.shape[0],
                    is_validation=True,
                    description="Validation step"
                )
            
            # Move batch to device
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            
            # Accumulate loss
            total_loss += loss.item()
    
    # Calculate average loss
    avg_loss = total_loss / len(val_loader)
    
    return avg_loss


def save_lora_model(model, output_path):
    """
    Save LoRA weights only.
    
    Args:
        model: Model with LoRA layers
        output_path: Path to save the model weights
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Extract and save LoRA weights
    lora_state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state_dict[f"{name}.lora_A"] = module.lora_A.data.cpu()
            lora_state_dict[f"{name}.lora_B"] = module.lora_B.data.cpu()
    
    # Save weights
    torch.save(lora_state_dict, os.path.join(output_path, "lora_weights.pt"))
    
    # Save config
    config = {
        "model_type": "Qwen2.5-0.5B-Instruct",
        "lora_applied_modules": ["q_proj", "v_proj"],
        "timestamp": torch.datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    }
    
    with open(os.path.join(output_path, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)


def load_lora_weights(model, weights_path):
    """
    Load LoRA weights into a model.
    
    Args:
        model: Model with LoRA layers
        weights_path: Path to the saved weights
        
    Returns:
        Model with loaded weights
    """
    lora_state_dict = torch.load(weights_path, map_location="cpu")
    
    # Load weights into model
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            if f"{name}.lora_A" in lora_state_dict:
                module.lora_A.data.copy_(lora_state_dict[f"{name}.lora_A"])
            if f"{name}.lora_B" in lora_state_dict:
                module.lora_B.data.copy_(lora_state_dict[f"{name}.lora_B"])
    
    return model


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
    args = parser.parse_args()
    
    # Initialize FLOP tracker
    flop_tracker = FLOPTracker(
        experiment_name=f"lora_r{args.lora_r}_a{args.lora_alpha}_lr{args.learning_rate:.0e}",
        hidden_size=896,  # Qwen2.5-0.5B-Instruct
        num_attention_heads=14,
        num_hidden_layers=24,
        intermediate_size=4864,
        head_dim=64,
        vocab_size=151936,
        max_budget=1e17
    )
    
    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_qwen()
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    train_texts, val_texts = load_and_preprocess(
        "data/lotka_volterra_data.h5",
        tokenizer=None,  # We'll tokenize later
        alpha=args.alpha,
        precision=args.precision
    )
    
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