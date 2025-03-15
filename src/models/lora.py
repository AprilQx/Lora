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
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.preprocessor import text_to_numeric, numeric_to_text, load_data
from src.models.qwen import load_qwen

from utils.lora_flop_tracker import LoRAFLOPTracker

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


def get_grad_norm(model):
        total_norm = 0.0
        parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
        for p in parameters:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm



def evaluate(model, val_loader, device, tokenizer=None, flop_tracker=None):
    """Enhanced evaluation with additional metrics"""
    model.eval()
    total_loss = 0
    total_batches = 0
    
    # New metrics
    total_tokens = 0
    correct_tokens = 0
    
    with torch.no_grad():
        for input_ids, labels in tqdm(val_loader, desc="Evaluating"):
            # Track FLOPS if tracker provided
            if flop_tracker is not None:
                flop_tracker.log_validation_step(
                    seq_len=input_ids.shape[1],
                    batch_size=input_ids.shape[0],
                    description="Validation step"
                )
            
            # Move batch to device
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            
            # Get logits for token accuracy
            logits = outputs.logits
            
            # Token accuracy calculation
            pred_tokens = torch.argmax(logits, dim=-1)
            mask = (labels != -100)  # Ignore padding tokens
            correct = ((pred_tokens == labels) & mask).sum().item()
            total = mask.sum().item()
            
            correct_tokens += correct
            total_tokens += total
            
            # Accumulate loss
            total_loss += loss.item()
            total_batches += 1
    
    # Calculate metrics
    avg_loss = total_loss / max(total_batches, 1)
    perplexity = math.exp(avg_loss)
    token_accuracy = correct_tokens / max(total_tokens, 1)
    
    metrics = {
        "val_loss": avg_loss,
        "val_perplexity": perplexity,
        "val_token_accuracy": token_accuracy,
    }
    
    # Skip time series specific metrics for now until text_to_numeric works
    
    return metrics


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
    lora_r = None
    lora_alpha = None
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state_dict[f"{name}.lora_A"] = module.lora_A.data.cpu()
            lora_state_dict[f"{name}.lora_B"] = module.lora_B.data.cpu()

            # Capture parameters from the first LoRA module
            if lora_r is None:
                lora_r = module.r
                lora_alpha = module.alpha
    
    
    # Save weights
    torch.save(lora_state_dict, os.path.join(output_path, "lora_weights.pt"))
    
    # Save config
    config = {
        "model_type": "Qwen2.5-0.5B-Instruct",
        "lora_applied_modules": ["q_proj", "v_proj"],
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "timestamp": datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
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
    output_dir="../../results/models",
    flop_tracker=None,
    use_wandb=True,
    wandb_project="lora-finetuning",
    wandb_entity=None
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
        use_wandb: Whether to use Weights & Biases for tracking
        wandb_project: Weights & Biases project name
        wandb_entity: Weights & Biases entity name
        
    Returns:
        Trained model and training history
    """
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    logger.info(f"Using device: {device}")

    # Initialize wandb
    if use_wandb:
        run_name = f"lora_r{lora_r}_a{lora_alpha}_lr{learning_rate:.0e}_bs{batch_size}"
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            config={
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "max_steps": max_steps,
                "max_length": max_length,
                "eval_steps": eval_steps,
                "save_steps": save_steps,
                "model": "Qwen2.5-0.5B-Instruct",
                "device": device
            },
            name=run_name
        )
    
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
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate,weight_decay=
                                  0.01)
    
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
    


    # Track gradient norms
    
    while step < max_steps:
        # Training epoch
        progress_bar = tqdm(train_loader, desc=f"Step {step}")
        epoch_loss = 0
        num_batches = 0
        
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
            
            # Calculate gradient norm before clipping
            grad_norm_before_clip = get_grad_norm(model)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            
            # Calculate gradient norm after clipping
            grad_norm_after_clip = get_grad_norm(model)
            
            # Update parameters
            optimizer.step()
            scheduler.step()
            
            # Log progress
            epoch_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])
            
            # Log to wandb
            if use_wandb:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/step": step,
                    "train/grad_norm_before_clip": grad_norm_before_clip,
                    "train/grad_norm_after_clip": grad_norm_after_clip,
                    "train/grad_norm_ratio": grad_norm_after_clip / grad_norm_before_clip
                })
                
                if flop_tracker is not None:
                    wandb.log({
                        "flops/total": flop_tracker.total_flops,
                        "flops/percent_used": (flop_tracker.total_flops / flop_tracker.max_budget) * 100
                    })
            
            # Periodic evaluation
            if step > 0 and step % eval_steps == 0:
                # In the training loop, when calling evaluate:
                val_metrics = evaluate(model, val_loader, device, tokenizer, flop_tracker)

                # Log all metrics
                if use_wandb:
                    wandb.log({
                        f"eval/{k}": v for k, v in val_metrics.items()
                    })

                # Also track learning curves
                if use_wandb:
                    wandb.log({
                        "curves/train_val_loss_ratio": loss.item() / val_metrics["val_loss"],
                        "curves/generalization_gap": abs(loss.item() - val_metrics["val_loss"])
                    })
                
                # Save best model
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    model_path = os.path.join(output_dir, f"best_lora_r{lora_r}_a{lora_alpha}_lr{learning_rate:.0e}")
                    save_lora_model(model, model_path)
                    
                    if use_wandb:
                        # Save the model to wandb
                        artifact = wandb.Artifact(f"best_model_step_{step}", type="model")
                        artifact.add_dir(model_path)
                        wandb.log_artifact(artifact)
                
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
    final_val_loss, final_perplexity = evaluate(model, val_loader, device, flop_tracker)
    logger.info(f"Final validation loss: {final_val_loss:.4f}, Perplexity: {final_perplexity:.2f}")
    
    if use_wandb:
        wandb.log({
            "eval/final_loss": final_val_loss,
            "eval/final_perplexity": final_perplexity
        })
    
    # Save final model
    final_model_path = os.path.join(output_dir, f"final_lora_r{lora_r}_a{lora_alpha}_lr{learning_rate:.0e}")
    save_lora_model(model, final_model_path)
    
    # Save training history
    history_path = os.path.join(output_dir, f"history_r{lora_r}_a{lora_alpha}_lr{learning_rate:.0e}.json")
    with open(history_path, 'w') as f:
        json.dump(history, f)
    
    if use_wandb:
        # Save the history to wandb
        wandb.save(history_path)
        
        # Save final flop report if available
        if flop_tracker is not None:
            flop_report = flop_tracker.generate_report()
            wandb.log({
                "flops/final_total": flop_report["total_flops"],
                "flops/final_percent_used": flop_report["budget_used_percent"],
                "flops/training_flops": flop_report.get("training_flops", 0),
                "flops/validation_flops": flop_report.get("validation_flops", 0)
            })
        
        # Finish the wandb run
        wandb.finish()
    
    return model, history
