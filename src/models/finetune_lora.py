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
from src.evaluation.evaluation_v2 import evaluate_forecasting, calculate_summary_metrics

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



from src.models.lora import apply_lora_to_model, save_lora_model, get_grad_norm, process_sequences, load_validation_data_from_file


def evaluate(model, tokenizer, validation_data, device, flop_tracker=None):
    """
    Evaluate the model using inference-style forecasting on validation data
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        validation_data: List of validation sequences
        device: Device to run evaluation on
        flop_tracker: FLOPTracker instance for monitoring FLOP usage
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    # Configuration for evaluation
    eval_config = {
        "input_steps": 50,
        "forecast_steps": 3,
        "alpha": 10.0,
        "precision": 3,
        "max_tokens": 48
    }
    
    # If validation_data is a file path, load it
    if isinstance(validation_data, str) and os.path.isfile(validation_data):
        validation_data = load_validation_data_from_file(
            validation_data,
            input_steps=eval_config["input_steps"],
            forecast_steps=eval_config["forecast_steps"],
            num_samples=10  # Limit validation to 10 samples to save time
        )
    
    # Ensure model is on the correct device
    model = model.to(device)
    
    all_results = []
    for i, seq in enumerate(tqdm(validation_data, desc="Evaluating")):
        # Use input_sequence and ground_truth from the sequence
        input_text = seq["input_sequence"]
        ground_truth = text_to_numeric(seq["ground_truth"])
        
        # Perform forecasting evaluation
        result = evaluate_forecasting(
            model=model,
            tokenizer=tokenizer,
            input_text=input_text,
            ground_truth=ground_truth,
            forecast_steps=eval_config["forecast_steps"],
            alpha=eval_config["alpha"],
            precision=eval_config["precision"],
            max_tokens=eval_config["max_tokens"]
        )
        
        # Add metadata
        result["trajectory_idx"] = seq.get("original_idx", i)
        
        # Calculate FLOPs (if using flop tracker)
        if flop_tracker and result.get("input_token_length") is not None:
            # Using the flop tracker to account for FLOPs
            flops = flop_tracker.log_inference(
                context_len=result["input_token_length"],
                gen_len=result.get("generated_token_length", 0),
                batch_size=1,
                description=f"Validation inference on sequence {i}"
            )
            result["flops"] = flops
            
        all_results.append(result)
    
    # Calculate summary metrics
    metrics = calculate_summary_metrics(all_results)
    
    return metrics


def train_lora(
    model, 
    tokenizer, 
    train_texts, 
    val_data_path, 
    lora_r=8, 
    lora_alpha=16, 
    lora_dropout=0.0,
    learning_rate=1e-4,
    batch_size=4,
    max_steps=2000,
    max_length=128,
    eval_steps=500,
    save_steps=500,
    device=None,
    output_dir="../../results/models",
    flop_tracker=None,
    use_wandb=True,
    wandb_project="lora-finetuning",
    wandb_entity=None,
    random_seed=42
):
    """
    Train a model with LoRA.
    
    Args:
        model: The base model
        tokenizer: The tokenizer
        train_texts: Training text samples
        val_data_path: Path to validation data file
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

    # Set global random seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

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
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(train_input_ids, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)
    
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
    
   
    # Load validation data
    logger.info(f"Loading validation data from {val_data_path}")
    validation_data = load_validation_data_from_file(
        val_data_path,
        input_steps=50,
        forecast_steps=3,
        num_samples=5,  # Limit validation to 5 samples
        random_seed=42,  # Use the same seed as the rest of the training
    )
    # Training loop
    model.train()
    step = 0
    best_val_loss = float("inf")
    history = {
        "train_loss": [],
        "val_metrics": [],
        "steps": []
    }
    
    logger.info(f"Starting training with batch_size={batch_size}, lr={learning_rate}, lora_r={lora_r}, lora_alpha={lora_alpha}")
    
    # Track gradient norms
    while step < max_steps:
        # Training epoch
        progress_bar = tqdm(train_loader, desc=f"Step {step}")
        epoch_loss = 0.0
        
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
            progress_bar.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])
            step+=1
            
            # Log to wandb
            if use_wandb:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/grad_norm_before_clip": grad_norm_before_clip,
                    "train/grad_norm_after_clip": grad_norm_after_clip,
                    "train/grad_norm_ratio": grad_norm_after_clip / grad_norm_before_clip
                },step=step)
                
                if flop_tracker is not None:
                    wandb.log({
                        "flops/total": flop_tracker.total_flops,
                        "flops/percent_used": (flop_tracker.total_flops / flop_tracker.max_budget) * 100
                    }  ,step=step)

                     # Check if we've reached 90% of the FLOP budget
                    if flop_tracker.total_flops >= 0.9 * flop_tracker.max_budget:
                        logger.warning(f"90% of FLOP budget reached ({flop_tracker.total_flops:.2e}/{flop_tracker.max_budget:.2e}). Terminating training.")
                        # Force the break from the training loop
                        step = max_steps  # This will cause the outer loop to exit
                        break  # Break from the current dataloader loop
            
            # Periodic evaluation
            if step > 0 and step % eval_steps == 0:
                logger.info(f"Evaluating at step {step}")
                # Use inference-style evaluation
                val_metrics = evaluate(
                    model, 
                    tokenizer, 
                    validation_data, 
                    device, 
                    flop_tracker
                )
                
                # Log all metrics
                if use_wandb:
                    wandb.log({
                        f"eval/{k}": v for k, v in val_metrics.items()
                    }, step=step)
                
                # Also track learning curves
                if use_wandb and 'mae' in val_metrics:
                    wandb.log({
                        "curves/train_loss": loss.item(),
                        "curves/val_mae": val_metrics["mae"],
                    },  step=step)
                
                # Save best model based on MAE
                current_val_metric = val_metrics.get('mae', float('inf'))
                if current_val_metric < best_val_loss:
                    best_val_loss = current_val_metric
                    logger.info(f"New best model with MAE: {best_val_loss}")
                    model_path = os.path.join(output_dir, f"best_lora_r{lora_r}_a{lora_alpha}_lr{learning_rate:.0e}")
                    save_lora_model(model, model_path)
                    
                    if use_wandb:
                        # Save the model to wandb
                        artifact = wandb.Artifact(f"best_model_step_{step}", type="model")
                        artifact.add_dir(model_path)
                        wandb.log_artifact(artifact)
                
                # Add to history
                history["val_metrics"].append(val_metrics)
                history["steps"].append(step)
                history["train_loss"].append(loss.item())
                
                # Reset for next evaluation
                model.train()
            
            # Periodic saving
            if step > 0 and step % save_steps == 0:
                 save_lora_model(model, os.path.join(output_dir, f"step_{step}_lora_r{lora_r}_a{lora_alpha}_lr{learning_rate:.0e}"))
            
            # Increment step
            if step >= max_steps:
                break
    
    # Final evaluation
    logger.info("Performing final evaluation")
    final_val_metrics = evaluate(model, tokenizer, validation_data, device, flop_tracker)
    logger.info(f"Final evaluation metrics: {final_val_metrics}")
    
    if use_wandb:
        wandb.log({
            f"eval/final_{k}": v for k, v in final_val_metrics.items()
        }, step=step)
    
    # Save final model
    final_model_path = os.path.join(output_dir, f"final_lora_r{lora_r}_a{lora_alpha}_lr{learning_rate:.0e}")
    save_lora_model(model, final_model_path)
    
    # Save training history
    history_path = os.path.join(output_dir, f"history_r{lora_r}_a{lora_alpha}_lr{learning_rate:.0e}.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    if use_wandb:
        # Save the history to wandb
        wandb.save(history_path)
        
        # Save final flop report if available
        if flop_tracker is not None:
            flop_report = flop_tracker.generate_report()
            flop_plot= flop_tracker.plot_usage()
            #wandb.save(flop_report)
            #wandb.save(flop_plot)

            wandb.log({
                "flops/final_total": flop_report["total_flops"],
                "flops/final_percent_used": flop_report["budget_used_percent"],
                "flops/training_flops": flop_report.get("training_flops", 0),
                "flops/validation_flops": flop_report.get("validation_flops", 0)
            }, step=step)
        
        # Finish the wandb run
        wandb.finish()
    
    return model, history

