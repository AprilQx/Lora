"""
Context length exploration using pretrained LoRA weights on Qwen2.5-Instruct model 
for Lotka-Volterra time series forecasting.

This script performs experiments with different context lengths:
- Context lengths: 128, 512, 768
- Uses previously trained LoRA weights as a starting point:
  - /Users/apple/Documents/GitLab_Projects/M2_coursework/results/hyperparameter/models/lr1e-04_rank8_prec2/best_lora_r8_a16_lr1e-04/lora_weights.pt
  - /Users/apple/Documents/GitLab_Projects/M2_coursework/results/hyperparameter/models/lr1e-04_rank4_prec3/best_lora_r4_a8_lr1e-04/lora_weights.pt

Each configuration is trained for up to 2,000 optimizer steps and evaluated
on the validation set to see the effect of context length.
"""

import torch
import numpy as np
import os
import json
import logging
from pathlib import Path
import argparse
import sys
import wandb
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import custom modules
from src.models.qwen import load_qwen
from src.data.preprocessor import load_data, numeric_to_text
from src.models.finetune_lora import train_lora
from utils.lora_flop_tracker import LoRAFLOPTracker
from src.models.lora import apply_lora_to_model, load_lora_weights

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("context_length_finetuning.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create results directory
RESULTS_DIR = Path(project_root) / "results"
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR = RESULTS_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
SEARCH_DIR = RESULTS_DIR / "context_length_finetuning"
SEARCH_DIR.mkdir(exist_ok=True)

# Also ensure the hyperparameter paths exist
HYPERPARAMETER_DIR = RESULTS_DIR / "hyperparameter"
HYPERPARAMETER_MODELS_DIR = HYPERPARAMETER_DIR / "models"

def run_context_length_finetuning(pretrained_model_path, lora_r, learning_rate, precision):
    """
    Run experiments with different context lengths, using pretrained LoRA weights.
    
    Args:
        pretrained_model_path: Path to pretrained LoRA model weights
        lora_r: LoRA rank
        learning_rate: Learning rate
        precision: Precision for text representation
    """
    # Define context lengths to explore
    context_lengths = [128, 512, 768]
    
    # Fixed hyperparameters
    lora_alpha = lora_r * 2  # Calculate alpha based on rank
    lora_dropout = 0.05
    batch_size = 4
    max_steps = 2000  # Limited to 2,000 steps as specified
    eval_steps = 500
    random_seed = 42
    alpha = 10.0  # Scaling factor for numeric values
    
    # Maximum FLOPs budget
    max_flops = 1e17
    
    # Data files based on precision
    data_paths = {
        2: {
            "train": project_root / "data" / "processed2" / "train_texts.txt",
            "val": project_root / "data" / "processed2" / "val_texts.txt"
        },
        3: {
            "train": project_root / "data" / "processed3" / "train_texts.txt",
            "val": project_root / "data" / "processed3" / "val_texts.txt"
        }
    }
    
    # Get appropriate data files
    train_file = data_paths[precision]["train"]
    val_file = data_paths[precision]["val"]
    
    # Check if pretrained model exists
    pretrained_weights_path = Path(pretrained_model_path) / "lora_weights.pt"
    if not pretrained_weights_path.exists():
        logger.error(f"Pretrained model weights not found at {pretrained_weights_path}. Exiting.")
        return None
    
    # Check if data files exist
    if not train_file.exists() or not val_file.exists():
        logger.error(f"Data files for precision {precision} not found. Exiting.")
        return None
    
    # Load training data for this precision
    try:
        with open(train_file, 'r') as f:
            train_texts = [line.strip() for line in f]
        logger.info(f"Loaded {len(train_texts)} training samples from {train_file}")
    except Exception as e:
        logger.error(f"Error loading training data: {str(e)}")
        return None
    
    # Track results
    search_results = {
        "configurations": [],
        "metrics": [],
        "flops_used": [],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "base_hyperparameters": {
            "learning_rate": learning_rate,
            "lora_rank": lora_r,
            "precision": precision,
            "lora_alpha": lora_alpha,
            "pretrained_model_path": str(pretrained_model_path)
        }
    }
    
    # Main search loop
    for context_length in context_lengths:
        # Set run name
        run_name = f"continued_ctx{context_length}_lr{learning_rate:.0e}_rank{lora_r}_prec{precision}"
        logger.info(f"Starting context length experiment: {run_name}")
        
        # Initialize wandb
        wandb.init(
            project="lora-context-length-finetuning",
            name=f"continued_ctx_{context_length}",
            config={
                "learning_rate": learning_rate,
                "lora_rank": lora_r,
                "precision": precision,
                "context_length": context_length,
                "max_steps": max_steps,
                "eval_steps": eval_steps,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "batch_size": batch_size,
                "max_flops": max_flops,
                "alpha": alpha,
                "pretrained_model_path": str(pretrained_model_path)
            },
            reinit=True
        )
        
        # Create FLOPs tracker for this run
        flop_tracker = LoRAFLOPTracker(
            max_budget=max_flops,
            experiment_name=run_name,
            log_path=str(SEARCH_DIR / "flop_logs"),
            lora_r=lora_r,
            lora_target_modules=["q_proj", "v_proj"],
            # Qwen2.5-0.5B parameters
            hidden_size=896,
            num_attention_heads=14,
            num_hidden_layers=24,
            intermediate_size=4864,
            head_dim=64,
            vocab_size=151936
        )
        
        # Load fresh model
        model, tokenizer = load_qwen()
        
        # Apply LoRA to model
        model, trainable_params = apply_lora_to_model(
            model, 
            r=lora_r, 
            alpha=lora_alpha, 
            dropout=lora_dropout
        )
        
        # Load pretrained LoRA weights
        logger.info(f"Loading pretrained LoRA weights from {pretrained_weights_path}")
        model = load_lora_weights(model, pretrained_weights_path)
        
        # Create output directory for this run
        run_dir = MODELS_DIR / run_name
        run_dir.mkdir(exist_ok=True)
        
        try:
            # Train the model with this configuration
            trained_model, history = train_lora(
                model=model,
                tokenizer=tokenizer,
                train_texts=train_texts,
                val_data_path=val_file,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                learning_rate=learning_rate,
                batch_size=batch_size,
                max_steps=max_steps,
                max_length=context_length,  # This is what we're testing
                eval_steps=eval_steps,
                save_steps=eval_steps,
                output_dir=str(run_dir),
                flop_tracker=flop_tracker,
                use_wandb=True,
                wandb_project="lora-context-length-finetuning",
                wandb_entity=None,
                random_seed=random_seed,
                precision=precision
            )
            
            # Extract best validation metrics
            best_val_mae = min([metrics.get('mae', float('inf')) for metrics in history['val_metrics']])
            best_val_prey_mae = min([metrics.get('prey_mae', float('inf')) for metrics in history['val_metrics']])
            best_val_predator_mae = min([metrics.get('predator_mae', float('inf')) for metrics in history['val_metrics']])
            
            # Get FLOPs used
            flops_used = flop_tracker.total_flops
            
            # Save results
            run_results = {
                "context_length": context_length,
                "best_val_mae": best_val_mae,
                "best_val_prey_mae": best_val_prey_mae,
                "best_val_predator_mae": best_val_predator_mae,
                "flops_used": flops_used,
                "steps_trained": len(history['train_loss'])
            }
            
            # Add to overall results
            search_results["configurations"].append({
                "context_length": context_length
            })
            search_results["metrics"].append({
                "best_val_mae": best_val_mae,
                "best_val_prey_mae": best_val_prey_mae,
                "best_val_predator_mae": best_val_predator_mae,
                "steps_trained": len(history['train_loss'])
            })
            search_results["flops_used"].append(flops_used)
            
            logger.info(f"Completed run {run_name} - Best val MAE: {best_val_mae:.4f}, FLOPs used: {flops_used:.2e}")
            
        except Exception as e:
            logger.error(f"Error in run {run_name}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        
        finally:
            # End wandb run
            wandb.finish()
    
    # Save overall search results
    model_name = Path(pretrained_model_path).parent.name
    results_file = SEARCH_DIR / f"context_length_results_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(search_results, f, indent=2)
    
    # Find the best configuration
    best_idx = np.argmin([m["best_val_mae"] for m in search_results["metrics"]])
    best_config = search_results["configurations"][best_idx]
    best_metric = search_results["metrics"][best_idx]
    
    logger.info(f"Best context length found: {best_config['context_length']}")
    logger.info(f"Best validation MAE: {best_metric['best_val_mae']}")
    
    # Create visualization of results
    create_context_length_visualization(search_results, SEARCH_DIR, model_name)
    
    return search_results


def create_context_length_visualization(search_results, output_dir, model_name):
    """
    Create visualizations of the context length search results.
    
    Args:
        search_results: Dictionary containing search results
        output_dir: Directory to save visualizations
        model_name: Name of the model for file naming
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    
    # Convert results to DataFrame
    data = []
    for i in range(len(search_results["configurations"])):
        config = search_results["configurations"][i]
        metrics = search_results["metrics"][i]
        flops = search_results["flops_used"][i]
        
        data.append({
            "context_length": config["context_length"],
            "best_val_mae": metrics["best_val_mae"],
            "best_val_prey_mae": metrics["best_val_prey_mae"],
            "best_val_predator_mae": metrics["best_val_predator_mae"],
            "flops_used": flops,
            "steps_trained": metrics["steps_trained"]
        })
    
    df = pd.DataFrame(data)
    
    # Create line plot for MAE vs context length
    plt.figure(figsize=(10, 6))
    plt.plot(df["context_length"], df["best_val_mae"], 'o-', linewidth=2, markersize=10)
    plt.title(f"Effect of Context Length on Validation MAE ({model_name})")
    plt.xlabel("Context Length")
    plt.ylabel("Best Validation MAE")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / f"context_length_mae_{model_name}.png")
    
    # Create line plots for prey and predator MAEs
    plt.figure(figsize=(12, 6))
    plt.plot(df["context_length"], df["best_val_prey_mae"], 'o-', label="Prey MAE", linewidth=2, markersize=8)
    plt.plot(df["context_length"], df["best_val_predator_mae"], 's-', label="Predator MAE", linewidth=2, markersize=8)
    plt.title(f"Effect of Context Length on Prey and Predator MAE ({model_name})")
    plt.xlabel("Context Length")
    plt.ylabel("MAE")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / f"context_length_prey_predator_mae_{model_name}.png")
    
    # Create bar chart for FLOPs usage
    plt.figure(figsize=(10, 6))
    sns.barplot(x="context_length", y="flops_used", data=df)
    plt.title(f"FLOPs Usage by Context Length ({model_name})")
    plt.xlabel("Context Length")
    plt.ylabel("FLOPs Used")
    for i, v in enumerate(df["flops_used"]):
        plt.text(i, v + v*0.01, f"{v:.2e}", ha='center')
    plt.tight_layout()
    plt.savefig(output_dir / f"context_length_flops_{model_name}.png")
    
    # Create a combined visualization
    plt.figure(figsize=(14, 8))
    
    # Plot 1: MAE
    plt.subplot(1, 2, 1)
    plt.plot(df["context_length"], df["best_val_mae"], 'o-', color='blue', linewidth=2, markersize=10)
    plt.title(f"Validation MAE vs Context Length ({model_name})")
    plt.xlabel("Context Length")
    plt.ylabel("Best Validation MAE")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: FLOPs
    plt.subplot(1, 2, 2)
    sns.barplot(x="context_length", y="flops_used", data=df, palette="viridis")
    plt.title(f"FLOPs Usage by Context Length ({model_name})")
    plt.xlabel("Context Length")
    plt.ylabel("FLOPs Used (Scientific Notation)")
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    plt.savefig(output_dir / f"context_length_combined_{model_name}.png")
    
    # Create a table view of results
    summary_table = df.copy()
    summary_table["flops_used_scientific"] = summary_table["flops_used"].apply(lambda x: f"{x:.2e}")
    summary_table = summary_table.sort_values("best_val_mae")
    
    # Save the summary table as CSV
    summary_table.to_csv(output_dir / f"context_length_results_summary_{model_name}.csv", index=False)
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run context length fine-tuning with pretrained LoRA weights")
    
    # Define command-line arguments
    parser.add_argument("--model_1", action="store_true", help="Use Model 1: lr1e-04_rank8_prec2")
    parser.add_argument("--model_2", action="store_true", help="Use Model 2: lr1e-04_rank4_prec3")
    parser.add_argument("--custom_path", type=str, help="Custom path to pretrained model weights")
    parser.add_argument("--lr", type=float, help="Custom learning rate (default: same as pretrained)")
    parser.add_argument("--rank", type=int, help="Custom LoRA rank (default: same as pretrained)")
    parser.add_argument("--precision", type=int, help="Custom precision (default: same as pretrained)")
    
    args = parser.parse_args()
    
    # Define paths to pretrained models using relative paths
    model_1_path = project_root / "results" / "hyperparameter" / "models" / "lr1e-04_rank8_prec2" / "best_lora_r8_a16_lr1e-04"
    model_2_path = project_root / "results" / "hyperparameter" / "models" / "lr1e-04_rank4_prec3" / "best_lora_r4_a8_lr1e-04"
    
    if args.model_1:
        # Run fine-tuning with model 1
        logger.info("Running context length fine-tuning with Model 1 (lr1e-04_rank8_prec2)")
        lr = args.lr if args.lr is not None else 1e-4
        rank = args.rank if args.rank is not None else 8
        precision = args.precision if args.precision is not None else 2
        search_results = run_context_length_finetuning(model_1_path, rank, lr, precision)
    elif args.model_2:
        # Run fine-tuning with model 2
        logger.info("Running context length fine-tuning with Model 2 (lr1e-04_rank4_prec3)")
        lr = args.lr if args.lr is not None else 1e-4
        rank = args.rank if args.rank is not None else 4
        precision = args.precision if args.precision is not None else 2
        search_results = run_context_length_finetuning(model_2_path, rank, lr, precision)
    elif args.custom_path:
        # Run fine-tuning with custom path
        logger.info(f"Running context length fine-tuning with custom model at {args.custom_path}")
        if args.lr is None or args.rank is None or args.precision is None:
            logger.error("Custom path requires specifying --lr, --rank, and --precision. Exiting.")
            sys.exit(1)
        search_results = run_context_length_finetuning(args.custom_path, args.rank, args.lr, args.precision)
    else:
        # Run both models sequentially
        logger.info("Running context length fine-tuning with both models")
        
        # Model 1
        logger.info("Starting with Model 1 (lr1e-04_rank8_prec2)")
        search_results_1 = run_context_length_finetuning(model_1_path, 8, 1e-4, 2)
        
        # Model 2
        logger.info("Starting with Model 2 (lr1e-04_rank4_prec3)")
        search_results_2 = run_context_length_finetuning(model_2_path, 4, 1e-4, 2)
        
        # Print summary of both runs
        print("\nContext Length Fine-tuning Complete!")
        
        if search_results_1:
            best_idx_1 = np.argmin([m["best_val_mae"] for m in search_results_1["metrics"]])
            best_config_1 = search_results_1["configurations"][best_idx_1]
            best_metric_1 = search_results_1["metrics"][best_idx_1]
            print(f"\nModel 1 (lr1e-04_rank8_prec2):")
            print(f"Best Context Length: {best_config_1['context_length']}")
            print(f"Best Validation MAE: {best_metric_1['best_val_mae']:.6f}")
        
        if search_results_2:
            best_idx_2 = np.argmin([m["best_val_mae"] for m in search_results_2["metrics"]])
            best_config_2 = search_results_2["configurations"][best_idx_2]
            best_metric_2 = search_results_2["metrics"][best_idx_2]
            print(f"\nModel 2 (lr1e-04_rank4_prec3):")
            print(f"Best Context Length: {best_config_2['context_length']}")
            print(f"Best Validation MAE: {best_metric_2['best_val_mae']:.6f}")
        
        # Determine the overall best model
        if search_results_1 and search_results_2:
            best_mae_1 = min([m["best_val_mae"] for m in search_results_1["metrics"]])
            best_mae_2 = min([m["best_val_mae"] for m in search_results_2["metrics"]])
            
            if best_mae_1 < best_mae_2:
                print("\nOverall best model: Model 1 (lr1e-04_rank8_prec2)")
                print(f"Best Validation MAE: {best_mae_1:.6f}")
            else:
                print("\nOverall best model: Model 2 (lr1e-04_rank4_prec3)")
                print(f"Best Validation MAE: {best_mae_2:.6f}")