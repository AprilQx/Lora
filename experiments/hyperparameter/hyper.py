"""
Hyperparameter search for LoRA fine-tuning on Qwen2.5-Instruct model 
for Lotka-Volterra time series forecasting.

This script performs a grid search over:
- Learning rates: 1e-5, 5e-5, 1e-4
- LoRA ranks: 2, 4, 8
- Context lengths: 128, 512, 768

For each configuration, we train for up to 10,000
optimizer steps and evaluate on the validation set.
"""

import torch
import numpy as np
import os
import json
import logging
from pathlib import Path
import argparse
import sys
from itertools import product
import wandb
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import custom modules
from src.models.qwen import load_qwen
from src.data.preprocessor import load_data
from src.models.finetune_lora import train_lora
from utils.lora_flop_tracker import LoRAFLOPTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hyperparameter_search.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create results directory
RESULTS_DIR = Path(project_root) / "results"
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR = RESULTS_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
SEARCH_DIR = RESULTS_DIR / "hyperparameter_search"
SEARCH_DIR.mkdir(exist_ok=True)

def run_hyperparameter_search():
    """
    Run hyperparameter search over learning rate, LoRA rank, and context length.
    """
    # Define hyperparameter grid
    learning_rates = [1e-5, 5e-5, 1e-4]
    lora_ranks = [2, 4, 8]
    context_lengths = [128, 512, 768]
    
    # Fixed hyperparameters
    lora_alpha = 16  # Scale factor is twice the rank by default
    lora_dropout = 0.05
    batch_size = 16
    max_steps = 5000  # Full training budget, will use early stopping
    eval_steps = 500
    random_seed = 42
    
    # Maximum FLOPs budget
    max_flops = 1e17
    
    # Load training and validation data
    train_file = project_root / "data" / "processed" / "train_texts.txt"
    val_file = project_root / "data" / "processed" / "val_texts.txt"
    
    with open(train_file, 'r') as f:
        train_texts = [line.strip() for line in f]
    
    # Track results
    search_results = {
        "hyperparameters": [],
        "metrics": [],
        "flops_used": [],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Initialize wandb
    wandb.init(
        project="lora-hyperparameter-search",
        name=f"grid_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "learning_rates": learning_rates,
            "lora_ranks": lora_ranks,
            "context_lengths": context_lengths,
            "max_steps": max_steps,
            "eval_steps": eval_steps,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "batch_size": batch_size,
            "max_flops": max_flops
        }
    )
    
    # Main hyperparameter search loop
    for lr, rank, ctx_len in product(learning_rates, lora_ranks, context_lengths):
        # Set run name
        run_name = f"lr{lr:.0e}_rank{rank}_ctx{ctx_len}"
        logger.info(f"Starting hyperparameter search run: {run_name}")
        
        # Create FLOPs tracker for this run
        flop_tracker = LoRAFLOPTracker(
            max_budget=max_flops,
            experiment_name=run_name,
            log_path=str(SEARCH_DIR / "flop_logs"),
            lora_r=rank,
            lora_target_modules=["q_proj", "v_proj"]
        )
        
        # Calculate the appropriate alpha based on the rank
        current_alpha = rank * 2
        
        # Load fresh model for each run
        model, tokenizer = load_qwen()
        
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
                lora_r=rank,
                lora_alpha=current_alpha,
                lora_dropout=lora_dropout,
                learning_rate=lr,
                batch_size=batch_size,
                max_steps=max_steps,
                max_length=ctx_len,
                eval_steps=eval_steps,
                save_steps=eval_steps,
                output_dir=str(run_dir),
                flop_tracker=flop_tracker,
                use_wandb=True,
                wandb_project="lora-hyperparameter-search",
                wandb_entity=None,
                random_seed=random_seed
            )
            
            # Extract best validation metrics
            best_val_mae = min([metrics.get('mae', float('inf')) for metrics in history['val_metrics']])
            best_val_prey_mae = min([metrics.get('prey_mae', float('inf')) for metrics in history['val_metrics']])
            best_val_predator_mae = min([metrics.get('predator_mae', float('inf')) for metrics in history['val_metrics']])
            
            # Get FLOPs used
            flops_used = flop_tracker.total_flops
            
            # Save results
            run_results = {
                "learning_rate": lr,
                "lora_rank": rank,
                "context_length": ctx_len,
                "lora_alpha": current_alpha,
                "best_val_mae": best_val_mae,
                "best_val_prey_mae": best_val_prey_mae,
                "best_val_predator_mae": best_val_predator_mae,
                "flops_used": flops_used,
                "steps_trained": len(history['train_loss'])
            }
            
            # Log to wandb
            wandb.log({
                "grid_search/learning_rate": lr,
                "grid_search/lora_rank": rank,
                "grid_search/context_length": ctx_len,
                "grid_search/best_val_mae": best_val_mae,
                "grid_search/best_val_prey_mae": best_val_prey_mae,
                "grid_search/best_val_predator_mae": best_val_predator_mae,
                "grid_search/flops_used": flops_used
            })
            
            # Add to overall results
            search_results["hyperparameters"].append({
                "learning_rate": lr,
                "lora_rank": rank,
                "context_length": ctx_len,
                "lora_alpha": current_alpha
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
    
    # Save overall search results
    results_file = SEARCH_DIR / f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(search_results, f, indent=2)
    
    # Find the best configuration
    best_idx = np.argmin([m["best_val_mae"] for m in search_results["metrics"]])
    best_config = search_results["hyperparameters"][best_idx]
    best_metric = search_results["metrics"][best_idx]
    
    logger.info(f"Best hyperparameter configuration found:")
    logger.info(f"Learning rate: {best_config['learning_rate']}")
    logger.info(f"LoRA rank: {best_config['lora_rank']}")
    logger.info(f"Context length: {best_config['context_length']}")
    logger.info(f"Best validation MAE: {best_metric['best_val_mae']}")
    
    # Log best configuration to wandb
    wandb.log({
        "best_config/learning_rate": best_config['learning_rate'],
        "best_config/lora_rank": best_config['lora_rank'],
        "best_config/context_length": best_config['context_length'],
        "best_config/best_val_mae": best_metric['best_val_mae']
    })
    
    # Create visualization of results
    create_search_visualization(search_results, SEARCH_DIR)
    
    wandb.finish()
    
    return best_config, best_metric


def create_search_visualization(search_results, output_dir):
    """
    Create visualizations of the hyperparameter search results.
    
    Args:
        search_results: Dictionary containing search results
        output_dir: Directory to save visualizations
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    
    # Convert results to DataFrame
    data = []
    for i in range(len(search_results["hyperparameters"])):
        params = search_results["hyperparameters"][i]
        metrics = search_results["metrics"][i]
        flops = search_results["flops_used"][i]
        
        data.append({
            "learning_rate": params["learning_rate"],
            "lora_rank": params["lora_rank"],
            "context_length": params["context_length"],
            "best_val_mae": metrics["best_val_mae"],
            "best_val_prey_mae": metrics["best_val_prey_mae"],
            "best_val_predator_mae": metrics["best_val_predator_mae"],
            "flops_used": flops,
            "steps_trained": metrics["steps_trained"]
        })
    
    df = pd.DataFrame(data)
    
    # Convert learning rate to string for better display
    df["learning_rate_str"] = df["learning_rate"].apply(lambda x: f"{x:.0e}")
    
    # Create heatmap of validation MAE by learning rate and rank
    plt.figure(figsize=(12, 10))
    for i, ctx_len in enumerate(df["context_length"].unique()):
        plt.subplot(1, 3, i+1)
        pivot = df[df["context_length"] == ctx_len].pivot(
            index="lora_rank", 
            columns="learning_rate_str", 
            values="best_val_mae"
        )
        sns.heatmap(pivot, annot=True, fmt=".4f", cmap="viridis_r")
        plt.title(f"Context Length: {ctx_len}")
        plt.xlabel("Learning Rate")
        plt.ylabel("LoRA Rank")
    
    plt.tight_layout()
    plt.savefig(output_dir / "heatmap_mae_by_ctx_len.png")
    
    # Create bar plot comparing all configurations
    plt.figure(figsize=(14, 8))
    
    # Create configuration labels
    df["config"] = df.apply(
        lambda row: f"lr={row['learning_rate_str']}\nrank={row['lora_rank']}\nctx={row['context_length']}", 
        axis=1
    )
    
    # Sort by MAE
    df_sorted = df.sort_values("best_val_mae")
    
    # Plot
    sns.barplot(x="config", y="best_val_mae", data=df_sorted)
    plt.xticks(rotation=90)
    plt.title("Validation MAE by Configuration")
    plt.tight_layout()
    plt.savefig(output_dir / "bar_plot_configs.png")
    
    # Plot FLOPs usage vs. performance
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x="flops_used", 
        y="best_val_mae", 
        hue="lora_rank", 
        size="context_length",
        style="learning_rate_str", 
        data=df
    )
    plt.xscale("log")
    plt.xlabel("FLOPs Used")
    plt.ylabel("Validation MAE")
    plt.title("Performance vs. Computational Cost")
    plt.tight_layout()
    plt.savefig(output_dir / "flops_vs_performance.png")
    
    # Save the DataFrame as CSV
    df.to_csv(output_dir / "search_results.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hyperparameter search for LoRA fine-tuning")
    
    # Define any additional command-line arguments here
    args = parser.parse_args()
    
    # Run the search
    best_config, best_metric = run_hyperparameter_search()
    
    # Print the best configuration
    print("\nBest Hyperparameter Configuration:")
    print(f"Learning Rate: {best_config['learning_rate']}")
    print(f"LoRA Rank: {best_config['lora_rank']}")
    print(f"Context Length: {best_config['context_length']}")
    print(f"Best Validation MAE: {best_metric['best_val_mae']:.6f}")