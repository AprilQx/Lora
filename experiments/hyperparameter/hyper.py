"""
Hyperparameter search for LoRA fine-tuning on Qwen2.5-Instruct model 
for Lotka-Volterra time series forecasting.

This script performs a grid search over:
- Learning rates: 1e-5, 5e-5, 1e-4
- LoRA ranks: 2, 4, 8
- Precision values: 2, 3

For a total of 18 configurations. Each configuration is trained
for up to 10,000 optimizer steps and evaluated on the validation set.
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
from src.data.preprocessor import load_data, numeric_to_text
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
    Run hyperparameter search over learning rate, LoRA rank, and precision.
    """
    # Define hyperparameter grid
    learning_rates = [1e-5, 5e-5, 1e-4]
    lora_ranks = [2, 4, 8]
    precision_values = [2, 3]  # Added precision parameter
    
    # Fixed hyperparameters
    context_length = 128  # Using default context length
    lora_dropout = 0.05
    batch_size = 4
    max_steps = 2000  # Full training budget
    eval_steps = 500
    random_seed = 42
    alpha = 10.0  # Scaling factor for numeric values
    
    # Maximum FLOPs budget
    max_flops = 1e17
    
    # Data files - processed2 for precision 2, processed3 for precision 3
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
    
    # Track results
    search_results = {
        "hyperparameters": [],
        "metrics": [],
        "flops_used": [],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Calculate total configurations
    total_configs = len(learning_rates) * len(lora_ranks) * len(precision_values)
    current_config = 0
    
    # Main hyperparameter search loop
    for lr, rank, precision in product(learning_rates, lora_ranks, precision_values):
        current_config += 1
        # Set run name to include precision
        run_name = f"lr{lr:.0e}_rank{rank}_prec{precision}"
        logger.info(f"Starting hyperparameter search run {current_config}/{total_configs}: {run_name}")

        # Get the appropriate train and validation files based on precision
        train_file = data_paths[precision]["train"]
        val_file = data_paths[precision]["val"]
        
        # Check if files exist
        if not train_file.exists() or not val_file.exists():
            logger.error(f"Data files for precision {precision} not found. Skipping this configuration.")
            continue
        
        # Load training data for this precision
        try:
            with open(train_file, 'r') as f:
                train_texts = [line.strip() for line in f]
            logger.info(f"Loaded {len(train_texts)} training samples from {train_file}")
        except Exception as e:
            logger.error(f"Error loading training data for precision {precision}: {str(e)}")
            continue

        # Initialize wandb for this specific run
        wandb.init(
            project="lora-hyperparameter-search",
            name=f"grid_search_{run_name}",
            config={
                "learning_rate": lr,
                "lora_rank": rank,
                "precision": precision,
                "context_length": context_length,
                "max_steps": max_steps,
                "eval_steps": eval_steps,
                "lora_alpha": rank * 2,  # Calculate alpha based on rank
                "lora_dropout": lora_dropout,
                "batch_size": batch_size,
                "max_flops": max_flops,
                "alpha": alpha
            },
            # Important: reinitialize wandb for each run
            reinit=True
        )
        
        # Create FLOPs tracker for this run
        flop_tracker = LoRAFLOPTracker(
            max_budget=max_flops,
            experiment_name=run_name,
            log_path=str(SEARCH_DIR / "flop_logs"),
            lora_r=rank,
            lora_target_modules=["q_proj", "v_proj"],
            # Add more specific model parameters for Qwen2.5-0.5B
            hidden_size=896,
            num_attention_heads=14,
            num_hidden_layers=24,
            intermediate_size=4864,
            head_dim=64,
            vocab_size=151936
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
                max_length=context_length,
                eval_steps=eval_steps,
                save_steps=eval_steps,
                output_dir=str(run_dir),
                flop_tracker=flop_tracker,
                use_wandb=True,
                wandb_project="lora-hyperparameter-search",
                wandb_entity=None,
                random_seed=random_seed,
                precision=precision  # Pass precision to train_lora
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
                "precision": precision,
                "context_length": context_length,
                "lora_alpha": current_alpha,
                "best_val_mae": best_val_mae,
                "best_val_prey_mae": best_val_prey_mae,
                "best_val_predator_mae": best_val_predator_mae,
                "flops_used": flops_used,
                "steps_trained": len(history['train_loss'])
            }
        
            
            # Add to overall results
            search_results["hyperparameters"].append({
                "learning_rate": lr,
                "lora_rank": rank,
                "precision": precision,
                "context_length": context_length,
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
    logger.info(f"Precision: {best_config['precision']}")
    logger.info(f"Best validation MAE: {best_metric['best_val_mae']}")
    
    # Log best configuration to wandb
    wandb.log({
        "best_config/learning_rate": best_config['learning_rate'],
        "best_config/lora_rank": best_config['lora_rank'],
        "best_config/precision": best_config['precision'],
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
            "precision": params["precision"],
            "best_val_mae": metrics["best_val_mae"],
            "best_val_prey_mae": metrics["best_val_prey_mae"],
            "best_val_predator_mae": metrics["best_val_predator_mae"],
            "flops_used": flops,
            "steps_trained": metrics["steps_trained"]
        })
    
    df = pd.DataFrame(data)
    
    # Convert learning rate to string for better display
    df["learning_rate_str"] = df["learning_rate"].apply(lambda x: f"{x:.0e}")
    
    # Create heatmap of validation MAE by learning rate and rank for each precision
    for precision in df['precision'].unique():
        df_prec = df[df['precision'] == precision]
        
        plt.figure(figsize=(10, 8))
        pivot = df_prec.pivot(
            index="lora_rank", 
            columns="learning_rate_str", 
            values="best_val_mae"
        )
        sns.heatmap(pivot, annot=True, fmt=".4f", cmap="viridis_r")
        plt.title(f"Validation MAE by Learning Rate and LoRA Rank (Precision = {precision})")
        plt.xlabel("Learning Rate")
        plt.ylabel("LoRA Rank")
        
        plt.tight_layout()
        plt.savefig(output_dir / f"heatmap_mae_precision_{precision}.png")
    
    # Create bar plot comparing all configurations
    plt.figure(figsize=(15, 8))
    
    # Create configuration labels
    df["config"] = df.apply(
        lambda row: f"lr={row['learning_rate_str']}\nrank={row['lora_rank']}\nprec={row['precision']}", 
        axis=1
    )
    
    # Sort by MAE
    df_sorted = df.sort_values("best_val_mae")
    
    # Plot
    sns.barplot(x="config", y="best_val_mae", data=df_sorted)
    plt.xticks(rotation=45)
    plt.title("Validation MAE by Configuration")
    plt.tight_layout()
    plt.savefig(output_dir / "bar_plot_configs.png")
    
    # Plot for different precision values
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="precision", y="best_val_mae", data=df)
    plt.title("Effect of Precision on Validation MAE")
    plt.tight_layout()
    plt.savefig(output_dir / "precision_effect.png")
    
    # Plot interactions between learning rate and rank for each precision
    g = sns.FacetGrid(df, col="precision", height=6, aspect=1.2)
    g.map_dataframe(sns.scatterplot, x="learning_rate", y="lora_rank", 
                   size="best_val_mae", hue="best_val_mae", 
                   sizes=(100, 400), size_norm=(df['best_val_mae'].min(), df['best_val_mae'].max()),
                   palette="viridis_r")
    g.set_axis_labels("Learning Rate", "LoRA Rank")
    g.add_legend(title="MAE", bbox_to_anchor=(1.05, 0.5), loc="center left")
    plt.tight_layout()
    plt.savefig(output_dir / "lr_rank_precision_interaction.png")
    
    # Create a combined heatmap with precision as rows
    combined_pivot = pd.pivot_table(
        df, 
        values='best_val_mae',
        index=['precision', 'lora_rank'],
        columns='learning_rate_str'
    )
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(combined_pivot, annot=True, fmt=".4f", cmap="viridis_r")
    plt.title("Validation MAE by Configuration (lower is better)")
    plt.tight_layout()
    plt.savefig(output_dir / "combined_heatmap.png")
    
    # Save the DataFrame as CSV
    df.to_csv(output_dir / "search_results.csv", index=False)
    
    # Create a summary table
    summary = df.groupby('precision')['best_val_mae'].agg(['mean', 'min', 'max', 'std'])
    summary.to_csv(output_dir / "precision_summary.csv")
    
    # Plot summary
    plt.figure(figsize=(10, 6))
    summary.plot(kind='bar', y=['mean', 'min', 'max'])
    plt.title("Summary Statistics by Precision")
    plt.tight_layout()
    plt.savefig(output_dir / "precision_summary.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hyperparameter search for LoRA fine-tuning")
    
    # Define any additional command-line arguments here
    parser.add_argument("--output_dir", type=str, default=str(SEARCH_DIR), 
                        help="Directory to save search results")
    parser.add_argument("--use_wandb", action="store_true", 
                        help="Whether to use Weights & Biases for tracking")
    
    args = parser.parse_args()
    
    # Run the search
    best_config, best_metric = run_hyperparameter_search()
    
    # Print the best configuration
    print("\nBest Hyperparameter Configuration:")
    print(f"Learning Rate: {best_config['learning_rate']}")
    print(f"LoRA Rank: {best_config['lora_rank']}")
    print(f"Precision: {best_config['precision']}")
    print(f"Best Validation MAE: {best_metric['best_val_mae']:.6f}")