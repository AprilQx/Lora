import torch
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import logging
# Configure logging
logger = logging.getLogger(__name__)
import sys
# Create results directory
# Add project root to Python path
project_root = Path(__file__).parent.parent
print(f"Project root: {project_root}")
sys.path.append(str(project_root))
# Create results directory
RESULTS_DIR = Path(project_root) / "results"/"figures"  # Use the project_root variable




def setup_device(model):
    """
    Set up the device for the model.
    
    Args:
        model: The model to move to the appropriate device
        
    Returns:
        The device and model
    """
    # Check if CUDA is available (for GPU acceleration)
    if torch.cuda.is_available():
        logger.info("CUDA is available. Moving model to CUDA device.")
        device = torch.device("cuda")
    # Check if MPS is available (for Mac with Apple Silicon)
    elif torch.backends.mps.is_available():
        logger.info("MPS backend is available. Moving model to MPS device.")
        device = torch.device("mps")
    else:
        logger.info("Using CPU for computation.")
        device = torch.device("cpu")
    
    model = model.to(device)
    logger.info(f"Using device: {device}")
    
    return device, model


def save_results(all_results, successful_results, config, val_size):
    """
    Save evaluation results to a file and create visualizations.
    
    Args:
        all_results: List of all evaluation results
        successful_results: List of successful evaluation results
        config: Configuration parameters
        val_size: Size of the validation set
    """
    successful_count = len(successful_results)
    
    # Calculate metrics
    metrics = {
        "overall_mse": np.mean([r["mse"] for r in successful_results]),
        "overall_mae": np.mean([r["mae"] for r in successful_results]),
        "prey_mse": np.mean([r["prey_mse"] for r in successful_results]),
        "prey_mae": np.mean([r["prey_mae"] for r in successful_results]),
        "predator_mse": np.mean([r["predator_mse"] for r in successful_results]),
        "predator_mae": np.mean([r["predator_mae"] for r in successful_results]),
        "success_rate": successful_count / val_size
    }
    
    # Log metrics
    logger.info(f"Overall evaluation metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save results to JSON file
    results_file = RESULTS_DIR / f"untrained_model_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump({
            "metadata": {
                "model": "Qwen2.5-0.5B-Instruct (untrained)",
                "dataset": "Lotka-Volterra",
                **config,
                "validation_size": val_size,
                "timestamp": timestamp
            },
            "metrics": metrics,
            "results": [{k: v for k, v in r.items() if k != "ground_truth" and k != "predictions"} 
                       for r in all_results]
        }, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    # Create distribution visualizations
    metrics_df = pd.DataFrame([{
        "MSE": r["mse"],
        "MAE": r["mae"],
        "Prey MSE": r["prey_mse"],
        "Prey MAE": r["prey_mae"],
        "Predator MSE": r["predator_mse"],
        "Predator MAE": r["predator_mae"]
    } for r in successful_results])
    
    # Create histogram of errors
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    metrics_to_plot = ["MSE", "MAE", "Prey MSE", "Prey MAE", "Predator MSE", "Predator MAE"]
    
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i // 3, i % 3]
        sns.histplot(metrics_df[metric], kde=True, ax=ax)
        ax.set_title(f'Distribution of {metric}')
        ax.set_xlabel(metric)
        ax.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"error_distributions_{timestamp}.png")
    plt.close(fig)
    
    logger.info(f"Error distribution visualization saved")
