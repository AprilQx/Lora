"""
Evaluation of the untrained Qwen2.5-Instruct model's forecasting ability 
on the Lotka-Volterra dataset.
"""

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import json
from datetime import datetime
from pathlib import Path
import sys
# Add project root to Python path
project_root = Path(__file__).parent.parent  # Go up one level from src
sys.path.append(str(project_root))

# Import custom modules
from utils.qwen import load_qwen
from utils.preprocessor import load_data
from utils.evaluation import evaluate_forecasting
from utils.visualization import visualize_predictions
from utils.saving import setup_device,save_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create results directory
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Set random seed for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def main():
    """
    Main function to evaluate the untrained model's forecasting ability.
    """
    print("Entering main function")
    logger.info("Starting evaluation of untrained Qwen2.5 model")
    
    # 1. Load the model and tokenizer
    logger.info("Loading model and tokenizer")
    model, tokenizer = load_qwen()

    # 2. Setup device
    _, model = setup_device(model)
    
    # 3. Load the Lotka-Volterra dataset
    logger.info("Loading Lotka-Volterra dataset")
    file_path = "data/lotka_volterra_data.h5"
    trajectories, time_points = load_data(file_path)
    
    # 4. Create validation set
    logger.info("Creating validation set")
    val_size = 50  # Use 50 trajectories for evaluation
    indices = np.random.permutation(trajectories.shape[0])
    val_indices = indices[:val_size]
    val_trajectories = trajectories[val_indices]
    
    # 5. Define parameters for preprocessing and evaluation
    config = {
        "alpha": 10.0,  # Scaling parameter
        "precision": 3,  # Decimal points to keep
        "input_steps": 50,  # Use first 50 steps as input
        "forecast_steps": 50,  # Forecast the next 50 steps
    }
    
    # 6. Evaluate on validation set
    logger.info(f"Evaluating model on {val_size} validation trajectories")
    
    all_results = []
    successful_count = 0
    
    for i, idx in enumerate(tqdm(val_indices)):
        trajectory = trajectories[idx]
        result = evaluate_forecasting(
            model=model,
            tokenizer=tokenizer,
            trajectory=trajectory,
            input_steps=config["input_steps"],
            forecast_steps=config["forecast_steps"],
            alpha=config["alpha"],
            precision=config["precision"]
        )
        
        result["trajectory_idx"] = int(idx)
        all_results.append(result)
        
        if result["success"]:
            successful_count += 1
            # Visualize a few examples
            if i < 5:  # Visualize first 5 successful predictions
                save_path = FIGURES_DIR / f"trajectory_{idx}_prediction.png"
                visualize_predictions(result, idx, file_path=file_path, save_path=save_path)
    
    # 7. Log overall results
    logger.info(f"Successfully generated predictions for {successful_count}/{val_size} trajectories")
    
    if successful_count > 0:
        # Extract metrics from successful runs
        successful_results = [r for r in all_results if r["success"]]
        
        # 8. Save results and create visualizations
        save_results(all_results, successful_results, config, val_size)
    else:
        logger.error("No successful predictions. Cannot calculate metrics.")
    
    logger.info("Evaluation completed")


if __name__ == "__main__":
    main()
