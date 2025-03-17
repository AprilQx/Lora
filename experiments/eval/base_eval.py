"""
Evaluation of the untrained Qwen2.5-Instruct model's forecasting ability 
on the Lotka-Volterra dataset with FLOP tracking.
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
from src.models.qwen import load_qwen
from src.data.preprocessor import load_data
from src.evaluation.evaluation import evaluate_forecasting
from src.evaluation.visualization import visualize_predictions
from utils.saving import setup_device, save_results
from utils.flop_tracker import FLOPTracker

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
    Main function to evaluate the untrained model's forecasting ability with FLOP tracking.
    """
    print("Entering main function")
    logger.info("Starting evaluation of untrained Qwen2.5 model with FLOP tracking")
    
    # Initialize FLOP tracker
    logger.info("Initializing FLOP tracker")
    tracker = FLOPTracker(
        hidden_size=896,
        num_attention_heads=14,
        num_hidden_layers=24,
        intermediate_size=4864,
        head_dim=64,
        vocab_size=151936,
        max_budget=1e17,
        log_path="flop_logs",
        experiment_name="baseline_evaluation"
    )
    
    # 1. Load the model and tokenizer
    logger.info("Loading model and tokenizer")
    model, tokenizer = load_qwen()

    # 2. Setup device
    device, model = setup_device(model)
    
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
        "forecast_steps": 50,  # Forecast the next 50 steps (target)
    }
    
    # 6. Evaluate on validation set and track FLOPs
    logger.info(f"Evaluating model on {val_size} validation trajectories")
    
    all_results = []
    successful_count = 0
    total_inference_flops = 0
    
    for i, idx in enumerate(tqdm(val_indices)):
        trajectory = trajectories[idx]
        
        # First evaluate to get the actual output length
        result = evaluate_forecasting(
            model=model,
            tokenizer=tokenizer,
            trajectory=trajectory,
            input_steps=config["input_steps"],
            forecast_steps=config["forecast_steps"],
            alpha=config["alpha"],
            precision=config["precision"]
        )
        
        # Determine actual generation length
        actual_gen_length = 0
        if result["success"] and "generated_token_length" in result:
            actual_gen_length = result["generated_token_length"]
        else:
            # If unsuccessful, use target length as fallback
            actual_gen_length = config["forecast_steps"]
        
        # Log FLOPs for this inference step using actual output length
        inference_flops = tracker.log_inference(
            context_len=result["input_token_length"],
            gen_len=actual_gen_length,
            batch_size=1,
            description=f"Inference on trajectory {idx} (generated {actual_gen_length} steps)",
            use_sliding_window=False
        )
        total_inference_flops += inference_flops
        
        result["trajectory_idx"] = int(idx)
        result["flops_used"] = float(inference_flops)
        result["actual_gen_length"] = actual_gen_length
        all_results.append(result)
        
        if result["success"]:
            successful_count += 1
            # Visualize a few examples
            if i < 5:  # Visualize first 5 successful predictions
                save_path = FIGURES_DIR / f"trajectory_{idx}_prediction.png"
                visualize_predictions(result, idx, file_path=file_path, save_path=save_path)
    
    # 7. Log overall results
    logger.info(f"Successfully generated predictions for {successful_count}/{val_size} trajectories")
    logger.info(f"Total FLOPs used: {tracker.total_flops:.2e}")
    logger.info(f"FLOPs budget remaining: {tracker.max_budget - tracker.total_flops:.2e}")
    
    if successful_count > 0:
        # Extract metrics from successful runs
        successful_results = [r for r in all_results if r["success"]]
        
        # Calculate average generation length
        avg_gen_length = sum(r.get("actual_gen_length", 0) for r in successful_results) / len(successful_results)
        logger.info(f"Average generation length: {avg_gen_length:.2f} steps")
        
        # 8. Save results and create visualizations
        save_results(all_results, successful_results, config, val_size)
        
        # 9. Generate FLOP usage report and plot
        report = tracker.generate_report(RESULTS_DIR / "flop_report.json")
        tracker.plot_usage(RESULTS_DIR / "flop_usage.png")
        
        # Print summary of FLOP usage
        logger.info("FLOP Usage Summary:")
        logger.info(f"  Total FLOPs: {report['total_flops']:.2e}")
        logger.info(f"  Budget used: {report['budget_used_percent']:.2f}%")
        logger.info(f"  Inference FLOPs: {report['inference_flops']:.2e}")
        logger.info(f"  Average FLOPs per trajectory: {total_inference_flops/val_size:.2e}")
    else:
        logger.error("No successful predictions. Cannot calculate metrics.")
    
    logger.info("Evaluation completed")


if __name__ == "__main__":
    main()