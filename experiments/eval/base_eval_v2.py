"""
Evaluation of the untrained Qwen2.5-Instruct model's forecasting ability 
on the Lotka-Volterra dataset using precision 3.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import logging
import json
from datetime import datetime
from pathlib import Path
import sys
from tqdm import tqdm

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import custom modules
from src.models.qwen import load_qwen
from src.data.preprocessor import load_data, numeric_to_text, text_to_numeric
from utils.saving import setup_device, save_results
from utils.flop_tracker import FLOPTracker

from src.evaluation.evaluation_v2 import evaluate_model_on_dataset, evaluate_forecasting,calculate_summary_metrics

from src.evaluation.visualization import plot_trajectory_prediction, create_metrics_dataframe, plot_error_distributions_log_scale,plot_error_comparison_log_scale, plot_error_boxplots,plot_trajectory_errors


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create results directory
RESULTS_DIR = Path(project_root) / "results"  # Use the project_root variable
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR = RESULTS_DIR / "base_figures_precision3"
FIGURES_DIR.mkdir(exist_ok=True)
Data_DIR=Path(project_root)/"data"/"processed3"/"test_texts.txt"

# Set random seed for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def main(args):
    """
    Main function to evaluate the untrained model's forecasting ability.
    """
    logger.info("Starting evaluation of untrained Qwen2.5 model")
    
    # Initialize FLOP tracker
    # logger.info("Initializing FLOP tracker")
    # tracker = FLOPTracker(
    #     hidden_size=896,
    #     num_attention_heads=14,
    #     num_hidden_layers=24,
    #     intermediate_size=4864,
    #     head_dim=64,
    #     vocab_size=151936,
    #     max_budget=1e17,
    #     log_path=RESULTS_DIR/'flop_logs',
    #     experiment_name="baseline_evaluation"
    # )
    
    # 1. Load the model and tokenizer
    logger.info("Loading model and tokenizer")
    model, tokenizer = load_qwen()

    # 2. Setup device
    device, model = setup_device(model)
    
    # Configuration parameters
    config = {
        "alpha": args.alpha,
        "precision": args.precision,
        "input_steps": args.input_steps,
        "forecast_steps": args.forecast_steps,
        "file_path": args.file_path
    }
    
    # Depending on evaluation mode, either use text files or HDF5 data
    if args.use_text_files:
        logger.info(f"Evaluating using preprocessed text files from {args.text_file_path}")
        all_results, successful_results = evaluate_model_on_dataset(
            model=model,
            tokenizer=tokenizer,
            text_file_path=args.text_file_path,
            num_samples=args.num_samples,
            config=config,
            visualize_first_n=args.visualize_first_n
        )
    # else:
        # # Load the Lotka-Volterra dataset
        # logger.info(f"Loading Lotka-Volterra dataset from {args.file_path}")
        # trajectories, time_points = load_data(args.file_path)
        
        # # Create validation set
        # logger.info(f"Creating evaluation set with {args.num_samples} trajectories")
        # indices = np.random.permutation(trajectories.shape[0])
        # eval_indices = indices[:args.num_samples]
        # eval_trajectories = trajectories[eval_indices]
        
        # # Evaluate on the set
        # all_results, successful_results, total_flops = evaluate_model_on_dataset(
        #     model=model,
        #     tokenizer=tokenizer,
        #     trajectories=eval_trajectories,
        #     indices=eval_indices,
        #     config=config,
        #     tracker=tracker,
        #     visualize_first_n=args.visualize_first_n
        # )

    # Save results
    if successful_results:
        # Save full results
        save_results(all_results, successful_results, config, len(all_results))

        # Create visualization of metric distributions
        metrics_df = create_metrics_dataframe({"results": successful_results})
        plot_error_distributions_log_scale(
            metrics_df,
            save_path=FIGURES_DIR / "error_distributions.png"
        )
        plot_error_comparison_log_scale(
            metrics_df,
            save_path=FIGURES_DIR / "prey_vs_predator_errors.png"
        )
        plot_error_boxplots(
            metrics_df,
            save_path=FIGURES_DIR / "error_boxplots.png"
        )
        plot_trajectory_errors(
            metrics_df,
            save_path=FIGURES_DIR / "trajectory_errors.png"
        )
        # Save full results
        save_results(all_results, successful_results, config, len(all_results))
        
        # Calculate and log summary metrics
        summary = calculate_summary_metrics(all_results)
        logger.info("Summary metrics:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
        
        # Generate FLOP usage report and plot
        # report = tracker.generate_report(RESULTS_DIR / "flop_report.json")
        # tracker.plot_usage(RESULTS_DIR / "flop_usage.png")
        
        # Save summary to file
        with open(RESULTS_DIR / "evaluation_summary.json", "w") as f:
            json.dump({
                "date": datetime.now().isoformat(),
                "config": config,
                "results": summary,
                # "flops": {
                #     "total": tracker.total_flops,
                #     "budget_percent": (tracker.total_flops / tracker.max_budget) * 100,
                #     "per_trajectory": total_flops / len(all_results) if len(all_results) > 0 else 0
                # }
            }, f, indent=2)
        
        # logger.info(f"FLOP usage: {tracker.total_flops:.2e} ({(tracker.total_flops / tracker.max_budget) * 100:.2f}% of budget)")
    else:
        logger.error("No successful predictions. Cannot calculate metrics.")
    
    logger.info("Evaluation completed")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5-Instruct on Lotka-Volterra data")
    
    # Data options
    parser.add_argument("--file_path", type=str, default="../data/lotka_volterra_data.h5",
                         help="Path to the HDF5 data file")
    parser.add_argument("--use_text_files", action="store_true",
                        help="Use preprocessed text files instead of HDF5 data")
    parser.add_argument("--text_file_path", type=str, default=Data_DIR,
                        help="Path to preprocessed text file (used with --use_text_files)")
    
    # Evaluation options
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Number of trajectories to evaluate")
    parser.add_argument("--input_steps", type=int, default=50,
                        help="Number of timesteps to use as input")
    parser.add_argument("--forecast_steps", type=int, default=50,
                        help="Number of timesteps to forecast")
    parser.add_argument("--alpha", type=float, default=10.0,
                        help="Scaling parameter for numeric values")
    parser.add_argument("--precision", type=int, default=3,
                        help="Decimal precision for text representation")
    parser.add_argument("--visualize_first_n", type=int, default=5,
                        help="Number of successful predictions to visualize")
    
    args = parser.parse_args()
    main(args)