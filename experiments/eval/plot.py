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

FIGURES_DIR=project_root/"results/base_figures_precision3"
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

#read the json file
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data
successful_results=read_json_file(project_root/"results/base_figures_precision3/untrained_model_results_20250404_141713.json")
metrics_df = create_metrics_dataframe(successful_results)
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
    

# Calculate and log summary metrics
summary = calculate_summary_metrics(successful_results["results"])
print("Summary of successful results:")
print(summary)