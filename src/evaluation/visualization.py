"""
Visualization utilities for time series forecasting.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import logging
import pandas as pd
import seaborn as sns
from src.data.preprocessor import text_to_numeric


# Configure logging
logger = logging.getLogger(__name__)

def load_trajectory_from_text(file_path, trajectory_idx, context_steps=50):
    """
    Load a specific trajectory from the text file.
    
    Args:
        file_path: Path to the text file with trajectories
        trajectory_idx: Index of the trajectory to load
        context_steps: Number of context steps to include
        
    Returns:
        NumPy array of the trajectory data
    """
    try:
        # Load the text file
        with open(file_path, 'r') as f:
            # Assuming each line is a separate trajectory
            trajectories = f.readlines()
        
        if trajectory_idx >= len(trajectories):
            logger.error(f"Trajectory index {trajectory_idx} out of range (max: {len(trajectories)-1})")
            return np.zeros((context_steps, 2))
        
        # Get the specified trajectory
        trajectory_text = trajectories[trajectory_idx].strip()
        
        # Convert text to numeric
        numeric_data = text_to_numeric(trajectory_text)
        
        return numeric_data
    except Exception as e:
        logger.error(f"Error loading trajectory from text: {str(e)}")
        return np.zeros((context_steps, 2))

def plot_trajectory_prediction(result, trajectory_idx, save_path=None, config=None, 
                               text_file_path="../data/processed/test_texts.txt"):
    """
    Create a visualization of model predictions vs ground truth.
    
    Args:
        result: Dictionary containing evaluation results
        trajectory_idx: Index of the trajectory being visualized
        save_path: Path to save the figure (if None, just displays it)
        config: Configuration parameters
        file_path: Path to the HDF5 file with the original data
    """
    if not result.get("success", False):
        logger.error(f"Cannot visualize unsuccessful prediction: {result.get('error', 'Unknown error')}")
        return
    
    try:
        ground_truth = np.array(result["ground_truth"])
        predictions = np.array(result["predictions"])
        
        # Number of steps to show before the prediction point (for context)
        context_steps = 50
        
        try:
            # Load the original trajectory to get context
            full_trajectory = load_trajectory_from_text(text_file_path, trajectory_idx)
            
            
            # Get the input portion (steps input_steps-context_steps to input_steps)
            input_steps = config.get("input_steps", 50) if config else 50
            input_context = full_trajectory[input_steps-context_steps:input_steps]
            
            # Combine context, ground truth and prediction for visualization
            context_time_points = np.arange(-context_steps, 0)
            forecast_time_points_gt = np.arange(len(ground_truth))
            forecast_time_points_pred = np.arange(len(predictions))
            forecast_time_points=min(len(forecast_time_points_gt),len(forecast_time_points_pred))
            
        except Exception as e:
            logger.warning(f"Could not load context from original data: {str(e)}")
            # Use synthetic context if original data is not available
            # input_context = np.zeros((context_steps, 2))
            # context_time_points = np.arange(-context_steps, 0)
            # forecast_time_points = np.arange(len(ground_truth))
        
        # Create figure with two subplots (one for prey, one for predator)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot prey population
        if len(input_context) > 0:
            ax1.plot(context_time_points, input_context[:, 0], 'b-', label='Input (Prey)')
        ax1.plot(forecast_time_points_gt[:forecast_time_points], ground_truth[:forecast_time_points, 0], 'g-', label='Ground Truth (Prey)')
        ax1.plot(forecast_time_points_pred[:forecast_time_points], predictions[:forecast_time_points, 0], 'r--', label='Prediction (Prey)')
        ax1.set_ylabel('Prey Population')
        ax1.set_title(f'Trajectory {trajectory_idx} - Prey Population')
        ax1.legend()
        ax1.grid(True)
        
        # Plot predator population
        if len(input_context) > 0:
            ax2.plot(context_time_points, input_context[:, 1], 'b-', label='Input (Predator)')
        ax2.plot(forecast_time_points_gt[:forecast_time_points], ground_truth[:forecast_time_points, 1], 'g-', label='Ground Truth (Predator)')
        ax2.plot(forecast_time_points_pred[:forecast_time_points], predictions[:forecast_time_points, 1], 'r--', label='Prediction (Predator)')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Predator Population')
        ax2.set_title(f'Trajectory {trajectory_idx} - Predator Population')
        ax2.legend()
        ax2.grid(True)
        
        # Add metrics as text annotation
        metrics_text = (
            f"MSE: {result.get('mse', 'N/A'):.4f}\n"
            f"MAE: {result.get('mae', 'N/A'):.4f}\n"
            f"Prey MSE: {result.get('prey_mse', 'N/A'):.4f}\n"
            f"Prey MAE: {result.get('prey_mae', 'N/A'):.4f}\n"
            f"Predator MSE: {result.get('predator_mse', 'N/A'):.4f}\n"
            f"Predator MAE: {result.get('predator_mae', 'N/A'):.4f}"
        )
        
        plt.figtext(0.85, 0.5, metrics_text, fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close(fig)
    except Exception as e:
        logger.error(f"Error in visualization: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

def plot_distribution_of_metrics(successful_results, save_path=None):
    """
    Create a distribution plot of evaluation metrics.
    
    Args:
        successful_results: List of successful evaluation results
        save_path: Path to save the figure
    """
    try:
        # Create a DataFrame with metrics
        metrics_df = pd.DataFrame([{
            "MSE": r.get("mse", np.nan),
            "MAE": r.get("mae", np.nan),
            "Prey MSE": r.get("prey_mse", np.nan),
            "Prey MAE": r.get("prey_mae", np.nan),
            "Predator MSE": r.get("predator_mse", np.nan),
            "Predator MAE": r.get("predator_mae", np.nan)
        } for r in successful_results])
        
        # Create histogram of errors
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        metrics_to_plot = ["MSE", "MAE", "Prey MSE", "Prey MAE", "Predator MSE", "Predator MAE"]
        
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i // 3, i % 3]
            sns.histplot(metrics_df[metric].dropna(), kde=True, ax=ax)
            ax.set_title(f'Distribution of {metric}')
            ax.set_xlabel(metric)
            ax.set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Metric distribution saved to {save_path}")
        else:
            plt.show()
            
        plt.close(fig)
        
    except Exception as e:
        logger.error(f"Error creating distribution plot: {str(e)}")
