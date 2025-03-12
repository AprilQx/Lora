"""
Visualization utilities for time series forecasting.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import logging

# Configure logging
logger = logging.getLogger(__name__)


def visualize_predictions(results, trajectory_idx, file_path="data/lotka_volterra_data.h5", save_path=None):
    """
    Create a visualization of model predictions vs ground truth.
    
    Args:
        results: Dictionary containing evaluation results
        trajectory_idx: Index of the trajectory being visualized
        file_path: Path to the HDF5 file with the original data
        save_path: Path to save the figure (if None, just displays it)
    """
    if not results["success"]:
        logger.error(f"Cannot visualize unsuccessful prediction: {results['error']}")
        return
    
    ground_truth = np.array(results["ground_truth"])
    predictions = np.array(results["predictions"])
    
    # Number of steps to show before the prediction point (for context)
    context_steps = 10
    
    try:
        # Load the original trajectory to get context
        with h5py.File(file_path, "r") as f:
            full_trajectory = f["trajectories"][trajectory_idx]
        
        # Get the input portion (steps 50-context_steps to 50)
        input_context = full_trajectory[50-context_steps:50]
        
        # Combine context, ground truth and prediction for visualization
        context_time_points = np.arange(-context_steps, 0)
        forecast_time_points = np.arange(len(ground_truth))
        
        # Create figure with two subplots (one for prey, one for predator)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot prey population
        ax1.plot(context_time_points, input_context[:, 0], 'b-', label='Input (Prey)')
        ax1.plot(forecast_time_points, ground_truth[:, 0], 'g-', label='Ground Truth (Prey)')
        ax1.plot(forecast_time_points, predictions[:, 0], 'r--', label='Prediction (Prey)')
        ax1.set_ylabel('Prey Population')
        ax1.set_title(f'Trajectory {trajectory_idx} - Prey Population')
        ax1.legend()
        ax1.grid(True)
        
        # Plot predator population
        ax2.plot(context_time_points, input_context[:, 1], 'b-', label='Input (Predator)')
        ax2.plot(forecast_time_points, ground_truth[:, 1], 'g-', label='Ground Truth (Predator)')
        ax2.plot(forecast_time_points, predictions[:, 1], 'r--', label='Prediction (Predator)')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Predator Population')
        ax2.set_title(f'Trajectory {trajectory_idx} - Predator Population')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close(fig)
    except Exception as e:
        logger.error(f"Error in visualization: {str(e)}")