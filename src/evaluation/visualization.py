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
                               text_file_path="../../data/processed2/test_texts.txt"):
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

def create_metrics_dataframe(results_data):
    """Create a DataFrame from the results data"""
    if not results_data:
        logger.error("No results data provided")
        return None
        
    # Extract successful results
    successful_results = [r for r in results_data["results"] if r.get("success", False)]
    if not successful_results:
        logger.error("No successful results found")
        return None
        
    # Create DataFrame
    metrics_df = pd.DataFrame([{
        "MSE": r.get("mse", np.nan),
        "MAE": r.get("mae", np.nan),
        "Prey MSE": r.get("prey_mse", np.nan),
        "Prey MAE": r.get("prey_mae", np.nan),
        "Predator MSE": r.get("predator_mse", np.nan),
        "Predator MAE": r.get("predator_mae", np.nan),
        "Trajectory": r.get("trajectory_idx", np.nan),
        "Predicted Steps": r.get("predicted_steps", np.nan)
    } for r in successful_results])
    
    return metrics_df

def plot_error_distributions_log_scale(metrics_df, save_path=None):
    """Create distribution plots with appropriate log scaling"""
    if metrics_df is None or metrics_df.empty:
        return
        
    # Create histogram of errors
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    metrics_to_plot = ["MSE", "MAE", "Prey MSE", "Prey MAE", "Predator MSE", "Predator MAE"]
    
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i // 3, i % 3]
        
        # Use log scale for MSE metrics which have larger ranges
        if "MSE" in metric:
            # Add small constant to handle zeros
            log_values = np.log10(metrics_df[metric] + 1e-10)
            sns.histplot(log_values, kde=True, ax=ax)
            ax.set_title(f'Distribution of {metric} (log10 scale)')
            ax.set_xlabel(f'log10({metric})')
            
            # Create custom ticks for easier interpretation
            power_ticks = [-2, 0, 2, 4, 6]  # log10 scale
            ax.set_xticks(power_ticks)
            ax.set_xticklabels([f"10^{p}" for p in power_ticks])
        else:
            # For MAE, use regular scale with outlier clipping
            q95 = metrics_df[metric].quantile(0.95)
            clipped_values = metrics_df[metric].clip(upper=q95)
            sns.histplot(clipped_values, kde=True, ax=ax)
            ax.set_title(f'Distribution of {metric} (clipped at 95th percentile)')
            ax.set_xlabel(metric)
        
        ax.set_ylabel('Frequency')
    
    plt.suptitle("Error Distribution for Lotka-Volterra Predictions", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Log-scale error distribution saved to {save_path}")
    else:
        plt.show()
    
    plt.close(fig)

def plot_error_comparison_log_scale(metrics_df, save_path=None):
    """Create a comparison plot with log scaling"""
    if metrics_df is None or metrics_df.empty:
        return
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # MSE comparison - log scale
    ax1.scatter(np.log10(metrics_df["Prey MSE"] + 1e-10), 
               np.log10(metrics_df["Predator MSE"] + 1e-10), alpha=0.7)
    ax1.set_xlabel("Prey MSE (log10)")
    ax1.set_ylabel("Predator MSE (log10)")
    ax1.set_title("Prey vs Predator MSE (log scale)")
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Add diagonal line for reference
    min_val = min(np.log10(metrics_df["Prey MSE"].min() + 1e-10), 
                 np.log10(metrics_df["Predator MSE"].min() + 1e-10))
    max_val = max(np.log10(metrics_df["Prey MSE"].max() + 1e-10), 
                 np.log10(metrics_df["Predator MSE"].max() + 1e-10))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
    
    # MAE comparison - clipped scale
    prey_mae_clip = metrics_df["Prey MAE"].clip(upper=metrics_df["Prey MAE"].quantile(0.95))
    predator_mae_clip = metrics_df["Predator MAE"].clip(upper=metrics_df["Predator MAE"].quantile(0.95))
    
    ax2.scatter(prey_mae_clip, predator_mae_clip, alpha=0.7)
    ax2.set_xlabel("Prey MAE (clipped at 95th percentile)")
    ax2.set_ylabel("Predator MAE (clipped at 95th percentile)")
    ax2.set_title("Prey vs Predator MAE")
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Add diagonal line for reference
    max_val = max(prey_mae_clip.max(), predator_mae_clip.max())
    ax2.plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
    
    plt.suptitle("Comparison of Prey vs Predator Prediction Errors", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Log-scale error comparison saved to {save_path}")
    else:
        plt.show()
    
    plt.close(fig)

def plot_error_boxplots(metrics_df, save_path=None):
    """Create boxplots for error distributions to better handle outliers"""
    if metrics_df is None or metrics_df.empty:
        return
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Prepare data in long format for boxplots
    mse_data = pd.DataFrame({
        'Population': ['Prey'] * len(metrics_df) + ['Predator'] * len(metrics_df),
        'MSE': list(np.log10(metrics_df['Prey MSE'] + 1e-10)) + list(np.log10(metrics_df['Predator MSE'] + 1e-10))
    })
    
    mae_data = pd.DataFrame({
        'Population': ['Prey'] * len(metrics_df) + ['Predator'] * len(metrics_df),
        'MAE': list(metrics_df['Prey MAE']) + list(metrics_df['Predator MAE'])
    })
    
    # Plot MSE boxplot with log scale
    sns.boxplot(x='Population', y='MSE', data=mse_data, ax=ax1)
    ax1.set_title('MSE Distribution (log10 scale)')
    ax1.set_ylabel('log10(MSE)')
    
    # Plot MAE boxplot
    sns.boxplot(x='Population', y='MAE', data=mae_data, ax=ax2)
    ax2.set_title('MAE Distribution')
    ax2.set_ylabel('MAE')
    
    # Set ylim for MAE to exclude extreme outliers
    q95 = mae_data['MAE'].quantile(0.95)
    ax2.set_ylim(0, q95 * 1.1)  # Give a little extra space
    
    plt.suptitle("Error Distributions by Population Type", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Error boxplots saved to {save_path}")
    else:
        plt.show()
    
    plt.close(fig)

def plot_trajectory_errors(metrics_df, save_path=None):
    """Plot errors by trajectory to identify problematic cases"""
    if metrics_df is None or metrics_df.empty:
        return
    
    # Sort by total MSE for better visualization
    metrics_df = metrics_df.sort_values('MSE')
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot bar chart of MSE for each trajectory (log scale)
    x = np.arange(len(metrics_df))
    width = 0.35
    
    log_prey_mse = np.log10(metrics_df['Prey MSE'] + 1e-10)
    log_predator_mse = np.log10(metrics_df['Predator MSE'] + 1e-10)
    
    ax.bar(x - width/2, log_prey_mse, width, label='Prey MSE')
    ax.bar(x + width/2, log_predator_mse, width, label='Predator MSE')
    
    # Add trajectory numbers to x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df['Trajectory'], rotation=90)
    
    ax.set_xlabel('Trajectory Index')
    ax.set_ylabel('log10(MSE)')
    ax.set_title('MSE by Trajectory (log10 scale)')
    ax.legend()
    
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Trajectory errors saved to {save_path}")
    else:
        plt.show()
    
    plt.close(fig)