"""
Evaluation utilities for time series forecasting.
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
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import custom modules
from src.models.qwen import load_qwen
from src.data.preprocessor import load_data, numeric_to_text, text_to_numeric
from utils.saving import setup_device, save_results
from utils.flop_tracker import FLOPTracker
from src.evaluation.visualization import plot_trajectory_prediction, plot_distribution_of_metrics
from utils.postprocessing import fix_numeric_text

project_root = Path(__file__).parent.parent.parent

RESULTS_DIR = Path(project_root) / "results"  # Use the project_root variable
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Set random seed for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def evaluate_forecasting(model, tokenizer, trajectory=None, input_text=None, ground_truth=None,
                        input_steps=50, forecast_steps=50, 
                        alpha=10.0, precision=3, max_tokens=600):
    """
    Evaluate the model's forecasting ability on a time series.
    
    Args:
        model: The Qwen model
        tokenizer: The tokenizer for the model
        trajectory: Single trajectory from the dataset (time_steps, variables) (optional)
        input_text: Preprocessed input text (optional, used instead of trajectory if provided)
        ground_truth: Ground truth values for evaluation (optional, required if input_text is provided)
        input_steps: Number of steps to use as input (only used if trajectory is provided)
        forecast_steps: Number of steps to forecast
        alpha: Scaling parameter
        precision: Decimal precision for text representation
        max_tokens: Maximum tokens to generate
        
    Returns:
        dict: Dictionary containing evaluation metrics and predictions
    """
    try:
        input_token_length = None  # Initialize for error cases
        
        # Prepare input text from trajectory if not provided directly
        if input_text is None and trajectory is not None:
            # Convert trajectory to text
            full_text = numeric_to_text(trajectory, alpha=alpha, precision=precision)
            
            # Split into timesteps
            timesteps = full_text.split(";")
            
            # Take only the first input_steps for input
            input_text = ";".join(timesteps[:input_steps])
            
            # If ground truth not provided, extract it from trajectory
            if ground_truth is None:
                ground_truth = trajectory[input_steps:input_steps+forecast_steps]
        
        # Validation
        if input_text is None:
            raise ValueError("Either input_text or trajectory must be provided")
            
        if ground_truth is None:
            raise ValueError("Ground truth is required for evaluation (either via trajectory or ground_truth parameter)")
        
        # Ensure input text ends with a semicolon
        if not input_text.endswith(';'):
            input_text += ';'
        
        # Generate forecasting prompt
        # system_message = "You are a helpful assistant that performs time series predictions. The user will provide a sequence and you will predict the remaining sequence. The sequence is represented by decimal strings separated by commas."
        # user_prompt = f"Please continue the following sequence without producing any additional text. Do not say anything like 'the next terms in the sequence are', just return the next {forecast_steps} timestep numbers. Sequence:\n{input_text}"
        
        # For Qwen models, we need to combine these into a single prompt
        # prompt = f"{system_message}\n\n{user_prompt}"

        # Tokenize prompt
        inputs = tokenizer(input_text, return_tensors="pt")
        
        # Get the device from the model
        device = model.device
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # prompt_length = inputs["input_ids"].shape[1]
        input_token_length = inputs["input_ids"].shape[1]
        
        # Generate prediction
        with torch.no_grad():
            # Set pad_token_id to eos_token_id if pad_token_id is None (for MPS compatibility)
            pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
            
            output = model.generate(
                inputs["input_ids"],
                max_new_tokens=max_tokens,
                temperature=0.9,
                top_p=0.9,
                pad_token_id=pad_token_id,
                do_sample=True,  # whether select the top_p samples in sampling
                renormalize_logits=True  # renormalize logits to avoid repetition
            )
        
        # Make sure output is moved to CPU before further processing
        output = output.cpu()
        
        # Decode the output
        generated_token_length = output.shape[1] -input_token_length #- prompt_length
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        # Extract the forecasted part (everything after the input)
        start_idx = generated_text.find(input_text) + len(input_text)
        forecasted_text = generated_text[start_idx:].strip()

        # Clean up the forecasted text
        forecasted_text = fix_numeric_text(forecasted_text)
        
        # Check if we have valid forecasted text
        if not forecasted_text or not any(c.isdigit() for c in forecasted_text):
            logger.warning("No valid numeric data in model output")
            return {
                "success": False,
                "input_token_length": input_token_length,
                "error": "No valid numeric predictions generated",
                "generated_token_length": generated_token_length,
                "predicted_timesteps": 0
            }
        
        # Convert forecasted text to numeric
        forecasted_numeric = text_to_numeric(forecasted_text)
        
        # If we couldn't generate any valid numeric data
        if len(forecasted_numeric) == 0:
            logger.warning("Could not convert forecasted text to valid numeric data")
            return {
                "success": False,
                "input_token_length": input_token_length,
                "error": "Failed to convert forecasted text to numeric data",
                "generated_token_length": generated_token_length,
                "predicted_timesteps": 0
            }
        
        # Calculate metrics - use only as many steps as we have in both ground truth and predictions
        predicted_length = len(forecasted_numeric)
        useful_length=min(len(forecasted_numeric),len(ground_truth))
        
        # If we couldn't generate enough steps
        if predicted_length == 0:
            logger.warning("No matching timesteps between predictions and ground truth")
            return {
                "success": False,
                "input_token_length": input_token_length,
                "error": "No matching timesteps between predictions and ground truth",
                "generated_token_length": generated_token_length,
                "predicted_timesteps": 0
            }
        
        # Ensure consistent dimensions for comparison
        comparison_ground_truth = ground_truth[:useful_length]
        comparison_predictions = forecasted_numeric[:useful_length]
        
        # Calculate metrics
        metrics = calculate_metrics(comparison_ground_truth, comparison_predictions)
        
        return {
            "success": True,
            "input_token_length": input_token_length,
            "predicted_steps": predicted_length,
            "generated_token_length": generated_token_length,
            **metrics,
            "ground_truth": ground_truth.tolist(),
            "predictions": forecasted_numeric.tolist()
        }
    
    except Exception as e:
        logger.error(f"Error in forecasting: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "input_token_length": input_token_length,
            "error": str(e),
            "generated_token_length": 0,
            "predicted_timesteps": 0
        }


def calculate_metrics(ground_truth, predictions):
    """
    Calculate various error metrics between ground truth and predictions.
    
    Args:
        ground_truth: Ground truth values
        predictions: Predicted values
        
    Returns:
        Dictionary of calculated metrics
    """
    try:
        # Convert inputs to numpy arrays if they're not already
        if not isinstance(ground_truth, np.ndarray):
            ground_truth = np.array(ground_truth)
        if not isinstance(predictions, np.ndarray):
            predictions = np.array(predictions)
            
        # Ensure both arrays have the same shape
        min_length = min(len(ground_truth), len(predictions))
        
        if min_length == 0:
            logger.warning("Empty arrays for metric calculation")
            return {
                "mse": float('nan'),
                "mae": float('nan'),
                "prey_mse": float('nan'),
                "prey_mae": float('nan'),
                "predator_mse": float('nan'),
                "predator_mae": float('nan')
            }
            
        ground_truth = ground_truth[:min_length]
        predictions = predictions[:min_length]
        
        # Check if arrays are 2D (with prey and predator populations)
        if len(ground_truth.shape) > 1 and ground_truth.shape[1] >= 2:
            # Calculate overall metrics
            mse = mean_squared_error(ground_truth.flatten(), predictions.flatten())
            mae = mean_absolute_error(ground_truth.flatten(), predictions.flatten())
            
            # Calculate separate metrics for prey and predator populations
            prey_mse = mean_squared_error(ground_truth[:, 0], predictions[:, 0])
            prey_mae = mean_absolute_error(ground_truth[:, 0], predictions[:, 0])
            predator_mse = mean_squared_error(ground_truth[:, 1], predictions[:, 1])
            predator_mae = mean_absolute_error(ground_truth[:, 1], predictions[:, 1])
        else:
            # Single variable case
            mse = mean_squared_error(ground_truth, predictions)
            mae = mean_absolute_error(ground_truth, predictions)
            prey_mse = mse
            prey_mae = mae
            predator_mse = float('nan')
            predator_mae = float('nan')
        
        return {
            "mse": float(mse),
            "mae": float(mae),
            "prey_mse": float(prey_mse),
            "prey_mae": float(prey_mae),
            "predator_mse": float(predator_mse),
            "predator_mae": float(predator_mae)
        }
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        return {
            "mse": float('nan'),
            "mae": float('nan'),
            "prey_mse": float('nan'),
            "prey_mae": float('nan'),
            "predator_mse": float('nan'),
            "predator_mae": float('nan')
        }


def evaluate_model_on_dataset(model, tokenizer, trajectories=None, indices=None, 
                            text_file_path=None, num_samples=None, config=None, 
                            tracker=None, visualize_first_n=20): #only use text_file_path
    """
    Evaluate model on dataset - from trajectories or complete sequence text files.
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        trajectories: List of trajectories (optional)
        indices: Indices of trajectories (optional)
        text_file_path: Path to preprocessed text file (optional)
        num_samples: Number of samples to evaluate (for text files)
        config: Configuration parameters
        tracker: FLOP tracker
        visualize_first_n: Number of successful predictions to visualize
        
    Returns:
        Tuple of (all_results, successful_results, total_flops)
    """
    all_results = []
    successful_count = 0
    total_flops = 0
    
    # Copy config and add tracker
    eval_config = config.copy() if config else {}
    if tracker:
        eval_config["flop_tracker"] = tracker
    
    # Get important parameters from config
    input_steps = config.get("input_steps", 50) if config else 50
    forecast_steps = config.get("forecast_steps", 50) if config else 50
    alpha = config.get("alpha", 10.0) if config else 10.0
    precision = config.get("precision", 3) if config else 3
    
    # Evaluate from text files with complete sequences
    if text_file_path:
        # Load text data
        logger.info(f"Loading data from {text_file_path}")
        try:
            with open(text_file_path, 'r') as f:
                lines = f.readlines()
            
            # Process complete sequences
            sequences = []
            logger.info(f"Using first {input_steps} steps as input from complete sequences")
            
            for i, line in enumerate(lines):
                if i >= num_samples and num_samples is not None: #we just test on the first 50 samples from test
                    break
                
                # Get the full sequence
                full_sequence = line.strip()
                
                # Split by semicolon to get individual timesteps
                timesteps = full_sequence.split(';')
                
                # Skip if too short
                if len(timesteps) <= input_steps:
                    logger.warning(f"Sequence {i} is too short ({len(timesteps)} steps), skipping")
                    continue
                
                # Take only the first input_steps for input
                input_sequence = ';'.join(timesteps[:input_steps])
                
                # The rest is ground truth (up to forecast_steps)
                ground_truth = ';'.join(timesteps[input_steps:input_steps+forecast_steps])
                
                if input_sequence and ground_truth:
                    sequences.append({
                        "input_sequence": input_sequence,
                        "ground_truth": ground_truth,
                        "original_idx": i
                    })
            
            logger.info(f"Loaded {len(sequences)} sequences from text file")
            
        except Exception as e:
            logger.error(f"Error loading text file: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return [], [], 0
        
    # Evaluate from trajectories
    elif trajectories is not None:
        sequences = trajectories
        logger.info(f"Using {len(sequences)} trajectories for evaluation")
    else:
        logger.error("Either text_file_path or trajectories must be provided")
        return [], [], 0
    
    # Process each sequence
    logger.info("Starting evaluation...")
    for i, seq in enumerate(tqdm(sequences, desc="Evaluating")):
        if indices is not None:
            idx = indices[i]
        else:
            idx = i
            
        logger.info(f"Evaluating sequence {i+1}/{len(sequences)}")
        
        # Determine if we're working with preprocessed sequences or raw trajectories
        if isinstance(seq, dict) and "input_sequence" in seq and "ground_truth" in seq:
            # Using preprocessed sequences
            input_text = seq["input_sequence"]
            ground_truth = text_to_numeric(seq["ground_truth"])
            result = evaluate_forecasting(
                model=model,
                tokenizer=tokenizer,
                input_text=input_text,
                ground_truth=ground_truth,
                forecast_steps=config.get("forecast_steps", 50),
                alpha=config.get("alpha", 10.0),
                precision=config.get("precision", 3),
                max_tokens=config.get("max_tokens", 1000)
            )
            # Add metadata
            result["trajectory_idx"] = seq.get("original_idx", idx)
            
        else:
            # Using raw trajectories
            result = evaluate_forecasting(
                model=model,
                tokenizer=tokenizer,
                trajectory=seq,
                input_steps=config.get("input_steps", 50),
                forecast_steps=config.get("forecast_steps", 50),
                alpha=config.get("alpha", 10.0),
                precision=config.get("precision", 3),
                max_tokens=config.get("max_tokens", 1000)
            )
            result["trajectory_idx"] = idx
            
        # Track successful evaluations
        if result.get("success", False):
            successful_count += 1
            
            # Visualize some results
            if successful_count <= visualize_first_n:
                try:
                    plot_trajectory_prediction(
                        result, 
                        idx, 
                        save_path=FIGURES_DIR / f"trajectory_{idx}_prediction.png",
                        config=eval_config
                    )
                except Exception as e:
                    logger.error(f"Error plotting trajectory {idx}: {str(e)}")
            
        # Calculate FLOPs (if using flop tracker)
        if tracker and result.get("input_token_length") is not None:
            # Using the flop tracker to account for FLOPs
            flops = tracker.log_inference(
                context_len=result["input_token_length"],
                gen_len=result.get("generated_token_length", 0),
                batch_size=1,
                description=f"Inference on sequence {idx}"
            )
            result["flops"] = flops
            total_flops += flops
            
        all_results.append(result)
        
    logger.info(f"Evaluation complete: {successful_count}/{len(sequences)} successful")
    if tracker:
        logger.info(f"Total FLOPs used: {total_flops:.2e}")
    
    successful_results = [r for r in all_results if r.get("success", False)]    
    return all_results, successful_results, total_flops


def calculate_summary_metrics(results):
    """
    Calculate summary metrics from a list of evaluation results.
    
    Args:
        results: List of evaluation results
    
    Returns:
        Dictionary of summary metrics
    """
    successful_results = [r for r in results if r.get("success", False)]
    
    if len(successful_results) == 0:
        logger.warning("No successful evaluations for summary metrics")
        return {
            "mse": float('nan'),
            "mae": float('nan'),
            "prey_mse": float('nan'),
            "prey_mae": float('nan'),
            "predator_mse": float('nan'),
            "predator_mae": float('nan'),
            "successful_count": 0,
            "total_count": len(results),
            "total_flops": 0
        }
    
    # Calculate average metrics
    mse_values = [r.get("mse", float('nan')) for r in successful_results]
    mae_values = [r.get("mae", float('nan')) for r in successful_results]
    prey_mse_values = [r.get("prey_mse", float('nan')) for r in successful_results]
    prey_mae_values = [r.get("prey_mae", float('nan')) for r in successful_results]
    predator_mse_values = [r.get("predator_mse", float('nan')) for r in successful_results]
    predator_mae_values = [r.get("predator_mae", float('nan')) for r in successful_results]
    
    # Filter out NaN values
    mse_values = [v for v in mse_values if not np.isnan(v)]
    mae_values = [v for v in mae_values if not np.isnan(v)]
    prey_mse_values = [v for v in prey_mse_values if not np.isnan(v)]
    prey_mae_values = [v for v in prey_mae_values if not np.isnan(v)]
    predator_mse_values = [v for v in predator_mse_values if not np.isnan(v)]
    predator_mae_values = [v for v in predator_mae_values if not np.isnan(v)]
    
    avg_mse = np.mean(mse_values) if mse_values else float('nan')
    avg_mae = np.mean(mae_values) if mae_values else float('nan')
    avg_prey_mse = np.mean(prey_mse_values) if prey_mse_values else float('nan')
    avg_prey_mae = np.mean(prey_mae_values) if prey_mae_values else float('nan')
    avg_predator_mse = np.mean(predator_mse_values) if predator_mse_values else float('nan')
    avg_predator_mae = np.mean(predator_mae_values) if predator_mae_values else float('nan')
    
    return {
        "mse": float(avg_mse),
        "mae": float(avg_mae),
        "prey_mse": float(avg_prey_mse),
        "prey_mae": float(avg_prey_mae),
        "predator_mse": float(avg_predator_mse),
        "predator_mae": float(avg_predator_mae),
        "successful_count": len(successful_results),
        "total_count": len(results),
        "total_flops": sum(r.get("flops", 0) for r in results),
        "success_rate": len(successful_results) / len(results) if len(results) > 0 else 0
    }


