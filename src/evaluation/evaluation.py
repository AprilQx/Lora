"""
Evaluation utilities for time series forecasting.
"""

import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
from utils.postprocessing import fix_numeric_text
from src.data.preprocessor import text_to_numeric,numeric_to_text

# Configure logging
logger = logging.getLogger(__name__)


def evaluate_forecasting(model, tokenizer, trajectory, input_steps=50, forecast_steps=50, 
                        alpha=10.0, precision=3, max_tokens=400):
    """
    Evaluate the model's forecasting ability on a single trajectory.
    
    Args:
        model: The untrained Qwen model
        tokenizer: The tokenizer for the model
        trajectory: Single trajectory from the dataset (time_steps, variables)
        input_steps: Number of steps to use as input
        forecast_steps: Number of steps to forecast
        alpha: Scaling parameter
        precision: Decimal precision for text representation
        max_tokens: Maximum tokens to generate
        
    Returns:
        dict: Dictionary containing evaluation metrics and predictions
    """
    try:
        # Convert trajectory to text
        full_text = numeric_to_text(trajectory, alpha=alpha, precision=precision)
        
        # Split into timesteps
        timesteps = full_text.split(";")
        
        # Take only the first input_steps for input
        input_text = ";".join(timesteps[:input_steps])
        
        # Generate forecasting prompt
        system_message = "You are a helpful assistant that performs time series predictions. The user will provide a sequence and you will predict the remaining sequence. The sequence is represented by decimal strings separated by commas."
        user_prompt = f"Please continue the following sequence without producing any additional text. Do not say anything like 'the next terms in the sequence are', just return the next 50 timestep numbers. Sequence:\n{input_text};"
        
        # For Qwen models, we need to combine these into a single prompt
        prompt = f"{system_message}\n\n{user_prompt}"

        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Get the device from the model
        device = model.device
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
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
                do_sample=True, #whether select the top_p samples in sampling
                renormalize_logits=True #this is done to renormlise the logits to avoid the model to generate the same token again and again
            )
        
        # Make sure output is moved to CPU before further processing
        output = output.cpu()
        
        # Decode the output
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        # Extract the forecasted part (everything after the input)
        input_marker = input_text + ";"
        start_idx = generated_text.find(input_marker) + len(input_marker)
        forecasted_text = generated_text[start_idx:].strip()

        # Clean up the forecasted text
        forecasted_text = fix_numeric_text(forecasted_text)
        
        # Check if we have valid forecasted text
        if not forecasted_text or not any(c.isdigit() for c in forecasted_text):
            logger.warning("No valid numeric data in model output")
            return {
                "success": False,
                "error": "No valid numeric predictions generated"
            }
        
        # Add the input text back to get the full sequence
        full_generated = input_text + ";" + forecasted_text
        
        # Convert back to numeric
        forecasted_numeric = text_to_numeric(full_generated)
        
        # Extract only the forecasted part (ensuring we don't exceed array bounds)
        predicted_length = min(forecast_steps, max(0, len(forecasted_numeric) - input_steps))
        forecasted_numeric = forecasted_numeric[input_steps:input_steps+predicted_length]
        
        # Get the ground truth for the same period
        ground_truth = trajectory[input_steps:input_steps+predicted_length]
        
        # If we couldn't generate enough steps
        if predicted_length == 0:
            logger.warning("Could not generate any valid predictions")
            return {
                "success": False,
                "error": "No valid predictions generated"
            }
        
        # Calculate metrics
        metrics = calculate_metrics(ground_truth, forecasted_numeric)
        
        return {
            "success": True,
            "predicted_steps": predicted_length,
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
            "error": str(e)
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
    # Calculate overall metrics
    mse = mean_squared_error(ground_truth.flatten(), predictions.flatten())
    mae = mean_absolute_error(ground_truth.flatten(), predictions.flatten())
    
    # Calculate separate metrics for prey and predator populations
    prey_mse = mean_squared_error(ground_truth[:, 0], predictions[:, 0])
    prey_mae = mean_absolute_error(ground_truth[:, 0], predictions[:, 0])
    predator_mse = mean_squared_error(ground_truth[:, 1], predictions[:, 1])
    predator_mae = mean_absolute_error(ground_truth[:, 1], predictions[:, 1])
    
    return {
        "mse": float(mse),
        "mae": float(mae),
        "prey_mse": float(prey_mse),
        "prey_mae": float(prey_mae),
        "predator_mse": float(predator_mse),
        "predator_mae": float(predator_mae)
    }