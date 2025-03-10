"""
Preprocessor for time series data based on the LLMTIME approach.
Handles loading, preprocessing, and converting between numerical arrays and text representations.
"""

import h5py
import numpy as np
from typing import Tuple, List, Union, Dict, Optional


def load_and_preprocess(
    file_path: str,
    val_split: float = 0.2,
    alpha: float = 0.99,
    precision: int = 3,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Load Lotka-Volterra data from HDF5 file and preprocess it into text format.
    
    Args:
        file_path: Path to the HDF5 file
        val_split: Fraction of data to use for validation
        alpha: Scaling parameter to normalize values
        precision: Number of decimal places to round to
        seed: Random seed for shuffling
        
    Returns:
        Tuple of (train_texts, val_texts) where each is a list of formatted text sequences
    """
    # Load the data
    trajectories, time_points = load_data(file_path)
    
    # Split the data into training and validation sets
    np.random.seed(seed)
    indices = np.random.permutation(trajectories.shape[0])
    split_idx = int(trajectories.shape[0] * (1 - val_split))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_trajectories = trajectories[train_indices]
    val_trajectories = trajectories[val_indices]
    
    # Preprocess the data
    train_texts = []
    for trajectory in train_trajectories:
        train_texts.append(numeric_to_text(trajectory, alpha=alpha, precision=precision))
    
    val_texts = []
    for trajectory in val_trajectories:
        val_texts.append(numeric_to_text(trajectory, alpha=alpha, precision=precision))
    
    return train_texts, val_texts


def load_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load data from HDF5 file.
    
    Args:
        file_path: Path to the HDF5 file
        
    Returns:
        Tuple of (trajectories, time_points)
    """
    with h5py.File(file_path, "r") as f:
        trajectories = f["trajectories"][:]  # Shape: (1000, 100, 2)
        time_points = f["time"][:]          # Shape: (100,)
    
    return trajectories, time_points


def numeric_to_text(
    trajectory: np.ndarray,
    alpha: float = 0.99,
    precision: int = 3,
) -> str:
    """
    Convert a numeric trajectory to text format following the LLMTIME scheme.
    
    Args:
        trajectory: NumPy array of shape (time_steps, variables)
        alpha: Scaling parameter (percentile used for scaling)
        precision: Number of decimal places to keep
    
    Returns:
        Formatted text string
    """
    # Scale the data
    scaled_trajectory = scale_data(trajectory, alpha)
    
    # Round to specified precision
    scaled_trajectory = np.round(scaled_trajectory, precision)
    
    # Convert to text format
    text_parts = []
    for timestep in scaled_trajectory:
        # Join variables at this timestep with commas
        timestep_str = ",".join([f"{value:.{precision}f}" for value in timestep])
        text_parts.append(timestep_str)
    
    # Join timesteps with semicolons
    return ";".join(text_parts)


def scale_data(
    data: np.ndarray,
    alpha: float = 0.99,
) -> np.ndarray:
    """
    Scale the data using a percentile-based approach.
    
    Args:
        data: NumPy array of data
        alpha: Percentile to use for scaling (0.99 means scale so 99th percentile equals 1)
    
    Returns:
        Scaled data
    """
    # Find the scaling factor based on the specified percentile
    percentile_value = np.percentile(data, alpha * 100)
    
    # Avoid division by zero
    if percentile_value == 0:
        percentile_value = 1.0
    
    # Scale the data
    return data / percentile_value


def text_to_numeric(
    text: str,
    alpha: float = 0.99,
    precision: int = 3,
) -> np.ndarray:
    """
    Convert text representation back to numeric array.
    
    Args:
        text: Text representation of time series
        alpha: Scaling parameter used during encoding
        precision: Precision used in encoding
    
    Returns:
        NumPy array of shape (time_steps, variables)
    """
    # Split into timesteps
    timesteps = text.split(";")
    
    # Initialize list to hold numeric values
    numeric_data = []
    
    # Process each timestep
    for timestep in timesteps:
        # Split variables at this timestep
        variables = timestep.split(",")
        
        # Convert to float
        try:
            variables_numeric = [float(v) for v in variables]
            numeric_data.append(variables_numeric)
        except ValueError:
            # Handle potential parsing errors (e.g., 'NaN' or incomplete values)
            continue
    
    # Convert to NumPy array
    return np.array(numeric_data)


def generate_forecast(
    model, 
    tokenizer, 
    input_text: str, 
    forecast_steps: int = 10, 
    alpha: float = 0.99,
    precision: int = 3,
    temperature: float = 0.8,
    max_length: int = 512,
) -> np.ndarray:
    """
    Generate forecasts using the model.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        input_text: Text representation of input time series
        forecast_steps: Number of steps to forecast
        alpha: Scaling parameter
        precision: Decimal precision
        temperature: Sampling temperature for generation
        max_length: Maximum sequence length
        
    Returns:
        NumPy array of forecasted values
    """
    # Create input including a trailing semicolon to indicate continuation
    if not input_text.endswith(";"):
        input_text += ";"
    
    # Tokenize the input
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
    
    # Count variables per timestep (for knowing how many tokens to generate)
    timestep_example = input_text.split(";")[0]
    vars_per_timestep = len(timestep_example.split(","))
    
    # Estimate tokens needed per forecast step
    # Each value has ~precision+2 tokens (digits + decimal + comma/semicolon)
    tokens_per_step = vars_per_timestep * (precision + 2)
    
    # Generate forecasts
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=input_ids.shape[1] + tokens_per_step * forecast_steps,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode the generated tokens
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract only the newly generated part
    generated_part = generated_text[len(input_text):]
    
    # Combine with input for context, but ensure we have a complete format
    complete_text = input_text + generated_part
    
    # Convert back to numeric
    predictions = text_to_numeric(complete_text, alpha, precision)
    
    # Return only the forecasted steps
    input_steps = len(input_text.split(";"))
    if len(predictions) > input_steps:
        return predictions[input_steps:input_steps+forecast_steps]
    else:
        # In case generation was incomplete
        return predictions[input_steps:]


if __name__ == "__main__":
    # Simple test to demonstrate usage
    import torch
    from qwen import load_qwen
    
    model, tokenizer = load_qwen()
    
    # Load test data
    train_texts, val_texts = load_and_preprocess("data/lotka_volterra_data.h5")
    
    # Print an example
    print("Example text representation:")
    print(train_texts[0][:100] + "...")
    
    # Convert back to numeric
    numeric_data = text_to_numeric(train_texts[0])
    print(f"\nConverted back to numeric, shape: {numeric_data.shape}")
    print(numeric_data[:3])
    
    # Generate a simple forecast
    short_example = train_texts[0].split(";")[:10]
    short_example = ";".join(short_example) + ";"
    forecast = generate_forecast(model, tokenizer, short_example, forecast_steps=5)
    print("\nExample forecast:")
    print(forecast)